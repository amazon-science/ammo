# Architecture

How the AMMO Sessions Server is built: components, session lifecycle, GPU
management, security, persistence, and deployment. For setup and first-run
instructions, see the [README](../README.md).

## System Overview

```
Client → API Key Middleware → FastAPI (app.py) → Session Manager
         (if AMMO_API_KEY set)                        ↓
                        ┌──────────────────┬──────────┴────────┬──────────────────┐
                        ↓                  ↓                   ↓                  ↓
                 Worktree Manager    GPU Allocation      CLI Tool Manager   Terminal Manager
                 (git worktrees)  (server + session)  (Claude/Codex launch) (tmux + ttyd)
```

Each session is an isolated git worktree of a vLLM checkout, with its own venv,
a slice of the host's GPUs, and an AI CLI process (Claude Code or Codex) running
inside a dedicated tmux server exposed through ttyd over WebSocket.

### Components

| Component | File | Responsibility |
|-----------|------|----------------|
| API server | `app.py` | FastAPI routes, API key middleware, ownership validation, WebSocket proxy |
| Entry point | `main.py` | CLI arguments, server startup |
| Session Manager | `orchestration/session_manager.py` | Lifecycle (create/pause/resume/terminate), per-client limits, fork builds |
| Worktree Manager | `orchestration/worktree_manager.py` | Git worktree creation, cleanup, S3 restore repair |
| CLI Tool Manager | `orchestration/cli_tool_manager.py` | CLI process config, template setup, launch argv |
| Terminal Manager | `orchestration/terminal_manager.py` | Dedicated tmux server + ttyd per session, port allocation |
| Session State | `orchestration/session_state.py` | S3 persistence, download archive sanitization |
| Inactivity Monitor | `orchestration/inactivity_monitor.py` | Auto-pause on idle |
| Checkpoint Manager | `orchestration/checkpoint_manager.py` | S3 sync on WebSocket disconnect |
| GPU Resource Manager | `shared/gpu_resource_manager.py` | Unified GPU allocation registry |
| GPU File Locks | `shared/gpu_file_lock.py` | File-based GPU locks (`fcntl.flock`) |
| Session Models | `shared/session_models.py` | Pydantic models, env var names and defaults |
| Web UI | `frontend/index.html` | LIGHTGRID single-page app (Alpine.js + Tailwind) |
| Session template | `ai_cli_session/` | The AMMO workspace copied into every session |

### The Session Template

`ai_cli_session/` holds everything copied into a new session's worktree: the
AMMO skill, agent definitions, enforcement hooks, and state schemas. It exists
in two parallel ports — `.claude/` (Claude Code) and `.codex/` (Codex CLI) —
both rendered from one canonical source tree, `ai_cli_session/_ammo_shared/`,
by `scripts/render_ammo_variants.py`. `render_ammo_variants.py --check` fails
if the variants drift from the canonical files.

How the AMMO optimization campaign itself works (stages, agents, gates, hooks)
is documented in [AMMO_DEEP_DIVE.md](AMMO_DEEP_DIVE.md).

## Session Lifecycle

```
CREATE → ACTIVE ↔ PAUSED → TERMINATED
   └→ BUILDING → ACTIVE | FAILED     (custom vLLM forks only)
```

1. **Create** — provision an isolated git worktree, copy the CLI template,
   set up the venv, allocate GPUs. Enforces the per-client session limit
   (429) and returns 503 when the server has insufficient GPUs.
2. **Active** — the user drives the CLI through the web terminal. The tmux
   config sets `destroy-unattached off`, so the session and all agent-team
   panes survive a browser disconnect; reconnecting reattaches via
   `ensure_terminal_healthy` → `restart_ttyd_with_tmux_attach`.
3. **Pause** — two distinct semantics:
   - *User-initiated* (`POST /sessions/{id}/pause`): `stop_terminal()`
     explicitly kills the tmux session. Agent teams are lost; `--continue`
     on resume does not restore them. The worktree is preserved and synced
     to S3 when configured.
   - *Server restart*: `_load_sessions` may mark a previously ACTIVE session
     PAUSED while its tmux session is still alive in the container. Resume
     then reattaches instead of restarting the CLI, so teams are preserved.
4. **Resume** — restarts the CLI with `--continue` (Claude) or
   `codex resume --last` (Codex, when history exists). S3 restore
   calls `repair_worktree_linkage()` to fix the worktree's
   `.git` pointer, and renames the claude-config project dir to match the new
   worktree path. If the terminal fails to start, the session stays PAUSED
   and GPUs are released — never stuck ACTIVE with leaked GPUs.
5. **Terminate** — removes the worktree, tmux socket, GPU reservation state
   dir, S3 data, and all other resources.

### Auto-Resume of Conversations

Claude Code state lives in a per-session `CLAUDE_CONFIG_DIR`
(`{SESSION_DATA_DIR}/{id}/claude-config/`), pre-seeded with a `.claude.json`
that trusts the worktree and skips onboarding. On resume the CLI starts with
`--continue`, so the conversation picks up where it left off. Codex state
lives in a per-session `CODEX_HOME`; `codex-home/auth.json` is never
checkpointed or included in downloads.

### Directory Layout

```
/data/
├── repos/                          # Base repository checkouts
│   ├── vllm/                       # Baked-in vLLM clone
│   └── forks/{sha256(url)[:16]}/   # Per-fork base repos (custom forks)
│
└── sessions/
    ├── .base_venvs/vllm/           # Cached venv for hardlinking
    └── {session_id}/
        ├── session.json            # Metadata
        ├── claude-config/          # Claude Code state (or codex-home/)
        ├── worktree/               # Git worktree (+ .venv hardlinked in)
        └── logs/
```

S3 layout (when `SESSION_S3_BUCKET` is set):

```
s3://{bucket}/{SESSION_S3_PREFIX}/{session_id}/
├── session.json
└── worktree + claude-config/codex-home (aws s3 sync)
```

`.git` is included in the S3 copy so `repair_worktree_linkage()` can rebuild
worktree pointers after a cross-host restore. venv and build artifacts are
excluded and recreated on resume.

## GPU Management

### Two-Layer GPU Locks

The server controls which physical GPUs a session owns; agents inside the
session subdivide that pool:

- **Layer 1 — server lock** (`shared/gpu_resource_manager.py` +
  `shared/gpu_file_lock.py`): allocates physical GPU IDs to a session at
  create/resume, released on pause/terminate. Lock files (`gpu_{N}.lock`,
  `fcntl.flock`) are discovered on init, with a `torch.cuda.device_count()`
  fallback that creates them.
- **Layer 2 — session reservation**
  (`ai_cli_session/.claude/skills/ammo/scripts/gpu_reservation.py`): agents
  inside an active session call `gpu_reservation.py reserve` to claim GPUs
  from the session's pool. State lives in `AMMO_GPU_RES_DIR`
  (`/tmp/ammo_gpu_res_{session_id}`), injected by the session manager and
  mirrored into `settings.local.json` so subagents inherit it. A
  `PostToolUse/Bash` hook auto-releases reservations after each tool use.

### GPU Visibility

Sessions are isolated via `CUDA_VISIBLE_DEVICES` in the ttyd process env:
`gpu_count=0` sets `-1` (no GPU access); `gpu_count>0` sets the allocated IDs.
`gpu_reservation.py` reads `CUDA_VISIBLE_DEVICES` to map logical → physical
IDs — `nvidia-smi` queries the driver directly and always shows every host
GPU, so it cannot be used for isolation checks. Verify actual access with:

```bash
python3 -c "import torch; print(torch.cuda.device_count())"
```

### Parallelism Env Vars

`AMMO_TP_SIZE` / `AMMO_DP_SIZE` are injected into the ttyd env on
create/resume and mirrored into `settings.local.json` so subagents inherit
them. Agents compute `NUM_GPUS=$(( AMMO_TP_SIZE * AMMO_DP_SIZE ))` without
reopening `target.json`. Both keys are omitted when `tp_size=0` (the legacy
"unknown" sentinel) so shell guards like `[ -z "${AMMO_TP_SIZE:-}" ]` still
fire.

## Security Model

### API Key Authentication

When `AMMO_API_KEY` is set, middleware gates access:

- **Protected** (prefix match): `/sessions`, `/api/`, `/docs`,
  `/redoc`, `/openapi.json`
- **Open**: `/health`, plus the exact paths `/ui` (login page) and
  `/api/changelog`
- **Key lookup order**: `Authorization: Bearer`, `X-API-Key` header,
  `ammo_api_key` cookie, `?token=` query param (WebSocket/terminal)
- Unset means auth is disabled (dev mode).

### Session Ownership

Sessions are isolated per client via the `X-Client-ID` header (a UUID the web
UI generates and persists in localStorage). The terminal proxy, download, and
pause/resume endpoints all validate ownership; a wrong client gets 404. With
`AMMO_API_KEY` set, enforcement is strict — a null client ID sees only legacy
sessions. Client IDs are isolation, not authentication: for multi-tenant
production, put real auth in front.

### Other Controls

- **Per-client limit** — `MAX_SESSIONS_PER_CLIENT` (default 8); paused
  sessions do not count. Returns 429 when exceeded.
- **Tmux hardening** — a dedicated tmux server per session (`-S` socket
  path) with a hardened config: prefix moved to `C-Space` (freeing `C-b` for
  application pass-through), `unbind-key -a -T prefix` (mouse bindings
  preserved), no status bar. The per-session socket owns the isolation; there
  is no `Bash(tmux:*)` deny rule, so agents can close their own stuck panes.
- **Download sanitization** — prepared archives strip `.claude/`,
  `claude-config/`, `codex-home/`, `CLAUDE.md`, and any encrypted fork
  token.
- **Filesystem isolation** — session processes run as `session_user`
  (uid 1000); the server runs as root and drops privileges for them.
- **Graceful failure handling** — session and build failures are reported as
  state, not HTTP 500. Only genuine server errors return 500.

## Custom vLLM Forks

A create request may include `vllm_fork_url` (and `vllm_fork_token` for
private forks) to build the session from a user-owned fork instead of the
baked-in repo:

- **URL allowlist** (`shared/fork_url_validator.py`): `https://github.com/...`
  only — no other hosts, no SSH, no localhost/IP. Invalid URLs are rejected
  at `POST /sessions`.
- **Token encryption** (`shared/fork_token_crypto.py`): the token is
  Fernet-encrypted at rest. Supplying one requires `AMMO_FORK_TOKEN_KEY` on
  the server; the raw token never touches disk or argv — it reaches `git`
  via `GIT_ASKPASS` only.
- **Async BUILDING status**: a fork needs a full C++/CUDA source build
  (`VLLM_USE_PRECOMPILED=0`), so `create_session` allocates GPUs, sets
  `BUILDING`, and returns immediately. A background task clones the fork,
  opens a build-console terminal that tails the live log, and builds with
  retry-once. On success the console `exec`s the CLI and the session becomes
  ACTIVE; on a second failure the session is marked FAILED and GPUs are
  released. GPUs are held through the build.
- **Shared fork bases**: each fork is cloned once into a hash-keyed base repo
  under `{SESSION_REPOS_DIR}/forks/`, reused across sessions on the same
  fork (per-fork `fcntl` clone lock), and removed when the last session
  using it terminates.
- **Resume rebuild**: venv and build artifacts are not synced to S3, so
  resuming a fork session re-clones and rebuilds. ccache is persisted on
  pause and restored first, so a warm rebuild takes ~1–3 min versus
  ~15–20 min cold.

## vLLM Environment and venv Caching

The Docker image ships a Python-ready vLLM checkout (no C++ build — that
keeps the image ~42 GB instead of ~62 GB) plus a `CMakeUserPresets.json` for
on-demand kernel builds inside a session:

```bash
source .venv/bin/activate
cmake --preset release
cmake --build --preset release --target install
```

venv provisioning is cached for speed:

| Operation | Time |
|-----------|------|
| First session on the main branch (populates `.base_venvs/vllm`) | ~25 s |
| Subsequent sessions (hardlinks from the cache) | ~10 s |
| First session on a non-main branch (fresh venv, no cache) | ~5–10 min |
| On-demand C++ build, cold / ccache-warm | ~15–20 min / ~1–3 min |

The cache is validated by a version marker (torch version + vLLM commit);
stale caches rebuild automatically. Mutable files (`bin/`, `.pth`,
`.dist-info/`) are copied rather than hardlinked, and shebangs/activate
scripts are rewritten per session.

## S3 Persistence and Cross-Host Restore

All of this is opt-in: without `SESSION_S3_BUCKET`, sessions live only on
local disk and everything else works.

- **Pause sync** — pause syncs the session to S3 via `aws s3 sync`.
- **Checkpoint on disconnect** — when a terminal WebSocket disconnects, a
  60-second delayed checkpoint syncs to S3 without pausing; it is cancelled
  on reconnect or shutdown.
- **Startup discovery** — on boot, the server lists S3 and registers any
  sessions not already on local disk as PAUSED (terminated/failed sessions
  are skipped, local state wins). This is how sessions survive a server
  restart or move to another host that shares the same S3 bucket.
- **Cross-host repair** — after an S3 restore, `repair_worktree_linkage()`
  fixes the worktree's `.git` file and registers it in the base repo, and
  the claude-config project dir is renamed to the new worktree path.

## Reverse Proxy Notes

The server is local-only: it does not discover peer servers or forward session
requests. If you put it behind a reverse proxy, configure a WebSocket-capable
proxy path for `/sessions/{id}/terminal/ws` and use an idle timeout well above
60 seconds so idle terminals stay connected. A client disconnect may trigger a
delayed S3 checkpoint when S3 is configured.

## Graceful Shutdown

Shutdown completes in about a second (it previously could hang for minutes
in WebSocket loops):

1. Terminal Manager — kill ttyd processes (breaks WebSocket loops)
2. WebSocket Proxy — cancel `_active_websocket_tasks` (5 s timeout)
3. Checkpoint Tasks — cancel `_pending_checkpoint_tasks`
4. Background Tasks — cancel periodic cleanup
5. Inactivity Monitor — stop the monitoring loop

## Configuration Reference

### CLI Arguments (`main.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | 0.0.0.0 | Server host address |
| `--port` | 8000 | Server port |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |
| `--reload` | false | Auto-reload for development |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Auth for Claude Code inside sessions (inherited from the server env) |
| `OPENAI_API_KEY` | — | Seeds Codex auth when no `auth.json` is available |
| `CODEX_AUTH_JSON_PATH` | `~/.codex/auth.json` | Pre-existing Codex login to copy into sessions |
| `AMMO_API_KEY` | — | API key for session/UI endpoint auth (unset = auth disabled) |
| `MAX_SESSIONS_PER_CLIENT` | `8` | Max concurrent active sessions per client (paused excluded) |
| `GPU_TYPE` | auto | Override GPU type detection (`nvidia-smi`) |
| `SESSION_DATA_DIR` | `/data/sessions` | Local session storage |
| `SESSION_REPOS_DIR` | `/data/repos` | Base repos and fork clones |
| `SESSION_TEMPLATES_DIR` | `/data/templates` | CLI tool templates |
| `SESSION_TERMINAL_BASE_PORT` | `8001` | First port for ttyd instances |
| `SESSION_MAX_TERMINAL_PORTS` | `99` | Max concurrent terminals |
| `SESSION_INACTIVITY_TIMEOUT_MINS` | `1440` | Auto-pause after inactivity |
| `SESSION_S3_BUCKET` | — | S3 bucket for session persistence (optional) |
| `SESSION_S3_PREFIX` | `sessions` | S3 key prefix |
| `SESSION_S3_TTL_DAYS` | `30` | S3 object TTL for stale sessions |
| `AMMO_FORK_TOKEN_KEY` | — | Fernet key enabling private-fork tokens |
| `LOG_LEVEL` | INFO | Logging verbosity |

Environment injected *into* sessions by the server (not set by operators):
`CUDA_VISIBLE_DEVICES`, `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `HF_HOME`,
`AMMO_GPU_RES_DIR`, `AMMO_TP_SIZE`, `AMMO_DP_SIZE`.

## Implementation Notes

Hard-won details worth knowing before touching the terminal path:

1. Env vars are passed to ttyd both via process env and `/usr/bin/env` in the
   command for reliable propagation; `CUDA_VISIBLE_DEVICES` must be in ttyd's
   process env for GPU restriction to work.
2. `IS_SANDBOX` must be the string `"1"` (not `"true"`) for
   `--dangerously-skip-permissions` to work.
3. tmux 3.5a is built from source in the Dockerfile — Ubuntu 22.04 ships
   3.2a, which lacks `allow-passthrough` and reliable OSC 52 clipboard
   support needed for copy/paste in ttyd/xterm.js.
4. S3 delete uses `aws s3 rm --recursive` to handle >1000 objects (boto3
   pagination limit).
5. The tmux launcher for a session is written to `/tmp/{session_id}/start.sh`
   — read it to see the exact env and argv the CLI was exec'd with.
