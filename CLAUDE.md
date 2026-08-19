# CLAUDE.md

## Project Overview

**AMMO Sessions Server** — a FastAPI server that creates and manages isolated
AI-CLI coding sessions on GPUs. Each session is a git worktree with its own venv,
a slice of the host's GPUs, and a Claude Code or Codex CLI process running in a
hardened tmux + ttyd terminal that the user drives from a web UI.

Architecture, lifecycle internals, GPU management, security model, and the
full configuration reference live in `docs/ARCHITECTURE.md`. The AMMO
optimization workflow itself is documented in `docs/AMMO_DEEP_DIVE.md`.

## Development Environment

- NVIDIA GPU with CUDA capability 8.0+ and CUDA 12.0+
- Python 3.11+
- Linux (Ubuntu 22.04+ recommended)
- Docker with the NVIDIA container runtime, for the container path

GPU, Docker, and test commands must run on a machine with a GPU. There is no
CPU-only test path: the test suite imports `torch` and the session code queries
the driver.

## Common Commands

### Server Management

```bash
# Docker
./docker-build.sh && ./docker-run.sh --gpu all

# Restart if already running
docker kill ammo-server

# Without Docker
python main.py --host 0.0.0.0 --port 8000 --log-level info

# Health check
curl http://localhost:8000/health
```

### Testing

```bash
pytest tests/unit/                    # Fast, isolated
pytest tests/integration/             # Component interactions
pytest tests/e2e/                     # Full workflow (needs a running server)
pytest tests/playwright/              # Browser UI
pytest tests/                         # All tests
```

`tests/e2e/` and `tests/integration/` expect a server at `AMMO_SERVER_URL`
(default `http://localhost:8000`) and skip when it is unreachable.
`tests/playwright/` additionally needs `playwright` and `pytest-playwright`,
which are not in either requirements file.

**Session test timeouts:** E2E/integration session tests use
`SESSION_CREATION_TIMEOUT=300s` because the editable vLLM install can take 200s+
under concurrent load.

### Session-specific suites

```bash
# Lifecycle, GPU races, S3 restore, auto-pause
pytest tests/unit/test_gpu_allocation_races.py tests/unit/test_cross_pod_resume_flow.py \
       tests/unit/test_inactivity_auto_pause.py tests/unit/test_session_lifecycle_edges.py \
       tests/unit/test_session_state_s3_edges.py -v

# GPU reservation
pytest tests/unit/test_gpu_reservation_integration.py tests/unit/test_session_gpu_res_env.py \
       tests/integration/test_gpu_reservation_docker.py -v

# Security hardening
pytest tests/unit/test_api_key_middleware.py tests/unit/test_session_limit.py \
       tests/unit/test_websocket_ownership.py tests/unit/test_download_sanitization.py \
       tests/unit/test_download_ownership.py tests/unit/test_tmux_hardening.py \
       tests/unit/test_filesystem_isolation.py \
       tests/unit/test_dockerfile_validation.py -v
```

### Debugging

- Enable debug logging: `LOG_LEVEL=DEBUG python main.py`
- Session logs live under `{SESSION_DATA_DIR}/{session_id}/logs/`
- The tmux launcher for a session is written to `/tmp/{session_id}/start.sh` —
  read it to see the exact env and argv the CLI tool was exec'd with

## Key Files

- `app.py` — FastAPI server, API key middleware, session ownership validation
- `main.py` — entry point and CLI arguments
- `orchestration/session_manager.py` — session lifecycle, per-client limits, fork builds
- `orchestration/worktree_manager.py` — git worktree creation and repair
- `orchestration/cli_tool_manager.py` — CLI process config, template setup
- `orchestration/terminal_manager.py` — dedicated tmux server + ttyd per session
- `orchestration/session_state.py` — S3 persistence, download archive sanitization
- `orchestration/inactivity_monitor.py` — auto-pause
- `shared/gpu_resource_manager.py` — unified GPU allocation registry
- `shared/gpu_file_lock.py` — file-based GPU locks (`fcntl.flock`)
- `shared/session_models.py` — session data models, env var names and defaults
- `frontend/index.html` — LIGHTGRID web UI (Alpine.js + Tailwind)
- `ai_cli_session/` — the AMMO session template, in two parallel ports:
  `.claude/` (Claude Code) and `.codex/` (Codex CLI)

## The Session Template Is Rendered, Not Edited

`ai_cli_session/.claude/` and `ai_cli_session/.codex/` are rendered from the
canonical tree `ai_cli_session/_ammo_shared/` by
`scripts/render_ammo_variants.py`. Edit the canonical file, re-render, and
verify:

```bash
python scripts/render_ammo_variants.py           # render both variants
python scripts/render_ammo_variants.py --check   # fails on drift
```

Never hand-edit a rendered variant file that has a `_ammo_shared` source —
the parity test will fail.

## Testing Philosophy — "Always Works™"

### Core Principles
- "Should work" ≠ "does work"
- Untested code is a guess, not a solution
- Always run the actual code to verify it works

### 30-Second Reality Check
Before claiming something is fixed:
1. Did I run/build the code?
2. Did I trigger the exact feature I changed?
3. Did I see the expected result with my own observation?
4. Did I check for error messages?
5. Would I bet $100 this works?

### Test Requirements
- **API changes** — make the actual API call, not "the logic looks right"
- **Logic changes** — run the specific scenario
- **Integration tests** — start the server if not running, invoke the real test
- **Anything GPU-touching** — run it on a GPU host

### Phrases to Avoid
- "This should work now"
- "I've fixed the issue" (without testing)
- "Try it now" (without trying it myself)

## Related Documentation

- `README.md` — getting started and API overview
- `docs/ARCHITECTURE.md` — server architecture and configuration reference
- `docs/AMMO_DEEP_DIVE.md` — how the AMMO optimization workflow works
