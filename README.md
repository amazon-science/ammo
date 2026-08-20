# AMMO Sessions Server

**AMMO (Agentic Model-on-Machine Optimizer)** is a multi-agent system that
autonomously optimizes vLLM GPU kernels for a specific deployment — model,
hardware, dtype, parallelism — over multi-hour unattended campaigns. This
server creates and manages the isolated sessions those campaigns run in.

Each session is a git worktree of vLLM with its own venv, a slice of the
host's GPUs, and an AI CLI (Claude Code or Codex) running in a hardened
terminal you drive from your browser.

## What You Need

- **Hardware**: Linux host with an NVIDIA GPU (CUDA capability 8.0+ — A100,
  L40S, H100, H200, B200), CUDA 12.0+, and the NVIDIA container toolkit.
- **An AI CLI subscription or API key**: sessions run Claude Code
  (`ANTHROPIC_API_KEY`) or Codex (`OPENAI_API_KEY` / an existing
  `~/.codex/auth.json` login). The AMMO campaign template pins a
  high-effort frontier model and spawns agent teams; a full optimization
  campaign runs for hours and consumes API credits accordingly. The
  shipped model-per-agent assignments are the recommended setup, and every
  pin is user-changeable — see
  [Which Model Runs Which Agent](docs/AMMO_DEEP_DIVE.md#which-model-runs-which-agent)
  for the roster, the reasoning, and concrete ways to reduce cost.
- **Disk**: the Docker image is ~42 GB (CUDA toolchain + a Python-ready
  vLLM checkout).

### Host prerequisites for GPU profiling

Campaigns drive NVIDIA Nsight profilers (`ncu`, `nsys`) from unprivileged
session users. On a stock host the NVIDIA driver restricts GPU performance
counters to admin users, so `ncu` fails with `ERR_NVGPUCTRPERM` and the
profiling stages of a campaign cannot run. Enable counters once per host:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
  sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo sysctl -w kernel.perf_event_paranoid=2
# reboot (or reload the nvidia kernel modules) for the modprobe option to apply
```

## Quick Start

```bash
git clone <this-repo-url> ammo-server
cd ammo-server

# Make your AI CLI credentials available to sessions
export ANTHROPIC_API_KEY=sk-ant-...   # for Claude Code sessions
# and/or: log in to Codex once so ~/.codex/auth.json exists

./docker-build.sh && ./docker-run.sh --gpu all

# Verify
curl http://localhost:8000/health
# {"status": "healthy", "gpu_available": true, ...}
```

Then open **http://localhost:8000/ui**, click **New Session**, pick a
HuggingFace model (the server auto-suggests TP/DP/dtype), and you get a
browser terminal with the AI CLI running inside the session's vLLM worktree.
To start an optimization campaign, tell it:

```
Use $ammo for model_id=Qwen/Qwen3-8B TP=1 dtype=bf16
```

The campaign runs unattended from there — profiling, debating candidates,
writing kernels, validating, and shipping only what survives end-to-end
measurement. The final deliverable is a `REPORT.md` you can view from the UI
or fetch via `GET /sessions/{id}/report`.

### Without Docker

```bash
pip install -r requirements.txt -r requirements.server.txt
python main.py                 # serves on :8000
```

The local path expects `git`, `tmux`, `ttyd`, and the AI CLIs on the host;
the Docker image provisions all of this, so prefer it.

## Using the Server

### Web UI

`http://localhost:8000/ui` is the primary interface: create sessions from a
modal with live HuggingFace model search, watch up to two terminals
side-by-side, pause/resume/terminate, download session archives, and read
optimization reports. Set `AMMO_API_KEY` to put the UI and API behind a
login.

### API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sessions` | Create a session (503 if insufficient GPUs, 429 at the per-client limit) |
| `GET` | `/sessions` | List sessions |
| `GET` | `/sessions/{id}` | Get session info |
| `POST` | `/sessions/{id}/pause` | Pause (frees GPUs; state preserved) |
| `POST` | `/sessions/{id}/resume` | Resume (conversation auto-continues) |
| `DELETE` | `/sessions/{id}` | Terminate and clean up |
| `GET` | `/sessions/{id}/report` | The campaign's REPORT.md |
| `POST` | `/sessions/{id}/prepare-download` → `GET .../download` | Sanitized ZIP of the worktree |
| `GET` | `/health` | GPU + vLLM build info (no auth) |
| `GET` | `/api/hf-model-config/{model_id}` | Suggested TP/DP/dtype for an HF model |

```python
import requests

r = requests.post("http://localhost:8000/sessions", json={
    "model_name": "Qwen/Qwen3-8B",
    "dtype": "bf16",
    "gpu_count": 1,
    "cli_tool": "claude",
    "initial_prompt": "Use $ammo for model_id=Qwen/Qwen3-8B TP=1 dtype=bf16",
}, headers={"X-Client-ID": "my-client"})
print(r.json()["session_id"])
```

### Session Lifecycle

```
CREATE → ACTIVE ↔ PAUSED → TERMINATED
```

Sessions survive browser disconnects (the terminal runs in tmux). Pause
frees the GPUs and preserves the worktree; resume restarts the CLI with the
conversation intact. Idle sessions auto-pause after
`SESSION_INACTIVITY_TIMEOUT_MINS` (default 24 h).

### Custom vLLM Forks

Point a session at your own vLLM fork by passing `vllm_fork_url` (and
`vllm_fork_token` for private forks) at create time — the server clones it
and runs a full source build before the session opens. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#custom-vllm-forks).

## Configuration

The variables you are most likely to set:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Claude Code auth, inherited by sessions |
| `OPENAI_API_KEY` / `CODEX_AUTH_JSON_PATH` | — / `~/.codex/auth.json` | Codex auth |
| `AMMO_API_KEY` | — | Gate the API and UI behind a key (unset = open) |
| `MAX_SESSIONS_PER_CLIENT` | `8` | Concurrent active sessions per client |
| `SESSION_S3_BUCKET` | — | Optional: persist paused sessions to S3 (enables cross-host resume) |
| `SESSION_INACTIVITY_TIMEOUT_MINS` | `1440` | Auto-pause idle sessions |

The full configuration reference is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#configuration-reference). S3 is
optional; on a single machine without S3 configured, everything works locally.

## Troubleshooting

| Problem | Symptom | Fix |
|---------|---------|-----|
| GPU not found | `"gpu_available": false` | Check `nvidia-smi` and the NVIDIA container toolkit |
| Insufficient GPUs | `POST /sessions` → 503 | Free GPUs (pause/terminate sessions) |
| Session limit | `POST /sessions` → 429 | Pause/terminate, or raise `MAX_SESSIONS_PER_CLIENT` |
| Port in use | Server won't start | `python main.py --port 8001` |
| Anything else | — | `LOG_LEVEL=DEBUG`, session logs in `{SESSION_DATA_DIR}/{id}/logs/` |

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the server is built:
  components, lifecycle internals, two-layer GPU locks, security model, S3
  persistence, and full config reference.
- [docs/AMMO_DEEP_DIVE.md](docs/AMMO_DEEP_DIVE.md) — how the AMMO
  optimization campaign works: the 7-stage pipeline, agent roles, audit
  gates, and the hook system that keeps agents honest over 8-hour runs.
- [CLAUDE.md](CLAUDE.md) — build/test commands and conventions for working
  on this codebase.

## Development

```
ammo-server/
├── app.py                # FastAPI app, auth middleware, WebSocket proxy
├── main.py               # Entry point
├── orchestration/        # Session lifecycle, worktrees, terminals, GPU allocation
├── shared/               # Models, GPU resource manager, file locks
├── frontend/             # LIGHTGRID web UI (single-page)
└── ai_cli_session/       # The AMMO session template (skills, agents, hooks)
```

```bash
pytest tests/unit/         # fast, isolated (needs a GPU host — the suite imports torch)
pytest tests/              # everything (integration/e2e need a running server)
```

## Citation and Paper Artifacts

AMMO is described in the paper *"A Fast Kernel Is Not a Ship: Field Lessons
from Agentic Optimization in vLLM"* (under submission to the IAAI-27 track
of AAAI-27). Citation metadata is in [CITATION.cff](CITATION.cff):

```bibtex
@article{ammo2026,
  title  = {A Fast Kernel Is Not a Ship: Field Lessons from Agentic
            Optimization in vLLM},
  author = {Huang, Jin and Zhang, Shuai and Gai, Jiading and Patil, Vihang
            and Budhathoki, Kailash and Khetan, Ashish and Dabeer, Onkar},
  year   = {2026},
  note   = {Under submission}
}
```

The ablation evidence archive referenced by the paper's appendix
(zstd-compressed, ~430 MB: session transcripts, verification ledger, and
per-run investigation reports) is published as a versioned **release asset**
on this repository; each release lists the archive's SHA-256 in its notes.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for
information on reporting security issues.

## License

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Released under the Creative Commons Attribution-NonCommercial 4.0 International
License (CC BY-NC 4.0). See [LICENSE](LICENSE.txt) and [NOTICE](NOTICE.txt).

The session template bundles the GSM8K dataset, which is MIT-licensed and
carries its own copyright, and adapts GSM8K evaluation helpers from vLLM
(Apache-2.0). See
[ai_cli_session/.claude/skills/ammo/data/ATTRIBUTION.md](ai_cli_session/.claude/skills/ammo/data/ATTRIBUTION.md).
