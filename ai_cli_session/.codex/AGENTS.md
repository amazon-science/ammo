# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Communication Rules (all agents, including the orchestrator)

1. Speak in plain, concise English. Put simplicity first: prefer the simplest word and the shortest sentence that keeps the meaning.
2. Write technical documents in ASD-STE100 Simplified Technical English: short sentences (max 20 words for instructions, 25 for descriptions), one instruction per sentence, active voice, one meaning per word.
3. No preambles and no closing pleasantries. Start with the answer or the action. Stop when done — no "Let me know if…", no recaps of what you just did.

## Session Information

- **Session ID**: `{session_id}`
- **Repository**: `{repo_name}`
- **Branch**: `{branch}`
- **GPUs**: `{gpu_ids}`

## Workspace Structure

```
/data/sessions/{session_id}/
├── worktree/# Git worktree (your working directory)
│ ├── .codex/ # Codex configuration
│ └── ...# Repository files
├── logs/# Session logs
└── session.json # Session metadata
```

## Available Tools

### GPU Access

> **IMPORTANT: Understanding GPU Visibility**
>
> - `nvidia-smi` shows ALL GPUs on the system (it queries the driver directly)
> - **This does NOT reflect your session's GPU access!**
> - CUDA applications (PyTorch, CuPy, Triton) respect `CUDA_VISIBLE_DEVICES`
>
> **To check your actual GPU access, use:**
> ```bash
> python3 -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"
> ```
> If this shows `0`, your session has no GPU access (gpu_count=0 at creation).

This session has access to the following GPUs via `CUDA_VISIBLE_DEVICES`: `{gpu_ids}`
- If `{gpu_ids}` is empty, you have NO GPU access for CUDA workloads
- GPUs are locked to this session while active
- Use `ncu` (Nsight Compute) for kernel profiling (if GPUs allocated)

#### Parallelism Size Env Vars

When the session was created with explicit tensor/data parallelism, two env vars are set:

- `AMMO_TP_SIZE` — tensor-parallel size from `target.json`
- `AMMO_DP_SIZE` — data-parallel size from `target.json`

Each DP replica runs its own TP group, so an E2E sweep needs `TP × DP` GPUs reserved as one contiguous block. Compute the count without reopening `target.json`:

```bash
NUM_GPUS=$(( ${AMMO_TP_SIZE:-1} * ${AMMO_DP_SIZE:-1} ))
```

> **Pool size vs. model world size:** The session's GPU pool may be larger than
> `TP × DP` when extra GPUs were requested for parallel experiment tracks.
> Discover total pool size:
> ```bash
> POOL_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
> ```
> Or use `gpu_reservation.py status` to see allocated vs. free within the pool.
> `NUM_GPUS` above is the world size for one E2E sweep — not the pool ceiling.

If either var is unset, the session was created without explicit parallelism metadata (legacy path). Fall back to reading `target.json` directly.

### Git Operations

- This is an isolated git worktree from the base repository
- Changes are local to this session
- You can commit changes, but they won't affect the upstream repository
- The worktree is on branch `session/{session_id}`

## Session Lifecycle

1. **Working**: Make changes to code in the worktree
2. **Pause**: Session can be paused (saves state, releases GPUs)
3. **Resume**: Session can be resumed later with full state
4. **Terminate**: Session cleanup removes worktree and releases resources

## Session Environment

Use the pre-built `.venv` for Python commands. Never create another environment or install packages unless the user explicitly authorizes that change.
Report missing environments and import failures to the session owner.

Run tests that exercise the change. AMMO's required gates remain binding; add further tests only for a change, failure, or unresolved question.

## Sandbox Restrictions

This session runs in a sandboxed environment with the following restrictions:

### Allowed Operations

- **File Access**: Full read/write access within this worktree directory
- **Temp Files**: Full access to `/tmp` and `/tmp/{session_id}/`
- **GPU Tools**: `nvidia-smi`, `ncu` (Nsight Compute), `nsys` (Nsight Systems)
- **Development**: `python`, `git`, `pytest`
- **Build Tools**: `make`, `gcc`, `nvcc`, `cmake`, `ninja`
- **Utilities**: `curl` (localhost only), `ls`, `cat`, `grep`, `find`, etc.

### Denied Operations

- **Remote Access**: `ssh`, `scp`, `rsync` (remote connections blocked)
- **System Admin**: `sudo`, `su`, `apt`, `yum`, `systemctl`
- **Container Ops**: `docker`, `kubectl`
- **Cloud CLI**: `aws` CLI (use server APIs instead)
- **Other Sessions**: Cannot read/write other session directories
- **System Files**: Cannot write to `/etc`, `/root`, `/var`, `/usr`

### Security Notes

- GPU access is managed at the server level before session starts
- API keys are injected via environment variables (not stored in files)
- Session data is isolated from other sessions on the same server

## Source Builds and Validation

Read `.codex/skills/ammo/references/session-development.md` when a change needs an environment check or incremental C++/CUDA compilation.
Read `.codex/skills/ammo/references/gpu-pool.md` before GPU work. Use the active role and stage references for campaign validation.
