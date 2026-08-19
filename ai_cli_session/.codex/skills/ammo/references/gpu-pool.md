# GPU Pool Reservation

All GPU commands must use the pool reservation pattern. This ensures GPU isolation across concurrent agents: the session owns a pool of GPUs, several agents work in that pool at the same time, and a reservation is the only thing that keeps two of them off the same device. Read this file before you run any GPU command, and again before you size an E2E sweep. It gives you the reserve command, the GPU count each task type needs, who releases the GPUs afterwards, and what to do when the pool is full.

## Reservation Pattern

Reserve first, then run the real command against the device list the reserve call prints:

```bash
CVD=$(.venv/bin/python .codex/skills/ammo/scripts/gpu_reservation.py reserve \
  --num-gpus N --session-id {gpu_session_id} --no-auto-release) && \
  CUDA_VISIBLE_DEVICES=$CVD <command>
```

- `--session-id {gpu_session_id}`: the spawner supplies an ID unique per active agent (campaign + op + agent nonce). Never share the logical `op_id` between an implementer and delegates.
- `--no-auto-release`: marks a persistent child reservation, so PostToolUse does not release it at command completion. Use only the exact owner injected by `SubagentStart`; `SubagentStop` releases that owner. Command-scoped reservations omit this flag and release after the observed command.

**Session-id rule:** treat the exact injected owner as opaque and pass it unchanged. It may include a colon; do not parse, normalize, share, or reconstruct it.

If you minted an ad-hoc id, always release it explicitly before returning:

`.venv/bin/python .codex/skills/ammo/scripts/gpu_reservation.py release-session --session-id {gpu_session_id} || true`

Pass `--lease-hours 2` (or higher) on the reserve call for long sweeps or nsys captures that run >10 min, so a run that legitimately exceeds the **15 min default** lease does not expire mid-run. The script does NOT auto-extend leases for nsys.

## How Your GPUs Get Released

GPUs auto-release when a command-scoped reservation completes. For persistent child reservations, the Codex `SubagentStart` hook injects the deterministic owner `<session_id>:<agent_id>`; use that owner with `--no-auto-release`, and the `SubagentStop` hook releases it exactly. Explicit `release-session` calls, lease expiry, and final Stop cleanup with `--include-children` are the recovery paths.

## How Many GPUs to Reserve

| Task | `--num-gpus` | Notes |
|------|-------------|-------|
| Kernel benchmarks | 1 | Single GPU sufficient |
| Micro-experiments (debate) | 1 | Keep brief to minimize contention |
| Static analysis (`ncu --query-metrics`) | 1 | No kernel execution |
| nsys single-kernel traces | 1 | Existing binary only |
| E2E sweeps | `{tp*dp}` | Match total parallel world size (TP × DP) from target.json |
| Parallel tracks / debate experiments | 1-N | Pool may exceed TP×DP; use remaining GPUs for concurrent tracks |

An E2E sweep needs `N = TP × DP` GPUs reserved as one contiguous block, because each DP replica runs its own TP group. Reserve only TP and you starve the other DP replicas — torchrun spawns TP×DP ranks and the sweep deadlocks waiting for the missing workers. For E2E sweeps, `gpu_reservation.py` allocates a contiguous block of exactly `TP × DP` GPUs (the vLLM world size); if no contiguous block of `TP × DP` GPUs is free, the command fails — retry after other agents release.

The session environment exposes `AMMO_TP_SIZE` and `AMMO_DP_SIZE` so you can compute `N` without reopening `target.json`:

```bash
NUM_GPUS=$(( ${AMMO_TP_SIZE:-1} * ${AMMO_DP_SIZE:-1} ))
```

The pool is usually larger than one world size: its size equals the `gpu_count` requested at session creation, which may exceed `TP × DP`. The extra GPUs enable parallel experiment tracks — multiple agents can reserve `--num-gpus 1` concurrently for kernel benchmarks while one agent holds `TP × DP` for an E2E sweep, and the pool GPUs outside that E2E block stay available for other agents' parallel kernel work. Discover pool size via `gpu_reservation.py status` or:

```bash
POOL_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
```

## When the Pool Is Full

If the pool is exhausted, the reserve command fails immediately with a `ReservationError`. Yield control and retry on a later turn; do not hold the foreground in a long sleep loop. Continue useful CPU-only work while waiting when possible.

For CPU-only commands (file reads, roofline math, ISA inspection), no reservation is needed.

## Never Kill Another Agent's Processes

**Never kill processes on GPUs you don't own.** Multiple agents share the same GPU pool, so a process you did not start is somebody's live work:

- Only terminate processes YOU started. Do not `kill`, `pkill`, or `killall` processes belonging to other agents.
- If `nvidia-smi` shows a process on "your" GPU that you didn't start, it belongs to another agent whose reservation overlaps. Wait for them to finish — do not kill it.
- If you suspect a zombie/leaked process, report it to the orchestrator rather than killing it yourself.

## Diagnostics (Orchestrator Only)

Only the orchestrator runs these:

- `scripts/gpu_reservation.py status --table` — print current reservation state as a table (one row per pool GPU)
- `scripts/gpu_reservation.py force-clear --all --session-id <crashed_session_id>` — clear stale reservations after crashes (target and session flags are both required; `--gpu-ids 0,1` narrows the target, `--force-no-session` is the emergency override)
