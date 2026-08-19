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

## Best Practices

1. **Save frequently**: Commit your changes to preserve them
2. **Check GPU usage**: Before running GPU-intensive tasks, verify GPU availability
3. **Use the existing .venv**: This session has a pre-built virtual environment at `.venv/`. Always run `source .venv/bin/activate` before Python commands. NEVER create a new venv or run `pip install` / `uv pip install` unless explicitly instructed by the user.
4. **Monitor resources**: Check memory and disk usage periodically

## Sandbox Restrictions

This session runs in a sandboxed environment with the following restrictions:

### Allowed Operations

- **File Access**: Full read/write access within this worktree directory
- **Temp Files**: Full access to `/tmp` and `/tmp/{session_id}/`
- **GPU Tools**: `nvidia-smi`, `ncu` (Nsight Compute), `nsys` (Nsight Systems)
- **Development**: `python`, `pip`, `git`, `pytest`
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


# vLLM Development Guide

This section provides detailed information for vLLM kernel development, extraction, and benchmarking.

## Project Overview

vLLM is a fast and easy-to-use library for LLM inference and serving. It provides state-of-the-art serving throughput using PagedAttention for efficient KV cache management, continuous batching, and optimized CUDA kernels.

## Key Commands

### Build and Installation

#### Check Setup Status

Run these checks to determine what's available:

```bash
# Check if Python environment is ready
test -d .venv && echo "Python venv: READY" || echo "Python venv: MISSING"

# Check if CMake presets are available
test -f CMakeUserPresets.json && echo "CMake presets: READY" || echo "CMake presets: MISSING"

# Check if C++ build has been run
test -d cmake-build-release && echo "CMake configured: YES" || echo "CMake configured: NO"
test -f vllm/_C.abi3.so && echo "C++ extensions: BUILT" || echo "C++ extensions: NOT BUILT"
```

> **Environment setup timing:**
> - **`main` branch**: venv is hardlinked from a pre-built cache (~10s). Python + precompiled C extensions are ready immediately.
> - **Non-main branches** (releases, feature branches): a fresh venv is built from the branch's own `requirements/*.txt` (~5-10 min). This is expected — different branches may need different torch versions.

#### vLLM Incremental Compilation Workflow

This session comes with a Python-ready vLLM environment. C++ kernel compilation is available on-demand.

**CRITICAL FOR ALL AGENTS (including subagents):** 
- The `.venv` is **pre-built and ready**. Just run `source .venv/bin/activate`.
- **NEVER** run `pip install vllm`, `uv pip install`, or any package installation command. 
- **NEVER** create a new venv. The existing `.venv` contains all required packages.
- If `import vllm` fails, report the error — do NOT try to fix it by installing. 
- The "If .venv is Missing" section below is for **session infrastructure only**, not for agents.

##### For Python-Only Work (Default)

The session is ready for Python development out of the box:

```bash
source .venv/bin/activate
# Python vLLM is ready to use
python -c "import vllm; print(vllm.__version__)"
```

##### For CUDA Kernel Development (On-Demand)

If you need to modify C++ code in `csrc/`, build the extensions:

```bash
source .venv/bin/activate

# First-time build (~15-20 minutes, no cache, use max parallellism)
cmake --preset release
cmake --build --preset release --target install -j {cpu_cores}
```

- CUDA kernels in `csrc/`
- Python bindings in `vllm/_custom_ops.py`
- Incremental compilation is configured via `CMakeUserPresets.json` ([docs](https://docs.vllm.ai/en/latest/contributing/incremental_build/))

**After editing `csrc/` code:**
```bash
cmake --build --preset release --target install
```

**After adding new `csrc/` files:**
```bash
cmake --preset release && cmake --build --preset release --target install
```

Python changes take effect immediately (editable install).

`CMakeUserPresets.json` is automatically patched at session creation to point at this session's `.venv/bin/python` and to detect your GPU architecture. If you see CMake Python path errors, verify `.venv` exists and check that `CMakeUserPresets.json` has the correct Python paths (`grep -i python CMakeUserPresets.json`).

##### If .venv is Missing (SESSION INFRASTRUCTURE ONLY — NOT FOR AGENTS)

**WARNING**: The instructions below are for the session provisioning system.
If you are an agent or subagent, DO NOT follow these instructions.
Report the issue instead — the session setup should handle venv creation.

The session setup should have created a venv automatically. If it's missing:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="" \
  uv pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

##### Build Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `CCACHE_DIR` | `/home/session_user/.ccache` | Shared compiler cache |
| `CCACHE_MAXSIZE` | `10G` | Cache size limit |
| `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit location |

### Testing

```bash
pytest tests/# Run all tests
pytest tests/path/to/test_file.py # Run specific test file
pytest tests/path/to/test_file.py::test_function_name # Run specific test
pytest -v tests/ # Verbose output
pytest -k "pattern" tests/ # Run tests matching pattern
.buildkite/scripts/rerun-test.sh tests/path/to/test.py::test_name# Debug flaky tests
```

### Running vLLM

```bash
vllm serve <model_name> # Start API server
vllm bench {serve,latency,throughput} # Benchmarking CLI
```

### Benchmarking

```bash
python benchmarks/benchmark_throughput.py --model <model> # Throughput benchmark
python benchmarks/benchmark_latency.py --model <model># Latency benchmark
python benchmarks/benchmark_serving.py --model <model># Serving benchmark
```

## Architecture Overview

### V1 Engine Architecture (Primary)

The V1 architecture (`vllm/v1/`) is the primary engine implementation. V0 is deprecated.

**Engine Core Components** (`vllm/v1/engine/`):
- **EngineCore** (`core.py`): Inner loop containing scheduler, model executor, and structured output manager
- **Processor** (`processor.py`): Transforms raw inputs → EngineCoreRequests via validation, tokenization
- **OutputProcessor** (`output_processor.py`): Converts EngineCoreOutputs → RequestOutput for users
- **AsyncLLM** (`async_llm.py`): Async interface for serving

**Three-Stage Execution Loop**:
1. **Schedule**: Select requests for decode/prefill from waiting/running queues
2. **Forward Pass**: Run model and sample tokens
3. **Postprocess**: Update requests, check stop conditions, clean up finished requests

**Request Lifecycle**:
- Requests enter scheduler's waiting queue with `WAITING` status
- Move to `RUNNING` during execution
- Scheduler uses FCFS or priority policies

### KV Cache and Memory Management

**Paged Attention**:
- Block-based KV cache with 16-token blocks by default
- `KVCacheManager` (`vllm/v1/core/kv_cache_manager.py`): Maintains `free_block_queue` - pool of available blocks
- Block size calculation: `2 * block_size * num_kv_heads * head_size * dtype_bytes`

**Prefix Caching**:
- Hash-based block identification for prompt prefix reuse
- Splits prompts into 16-token chunks with `cached_block_hash_to_block` mapping

### Executor Hierarchy

- **UniProcExecutor** (`vllm/v1/executor/uniproc_executor.py`): Single-GPU execution
- **MultiProcExecutor** (`vllm/v1/executor/multiproc_executor.py`): Multi-GPU with `rpc_broadcast_mq` coordination
- Supports tensor parallelism (TP) and pipeline parallelism (PP)

### Key Directories

- `vllm/v1/`: Primary V1 engine implementation
- `vllm/v1/core/sched/`: Scheduler implementations
- `vllm/v1/attention/backends/`: V1 attention backends (FlashAttention, FlashInfer, MLA, etc.)
- `vllm/model_executor/models/`: Model implementations
- `vllm/attention/`: Attention mechanisms including PagedAttention
- `csrc/`: C++/CUDA kernels
- `tests/`: Test suite
- `benchmarks/`: Performance tools

### Model Support

Models in `vllm/model_executor/models/` include:
- Model architecture implementation
- Weight loading logic
- Forward pass implementation
- Parallelism strategy support

Registration in `vllm/model_executor/models/registry.py` maps HuggingFace architecture names to vLLM implementations.

## Development Patterns

### Testing Practices

- Use `pytest` fixtures from `tests/conftest.py`
- `@pytest.mark.parametrize` for multiple configurations
- `@pytest.mark.distributed` for distributed tests
- `@pytest.mark.skip_v1` for V1-incompatible tests

### CUDA Kernel Development

- CUDA kernels in `csrc/`
- Python bindings in `vllm/_custom_ops.py`
- Incremental compilation is configured via `CMakeUserPresets.json` ([docs](https://docs.vllm.ai/en/latest/contributing/incremental_build/))

**After editing `csrc/` code:**
```bash
cmake --build --preset release --target install
```

**After adding new `csrc/` files:**
```bash
cmake --preset release && cmake --build --preset release --target install
```

Python changes take effect immediately (editable install).

## Important Configuration

### Environment Variables

```bash
# V1 Engine
VLLM_USE_V1="1"# Use V1 engine (default)
VLLM_ENABLE_V1_MULTIPROCESSING="0" # Disable V1 multiprocessing

# Attention
VLLM_ATTENTION_BACKEND=<backend> # Force specific attention backend

# Debugging
VLLM_LOGGING_LEVEL=DEBUG # Enable debug logging

# Hardware
CUDA_VISIBLE_DEVICES=<ids> # Control GPU visibility
VLLM_CPU_ONLY=1# CPU-only mode
```

### Config Classes (`vllm/config.py`)

- `VllmConfig`: Main configuration container
- `ModelConfig`: Model-specific settings
- `ParallelConfig`: Parallelism configuration
- `SchedulerConfig`: Scheduling parameters
- `CacheConfig`: KV cache configuration
