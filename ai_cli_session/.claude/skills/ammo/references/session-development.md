# Session Development

Read this when a source change needs an environment check or an incremental C++/CUDA build.
The session provisions `.venv` and `CMakeUserPresets.json`. Use that environment for all Python commands.

## Check the environment

```bash
.venv/bin/python -c "import vllm; print(vllm.__file__)"
```

The import path must belong to the assigned worktree. Check again after resume or a worktree change.
Report a missing environment or import error to the session owner. Agents do not create environments or install packages.

## Build changed extensions

| Change | Action |
|---|---|
| Python only | Use the editable installation; no extension build is needed. |
| Existing `csrc/` code | Run `cmake --build --preset release --target install`. |
| New `csrc/` files or no configured build | Run `cmake --preset release`, then the build command above. |

Activate `.venv` before CMake commands. Use the configured build parallelism.
If CMake reports a Python-path error, check the session paths in `CMakeUserPresets.json`.
The session supplies `CCACHE_DIR`, `CCACHE_MAXSIZE`, and `CUDA_HOME`; preserve them unless the approved task changes them.

## Validate the change

Run the tests that exercise the changed behavior. Add or repeat tests when a change, failure, or unresolved question requires them.
AMMO's required gates remain binding. Read `validation-defaults.md` for those gates and `gpu-pool.md` before GPU work.
Use `run_vllm_bench_latency_sweep.py` for official campaign measurements. Generic benchmark commands do not replace its evidence contract.
