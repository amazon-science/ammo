# CUDA Graphs Safety Checklist (Custom CUDA Ops)

Read this before you write or review a custom CUDA op on the hot path. A custom op that works in eager mode can fail under CUDA graph capture, cause graph breaks, or silently regress latency under graphs. Apply the "(must)" rules to every new custom op on the hot path. Use the debug toolkit (§4) when capture fails. Run the verification procedure in §5 before you ship.

## 1) Launch on the current stream (must)

**Rule:** launch on the **current PyTorch CUDA stream** for the current device.

In C++/CUDA extensions, prefer:

```cpp
#include <ATen/cuda/CUDAContext.h>
cudaStream_t stream = at::cuda::getDefaultCUDAStream();   // usually NOT what you want
cudaStream_t stream = at::cuda::getCurrentCUDAStream();   // what you want for graph capture
```

Reasons:
- vLLM and PyTorch may run work on non-default streams, especially with graphs.
- A launch on the default stream can add hidden sync or break capture assumptions.

## 2) Set the device, and never synchronize (must)

- Use `at::cuda::CUDAGuard` (or equivalent) to set the correct device before you launch.
- Never call `cudaDeviceSynchronize()`. Never add other host-side synchronization in code that can run during capture.
- To debug with synchronization, gate it behind a debug env var and keep it off by default.

## 3) No allocations, stable shapes (must)

**Allocate nothing during capture.** Do not use `at::empty`, `new`, or `malloc`, and do not create temporary tensors, inside the captured region. Preallocate explicit workspaces and reuse them.

**Keep shapes stable per bucket.** CUDA graphs require stable shapes for captured paths. The op must see consistent shapes within a bucket: same hidden dims, same top_k, same quant format, same workspace sizes. If the kernel uses dynamic shared memory, keep the dynamic SMEM size constant within a bucket.

## 4) Make errors visible (debug toolkit)

When capture fails or the output is wrong, set these:
- `CUDA_LAUNCH_BLOCKING=1` — slow, but it surfaces the true failing op.
- `TORCH_SHOW_CPP_STACKTRACES=1` — better call stacks.

## 5) Minimum verification procedure

1. Run an eager correctness test (unit test or small harness).
2. Run the *same* correctness test under CUDA graphs, or inside `vllm bench latency` with graphs enabled.
3. Take a Nsight Systems trace. Confirm the op is captured (no graph breaks) and there are no unexpected gaps, memcpys, or sync nodes.

## 6) Common gotchas (quick scan)

- A default-stream launch in a graph-captured workload.
- A hidden device mismatch in a multi-GPU (TP/EP) setup.
- Implicit temporary allocations during capture.
- Control flow that diverges by bucket and changes the workspace or dynamic SMEM requirements.
