# Bottleneck Analysis (Stage 2, Round 1) — Qwen/Qwen3.5-4B, L40S, bf16, TP=1

Mining of EXISTING Stage-1 evidence only (no re-run). All timings are *measured*
from the Stage-1 clean E2E sweep and the bounded nsys node-mode traces; all
physical ceilings are grounded in roofline math against the L40S hardware spec
read directly from the trace metadata (`TARGET_INFO_GPU`: NVIDIA L40S / AD102 /
SM 8.9 / `memoryBandwidth = 864.096 GB/s` / 142 SMs / 96 MiB L2).

## Evidence Sources (primary artifacts)

- Clean E2E baseline (AUTHORITATIVE, profiler-free):
  `rounds/1/sweeps/baseline/e2e_latency_results.json` — BS=32, IL=64, OL=512,
  num_iters=10, `CompilationMode.VLLM_COMPILE` + `CUDAGraphMode.FULL_AND_PIECEWISE`.
- nsys node-mode traces at three decode depths (bottleneck attribution;
  profiler latency here is contaminated and NOT used for timing):
  - `rounds/1/profiling/nsys/baseline_il58_ol8_bs32.{nsys-rep,sqlite}` (step 2)
  - `rounds/1/profiling/nsys/baseline_il312_ol8_bs32.{nsys-rep,sqlite}` (step 256)
  - `rounds/1/profiling/nsys/baseline_il568_ol8_bs32.{nsys-rep,sqlite}` (step 512)
  - Sidecars confirm each is a genuine full-batch decode step: `batch_size=32`,
    `source_input_len=64`, `source_output_len=512`, `capture_output_step ∈ {2,256,512}`.
- Stage-1 architecture correction + component decode-GPU shares:
  `rounds/1/constraints.md` (Baseline Truth Snapshot). This analysis re-derives
  the per-component shares directly from the sqlite kernel tables and confirms them.

Stage-1 baseline and profiling commands are recorded in `rounds/1/constraints.md`
and `rounds/1/sweeps/{baseline,profiling}/logs/`. This is Round 1: no prior SHIP,
so the Round-1 baseline + attribution traces are the current evidence.

## Method Notes (how the steady-state step was isolated)

Each trace contains one CUDA-graph capture-window transient (index/copy/
slot-mapping/gather setup + chunked-prefill arm-up) followed by exactly one
steady-state full-batch decode step. The transient is separated from the step by
the single largest inter-kernel gap in the trace (~1.36–1.39 ms). Kernels before
that gap (idx 0–18, ~29 µs busy over ~1.83 ms) are the transient and are excluded
from per-step attribution; the steady-state region (563 kernels) is used for all
`f_decode` shares. This matches the Stage-1 snapshot's "first ~1.8 ms is a
capture-window transient" note.

**decode_busy is measured, not assumed.** Within the isolated steady-state step
the kernels run essentially gap-free: busy/span = **0.9985** at every depth
(idle 29.6 µs / 19.7 ms at step 512). This is the signature of full CUDA-graph
replay — there is almost no intra-step host/launch gap. The dilution `decode_busy`
below is computed at the E2E level (mean per-step kernel busy / measured tpot),
which is the E2E-relevant quantity.

Per-step kernel busy by depth (steady region, sum of kernel durations):
18.974 ms (step 2), 19.333 ms (step 256), 19.712 ms (step 512); mean **19.340 ms**.
This tracks the clean tpot of 19.411 ms/token (from `RequestOutput.metrics`),
confirming the trace step is representative of production decode.

Instance counts per step confirm the hybrid layer decomposition (trace-verified):
24 `fused_recurrent_gated_delta_rule_packed_decode` + 24 `_causal_conv1d_update`
(the 24 linear-attention layers), 8 `flash_fwd_splitkv` + 8 `flash_fwd_splitkv_combine`
(the 8 full-attention layers), 1 LM-head GEMM. No CUDA memcpy of note, no
collectives (TP=1, single stream id 7).

## Workload Dilution (per BS)

Phase times from `e2e_latency_results.json` (`prefill_avg_s`, `decode_avg_s`,
per-request means). `decode_busy = mean_per_step_kernel_busy / tpot` = 0.019340 / 0.019411.
`decode_share_of_e2e = decode_avg_s / (prefill_avg_s + decode_avg_s)`.
`inter_kernel_share = (1 - decode_busy) * decode_share_of_e2e`.
`prefill_share = prefill_avg_s / (prefill_avg_s + decode_avg_s)`.

| BS | total_e2e_s | prefill_s | decode_wall_s | decode_kernel_s | decode_busy | decode_share_of_e2e | inter_kernel_share | prefill_share |
|---|---|---|---|---|---|---|---|---|
| 32 | 10.0931 | 0.1093 | 9.9189 | 9.8826 | 0.9963 | 0.9891 | 0.00362 | 0.0109 |

`decode_kernel_s = decode_busy × decode_wall_s`. Four-slice check:
prefill 0.0108 + decode_kernel 0.9855 + inter_kernel 0.0036 + other 0.0064 ≈ 1.0
(`other` = 0.0649 s = 0.64% of E2E: request-level warmup/scheduling/cleanup
outside the profiled steady step; small, as expected). `decode_frac = 0.9891`.

This is a **decode-heavy, decode-bound, near-fully-CUDA-graphed** workload:
98.6% of E2E wall time is decode-step kernel execution; only 1.1% is prefill and
0.36% is decode inter-kernel slack. The dominant E2E lever is decode-step kernel
work, not host/launch overhead.

## Top Components (by f_e2e)

`f_e2e = f_decode × decode_busy × decode_share_of_e2e = f_decode × 0.9855`.
`f_decode` (= "decode-graph %") is the component's mean share of steady-state
decode-step kernel time across the three depths (diagnostic only, never the E2E
multiplier). `physical_ceiling` is the measured-kernel-time / roofline-floor ratio
(kernel speedup that would hit the hardware limit), grounded per the Physical
Ceilings section; components without a defensible closed-form roofline are marked
`(disclosed)` and carry no addressable number here.
Final column `f_e2e × (1 - 1/ceiling)` is the physically addressable E2E impact.

| Component | BS | decode-graph % | f_e2e | physical_ceiling | f_e2e × (1-1/ceiling) | prefill-active? |
|---|---|---|---|---|---|---|
| dense_proj_GEMM (all linear proj) | 32 | 69.19 | 0.6818 | 1.619x | 0.2607 | Yes |
| GDN_recurrent (fused_recurrent_gated_delta_rule) | 32 | 13.30 | 0.1310 | (disclosed) | (disclosed) | No |
| LMhead_GEMM (cutlass relu 256x128) | 32 | 10.18 | 0.1003 | 1.337x | 0.0253 | No |
| flash_attn (FA2 splitkv, 8 full-attn layers) | 32 | 2.56 | 0.0253 | (disclosed) | (disclosed) | Yes |
| act_silu (fused silu/sigmoid gate) | 32 | 1.05 | 0.0103 | (disclosed) | (disclosed) | Yes |
| rmsnorm (fused add+rms_norm) | 32 | 0.96 | 0.0095 | (disclosed) | (disclosed) | Yes |
| causal_conv1d_update (GDN short conv) | 32 | 0.59 | 0.0058 | (disclosed) | (disclosed) | No |
| elementwise_misc | 32 | 0.51 | 0.0050 | (disclosed) | (disclosed) | Yes |
| sampling (softmax/argmax/rng) | 32 | 0.51 | 0.0050 | (disclosed) | (disclosed) | No |
| gemm_splitK_reduce (cuBLASLt splitK) | 32 | 0.47 | 0.0046 | (disclosed) | (disclosed) | Yes |
| triton_misc (small fused elementwise) | 32 | 0.47 | 0.0046 | (disclosed) | (disclosed) | Yes |
| flash_combine (FA2 splitkv combine) | 32 | 0.13 | 0.0013 | (disclosed) | (disclosed) | Yes |
| kv_cache_write (reshape_and_cache + slot_mapping) | 32 | 0.08 | 0.0008 | (disclosed) | (disclosed) | Yes |
| gather_index | 32 | 0.01 | 0.0001 | (disclosed) | (disclosed) | Yes |
| **inter_kernel_slack** | 32 | — | 0.00362 | 1.0 | 0.00362 | — |
| **prefill (all)** | 32 | — | 0.0108 | (disclosed) | (disclosed) | (is prefill) |

Notes:
- `dense_proj_GEMM` aggregates every dense projection across all 32 layers:
  full-attn `qkv_proj`+`o_proj`, GDN `in_proj_qkvz`+`in_proj_ba`+`out_proj`, and
  MLP `gate_up`+`down`. It is `prefill-active` (the same weights run in prefill),
  so its decode-only `f_e2e` is a LOWER BOUND on true E2E contribution.
- `dense_proj_GEMM` is a family, not one kernel. Its per-kernel decomposition
  (below) matters because a mechanism usually targets one shape class, not all
  6.65 GiB of weights at once. The 0.2607 addressable figure is the family-level
  physical bound (see Physical Ceilings).
- `LMhead_GEMM` is decode-only in the profiled window (1 instance/step); it is
  not marked prefill-active because logits are computed once per generated token.
- The maximum addressable slice across non-overlapping slices is
  **dense_proj_GEMM = 0.2607 (26.07%)**; see the enrichment summary at the end.

### dense_proj_GEMM per-kernel decomposition (trace-grounded, step 512)

Distinct CUDA kernels inside the family (from `demangledName` + grid dims):

| Kernel symbol | n/step | tot µs/step | avg µs | grid / regs | maps to |
|---|---|---|---|---|---|
| `ampere_bf16_s16816gemm_bf16_64x128_ldg8_f2f_stages_64x3_tn` | 32 | 5898 | 184.3 | g288 / r158 | largest proj family (1/layer) |
| `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32` (bimodal) | 88 | 4236 | 2.5 & 48–94 | g8x10 / r90 | multiple proj GEMMs/layer |
| `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x3_tn` | 32 | 3230 | 100.9 | g96/g80 / r158 | proj family (1/layer) |
| `cublasLt::splitKreduce_kernel` | 56 | 91 | 1.6 | — | split-K epilogue reduce |

All dispatch through `torch.nn.functional.linear` →
`UnquantizedLinearMethod.apply` → `dispatch_unquantized_gemm()` →
`default_unquantized_gemm` (cuBLAS/cuBLASLt), confirmed in
`vllm/model_executor/layers/{linear.py:225, utils.py:332}`. bf16, no quantization.

## Chronological Kernel Chains (trace order, not architecture-inferred)

Within one steady-state decode step (step 512, timestamp order):

**Linear-attention (GDN) layer (×24):**
`[triton fused add+rms_norm]` → `in_proj_qkvz GEMM (ampere s16816)` →
`triton fused (to_copy/cat/rms/split — QKV+conv prep)` → `_causal_conv1d_update`
→ `fused_recurrent_gated_delta_rule_packed_decode` (grid 4×1024×1, 209 regs) →
`triton fused mul/silu gate + RMSNormGated` → `out_proj GEMM` → residual add →
`[rms_norm]` → `gate_up GEMM` → `triton_poi_fused_mul_silu` → `down GEMM`.

**Full-attention layer (×8):**
`[rms_norm]` → `qkv_proj GEMM` → `fused qk_rmsnorm+rope+gate (triton)` →
`reshape_and_cache_flash` → `flash_fwd_splitkv_kernel` (grid 1×2×128, 254 regs) →
`flash_fwd_splitkv_combine_kernel` → `triton mul_sigmoid gate` → `o_proj GEMM` →
residual add → `[rms_norm]` → `gate_up GEMM` → `silu` → `down GEMM`.

**Tail (once/step):** final `rms_norm` → **LM-head GEMM**
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128` (1.97 ms, grid 8×122,
256 regs) → `cunn_SoftMaxForward` → `distribution_elementwise` / `argmax` /
`reduce_kernel` (sampling). The `_compute_slot_mapping`, `vectorized_gather`,
and large `direct_copy`/`elementwise` (18–23 µs) kernels appear only in the
capture-window transient, not in the steady step.

## Phase Attribution

- **Prefill:** 0.1093 s = 1.09% of E2E. Small but retained as a first-class
  opportunity. Prefill exercises the same dense-proj GEMMs, GDN chunk kernel
  (`chunk_gated_delta_rule`, distinct from the decode recurrent kernel), FA2, and
  MLP; components flagged `prefill-active` above additionally accelerate prefill.
- **Decode (steady state):** 98.55% of E2E is decode-step kernel execution
  (`decode_busy × decode_share`), 0.36% is decode inter-kernel slack.
- **Warmup / graph capture:** not in the E2E measurement window (sweep uses warm
  compile caches; measured after warmup). The per-trace ~1.8 ms transient is a
  profiler capture artifact, excluded from steady-state attribution.
- **Other residual:** 0.64% (request-level scheduling/cleanup).

## Phase / Instance-Count Sanity (transient vs decode)

Kernels absent from the steady decode step, or with counts far above
`layers × decode_steps`, were treated as transient until confirmed. All large
one-off kernels (`direct_copy` 22.9 µs, `elementwise` 18.8 µs, `reduce_kernel`
15.8 µs, `vectorized_gather`, `_compute_slot_mapping` ×4) live only in the
capture-window transient and are excluded. The steady step's counts
(24/24/8/8/1) match `layers_of_type × 1 decode step`, confirming a clean
single-step attribution. No large full-trace/decode discrepancy remains after
transient removal.

## Physical Ceilings (roofline, grounded in L40S spec from trace metadata)

L40S (AD102, SM 8.9), from `TARGET_INFO_GPU`: HBM BW = **864.096 GB/s**,
142 SMs. Public bf16 dense tensor-core peak ≈ **362 TFLOP/s** (used only where
compute-bound). Decode at BS=32 is weight-GEMV-like: arithmetic intensity of a
weight matrix W read once for 32 tokens is `2·BS·W / (2·W) = BS = 32` FLOP/byte,
far below the L40S ridge `TC/HBM ≈ 419` FLOP/byte. **Decode is firmly
memory-bandwidth-bound**, so the dense-GEMM and LM-head floors are weight-read
floors (bytes / HBM BW), not compute floors.

**dense_proj_GEMM (all 32 layers):** total projection weights read per step =
3.57 G elements × 2 B = **6.649 GiB**. Weight-read floor = 6.649 GiB / 864.096 GB/s
= **8264 µs**. Measured family time = 13,378 µs. Ceiling = 13378 / 8264 =
**1.619×** (max kernel speedup if BW-bound floor were hit). Sub-floors:
full-attn qkv+o (×8) 680 µs, GDN in/out/conv (×24) 2341 µs, MLP gate_up+down
(×32) 5243 µs. These are the family's memory floor; a mechanism targeting one
shape class inherits only its sub-share.
- compute floor for reference: 228.5 GFLOP/step → 631 µs at 362 TFLOP/s (13× below
  the BW floor), confirming BW-bound.

**LMhead_GEMM:** weight = 248320 × 2560 × 2 B = **1.184 GiB** (tied embedding).
Weight-read floor = 1471 µs. Measured = 1968 µs. Ceiling = 1968 / 1471 = **1.337×**.
- compute floor: 40.7 GFLOP → 112 µs, again far below the BW floor → BW-bound.

**GDN_recurrent — disclosed, no clean closed-form ceiling asserted.** The
temporal SSM state is **fp32** (HF `text_config.mamba_ssm_dtype = "float32"`,
applied by vLLM `Qwen3_5ForConditionalGenerationConfig.verify_and_update_config`,
`vllm/model_executor/models/config.py:613-616`; conv state bf16). Per-step state
shapes (from `MambaStateShapeCalculator.gated_delta_net_state_shape`, TP=1):
temporal `(32,128,128)` fp32 = 2048 KiB/seq/layer, conv `(3,8192)` bf16 = 48 KiB.
A read-once state floor over 24 layers × BS=32 is ≈1908 µs; a naive read+write
floor is ≈3816 µs. Measured 2571 µs falls between these, so the exact byte
traffic (state read/write pattern, kernel-internal recompute, register/L2 reuse
of the 96 MiB L2) is not pinned down from nsys alone. **A hardware-counter
(achieved-bandwidth) NCU measurement would be required to assert a GDN physical
ceiling; I did not run one because no top-3 addressable ranking depends on it**
(GDN removable fraction is undetermined, so it cannot be the max-addressable
slice regardless). Grid = 4×1024×1, 209 regs/thread — high register pressure is
observable but an occupancy-limited claim would also need NCU.

**flash_attn — disclosed.** FA2 splitkv, grows with KV depth (132 → 502 → 867 µs
across steps 2/256/512), so its share is depth-dependent; only 0.7%→4.4% of
decode. head_dim=256 (grid 1×2×128). No ceiling asserted; a bandwidth ceiling for
paged-KV read would need NCU counters and is not top-3 addressable.

**inter_kernel_slack:** measured 0.36% of E2E; the whole slice is nominally
removable (removable_fraction = 1.0) → addressable 0.00362. It is tiny because
decode is already near-fully CUDA-graphed (busy/span 0.9985 within the step).

Small elementwise/norm components (act_silu, rmsnorm, triton_misc, etc.) are each
< 1.1% of decode and are BW-bound fused Triton/ATen ops; no individual physical
ceiling is asserted (disclosed).

## Technology Landscape

Top-3 addressable kernel opportunities by `f_e2e × removable_fraction`:
1. dense_proj_GEMM: 0.6818 × 0.382 = **0.2607**
2. LMhead_GEMM: 0.1003 × 0.252 = **0.0253**
3. GDN_recurrent: f_e2e = 0.1310, removable_fraction **undetermined** (no grounded
   ceiling). Ranked third by `f_e2e` because its removable fraction cannot be
   computed from retained evidence; disclosed as a gap. Included so champions see
   the second-largest raw decode consumer.

### dense_proj_GEMM (all dense projections; `torch.nn.functional.linear` → cuBLAS/cuBLASLt)
- Authoring class: **library:cuBLAS/cuBLASLt** (via PyTorch `F.linear`). Symbols
  are `ampere_bf16_s16816gemm_*` and `cutlass_80_wmma_tensorop_bf16_*` +
  `cublasLt::splitKreduce_kernel` — cuBLASLt-selected Ampere/CUTLASS bf16 tensor-op
  GEMMs.
- Evidence: dispatch `vllm/model_executor/layers/linear.py:225`
  (`dispatch_unquantized_gemm()`) → `vllm/model_executor/layers/utils.py:332`
  → `default_unquantized_gemm` = `torch.nn.functional.linear`. Kernel symbols and
  grid dims from `baseline_il568_ol8_bs32.sqlite` (see decomposition table).
- SM generation (this deployment): **SM89** (L40S / AD102, from trace `TARGET_INFO_GPU`).
- Op character: **structured tensor-core** GEMV/skinny-GEMM (BS=32 tokens ×
  weight matrices), memory-bandwidth-bound at this batch size (AI≈32 ≪ ridge 419).
- Library coverage for this op+shape+dtype: mature — cuBLASLt is the incumbent
  tuned path for bf16 dense GEMM on SM89; CUTLASS provides the wmma kernels
  already dispatched. No FP8/INT quant is in play (bf16 frozen), so quant-GEMM
  libraries (DeepGEMM — not importable here; Marlin/machete) do not apply without
  changing dtype. This is a memory-bound bf16 GEMM family; the removable fraction
  is bounded by the 1.619× BW-floor ceiling.

### LMhead_GEMM (tied-embedding logits projection)
- Authoring class: **library:CUTLASS** (cuBLASLt-selected). Symbol
  `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_...>`,
  1 instance/step, 1.97 ms, grid 8×122×1, 256 regs.
- Evidence: `vllm/model_executor/layers/logits_processor.py:96`
  (`lm_head.quant_method.apply`) → `UnquantizedLinearMethod` → same `F.linear`
  path. Weight is the tied embedding (`tie_word_embeddings=true`, vocab 248320).
  Kernel symbol from the sqlite trace.
- SM generation (this deployment): **SM89**.
- Op character: **structured tensor-core** GEMM, `[32 × 2560] × [2560 × 248320]`,
  memory-bandwidth-bound (1.184 GiB weight read dominates; compute floor 112 µs
  ≪ BW floor 1471 µs).
- Library coverage for this op+shape+dtype: mature — CUTLASS/cuBLASLt bf16 GEMM.
  The large vocab makes the weight read the floor; only a dtype change (weight
  quant) or vocab-partition/argmax-fusion changes the byte traffic. `logits_processor`
  already offers a vocab-parallel argmax path (`get_top_tokens`) that avoids
  materializing full logits under TP, but TP=1 here so the full 248320-wide GEMM
  runs. Removable fraction bounded by the 1.337× BW-floor ceiling.

### GDN_recurrent (`fused_recurrent_gated_delta_rule_packed_decode`)
- Authoring class: **Triton** (vendored flash-linear-attention / FLA).
- Evidence: `@triton.jit fused_recurrent_gated_delta_rule_packed_decode_kernel`
  in `vllm/model_executor/layers/fla/ops/fused_recurrent.py:256`, imported by
  `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`. 24 instances/step
  (the 24 linear-attention layers), grid 4×1024×1, 209 regs. bf16 I/O, **fp32
  temporal state**.
- SM generation (this deployment): **SM89**.
- Op character: **novel algorithm / irregular** — gated-delta-rule linear-attention
  recurrent state update (Mamba-style SSM scan), not a standard GEMM or softmax
  attention. State-bound (fp32 temporal state R/W dominates).
- Library coverage for this op+shape+dtype: the vendored FLA Triton kernels ARE
  the current production coverage for gated-delta-rule on this stack (decode:
  `fused_recurrent.py`; prefill: `chunk.py`/`chunk_delta_h.py`). No mature CUDA/
  CUTLASS library provides gated-delta-rule decode on SM89. FlashInfer is
  importable but does not cover GDN linear-attention recurrence. A specialized
  CUDA/CuTeDSL kernel would be a rewrite competing against the tuned FLA Triton
  core (beats-baseline rule applies). Removable fraction is **undetermined**
  without an NCU achieved-bandwidth measurement (see Physical Ceilings).

## Disclosed Gaps / Caveats

- **GDN removable fraction is undetermined.** The measured 2571 µs sits between a
  read-once (1908 µs) and read+write (3816 µs) fp32-state floor; the exact byte
  traffic is not resolvable from nsys node timing alone. No NCU counter run was
  performed (no top-3 *addressable* ranking depends on it, since an undetermined
  removable fraction cannot be the max). A future hardware-counter probe
  (achieved DRAM throughput on `fused_recurrent_gated_delta_rule_packed_decode`)
  would be needed to assert a GDN ceiling. This is the honest boundary of the
  retained evidence.
- **dense_proj_GEMM is a family.** The 0.2607 addressable figure is the aggregate
  BW-floor bound over 6.649 GiB of weights. A real mechanism typically targets one
  shape class (MLP gate_up+down is the largest sub-share at ~5243 µs / step); the
  family bound is an upper envelope, not a single-kernel promise. Champions must
  scope their target and re-derive the sub-share `f_e2e`.
- **prefill-active lower bound.** All GEMM/norm/attention components marked
  prefill-active have decode-only `f_e2e` values that are LOWER BOUNDS on true E2E
  contribution (they also run in prefill's 1.09%).
- **Greedy decode is non-deterministic on this stack** (Stage-1 self-consistency
  FAILED; `constraints.md` invariant #7). Not a mining number, but any A/B must
  use the accuracy-tolerance band, not exact-token equality.
- Roofline HBM BW (864.096 GB/s) is read directly from the trace `TARGET_INFO_GPU`
  metadata, not assumed; bf16 tensor peak (362 TFLOP/s) is a published L40S spec
  and is used only to show the workload is BW-bound (compute floors are 8–13× below).
