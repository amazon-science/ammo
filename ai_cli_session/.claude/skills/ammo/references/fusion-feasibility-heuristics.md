# Fusion Feasibility Heuristics

These five heuristics (H1-H5) are **filters**, not proofs. Read them before you propose a kernel fusion. They reject fusion ideas that cannot plausibly beat baseline under **production parity** (CUDA graphs / torch.compile), either because the *maximum* memory-hop savings are too small, or because the design is likely to underfill SMs / reduce occupancy. These filters guide what to propose; they are not proofs, and passing one is never evidence that a fusion will win.

## Contents
- Inputs (per bucket, per rank)
- H1: Memory-hop time-saved upper bound
- H2: DRAM vs L2 best-case and worst-case
- H3: W2-fusion ROI (when K is large)
- H4: Occupancy + underfill stop-ship signals
- H5: Kernel-count wins require API time to be material

## Search anchors
bytes_saved, time_saved_max_us, BW_eff, DRAM vs L2, occupancy, spills, grid.sync, CUDA graphs, required_savings_us.

## Inputs (per bucket, per rank)

Collect these numbers for one bucket on one rank:
- `M`: tokens on this rank for the bucket (decode: batch size; prefill: tokens in the microbatch)
- `top_k`: experts per token
- `P = M * top_k`: token–expert pairs processed on this rank
- `N`: MoE intermediate per partition for this rank (post-TP/EP)
- `b_mm1`: bytes/elt of W1 output (often BF16/FP16 = 2)
- `b_act`: bytes/elt of activation output (often BF16/FP16 = 2)
- `b_w2in`: bytes/elt of quantized W2 input (often FP8 = 1)
- `b_scale`: bytes/elt of per-token scale (often FP32 = 4; sometimes FP16/BF16)

Fill in the baseline “already fused?” checklist first. If the baseline already fuses a hop, you do **not** count it as savings, so leave it out of the formulas below.

---

## H1: Memory-hop time-saved upper bound

H1 gives you a ceiling on what the fusion can save. Do the three steps in order, then apply the filter.

### Step 1: Identify which hops the fusion removes
A common baseline pipeline (example):
- W1 GEMM produces `mm1_out[P, 2N]` in BF16
- activation produces `act_out[P, N]` in BF16
- quant produces `w2_in[P, N]` in FP8 + per-token scales

A fused W1-epilogue kernel that directly writes FP8 W2 input can remove:
- `mm1_out` write+read hop
- `act_out` write+read hop

A full fused kernel can additionally remove:
- `w2_in` write+read hop
- per-token scales write+read hop (if they exist and are no longer needed)

### Step 2: Compute bytes saved (count BOTH store + load)
Count each removed hop twice: once for the store, once for the load.

**W1-epilogue fusion (remove mm1_out and act_out hops):**
- `bytes_saved_w1ep = 2 * P * (2N) * b_mm1  +  2 * P * (N) * b_act`
- Typical BF16 case: `b_mm1=b_act=2` ⇒ `bytes_saved_w1ep = 12 * P * N`

**Full fused kernel additional savings (remove w2_in + scales hops):**
- `bytes_saved_full ≈ bytes_saved_w1ep + 2 * P * N * b_w2in + 2 * P * b_scale`
- Typical FP8 + FP32 scale: `b_w2in=1, b_scale=4` ⇒ `bytes_saved_full ≈ 14 * P * N + 8 * P`

### Step 3: Convert to a time upper bound
Divide the bytes by a conservative effective bandwidth:
- `time_saved_max_us ≈ (bytes_saved / BW_eff) * 1e6`

Compute `BW_eff` from NCU for the relevant baseline kernels. If you don’t have NCU yet, use `BW_eff = 1 TB/s` *(advisory BW fallback — hardware-class hint, NOT a gate)* as a conservative Hopper-class back-of-envelope.

### H1 filter (advisory, not a gate)
Apply the filter to the bucket(s) you’re optimizing:
- If `time_saved_max_us < 0.5 * required_savings_us` → fusion is **low probability** *(advisory heuristic — NOT a ship/retract gate)* unless it is a "true epilogue" with no occupancy/grid regression.
- If `time_saved_max_us < 1–2 us` → treat as **not worth attempting** for decode *(advisory heuristic — NOT a ship/retract gate)* unless you already have evidence the fused kernel matches baseline GEMM throughput.

---

## H2: DRAM vs L2 best-case and worst-case

Even if you remove a “global memory hop,” the real savings may be small if the intermediate is L2-resident. Use NCU to find out whether the intermediate hits DRAM. For the baseline kernels involved in the hop, collect:
- DRAM bytes read/write
- L2 bytes and hit rate

If DRAM bytes attributable to the hop is near zero, your upside from “HBM savings” is intrinsically limited. Exact metric names vary, so the principle is what matters: quantify DRAM traffic vs L2 traffic for the hop.

---

## H3: W2-fusion ROI is usually tiny when K is large

**Don’t fuse W2 just to remove the intermediate.** Eliminating the W2-input activation hop saves `O(P*N)` bytes, but W2 weight traffic per pair is `O(K*N)` bytes (FP8 weights), so at low reuse:
- savings ratio ≈ `(2*P*N) / (P*K*N) = 2 / K`

If `K` is thousands (common), this ratio is << 1%. Fuse W2 only when both of these already hold:
- you already have token-major ownership (no atomics),
- and you can keep the grid large and occupancy high.

---

## H4: Occupancy + underfill stop-ship signals

Any fused design that increases regs/SMEM can lose even if it removes hops. Compare against the baseline dominant GEMM and watch for these stop-ship signals:
- Active CTAs/SM drops by ≥ 1 step (e.g., 2→1 CTA/SM)
- Register spills appear
- Total grid is not “large” (e.g., not >> SM count) AND fusion adds significant per-CTA work

If any stop-ship signal triggers, pivot to:
- simpler split kernels + CUDA graphs, or
- partial fusion around baseline GEMMs.

---

## H5: Kernel-count wins require API time to be material

Under CUDA graphs, kernel launch overhead is reduced, so “reduce kernel count” rarely pays. Prioritize it only if Nsight Systems shows CUDA API time is a material share of the target kernel subgraph (e.g., >5-10%). Otherwise, focus on kernel-time reductions (tiling, occupancy, DRAM traffic).
