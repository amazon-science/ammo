# Optimization Category Taxonomy

Every Phase 0 proposal declares a `## Category`. This file is the canonical catalog of those categories, and it tells you what the declaration is for: it routes the proposal to a likely measurement boundary and an instrumentation route. Read it when you write a Phase 0 proposal, when you check one, or when you need to know where a mechanism gets measured.

Category is a **non-binding descriptor**, not the eligibility gate. Eligibility needs an authored mechanism that targets a profiled bottleneck and carries an admissible production-boundary projection. The category never supplies the mechanism.

This file is referenced from:
- `SKILL.md` — stage and role contract
- `references/e2e-delta-math.md` — projection authority (single source)
- `references/technology-selection.md` — eligible technologies per category
- `references/validation-defaults.md` — default Gate 5.2 form
- `references/debate-scoring-rubric.md` — EV / scope admissibility
- `.claude/agents/ammo-champion.md` — Category field requirement in Phase 0 template
- `.claude/agents/ammo-impl-champion.md` — per-category validation gate routing + scaffolds (the impl-champion runs the kernel correctness & speedup checks)
- `orchestration/debate-protocol.md` — Category eligibility gate, champion spawn context

---

## Schema-Version Guard (Legacy Campaigns)

The Category requirement applies only to campaigns with `state.json.campaign.schema_version >= "4.1"`. Legacy campaigns (`schema_version < "4.1"` or absent) keep the older Phase 0 eligibility rules: Authored-Mechanism Mandate, Technology Selection, and Precision Classification only. The lead reads `state.json` and skips every Category-related gate when the schema version is older than 4.1, because a paused or in-flight campaign must still resume.

This mirrors the schema-version guard in `new_target.py`.

---

## When the Category Is Chosen (Phase 0)

A category is a conclusion you draw from data. The data answers questions like these: does a fusable seam exist, can a kernel be replaced by a faster one, or do host-side dispatch gaps dominate? The existing profiling data for the component(s) a candidate attacks makes the answer clear.

The champion selects the category in **Phase 0**, after it analyzes the full profile. Each of a champion's 2-3 ranked candidates declares its own category. See `.claude/agents/ammo-champion.md` § Candidate Eligibility and Generation and `orchestration/debate-protocol.md` § Phase 0: Independent Proposals.

This shapes exhaustion filtering. During candidate generation, a champion skips a component only when its relevant mechanisms are unambiguously closed. The lead then applies the soft exhaustion check during Phase 0 eligibility, once the proposal declares a concrete category and technology. This matches the structured `exhausted_technologies[]` scope.

---

## The Category Catalog (a descriptor, not a gate)

The authored-mechanism and evidence rules decide eligibility, not this table. Each row gives you a descriptor, a likely boundary, and an instrumentation hint. The catalog is also **non-exhaustive**: a novel mechanism stays eligible, so choose the closest schema-defined descriptor and explain the mismatch instead of inventing a state value outside the schema enum.

**Decode-kernel-slice classes** (likely boundary: kernel / fused chain):

| Category | Scope / what's authored | Slice targeted |
|----------|-------------------------|----------------|
| `kernel_replacement` | Single kernel → faster alternative | `f_e2e(kernel)` |
| `kernel_fusion` | N kernels → 1 fused kernel | `f_e2e(chain)` |
| `custom_kernel` | New/rewritten kernel compute (Triton/CuTeDSL/CUTLASS/CUDA C++) — legacy alias of `kernel_replacement` | `f_e2e(kernel)` |
| `weight_layout_transform` | Load-time weight concat/repack/requant + slice, library does the matmul (e.g. QKV/gate-up weight-merge) | `f_e2e(chain)` |
| `attention_kv_layout` | Authored KV-cache layout / new AttentionBackend / metadata builder | `f_e2e(kernel)` |

**Inter-kernel-slice classes** (likely boundary: actual-runner host interval):

| Category | Scope / what's authored | Slice targeted |
|----------|-------------------------|----------------|
| `dispatch_optimization` | CPU↔GPU pipelining, dispatch elimination | host portion of `inter_kernel_share` |
| `execution_pipeline_restructuring` | Authored scheduling / H2D-D2H overlap / async-stream / CUDA-graph capture-boundary code — legacy alias of `dispatch_optimization` | host portion of `inter_kernel_share` |
| `communication_strategy` | Authored collective-comm algorithm / EP dispatch / EPLB / comm-compute overlap | host portion of `inter_kernel_share` |
| `compute_graph_pass` | Authored Inductor/FX pattern-matcher pass that rewrites the compiled graph | host portion of `inter_kernel_share` |

**Legacy aliases (retained, never deleted):** `kernel_replacement` conceptually folds into `custom_kernel`; `dispatch_optimization` conceptually folds into `execution_pipeline_restructuring`. Both old values stay valid so paused campaigns resume.

Retuned constants and existing-path flag flips are configuration, not authored mechanisms.

---

## Where the Projection and Evidence Rules Live

Projection formulas, the slice-split rule, and the evidence-scope ladder live in their own authority docs, never here. This file gives descriptors, likely boundaries, and instrumentation hints only.

See `references/e2e-delta-math.md` § Projection Authority — One Boundary Equation and § Pre-Implementation Magnitude — Slice Split. See `references/debate-rules.md` § Evidence-Scope Ladder for scope admissibility.

---

## Phase 0 Evidence (by slice)

Attach the strongest bounded evidence you have for the load-bearing claim, per `references/debate-rules.md` § Claim-Driven Experiments (run an experiment or declare the evidence gap).

See `references/e2e-delta-math.md` § Pre-Implementation Magnitude — Slice Split (which scope is EV-eligible per slice) and § Projection Authority. Tier caps: `references/debate-rules.md` § Execution Confidence.

---

## Validation Gate Routing

Gate 5.2 is defined once, and its boundary follows the mechanism (kernel, fusion, dispatch/graph, overlap); `references/validation-defaults.md` § Gate 5.2 owns the mechanism → boundary mapping. Gate 5.3a always uses production activation evidence for the mechanism at hand; host and inter-kernel work does not pretend to have a replacement-kernel proof.

---

## Technology Fit

A category never whitelists technologies, and it never implies a static abstraction ranking. Use the closest schema-defined category as a descriptor, then apply the capability-based production-baseline rule in `technology-selection.md` to the actual mechanism. Kernel, compiler, library, and host-side implementations are all valid when their evidence and production integration support the claim.

---

## Disambiguating Examples

These worked cases fix the two boundaries the table cannot: which slice a mechanism really targets, and where configuration ends and an authored mechanism begins.

### Example 3 — dispatch_optimization (accepted)

> Fuse the MTP target-model preamble (28 eager kernel launches: torch.compile op + H2D copies + elementwise updates) into a single Triton `delta_advance` kernel that eliminates 27 dispatches and caches structurally-constant H2D uploads.

- **Category: `dispatch_optimization`**. The win is dispatch elimination — the 28 kernels collectively compute ~107 µs but incur ~6800 µs of inter-kernel gap time.
- **Boundary**: actual-runner host interval (Gate 5.2 measures wall-time drop, not kernel-vs-kernel speedup — see `references/validation-defaults.md` § Gate 5.2). The kernel speedup is decorative; the preamble wall-time reduction drives E2E.
- **Technology**: Triton (fused dispatch kernel) + Python (H2D caching, sync deferral).

### Example 4 — Rejected proposal: env-var flip (no Category will save it)

> Set `VLLM_USE_FLASHINFER_SAMPLER=1` in production.

- **Rejected at authored-mechanism eligibility.** The env var selects an existing code path; no mechanism is authored.

### Example 4b — Rejected proposal: tuning constant in `.py` (no Category will save it)

> Change `num_stages=2` → `num_stages=3` (or `num_warps`, `BLOCK_SIZE_M`, an `@autotune` config tuple, or a TRTLLM tactic-table entry) in a Triton/kernel source file. Measured +0.69% E2E.

- **Rejected at authored-mechanism eligibility** even with a measured win. The kernel body is unchanged; a constant in `.py` is still configuration.

### Example 4c — Eligible: authored logic that achieves the same effect

> Hand-write a software-pipelined prefetch loop in the kernel body that achieves what `num_stages` would, with an explicit double-buffer and `cp.async` schedule you wrote.

- **Eligible at authored-mechanism eligibility.** The mechanism logic is authored; magnitude still requires admissible evidence.

---

## Cross-Category Anti-Regression

A new category string alone does not prove forward progress, because two aliases can describe the same measured slice and the same mechanism. Across rounds, require one of three things: a changed mechanism, a different measured slice, or new evidence that invalidates the prior failure. The structured exhaustion scope stays authoritative.

---

## Verifying a Proposal's Category Block

The lead's eligibility check in `orchestration/debate-protocol.md` § Eligibility, revision, and elimination validates the Category block on schema_version ≥ 4.1 campaigns. These fields are required, verbatim from `.claude/agents/ammo-champion.md`:

```markdown
## Category
- Selected: <a schema-defined category; for a novel mechanism, choose the closest descriptor and explain the mismatch>
- Slice targeted: <f_e2e(component) | f_e2e(chain) | host portion of inter_kernel_share>
- Projection formula: <the one boundary equation from `references/e2e-delta-math.md`, numeric>
- Justification: <1-2 sentences citing the champion's analysis of the existing profiling data for the assigned component (the seam/op-character that makes this the right mechanism), with workload-composition data from bottleneck_analysis.md as support>
- Expected validation gates: <list per `validation-defaults.md` Gates 5.1-5.3>
- Evidence scope: <bound | proxy | production_boundary | clean_e2e>
```

A missing block is rejected at the eligibility gate after one revision opportunity — the same severity as a missing Technology Selection block. Projection uses the one boundary equation in `references/e2e-delta-math.md`; proxy-scope magnitude is EV-ineligible. Category membership never makes an otherwise ineligible mechanism eligible.

---

## References

- `e2e-delta-math.md` — `f_e2e` definition, four-slice composition, projection authority (single source)
- `technology-selection.md` — eligible technologies per category (§5.5)
- `validation-defaults.md` — Gate 5.2 authority
- `debate-scoring-rubric.md` — EV / scope admissibility
- `audit-invariants.md` — Stage 2 invariants on category fields
- `.claude/agents/ammo-champion.md` — Phase 0 Category block template
- `orchestration/debate-protocol.md` — Phase 0 eligibility/exhaustion check and champion choreography
