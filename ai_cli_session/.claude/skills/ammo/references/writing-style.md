# Writing Style for AMMO Artifacts

This is the style contract for every markdown artifact you author — proposals, critiques, rebuttals, `validation_results.md`, monitor logs, decision records. Read it before you write one. Each artifact is read **once, by one busy engineer, to make one decision**. Write for that reader, not for a grader tallying coverage.

The pipeline already proves your diligence structurally: a populated Technology Selection block, a passing gate, a cited log file. You earn nothing by *narrating* that diligence in prose, because coverage, length, and visible effort are not scored (see `debate-scoring-rubric.md` § Scoring Is On Content, Not Volume). A staff engineer reviewing your doc would keep the numbers, the verdict, the non-obvious reasoning, and the evidence pointers — and delete roughly half of what these artifacts currently contain. Write the version they'd keep.

## The reader's test

Before you write a sentence, ask: *does this change what the reader does next?* Cut the sentence if the same conclusion already appears above, if it restates a rule you're following, or if it announces your own honesty.

## Rules

State each conclusion once. Put the verdict at the top, the evidence below it, and stop. Do not repeat it in YAML frontmatter *and* a bold `Verdict:` line *and* the end of every section *and* a closing "Recommendation." One `CLOSED` / `PASS` / `GATED_PASS`, in one place.

Cite each number once, then refer to it by name. Give a figure its source the first time (`file:line`, log path, or trace ref), then call it "the 4.12µs body" or "the BS1 win". Re-deriving the same µs figure in four sections is the single loudest slop signal in these documents.

Bold budget: about five spans per document. Reserve bold for the verdict and the one or two genuinely surprising facts. When the key phrase of every third sentence is bold, emphasis carries no signal. Never bold a whole multi-clause sentence.

Never narrate honesty or candor. Delete "honest," "honestly," "full disclosure," "I am explicit," "(read first)," "not advocacy," "self-falsification." State the caveat plainly instead — `The isolated 1.25µs is 2.7× faster than the trace; the gap is paged-cache scatter, not body compute.` is complete on its own. Framing a caveat as a moral act adds words and subtracts credibility.

Never echo the protocol you're following. Write the evidence, not the rule the evidence satisfies. Cut "satisfying the Component Dismissal Standard's two-independent-results bar," "this is the document-the-loop justification the Diversity Check requires," "clears NN#8." The reader knows the rules; they want to know what you found. The one exception is the machine-parsed blocks listed below: they keep their exact labels, because they are contracts, not prose.

Drop status-label theater and self-praise. No `ADDRESSED` / `CONCEDED` / `REBUTTED` / `EXEMPLARY` / `textbook` stamps. In a rebuttal, just concede or counter the point in a sentence. In a monitor log, "no flags" is the finding, not "NO RED FLAGS observed — methodology sound throughout."

Prefer short sentences to em-dash chains. Split any sentence that has three em-dashes, or nested parentheticals that each re-qualify the last. Write one claim per sentence, and put the qualification in the next sentence if it matters.

## Fixed forms for required disclosures

Put each required disclosure in a fixed compact form, not in prose:

- Methodology compliance → a one-line checklist: `parity ✓ (CUDA graphs + torch.compile) · warm+cold ✓ · baseline provenance ✓ (logs: …)`. This satisfies the feasibility caps; surrounding narrative earns nothing.
- Risks → a register table, ≤5 rows: `risk | likelihood | mitigation`.
- Fix attempts (impl FAIL paths) → one table row per attempt: `attempt | root cause | result`. Not a running narrative.
- Critiques addressed (rebuttal) → one row per critique: `critique | concede / rebut | one-line evidence`. A one-row concession fully counts as "addressed" — you do not need a paragraph per critique.

## Length targets

These targets cover the prose you author. They *exclude* required machine-parsed blocks and data tables, which stay complete. Going over is a signal to cut narration, not to delete evidence.

| Artifact | Target |
|---|---|
| Proposal | ≤ 900 words + required blocks |
| Critique | ≤ 500 words |
| Rebuttal | ≤ 600 words (table per critique) |
| `validation_results.md` | ≤ 700 words + gate tables |
| Decision record | ≤ 500 words |
| Monitor poll, no findings | **one line**: `Poll 12 (lines 296–468): smoke test pass, no flags` |
| Monitor poll, finding | the DA-MONITOR message + one line of context |

A clean kill or a FAIL is not a license to write more to "prove" it. A well-supported negative is *shorter*, because the evidence is decisive. The OP-012 FAIL ran 2,000 words for a track with zero code written; the human version is the verdict, the two independent negatives in a table, and the fallback-ladder result — under one page.

## What you must NOT cut

Brevity never overrides a machine contract. Gates, hooks, and the orchestrator parse the structures below, so keep their headings and field labels verbatim:

- Proposals: the `Technology Selection` block, the `Precision Classification` field (`lossless`/`lossy` + dtype-rule citation), the `## Category` block with all six labeled fields, and the authored-mechanism description.
- `validation_results.md`: the Decision/Overall heading with an explicit `PASS`/`FAIL`/`GATED_PASS` token, the `## Gate 5.3` E2E table, the Files-modified / Scope-adherence list, and (for GATED_PASS) the dispatch-mechanism + env-var + crossover + pre/post per-BS tables.
- `bottleneck_analysis.md`: the literal `## Technology Landscape` heading, the `## Workload Dilution` and `## Top Components` tables (with `f_e2e` and the per-BS `decode_busy` / `decode_share_of_e2e` cells).
- Rebuttals: the `## Open Items Declaration` with literal `[NONE]` / `[UNADDRESSED_CRITIQUE]` / `[NEW_EVIDENCE]` tokens.
- All artifacts: every real number, evidence pointer (`file:line`, log path, trace ref), and per-BS data row.

Cut narration. Keep evidence. The labeled blocks are the proof — let them do the proving.

## Before vs after

> **Before** (real proposal prose, 78 words):
> **Provenance note (honest):** the isolated 1.25µs is 2.7× *faster* than the trace's 3.42µs (>10% divergence — so I do **NOT** claim "production conv = 1.25µs"). The isolated bench lacks production L2 pressure and uses a small hot conv_state cache, whereas production scatters conv_state across a large paged KV-cache. The ~2.2µs gap is therefore **paged-cache scatter / memory-latency + exposure, not body compute** — confirmed by the isolated body sitting exactly at the launch floor.

> **After** (24 words):
> Isolated conv1d body is 1.25µs vs the trace's 3.42µs (`conv1d_floor_bs1.log`). The 2.2µs gap is paged-cache scatter, not body compute — the isolated body sits at the launch floor.
