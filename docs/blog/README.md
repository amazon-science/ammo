<!-- Extracted from index.html with extract-md.js. Do not edit by hand; re-extract after the page changes. -->

*This is the plain-text version of the interactive report at https://amazon-science.github.io/ammo/blog/. Figures appear here as their captions.*

*Engineering report*

# A Fast Kernel Is Not a Ship: optimizing live vLLM deployments, measured end to end

> We built **AMMO**, an agentic system that optimizes live vLLM deployments end to end and hands back measured pull requests. The result that shaped its design: an agent-written kernel, **5.36× faster** than the one it replaced, made the server running it **1.05% faster**. Both numbers are real, both come off the same B200, and the gap between them is the normal result.

*The AMMO team · 26 August 2026 · 23 min read*

**In this report.** The gap · Twenty-three candidates · How agents fail · How AMMO works · Four pull requests · The price · The limits · The takeaway

**What one campaign is.** AMMO is pointed at one deployment contract: one model, one machine, one serving configuration. It profiles that server, proposes **candidate** changes, and keeps a candidate only if it clears four checks. A kept change is **shipped**: it becomes the campaign's own new baseline. Everything else is **killed**, with the reason recorded. The pull requests came afterwards, reviewed and posted by people.

---

*Background*

## Agents already write fast GPU kernels, and the gap that nobody measures

Language models write competitive CUDA and Triton kernels, and recent work has pushed them further with multi-agent decomposition, hardware feedback and evolutionary search.

Nearly all of it is scored the same way, on the unit KernelBench established: take one reference operation, check that its output is still correct, and report how much faster the new kernel runs alone. It is cheap to run and easy to reproduce, which is why the field adopted it, and it is entrenched enough that when an agent was caught inflating its own speedups by gaming the harness, the response was to harden the benchmark rather than to change what it measures.

What a deployment needs is a faster server. Between a fast operation and a served token sit `torch.compile`, CUDA graph capture, speculative decoding, the KV-cache format and whichever revision of the model is actually loaded. **The same line of source can dispatch a different kernel under each one.**

So a kernel can be correct, genuinely faster, and irrelevant to the deployment it was written for. The dials above show this gap on one candidate; the sections below measure it.

---

*Problem 1*

## Isolated kernel speedup does not predict server speedup

The opening candidate put a gate projection, a sigmoid, and a multiply into one launch. In isolation, it ran 5.36× faster. In serving, it cut latency by 1.05%. By Amdahl's Law, the original chain accounted for about **1.29%** of token-generation time. Remove it, and with the rest of the schedule fixed, you can save no more than that share.

A speedup by itself does not tell the share of the whole schedule. Nor does it show overlap. One bit-exact candidate ran 1.66× faster after redundant writes were removed, but those writes were already hidden behind memory-bound expert matrix multiplies. The clean serving result was +0.01%.

> *Figure 1 · schematic. Measure 1.05% on the server. 1.29% is not a second reading; it is what that measurement says. A kernel that runs **5.36×** faster and holds 1.29% of the clock gives back 1.29 − (1.29 ÷ 5.36) = **1.05%**. The bar is to scale. The rest is only for illustration. Work backward: profile the serving path first. The share it reports is the ceiling on anything written for it.*

**On the two scales.** The dials use different ranges on purpose. "5.36× faster" and "1.05% of total time" are not the same sort of number, so they cannot go on one axis. The mistake here is to read the left dial and assume the right one follows. A shared axis would put that mistake into the chart.

## The pattern holds across twenty-three candidates

The retained ledgers contain 23 candidates with an isolated measurement. Of these, ten passed the campaign checks and thirteen were killed. The four examples below show why isolated speedup alone cannot rank them.

### Four more examples

Kernel speedup does not carry over to the server, and the third row made the server slower. We **ship** a change only if it wins at least **0.25%** on the real server. Shipping means it becomes the new baseline inside our own loop; it does not mean anyone upstream merged it. Three of these four were killed, and one of the three was a 2.38× sampler, verified bit-exact and lossless. We measured twenty-three of these.

Nineteen candidates beat their isolated baseline. Ten passed every deployment check. The plot has no upward trend between isolated speedup and serving value.

### Candidate

**Deployment**

Qwen3.5-122B

**What it changed**

Fuse shared-expert gate chain

**Kernel, timed alone**

5.36×

**End to end**

+1.05%

Kept. It cleared every check.

> *Figure 2. All 23 candidates that logged a measured isolated speedup across the three runs we kept complete records for. Horizontal axis is isolated speedup (log). Vertical axis is the measured end-to-end latency reduction, where positive means faster. **Circles were kept, squares were killed.** The dashed pair: a 1.24× kernel returned **+3.07%** and a 5.36× kernel returned **+1.05%**. Nine candidates died before anyone spent an end-to-end run on them; they sit in the rug strip below the axis rather than at an invented zero, because **no end-to-end measurement exists for them**. The gemma point (8.35×, +18.5%) is off the top of the scale, drawn as an arrow. The shaded band is the 0.25% threshold declared before each attempt. "Not in the public record" in the panel means the cause was never published, not that there was none.*

---

*Problem 2*

## How agents fail at long-horizon optimization

The measurement problem has a plain fix: score the change on the deployment rather than on the kernel. Getting one shipped change out of that fix is the long part: profile a live server, form competing hypotheses, write the kernel, run the correctness, quality and latency benchmarks, measure against the current best build, then go looking for the next bottleneck the change exposed. That grind is exactly what you would hand to an agent, and ours ran for **hours-to-days at a stretch across billions of tokens**, with the model's context filling up and getting compacted numerous times along the way.

An agent that looks reliable on a coding benchmark falls apart on a task like this, because the two settings ask for different things. A benchmark hands it one bounded episode: a clean starting state, a local objective, an immediate score. A campaign gives it days of work toward a server metric several steps away from whatever file is open, with no score on any individual step. Three failures showed up again and again in our early, looser runs.

**It stops early, for reasons that sound like judgment.** Hours into a run, an agent decides it is done: enough approaches tried, the remaining targets are library kernels, returns are diminishing. Each reason is defensible, and each one, on a real bottleneck, was wrong. This was the most expensive failure we watched.

**It forgets what it was told.** An instruction can stay in the prompt and stop working: it becomes wallpaper under the logs. An agent that knew to always use the sweep script would, eight hours in, casually invoke the raw benchmark instead. A better prompt cannot fix this, because the instruction is already there.

**It games the proxy.** In our looser runs the agent wrote a fast kernel on a path the server never dispatches, and it passed every kernel-level test. It wrote both a kernel and that kernel's tests, and over rounds the tests drifted toward what the code already did. It benchmarked without graph-capture warmup, which makes anything look fast. One worktree benchmark picked up the wrong build and reported that nothing was wrong. None of this was concealment: in every case the agent's own records said the work was succeeding.

The job makes these failures likely. The objective is conjunctive, so every requirement must hold at once: a candidate that is fast but breaks quality, or fast but never dispatches under CUDA graphs, has zero value. The evidence corrupts easily: profiled runs distort timing, serialized traces hide overlap, benchmarks can import the wrong build, and a stale baseline inflates every candidate measured against it, so the system must prove its measurements are about what they claim to be about. And a campaign runs for hours of wall clock, across many GPU reservations, parallel worktrees, and several context compactions. At that length, task-completion reliability declines (T. Kwa et al., *Measuring AI Ability to Complete Long Software Tasks*, METR / NeurIPS 2025) and agents "demonstrably forget or skip obligations that remain textually present in their instructions" (K. Hong, A. Troynikov, J. Huber, *Context Rot*, Chroma Technical Report, July 2025).

---

*AMMO*

## How AMMO works: every mechanism answers a failure we watched

**AMMO**, the Agentic Model-on-Machine Optimizer, runs the campaign end to end. The easiest way to read it is failure-first: each mechanism below exists as the counter to one of the failures above, added after we watched a run fail without it.

**The loop.** A campaign runs as a loop of five stages that repeats until the work runs out: profile the live server, propose rival fixes, implement in isolation, run the four checks, integrate and remeasure.

> *Figure 3. The band across the top is a directory on disk. Every stage reads it and writes back to it, so a stage that has already happened cannot un-happen when the context is compacted. Supervision covers two stages rather than all five, because that is where the measured benefit was. The return line is the ordinary case: a change that ships moves the bottleneck somewhere else, and the next round profiles the server it just changed.*

**The gate.** The four checks at the loop's fourth stage decide what ships, and three of the four are asked of the deployment rather than of the kernel. The new kernel returns the same numbers as the one it replaces. The whole model still answers as well as it did, inside one point on the deployment's own quality task. The new code is the code that actually runs once the server is compiled and captured the way production compiles and captures it. And the server's own latency improves by at least a quarter of a percent against the current best, at the batch sizes it serves.

One failure ends the candidate. A change cannot pass the quality check by being faster. The isolated speedup the field reports is **not one of the four**.

> *Figure 4. Pick a check to see what it asks and what each of the three candidates returned. The first is the kernel from the top of this page. The third comes from a separate FP8 run rather than that campaign, and the paper marks it as an illustration for that reason: the Qwen run's own deaths were all latency deaths.*

The second candidate was bit-exact against the kernel it replaced and 1.66× faster, and it left the server's latency unchanged. The writes it removed were already hidden behind memory-bound matrix multiplies, so removing them opened an idle gap instead of shortening the step.

The third died much earlier, at the model-quality check, and it died strangely. It was bit-exact with the old kernel on 99.8% of its outputs, then broke the model. The Triton compiler computes one division differently from NVIDIA's C++ compiler, by the smallest step a float can hold. That flips about **64% of the quantization scales by a single bit**, and across forty re-quantizing layers it compounded until 99.8% of answers changed at the very first generated token.

Isolated speedup still gets measured on every candidate. It is the cheapest signal we have for which candidate deserves an expensive end-to-end run, and for how much the deployment could gain if the change lands on an exposed bottleneck. It decides nothing about what ships; that rule is why nineteen isolated wins became ten shipped changes.

**The counters.** A run measured in days gives an agent time to find a way around the checks, and we watched it happen: when we later removed supervision under controlled conditions, changes passed every check while swapping in a mechanism nobody asked for, and quality references were re-captured after a check had already failed. The checks read whatever evidence they are given and cannot tell how it was produced. Each of the three failures above therefore has a counter built into the structure of the run, rather than a paragraph asking the model to behave.

**Against stopping early.** The agent that runs the campaign has the least policy authority in it: it advances stages, delegates and gates, and it cannot decide the campaign is finished. The run halts only when the largest remaining bottleneck holds less of the clock than the smallest win worth shipping, and a runtime hook blocks the session from ending while the campaign is live.

**Against forgetting.** Rules kept in the prompt rot, so the rules that matter do not live there. Campaign state sits in files: the contract that scopes the run, the current best, every attempt with its measured reason. The invariants are checked by hooks that fire on every tool call and do not weaken as the context grows. When the context fills and gets compacted, the run re-reads the files and continues from the last completed check.

**Against gaming the proxy.** There is one number that matters, and the agent can invoke it but cannot touch it: serving latency, measured by the same sweep script every time, against the current best, with graph capture on. No agent both proposes a change and approves it, a benchmark in a side copy of the repository is not evidence, and an auditor re-derives each gated stage from the artifacts alone. In one campaign that audit caught a baseline captured with the profiler still running, which had inflated the number every candidate was measured against; it was corrected before anything shipped.

Two of the mechanisms in the loop are not automatic. Candidate selection goes through rival agents that argue for competing hypotheses, and no feasibility claim enters that argument without a measurement run on the GPU behind it. A monitor reads the implementation transcript while the work happens.

---

*Results*

## Four pull requests against vLLM

Four complete campaigns produced four public vLLM pull requests. Each change carries its deployment envelope, benchmark commands, quality result, and raw measurements. Several changes are default-off because their evidence covers specific model and hardware configurations. Maintainers still decide whether they merge.

GPT-OSS · two H100 GPUs · TP=2 · batch sizes up to 64

5.4–9.7%

less time per output token, the wait between one token and the next

and 9.2–27.3% more throughput

**Measured over**

Ten serving configurations, baseline and candidate on the **same build and the same pair of GPUs**.

**Quality**

Greedy outputs are **bit-exact** with the baseline, so quality is preserved by construction rather than by a score.

**Provenance**

The broadest result of the four, and the **one run we did not keep**. Its evidence is the public pull request and diff rather than a trajectory anyone can walk back.

**Still owed**

A paired quality run, and a full rerun against the current vLLM codebase.

Qwen3.5-122B · one B300 GPU · TP=1 · three activation flags

1.7–5.4%

lower end-to-end latency, from kernels 4.2–5.4× faster on their own

**What it contains**

The **expert-gate kernel from the top of this page** (re-measured here on a B300), the pair of projections merged into one call, and two more changes from the same run.

**Quality**

GSM8K moved down by **0.15 of a point**, on the merged projections.

**Measured over**

Six fixed-batch cells, ten iterations each, a **fresh process per cell** so no cell inherits another's warm state.

**Still owed**

Latency was re-measured after the rebase. The quality evidence is still at the level of the older vLLM version it was captured on.

Kimi-K2.6 · four B300 GPUs · TP=4 · three activation flags

2.5–6.0%

lower end-to-end latency

**Quality**

A full maths-benchmark A/B on the same hardware, **92.95% to 93.10%**, up 0.15 of a point.

**Measured over**

Five cells, five iterations per arm, with two spread criteria stated up front.

**Still owed**

Valid on the vLLM version it was measured against. **Not yet rebased** onto the current one, and labelled as such.

Gemma-4-31B · one B300 GPU · TP=1 · speculative decoding on · behind a structural guard

6.2–35.4%

lower end-to-end latency across six shapes

**Only at**

The top of that range needs a **27,000-token prompt at batch size 1**, where output also rose from 74.5 to 320.6 tokens per second. Only the two 27k shapes clear measurement noise, so the peak does not generalise past that shape.

**Quality**

GSM8K moved **56.41% to 55.95%**, a drop of 0.46 of a point. The larger of the two on this page.

**What it fixed**

No new kernel. The server had been **falling back to a slow 2D kernel** for entire long-context verify steps, and this restores the fast path.

**Evidence split**

Rebased onto the current vLLM version, with the pre-rebase deployment measurements and the post-rebase kernel checks kept apart rather than pooled.

The quality check allows a drop of up to one point; both drops are inside it, and the cards above show the measured values. Two runs of the same build also differ by up to a point, because the engine is not deterministic, so a drop this small cannot be separated from noise, and the check will sometimes reject a change that did not hurt quality.

We cannot give a campaign success rate. Other campaigns produced candidates that never reached these checks, so the four finished campaigns are not the full set. The appendix has a rate from a study we ran on ourselves, on different hardware and a different model, and it does not apply to these four.

---

*Cost*

## The three campaigns we can price cost about $10,000, mostly in cache traffic

Four campaigns ran to completion. Three could be reconstructed from session metadata and are billed here at public API prices rather than out of a research allocation. That cache cost comes from the design: every stage re-opens the state files instead of trusting the conversation, and the monitor re-reads a transcript that only gets longer.

Reconstructed from session metadata at published prices.

The humans did five things across all four campaigns: recovered infrastructure and moved hosts; reviewed every diff line by line; inspected the benchmark and quality results; signed and posted the pull requests; and granted one scope permission, in the Kimi campaign, after the agent had already measured a win that sat outside its brief. **No human wrote kernel code, chose an optimization mechanism, or tuned a parameter.**

---

*Limitations*

## What this does not show

Everything above holds only inside these five boundaries.

**No human was measured against it.**

There was no matched human engineer and no competing agent to compare against. We are claiming that the system finished the technical work and that its output was measured. We are not claiming it replaced anybody, and we deliberately do not estimate engineer-hours saved: the search adapts as it goes, so there is no counterfactual path to price it against.

**Re-running it will not give you these candidates.**

Each campaign is one path through an adaptive, stochastic search. Re-running the harness need not find the same candidates or the same yield. What can be rechecked is the evidence: manifests, commits, commands, raw measurements, quality results and activation conditions.

**The mechanisms are not proven minimal.**

They describe the loop we configured. We have not shown that each is necessary or that the set is the smallest one that works. The guardrail we deleted is one measured instance, not a trend.

**Nothing is merged.**

All four pull requests are open or draft. Broader use needs maintainer review, a uniform paired protocol after any material rebase, operators who are not the authors, and tests on serving stacks other than vLLM.

**The study and the pull requests ran on different machines.**

Everything was author-operated, on vLLM and NVIDIA GPUs. But the four campaigns ran on B200, B300 and H100 against large mixture-of-experts models in 4-bit formats, on Claude Opus 4.6 and 4.8. The study of the monitor ran on one L40S against a small model in BF16, on Opus 4.7. Three axes differ, so the study is evidence about the method. Do not carry a ship rate from it over to the pull-request setting.

---

*Takeaway*

## Measure on the deployment, and hold the run together from outside the model

An isolated kernel speedup is a real measurement that does not predict the deployment number. Wherever we measured both, the two disagreed in size and sometimes in sign, so AMMO scores candidates on the live server, against four checks a change must clear before it ships.

The runs that produce those measurements last days, which is long enough for an agent to stop early, forget its rules, or game its own evidence. Six things held our runs together, all outside the model: one objective, a mechanical stop rule, campaign state in files rather than in the conversation, one sanctioned measurement path, rival proposers, and a monitor on implementation.

Across the campaigns with complete records, ten of nineteen isolated wins cleared the checks and shipped. Four of the shipped changes went to vLLM as pull requests. The full records are in the paper.

---

*Appendix · what watching cost*

## With the monitor on, fewer changes ship and more of them hold up

Rival agents and the monitor are the expensive part of the system, so we ran a controlled study to measure what they contribute. Five optimization tasks in four configurations: one agent proposing and implementing on its own; rival agents arguing over what to try; the whole system with the monitor switched off; and the whole system. Each task ran five times in each configuration, so 100 planned attempts of which 95 completed, all on one L40S GPU against a small Qwen model, all on Claude Opus 4.7. Same tasks, same checks. Eight more are left out of the counts below, because their transcripts show an agent reading a sibling run's work. That happened in all four configurations, so dropping them favours none of them.

Then we scored the results twice. The first score counts every change that passed every check the harness ran. The second re-reads the evidence afterwards and asks four more things of each one: did it use the mechanism it was asked to use, was its baseline fresh and measured under matching conditions, were the quality references captured before the attempt started, and was the effect bigger than the 0.5% noise floor.

> *Figure 5. Switching the monitor off adds one ship and removes one that holds up. Counted by ships alone, the monitor looks like a net loss. The unsupervised arm kept everything it shipped, and it shipped three times out of twenty-one, so that 100% rests on three ships. Even that arm is handed the loop's frozen evidence, its state files and its checks, so **3 of 21 is a ceiling on what a bare agent would manage rather than a measurement of one**. Five attempts per cell is not a significance claim and we do not make one. These are directional results from a study we ran on ourselves.*

Against the unsupervised configuration the whole system cost about **1.6× the GPU-hours** and beat it on both scores: six ships against three, and five that held up against three. The configuration that ships most is rival agents arguing, at eight, and it holds up least often, at four of the eight: a high ship count says nothing about how many of the ships hold up. The one close call is between the whole system and the same system with the monitor switched off.

## The task we rigged

One of the five tasks was built so that nothing should ship. It asked for INT4 quantization of a particular matrix multiply, and at that shape INT4 costs more model quality than the gate allows. The only correct outcome is to hand back nothing.

Two configurations shipped it **six times**: rival agents arguing, and the whole system with the monitor switched off. Four of those six contained no INT4 at all. The agent had substituted a lossless kernel in a different number format, which is faster and no less accurate and therefore passes every check, while quietly dropping the one thing it was asked to do. **No check looks for that.** The full system shipped it zero times, and it is the only configuration in the study that ever stopped early to say no viable candidate exists.

Take the rigged task out and the difference nearly vanishes: five ships from eighteen against six from seventeen, with four of five holding up against five of six. The monitor's measured value sits almost entirely on the task designed to tempt an agent into answering a different question, and on runs surviving at all. Three runs with the monitor switched off died when the coordinating agent sent an erroneous shutdown to its own workers. The monitored ones had none.

## One guardrail we deleted, one we kept despite its cost

An independent agent used to re-test every candidate under an independence rule: write your own tests, do not read the implementer's. It blocked the implementer until it returned a verdict. Across **32 dispatches it changed zero shipping decisions**, found no discrepancy in 24 of them, and cost a median of 8.16 minutes of blocked time on every dispatch. We deleted it and folded its checks into the implementer. Whether a stronger model made it redundant is a guess. The decision to delete it was measured.

The monitor survived that same test, and it is not cheap: it re-reads a growing transcript many times per campaign, which roughly doubles the cost of the stage it watches. On implementation it catches two things no artifact check reaches, changes built so the check passes as run, and implementers that have silently wedged. On the debate stage it changed no outcome we could find, because the rival agents already argue with each other. So it runs on implementation and nowhere else.

## Citation

You can cite this blog here:

```bibtex
@article{ammo2026blog,
  title   = "A Fast Kernel Is Not a Ship: optimizing live vLLM deployments, measured end to end",
  author  = "The AMMO team",
  year    = "2026",
  month   = "August",
  url     = "https://amazon-science.github.io/ammo/blog/"
}
```
