# Champion Common Patterns

Debate champions and implementation agents both follow the rules below. Your
role file owns the technical behavior. This file covers only four things: the
Python environment you must run in, what to delegate, how to stay reachable
during long runs, and what to do with a review finding that arrives mid-track.

## Worktree Environment

Run every Python command inside `.codex/worktrees/<op_id>/` with the
worktree-local `.venv/bin/python`, including pytest. The outer session venv can
import the session source tree instead of your edits, and it does so silently —
your correctness and performance evidence becomes worthless with no error to
warn you.

Prove the environment before any GPU or validation work:

```bash
.venv/bin/python -c "import vllm; print(vllm.__file__)"
```

The printed path must sit inside your assigned worktree. Check it again after
every resume. Never install a dependency and never build a second environment.

## Delegation

Hand bounded work to a delegate task: extraction, source tracing, arithmetic,
prior-art search, and experiment execution. Keep synthesis, interpretation, and
verdicts for yourself. Spawn independent tasks in parallel when that helps, and
use `fork_turns="none"` for AMMO roles.

Give each task the raw `op_id`, absolute worktree and artifact paths, the exact
question, the evidence you expect back, and pointers to the authorities that
apply.

Runtime task names use a lowercase safe slug; artifacts and `state.json` always
carry the raw identifier. Reuse an idle delegate for a related follow-up. Start
a fresh delegate whenever you need independence or a clean context.

## Long Runs and Responsiveness

Never wait in a foreground sleep loop, for an agent or for GPU capacity. Check
once, yield, then retry on a later turn. Do CPU-only work in between when you
can.

Run a long benchmark or profiler command in an execution cell you own, so you
can resume it with the wait mechanism. Yield between checks to let monitor and
lead messages reach you.

Do not detach a process you do not own. Do not start a second GPU run while one
is still active.

## Transcript Monitor

Only Stage 4-5 implementers get a paired transcript monitor. Debate quality
comes from critique, rebuttal, and the open-item ledger instead.

Monitor findings are evidence, not commands:

- `CRITICAL`: stop the affected approach until you have assessed it;
- `WARNING`: investigate before you commit to it;
- `INFO`: retain for later review.

Read the evidence the finding cites. Decide whether it saw a real defect or
only unfinished work. Test the simplest competing explanation. Verify a
straightforward fact inline yourself. For an ambiguous cross-system claim, or
after your response to the same issue has failed more than once, ask a fresh
skeptical delegate to assess correctness, root cause, repair, and verification.
The implementer owns the final decision, and records why it accepted or
rejected any consequential finding.

Never bury a monitor objection by rewriting the artifact it points at. The lead
acknowledges queue records, the implementer fixes the code or the evidence, and
the monitor keeps its own read-only log.

## When to Send a Message

Report completed work with a normal return. Send a message only for a genuine
blocker, a required proposal revision, or an acknowledged monitor handoff.
Routine phase completion needs no status chatter — atomically published
artifacts and durable state say it for you.
