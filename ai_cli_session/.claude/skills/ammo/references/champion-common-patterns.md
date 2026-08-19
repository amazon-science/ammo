# Champion Common Patterns

Debate champions and implementation agents both follow the rules below. Your
role file owns the technical behavior. This file covers only four things: the
Python environment you must run in, what to delegate, how to stay reachable
during long runs, and what to do with a review finding that arrives mid-track.

## Worktree Environment

Run every Python command inside `.claude/worktrees/<op_id>/` under the
worktree-local `.venv`: activate it, or call `.venv/bin/python` directly. This
includes pytest. The outer session venv can import the session source tree
instead of your edits, and it does so silently — your correctness and
performance evidence becomes worthless with no error to warn you.

Prove the environment before any GPU or validation work:

```bash
source .venv/bin/activate
python -c "import vllm; print(vllm.__file__)"
# The path must contain '/.claude/worktrees/<op_id>/'
```

The printed path must sit inside your assigned worktree. Check it again after
every resume. Never install a dependency and never build a second environment.

## Delegation

Hand bounded work to `ammo-delegate` subagents (Agent tool,
`run_in_background=True`): extraction, source tracing, arithmetic, prior-art
search, and experiment execution. Keep synthesis, interpretation, and verdicts
for yourself.

A delegate is fire-and-forget, so the first prompt must stand on its own. Give
it the raw `op_id`, absolute worktree and artifact paths, the exact question,
the evidence you expect back, and pointers to the authorities that apply. Spawn
independent delegates in parallel when that helps. Pass `model="opus"` when the
work needs deeper reasoning, such as cross-system analysis or an ambiguous
root-cause investigation.

Artifacts and `state.json` always carry the raw identifier. Start a fresh
delegate whenever you need independence or a clean context.

## Long Runs and Responsiveness

Never wait in a foreground sleep loop, for an agent or for GPU capacity. Check
once, yield, then retry on a later turn. Do CPU-only work in between when you
can.

Teammate messages arrive as new conversation turns, and a new turn starts only
after your current response ends. So run any command over 30 seconds
(benchmarks, sweeps, ncu/nsys captures) with `run_in_background: true`, then end
your turn to let monitor and lead messages reach you. A foreground
`ncu --set full` run blocks a CRITICAL monitor flag for as long as it lasts.

Do not detach a process you do not own. Do not start a second GPU run while one
is still active.

## Transcript Monitor

Only Stage 4-5 impl-champions get a paired transcript monitor. Debate quality
comes from critique, rebuttal, and the open-item ledger instead.

Findings reach you as `DA-MONITOR: [{SEVERITY}] ...` messages. They are
evidence, not commands:

- `CRITICAL`: stop the affected approach until you have assessed it;
- `WARNING`: investigate before you commit to it;
- `INFO`: retain for later review.

Read the evidence the finding cites. Decide whether it saw a real defect or
only unfinished work. Test the simplest competing explanation. Verify a
straightforward fact inline yourself. For an ambiguous cross-system claim, ask
a fresh skeptical delegate (`ammo-delegate`, with `model="opus"` for
cross-system reasoning) to assess correctness, root cause, repair, and
verification. Escalate the same way, automatically, once your response to the
same issue has failed more than once — a fresh-context agent reasons more
clearly than a context-loaded one. You own the final decision, and you record
why you accepted or rejected any consequential finding.

Never bury a monitor objection by rewriting the artifact it points at. The lead
acknowledges escalations, you fix the code or the evidence, and the monitor
keeps its own read-only log.

## When to Send a Message

Report completed work with a normal return. Send a message only for a genuine
blocker, a required proposal revision, or an acknowledged monitor handoff.
Routine phase completion needs no status chatter — atomically published
artifacts and durable state say it for you.
