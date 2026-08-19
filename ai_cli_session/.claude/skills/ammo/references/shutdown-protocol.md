# Shutdown Protocol (Agent Teardown)

This document owns one rule set: how a team member shuts down after the orchestrator decides its work is done. It is the canonical definition, so debate champions, implementation champions, and transcript monitors all point here instead of repeating it. Read it before you answer a shutdown request, and read it before you drain a round.

## How Teardown Works

1. The orchestrator sends a structured `shutdown_request` to the member: `SendMessage(to=<agent_id>, message={"type": "shutdown_request"})`.
2. The member replies with a structured `shutdown_response` that echoes the `request_id` and sets `approve` true or false.
3. **Approving terminates the member's process.** So reply only against your actual state: approve only when your work is genuinely complete.

## The `shutdown_response` Shape

Approve when your work is complete, and make no further tool calls afterward:

```json
{"type": "shutdown_response", "request_id": "<echo the request_id>", "approve": true}
```

Decline when work is outstanding — name what remains and give an ETA in `reason`, finish the work, then approve the next request:

```json
{"type": "shutdown_response", "request_id": "<echo>", "approve": false, "reason": "<what remains + ETA>"}
```

**Only the structured `shutdown_response` shuts you down. A prose reply does not.** A plain-text "ok" or "done" is not a teardown; the orchestrator waits for the structured `approve: true`.

## Per-Role Approve Conditions

| Role | Approve (`approve: true`) when | Decline (`approve: false`) when |
|---|---|---|
| **Debate champion** (`ammo-champion`) | Proposal written, debate rounds done, no open items you raised | Unwritten rebuttal, an open item you declared in Phase C, a mid-flight micro-experiment |
| **Impl champion** (`ammo-impl-champion`) | Track terminal — you have sent `TRACK_COMPLETE` with a PASS / GATED_PASS / FAIL verdict and `validation_results.md` is written | Gates still running, a fix in progress, fix-attempt budget not yet exhausted, `validation_results.md` unwritten |
| **Transcript monitor** (`ammo-transcript-monitor`) | Monitoring complete — your champion reached a completion signal and you have sent the DA-MONITOR SUMMARY | Champion still active — no completion signal yet, an interjection in flight |

When you decline, finish the outstanding work to a terminal state (terminal verdict / DA-MONITOR SUMMARY / written proposal), then approve the next request.

## Orchestrator Rules

Send `shutdown_request` to each member, and wait a bounded 5 minutes for the structured approval before you proceed.

- A member approves only when its work is complete, so if one replies `approve: false`, let it finish and re-send once it reports done.
- If no approval arrives within 5 minutes, check the member's transcript. No new events since the request: treat the member as drained — remove or kill its roster entry and proceed. Still active: wait one more bounded 5 minutes, then hard-stop the agent with `TaskStop` on its teammate name.
- Confirm each member left the team roster (its `shutdown_approved` arrived, or the bounded-wait rule declared it drained) before you spawn the next phase's agents.
- The round is fully drained once every member has replied `shutdown_approved` or was declared drained by the bounded-wait rule.
- Team teardown and worktree cleanup rules are owned by `orchestration/parallel-tracks.md § Shutdown and Cleanup`.
