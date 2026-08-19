# Shutdown Protocol (Current Codex Subagent Teardown)

This document owns one rule set: how the lead releases an AMMO subagent after its work is done. The rule is completion-first — evidence decides release, not a self-report — because current Codex exposes `send_message`, `followup_task`, and `interrupt_agent`, and has no structured self-shutdown handshake or close primitive. Read it before you release any agent, and read it before you drain a round.

## How Teardown Works

1. Verify the role-specific completion evidence below. A final prose message alone is insufficient when the role owns an artifact or state transition.
2. If an idle or completed agent needs another bounded task, use `followup_task`; use `send_message` only to deliver information without starting a turn.
3. If the evidence is complete and the agent is still running, invoke `interrupt_agent` for that agent id to stop the live turn and free its active slot. Record the returned prior status.
4. If the agent is completed or idle, no teardown call is required, and it remains available for a later `followup_task`. Remove it from the lead's active-round mapping only after evidence reconciliation.

A stopped, failed, or interrupted agent still needs its owned work recovered or reassigned before the workflow advances. `interrupt_agent` is not proof of completion and never substitutes for validation.

## Per-Role Release Conditions

| Role | Release only when | Keep running or reassign when |
|---|---|---|
| **Debate champion** (`ammo-champion`) | Proposal and required debate artifacts exist, open items are resolved, and winner inputs are readable | Rebuttal/revision is unwritten, an open item remains, or an experiment is in flight |
| **Implementation agent** (`ammo-implementer` / legacy `ammo-impl-champion`) | Track is terminal (`PASS`, `GATED_PASS`, or `FAIL`), reconciled, and report plus structured gates exist | Gates/fixes are running, fallback ladder is incomplete, or evidence is missing |
| **Transcript monitor** (`ammo-transcript-monitor`) | Target thread is terminal and its final observation/queue records exist | Target remains active, an intervention is pending, or summary is missing |
| **Research/report/review role** | Requested artifact/verdict exists and blocking findings reached the lead | Required evidence is absent or revision was requested |

## Orchestrator Rules

- Release debate champions after winner selection and before you spawn implementers.
- When you drain a round, wait a bounded 5 minutes for a running agent to finish its turn. If its transcript shows no new events since your request, treat the agent as drained: `interrupt_agent` it and remove it from the active-round mapping. A transcript still producing events gets one more bounded 5-minute wait, then a hard `interrupt_agent`.
- Release implementers and paired monitors only after every current-round track is terminal and reconciled; Stage 6 must not discard a late verdict.
- Batch `interrupt_agent` calls only for agents that are still running; completed or idle agents need no synthetic teardown.
- Agent release never substitutes for artifact validation, state reconciliation, worktree cleanup, or audit gates.
