#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# AMMO teammate exec wrapper — per-agent-type high-effort arming.
#
# Purpose: Claude Code agent-team teammates are spawned as SEPARATE claude
# processes in tmux panes. This wrapper is the one per-agent-type seam
# available at that spawn point, so it is where the high-effort orchestration
# configuration is turned ON for ONLY the two champion agent types
# (ammo-champion, ammo-impl-champion) — without making it session-wide.
#
# How it's wired: CLAUDE_CODE_TEAMMATE_COMMAND is set on the orchestrator's
# process env in SessionManager._build_extra_env
# (orchestration/session_manager.py); Claude Code uses it as the exec path for
# every tmux teammate it spawns, passing `--agent-type {type}` among the args.
#
# Scope: ONLY agent-team teammates (separate claude processes in tmux panes)
# route through this wrapper. Task-tool subagents run in-process inside their
# spawning teammate, so a champion's research/profiling/validator subagents
# inherit that champion's environment automatically.
#
# Binary resolution: the server launches claude by the hardcoded absolute path
# /usr/bin/claude (CLI_TOOL_CONFIGS[CLAUDE].command, cli_tool_manager.py). We
# resolve the SAME absolute path and deliberately do NOT do a PATH lookup
# (`command -v claude`): this wrapper may itself be on PATH, and a lookup could
# re-resolve to this script and infinite-loop. CLAUDE_REAL_BIN overrides for
# non-/usr/bin layouts (local dev / a different image).
#
# Why this is safe: arming is additive. It only appends argv (no command, no I/O
# on the critical path) and fires for exactly the two champion types; every
# other teammate execs unchanged.
#
# Failure posture: this wrapper sits on the critical path of EVERY teammate
# spawn. It must NEVER prevent a spawn. All logic is best-effort and guarded; on
# any internal error it falls through and execs the real binary with the
# orchestrator-provided env unchanged.
set -uo pipefail

REAL_CLAUDE="${CLAUDE_REAL_BIN:-/usr/bin/claude}"

# ---- champion arming (champions only) ----------------------------------------
# Agent types that get the high-effort configuration. Defaults to ONLY the two
# champion definition basenames (the exact value Claude Code passes as
# `--agent-type`). Space-separated; override via env to widen/narrow. Empty
# disables arming.
AMMO_ARM_AGENT_TYPES="${AMMO_ARM_AGENT_TYPES:-ammo-champion ammo-impl-champion}"
# Argv tokens appended (after "$@") to enable the configuration. A bash ARRAY
# so the JSON object arrives as ONE argv token with its braces intact. This
# single line is the one place to edit if the settings ever change.
ARM_ARGS=(--settings '{"ultracode":true,"enableWorkflows":true}')

# ---- Extract --agent-type from argv (the value AFTER the flag) ---------------
# Match the token after the flag specifically, never a bare arg or an
# --agent-name value that happens to equal a type. Claude Code always passes the
# definition name as `--agent-type <type>`. Single pass over argv; argv itself is
# never modified.
_agent_type=""
_prev=""
for arg in "$@"; do
    case "$_prev" in
        --agent-type) _agent_type="$arg" ;;
    esac
    _prev="$arg"
done

# ---- Arm champion teammates only --------------------------------------------
# Best-effort and additive: only the champion agent types get ARM_ARGS appended;
# on an empty/mismatched --agent-type (or an empty type list) we append nothing.
# This never blocks the spawn — worst case it passes one extra (valid)
# --settings flag.
_arm=0
if [ -n "$_agent_type" ] && [ -n "${AMMO_ARM_AGENT_TYPES// /}" ]; then
    for _at in $AMMO_ARM_AGENT_TYPES; do
        if [ "$_agent_type" = "$_at" ]; then
            _arm=1
            break
        fi
    done
fi

# Exec the real binary. exec replaces this process so the teammate IS claude (no
# extra PID / signal-forwarding gap). Champions get ARM_ARGS appended (winning
# last); every other teammate execs with the orchestrator's argv unchanged.
# "${ARM_ARGS[@]}" on a populated array is safe under set -u; the _arm gate keeps
# it off the non-champion path entirely.
if [ "$_arm" = "1" ]; then
    # Make the workflows feature available for armed champions. Scoped HERE
    # (not in settings.local.json env) so only armed champion processes get it.
    export CLAUDE_CODE_WORKFLOWS="${CLAUDE_CODE_WORKFLOWS:-1}"
    exec "$REAL_CLAUDE" "$@" "${ARM_ARGS[@]}"
fi
exec "$REAL_CLAUDE" "$@"
