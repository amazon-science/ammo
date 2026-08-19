#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PreToolUse hook — always-on, session-wide DENY of package-install commands.
#
# Matcher: Bash (configured in settings.local.json).
#
# The session .venv is pre-built with vLLM editable-installed. Per CLAUDE.md,
# agents MUST NEVER run `pip install`, `pip3 install`, `uv pip install`, or
# `python -m pip install` (nor the uninstall variants) — doing so corrupts the
# carefully provisioned environment. This guard blocks every such command.
#
# It is DISTINCT from the two sibling Bash guards:
#   - ammo-pretool-guard.sh    — wrong-tree one-shot venv block (campaign-scoped,
#                                fires once per session)
#   - ammo-venv-python-guard.sh — bare-python invocation of sweep scripts
# This guard is unconditional and fires on every Bash invocation.
#
# Block mechanism: `exit 2` + a clear stderr message pointing at the .venv
# policy — the established PreToolUse Bash deny convention in this repo
# (ammo-venv-python-guard.sh:37, ammo-pretool-guard.sh:113), DISTINCT from the
# PostToolUse decision:block JSON used by the state-validate / teamdelete hooks.
#
# Read-only pip forms (list / show / freeze / --version / config / cache /
# check) are ALLOWED — only install/uninstall are denied.
#
# Escape hatch: AMMO_ALLOW_PIP=1 → ALLOW (the "If .venv is Missing" session-
# infrastructure path in CLAUDE.md, for the provisioning system only).
# Fail-open: missing jq → exit 0.
set -euo pipefail

if ! command -v jq &>/dev/null; then exit 0; fi

# Escape hatch for session-infrastructure provisioning.
if [ "${AMMO_ALLOW_PIP:-}" = "1" ]; then exit 0; fi

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // .input.command // empty' 2>/dev/null) || true
[ -z "$COMMAND" ] && exit 0

# Match a package-install/uninstall invocation at COMMAND POSITION only:
#   - pip / pip3            install|uninstall
#   - uv pip                install|uninstall
#   - python[3] -m pip      install|uninstall
# Classification is delegated to skills/ammo/scripts/hook_cmd_classify.py
# (--mode install), which tokenizes the command and inspects each segment. It
# replaces a regex that required the verb to sit IMMEDIATELY after the invoker
# and matched the invoker name literally, so all of these passed:
#   pip -q install / pip3 --quiet install / python -m pip -q install
#   /usr/bin/pip install / .venv/bin/pip install   (path-qualified)
#   sudo pip install / env FOO=1 uv pip install    (wrappers)
#   bash -c "pip install ..."                      (nested shell)
# Quoted mentions (`grep "pip install"`, `echo "...pip install..."`) are
# arguments, not invokers, and still ALLOW. Read-only verbs (list/show/freeze/
# --version/config/cache/check) are not install verbs, so they ALLOW too.
#
# Fail-open on a missing classifier or interpreter (exit code 2 from the
# classifier means "could not classify", never "install"): this guard runs on
# EVERY Bash call, so a degraded env must not block every command. The pip
# policy is also carried in CLAUDE.md prose and the .venv provisioning path.
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFY="$HOOK_DIR/../skills/ammo/scripts/hook_cmd_classify.py"
if [ -x "$HOOK_DIR/../../.venv/bin/python" ]; then
    CLASSIFY_PY="$HOOK_DIR/../../.venv/bin/python"
else
    CLASSIFY_PY="python3"
fi

[ -f "$CLASSIFY" ] || exit 0
command -v "$CLASSIFY_PY" &>/dev/null || exit 0

_VERDICT=0
"$CLASSIFY_PY" "$CLASSIFY" --mode install "$COMMAND" >/dev/null 2>&1 || _VERDICT=$?

if [ "$_VERDICT" -eq 0 ]; then
    cat >&2 <<'EOF'
BLOCKED: package install/uninstall is forbidden in AMMO sessions.

The session .venv is pre-built with vLLM editable-installed. Running
  pip install / pip3 install / uv pip install / python -m pip install
(or the uninstall variants) corrupts the provisioned environment and
silently invalidates profiling/benchmark results.

Policy (CLAUDE.md): NEVER pip install / uv pip install / create a new venv.
Just `source .venv/bin/activate` — every required package is already present.
If `import vllm` fails, REPORT the error; do not try to fix it by installing.

Escape hatch (session-infrastructure provisioning ONLY):
  AMMO_ALLOW_PIP=1 <your command>
EOF
    exit 2
fi

exit 0
