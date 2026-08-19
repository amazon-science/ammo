#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# PostToolUse hook — non-blocking warning when files land outside the
# canonical AMMO V2 artifact layout under kernel_opt_artifacts/.
#
# Reference: ai_cli_session/.claude/skills/ammo/references/artifact-layout.md
#
# Matchers (configured in settings.local.json): Write, Edit, Bash.
#   - Write/Edit: extracts tool_input.file_path
#   - Bash: scans command for `> path`, `mkdir -p path`, `--out PATH`, etc.,
#     limited to paths containing kernel_opt_artifacts/.
#
# Non-blocking: only emits additionalContext warnings. Never returns
# {"decision": "block"} — layout drift is an organizational hazard, not a
# correctness violation.
#
# Fail-open: any internal error (jq missing, malformed JSON) exits 0 silently.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // ""' 2>/dev/null) || true
[ -z "$TOOL" ] && exit 0

# Allowed regex patterns relative to {artifact_dir}/, DERIVED from the shared
# manifest skills/ammo/scripts/artifact_layout.json. No path fact is restated
# here: the manifest owns every template, this hook owns detection only. Adding
# a slot is a one-file edit, and the Codex twin reads the same manifest.
#
# Emitted per slot: an exact `^...$` pattern for a file template, a
# `^...(/|$)` prefix pattern for a directory template (trailing `/`), plus an
# exact pattern for every proper directory ancestor, so a bare `rounds/3/sweeps`
# token reads as a directory reference and not as drift.
#
# Fail-open and LOUD: a missing or corrupt manifest disables checking with a
# stderr diagnostic. It never degrades to an empty allowlist, which would warn
# on every correct write.
LAYOUT_MANIFEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../skills/ammo/scripts/artifact_layout.json"

PATTERN_TEXT=$(AMMO_LAYOUT_MANIFEST="$LAYOUT_MANIFEST" python3 -c '
import json, os, sys

_ERE_SPECIAL = set(r".^$*+?()[]{}|\\")


def ere_escape(text):
    return "".join("\\" + c if c in _ERE_SPECIAL else c for c in text)


def compile_template(template, tokens, placeholders):
    out, i = [], 0
    while i < len(template):
        for token in tokens:
            if template.startswith(token, i):
                out.append(placeholders[token])
                i += len(token)
                break
        else:
            out.append(ere_escape(template[i]))
            i += 1
    return "".join(out)


manifest = json.load(open(os.environ["AMMO_LAYOUT_MANIFEST"], encoding="utf-8"))
placeholders = manifest["placeholders"]
tokens = sorted(placeholders, key=len, reverse=True)
slots = manifest["slots"]
if not slots:
    raise SystemExit("empty slots")

templates = [slot["path_template"] for slot in slots.values()]
templates += manifest.get("open_dirs", [])

patterns, ancestors = [], set()
for template in templates:
    body = compile_template(template.rstrip("/"), tokens, placeholders)
    patterns.append("^" + body + ("(/|$)" if template.endswith("/") else "$"))
    segments = template.strip("/").split("/")
    for cut in range(1, len(segments)):
        prefix = "/".join(segments[:cut])
        ancestors.add("^" + compile_template(prefix, tokens, placeholders) + "$")
for pattern in patterns + sorted(ancestors):
    sys.stdout.write(pattern + "\n")
' 2>/dev/null) || PATTERN_TEXT=""

if [ -z "$PATTERN_TEXT" ]; then
    echo "LAYOUT WARN HOOK DISABLED: cannot load layout manifest ${LAYOUT_MANIFEST}. Layout drift is unchecked until it is restored." >&2
    exit 0
fi

ALLOWED_PATTERNS=()
while IFS= read -r line; do
    [ -n "$line" ] && ALLOWED_PATTERNS+=("$line")
done <<< "$PATTERN_TEXT"

if [ "${#ALLOWED_PATTERNS[@]}" -eq 0 ]; then
    echo "LAYOUT WARN HOOK DISABLED: layout manifest ${LAYOUT_MANIFEST} produced no patterns." >&2
    exit 0
fi

# Helper: emit a warning for a single non-conforming relative path.
emit_warn() {
    local rel="$1"
    local msg
    msg="LAYOUT WARN: ${rel} is outside the canonical AMMO V2 layout. Expected: rounds/{N}/{profiling|sweeps|mining|debate|tracks|audits|_archive}/... See ai_cli_session/.claude/skills/ammo/references/artifact-layout.md § Prohibited Patterns."
    jq -c -n --arg msg "$msg" '
    {
        hookSpecificOutput: {
            hookEventName: "PostToolUse",
            additionalContext: $msg
        }
    }'
}

# Helper: classify one absolute path. Returns 0 (conforming) or 1 (warn-worthy).
# Prints the relative path on warning.
check_path() {
    local p="$1"
    # Accept both absolute paths and repo-relative tokens: the Bash extractor at
    # :111 yields `kernel_opt_artifacts/...` with no leading slash, which a
    # `*/kernel_opt_artifacts/*` guard alone would skip silently.
    case "$p" in
        */kernel_opt_artifacts/*) ;;
        kernel_opt_artifacts/*) p="/$p";;
        *) return 0;;
    esac

    # Strip the longest prefix up to and including kernel_opt_artifacts/{target}/.
    # The {target} segment is whatever follows kernel_opt_artifacts/.
    local tail="${p##*/kernel_opt_artifacts/}"
    # Drop the {target}/ segment, leaving the artifact-relative path.
    local rel="${tail#*/}"
    # If tail has no slash (path was just kernel_opt_artifacts/{file}), skip.
    [ "$rel" = "$tail" ] && return 0

    # Walk allowed patterns. A bare directory token — which the Bash extractor
    # below yields from ordinary `ls`/`grep` commands — matches either its own
    # directory-prefix pattern or one of the manifest-derived ancestor patterns,
    # so `rounds/1/mining` and `rounds/1/sweeps/opt` do not warn.
    rel="${rel%/}"
    for pat in "${ALLOWED_PATTERNS[@]}"; do
        if echo "$rel" | grep -Eq "$pat" || echo "$rel/" | grep -Eq "$pat"; then
            return 0
        fi
    done

    # A proper ancestor of a canonical slot is a directory, not drift:
    # rounds/1 and rounds/1/profiling are legitimate to reference.
    case "$rel" in
        rounds|rounds/[0-9]|rounds/[0-9][0-9]|\
        rounds/[0-9]/profiling|rounds/[0-9][0-9]/profiling|\
        rounds/[0-9]/sweeps|rounds/[0-9][0-9]/sweeps|\
        rounds/[0-9]/debate|rounds/[0-9][0-9]/debate|\
        rounds/[0-9]/tracks|rounds/[0-9][0-9]/tracks|\
        rounds/[0-9]/sweeps/opt|rounds/[0-9][0-9]/sweeps/opt|\
        rounds/[0-9]/sweeps/opt_correctness|rounds/[0-9][0-9]/sweeps/opt_correctness|\
        rounds/[0-9]/sweeps/opt_profiling|rounds/[0-9][0-9]/sweeps/opt_profiling)
            return 0;;
    esac

    emit_warn "$rel"
    return 1
}

case "$TOOL" in
    Write|Edit|MultiEdit)
        FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""' 2>/dev/null) || true
        [ -z "$FILE_PATH" ] && exit 0
        check_path "$FILE_PATH" || true
        ;;
    Bash)
        CMD=$(echo "$INPUT" | jq -r '.tool_input.command // ""' 2>/dev/null) || true
        [ -z "$CMD" ] && exit 0
        # Extract candidate paths under kernel_opt_artifacts/. We grep for
        # tokens that match `[^ ]*kernel_opt_artifacts/[^ ]+`, dedup, and
        # check each. Quotes are handled by the tokenizer below.
        # Use Python to pull out plausible path tokens robustly.
        PATHS=$(printf '%s' "$CMD" | python3 -c '
import sys, re
text = sys.stdin.read()
# Match tokens that contain kernel_opt_artifacts/ and look like paths.
# Stop at whitespace, backticks, or quote boundaries.
seen = set()
out = []
for m in re.finditer(r"[^\s`\"'\''<>|;]*kernel_opt_artifacts/[^\s`\"'\''<>|;)]+", text):
    p = m.group(0).rstrip("/.,;:")
    if p and p not in seen:
        seen.add(p)
        out.append(p)
for p in out:
    print(p)
' 2>/dev/null) || true
        if [ -n "$PATHS" ]; then
            while IFS= read -r p; do
                [ -z "$p" ] && continue
                check_path "$p" || true
            done <<< "$PATHS"
        fi
        ;;
    *)
        exit 0
        ;;
esac

exit 0
