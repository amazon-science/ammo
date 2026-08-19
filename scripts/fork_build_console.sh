#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Build console for custom vLLM fork sessions. ttyd launches this as the
# terminal command. It streams the server-written build log live, then:
#   - on success sentinel ("ok")  → exec the CLI argv passed after the paths
#   - on failure sentinel ("failed") → show a clear message and idle
#
# Usage: fork_build_console.sh <status_file> <build_log> -- <cli argv...>
set -u
STATUS_FILE="$1"; shift
BUILD_LOG="$1"; shift
[ "${1:-}" = "--" ] && shift   # drop the separator if present

touch "$BUILD_LOG" 2>/dev/null || true
echo "── Building vLLM from your fork (full source build) ──"
echo "   This can take 15-20 min cold, ~1-3 min with warm ccache."
echo ""
# Stream the build log as the server writes it.
tail -n +1 -F "$BUILD_LOG" 2>/dev/null &
TAIL_PID=$!

# Wait for the server to signal completion via a non-empty status file.
while [ ! -s "$STATUS_FILE" ]; do sleep 1; done
sleep 1                       # let tail flush the final lines
kill "$TAIL_PID" 2>/dev/null || true

STATUS="$(cat "$STATUS_FILE" 2>/dev/null)"
if [ "$STATUS" = "ok" ]; then
    echo ""
    echo "── Build succeeded. Starting agent... ──"
    exec "$@"
else
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  vLLM fork build FAILED — session marked FAILED."
    echo "  Review the build output above for the error."
    echo "════════════════════════════════════════════════════"
    # Idle so the pane keeps the log visible; the session is FAILED and holds
    # no GPUs (released by the server).
    sleep infinity
fi
