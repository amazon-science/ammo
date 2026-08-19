#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
set -euo pipefail

# WorktreeRemove hook for vLLM
# Cleans up a worktree and its associated git branch.
#
# Input (stdin JSON): { worktree_path, ... }
# All output goes to stderr.

INPUT=$(cat)
WORKTREE_PATH=$(echo "$INPUT" | jq -r '.worktree_path // empty')
[ -z "$WORKTREE_PATH" ] && exit 0
[ ! -d "$WORKTREE_PATH" ] && exit 0

MAIN_REPO="${CLAUDE_PROJECT_DIR:-}"
[ -z "$MAIN_REPO" ] && MAIN_REPO=$(git -C "$WORKTREE_PATH" rev-parse --path-format=absolute --git-common-dir 2>/dev/null | sed 's|/\.git$||')

WORKTREE_NAME=$(basename "$WORKTREE_PATH")
echo "Removing worktree '$WORKTREE_NAME'..." >&2

# Detect actual branch before removal (may be ammo/* or worktree-*)
BRANCH_NAME=$(git -C "$WORKTREE_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null || true)

# Warn about uncommitted changes
if ! git -C "$WORKTREE_PATH" diff --quiet 2>/dev/null || \
   ! git -C "$WORKTREE_PATH" diff --cached --quiet 2>/dev/null; then
    echo "WARN: Worktree has uncommitted changes" >&2
fi

# Remove the worktree
git -C "$MAIN_REPO" worktree remove --force "$WORKTREE_PATH" 2>&1 >&2 || {
    rm -rf "$WORKTREE_PATH" 2>/dev/null || true
    git -C "$MAIN_REPO" worktree prune 2>&1 >&2 || true
}

# Delete the branch (only worktree-* and ammo/* branches, never main/master)
if [[ "$BRANCH_NAME" == worktree-* || "$BRANCH_NAME" == ammo/* ]]; then
    git -C "$MAIN_REPO" branch -D "$BRANCH_NAME" 2>&1 >&2 && \
        echo "Deleted branch $BRANCH_NAME" >&2 || true
fi

echo "Done." >&2
