#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <worktree-path-or-name> [main-repo]" >&2
  exit 2
fi

INPUT_PATH="$1"
MAIN_REPO="${2:-$(git rev-parse --show-toplevel)}"
MAIN_REPO="$(cd "$MAIN_REPO" && pwd)"

if [[ "$INPUT_PATH" == */* ]]; then
  WORKTREE_PATH="$INPUT_PATH"
else
  WORKTREE_PATH="$MAIN_REPO/.codex/worktrees/$INPUT_PATH"
fi

if [[ ! -d "$WORKTREE_PATH" ]]; then
  exit 0
fi

BRANCH_NAME="$(git -C "$WORKTREE_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"

if ! git -C "$WORKTREE_PATH" diff --quiet 2>/dev/null || ! git -C "$WORKTREE_PATH" diff --cached --quiet 2>/dev/null; then
  echo "Warning: worktree has uncommitted changes: $WORKTREE_PATH" >&2
fi

git -C "$MAIN_REPO" worktree remove --force "$WORKTREE_PATH" >/dev/null 2>&1 || {
  rm -rf "$WORKTREE_PATH"
  git -C "$MAIN_REPO" worktree prune >/dev/null 2>&1 || true
}

if [[ "$BRANCH_NAME" == ammo/* || "$BRANCH_NAME" == worktree-* ]]; then
  git -C "$MAIN_REPO" branch -D "$BRANCH_NAME" >/dev/null 2>&1 || true
fi
