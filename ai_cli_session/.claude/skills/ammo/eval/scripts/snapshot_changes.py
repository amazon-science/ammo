#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Snapshot all code changes from an AMMO campaign for eval archiving.

Reads state.json to identify campaign worktrees, then captures:
- Full git diff (patch) for each worktree vs main
- Commit log for each worktree
- Worktree metadata (branch, merge status, changed files)
- Complete copy of the artifact directory (skipping huge binary files)

The output directory can be passed to archive_run.py via --changes-snapshot.

Usage:
  python snapshot_changes.py \
    --artifact-dir /path/to/kernel_opt_artifacts/auto_... \
    --output /tmp/ammo_eval_changes_snapshot
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_git(
    args: List[str], cwd: Optional[str] = None, timeout: int = 30
) -> Optional[str]:
    """Run a git command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _find_campaign_worktrees(
    artifact_dir: Path, repo_root: Path
) -> List[Dict[str, str]]:
    """Find campaign worktrees from every round's parallel_tracks.tracks."""
    state_path = artifact_dir / "state.json"
    if not state_path.exists():
        return []

    state = json.loads(state_path.read_text(encoding="utf-8"))
    rounds = state.get("campaign", {}).get("rounds", [])

    worktrees = []
    seen_paths = set()
    for rnd in rounds:
        if not isinstance(rnd, dict):
            continue
        tracks = rnd.get("parallel_tracks", {}).get("tracks", {})
        if not isinstance(tracks, dict):
            continue
        for track_id, track_data in tracks.items():
            if not isinstance(track_data, dict):
                continue
            wt_path = track_data.get("worktree") or track_data.get("worktree_path")
            branch = track_data.get("branch") or track_data.get("worktree_branch")
            if wt_path and wt_path not in seen_paths:
                seen_paths.add(wt_path)
                worktrees.append({
                    "path": wt_path,
                    "branch": branch or "unknown",
                    "track_id": track_id,
                    "status": track_data.get("status", "UNKNOWN"),
                })

    # Also check for additional worktrees under .claude/worktrees/ that match
    # the same session but aren't in parallel_tracks (e.g. researcher agents)
    session_ids = set()
    for wt in worktrees:
        branch = wt.get("branch", "")
        if "session/" in branch:
            parts = branch.split("/")
            if len(parts) >= 2:
                session_ids.add(parts[1])

    if session_ids:
        for wt_parent in [repo_root / ".claude" / "worktrees",
                          repo_root / ".codex" / "worktrees"]:
            if not wt_parent.exists():
                continue
            for wt_dir in sorted(wt_parent.iterdir()):
                if not wt_dir.is_dir():
                    continue
                wt_str = str(wt_dir)
                if wt_str in seen_paths:
                    continue
                branch = _run_git(
                    ["rev-parse", "--abbrev-ref", "HEAD"], cwd=wt_str
                )
                if not branch:
                    continue
                for sid in session_ids:
                    if sid in branch:
                        seen_paths.add(wt_str)
                        worktrees.append({
                            "path": wt_str,
                            "branch": branch,
                            "track_id": None,
                            "status": "session-member",
                        })
                        break

    return worktrees


def _capture_worktree_snapshot(
    wt_info: Dict[str, str], main_branch: str = "main"
) -> Dict[str, Any]:
    """Capture full metadata + patch for a single worktree."""
    wt_path = wt_info["path"]

    if not Path(wt_path).exists():
        return {
            **wt_info,
            "error": "worktree path does not exist",
            "commit": None,
            "patch": None,
        }

    commit = _run_git(["rev-parse", "HEAD"], cwd=wt_path) or "unknown"
    short_hash = _run_git(["rev-parse", "--short", "HEAD"], cwd=wt_path) or "unknown"

    # Commits since diverging from main
    merge_base = _run_git(["merge-base", main_branch, "HEAD"], cwd=wt_path)
    if merge_base:
        log = _run_git(["log", "--oneline", f"{merge_base}..HEAD"], cwd=wt_path) or ""
        diff = _run_git(
            ["diff", merge_base, "HEAD"], cwd=wt_path, timeout=60
        ) or ""
    else:
        log = _run_git(["log", "--oneline", "-10"], cwd=wt_path) or ""
        diff = _run_git(
            ["diff", f"{main_branch}...HEAD"], cwd=wt_path, timeout=60
        ) or ""

    # Uncommitted changes on top
    uncommitted = _run_git(["diff", "HEAD"], cwd=wt_path, timeout=60) or ""

    # Check merge status
    is_merged = False
    if merge_base:
        check = _run_git(
            ["branch", "--contains", commit, main_branch], cwd=wt_path
        )
        is_merged = check is not None and main_branch in (check or "")

    # Changed files list
    if merge_base:
        changed = _run_git(
            ["diff", "--name-only", merge_base, "HEAD"], cwd=wt_path
        ) or ""
    else:
        changed = _run_git(
            ["diff", "--name-only", f"{main_branch}...HEAD"], cwd=wt_path
        ) or ""

    return {
        "name": Path(wt_path).name,
        "path": wt_path,
        "branch": wt_info["branch"],
        "track_id": wt_info.get("track_id"),
        "track_status": wt_info.get("status"),
        "commit": commit,
        "short_hash": short_hash,
        "is_merged_to_main": is_merged,
        "commit_log": log,
        "changed_files": [f for f in changed.split("\n") if f],
        "patch": diff,
        "uncommitted_diff": uncommitted if uncommitted else None,
    }


# Files to skip when copying artifacts (large binaries, caches)
_SKIP_EXTENSIONS = {".nsys-rep", ".ncu-rep", ".sqlite", ".qdstrm"}
_SKIP_NAMES = {"__pycache__", ".git"}
_MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB


def _copy_ignore(directory: str, files: list) -> list:
    """shutil.copytree ignore callback: skip large binaries and caches."""
    ignored = []
    for f in files:
        if f in _SKIP_NAMES:
            ignored.append(f)
            continue
        fpath = Path(directory) / f
        if fpath.suffix in _SKIP_EXTENSIONS:
            ignored.append(f)
            continue
        if fpath.is_file():
            try:
                if fpath.stat().st_size > _MAX_FILE_SIZE:
                    ignored.append(f)
            except OSError:
                pass
    return ignored


def snapshot_changes(
    artifact_dir: Path,
    output_dir: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    """Create a full snapshot of all campaign changes.

    Returns the manifest dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- worktrees ---
    worktrees = _find_campaign_worktrees(artifact_dir, repo_root)
    wt_metadata = []
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(exist_ok=True)

    for wt in worktrees:
        snap = _capture_worktree_snapshot(wt)
        # Store patch as a separate file (can be large)
        safe_name = snap["name"]
        if snap.get("patch"):
            (patches_dir / f"{safe_name}.patch").write_text(
                snap["patch"], encoding="utf-8"
            )
        if snap.get("uncommitted_diff"):
            (patches_dir / f"{safe_name}_uncommitted.patch").write_text(
                snap["uncommitted_diff"], encoding="utf-8"
            )
        # Strip large text from metadata (it's in the files)
        meta = {k: v for k, v in snap.items()
                if k not in ("patch", "uncommitted_diff")}
        meta["has_patch"] = bool(snap.get("patch"))
        meta["has_uncommitted"] = bool(snap.get("uncommitted_diff"))
        wt_metadata.append(meta)

    # --- artifact directory ---
    artifacts_dest = output_dir / "campaign_artifacts"
    artifacts_copied = False
    skipped_files: List[str] = []
    if artifact_dir.exists():
        shutil.copytree(
            artifact_dir, artifacts_dest,
            ignore=_copy_ignore, dirs_exist_ok=True,
            symlinks=False,  # resolve symlinks so we capture actual data
        )
        artifacts_copied = True

    # --- main repo state ---
    main_commit = _run_git(["rev-parse", "HEAD"], cwd=str(repo_root)) or "unknown"
    main_short = _run_git(["rev-parse", "--short", "HEAD"], cwd=str(repo_root)) or "unknown"
    main_diff_stat = _run_git(["diff", "HEAD", "--stat"], cwd=str(repo_root)) or ""
    skill_diff = _run_git(
        ["diff", "HEAD", "--", ".claude/skills/ammo/"], cwd=str(repo_root)
    ) or ""
    if skill_diff:
        (output_dir / "skill.patch").write_text(skill_diff, encoding="utf-8")

    # --- manifest ---
    manifest = {
        "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "artifact_dir": str(artifact_dir),
        "main_branch": {
            "commit": main_commit,
            "short_hash": main_short,
            "has_uncommitted_changes": bool(main_diff_stat),
        },
        "worktrees": wt_metadata,
        "worktree_count": len(wt_metadata),
        "artifacts_copied": artifacts_copied,
        "has_skill_diff": bool(skill_diff),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifact-dir", type=str, required=True,
        help="Path to campaign artifact directory",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for the changes snapshot",
    )
    parser.add_argument(
        "--repo-root", type=str, default=None,
        help="Repository root (auto-detected if omitted)",
    )

    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if args.repo_root:
        repo_root = Path(args.repo_root).expanduser().resolve()
    else:
        detected = _run_git(["rev-parse", "--show-toplevel"])
        if detected:
            repo_root = Path(detected)
        else:
            print("ERROR: Could not detect git repo root. Use --repo-root.",
                  file=sys.stderr)
            return 1

    manifest = snapshot_changes(artifact_dir, output_dir, repo_root)

    print(f"Snapshot complete: {manifest['worktree_count']} worktrees captured")
    print(f"Artifacts copied: {manifest['artifacts_copied']}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
