#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Archive a scored AMMO eval run into the versioned eval repository.

Creates the repository structure if it doesn't exist. Each run is stored
under versions/{version_id}/runs/{target_slug}/run_{N}/.

Version IDs are derived as: {git_short_hash}_{slugified_description}

Usage:
  python archive_run.py \
    --scorecard scorecard.json \
    --snapshot artifacts_snapshot.json \
    --description "Improve debate convergence" \
    --repository ~/.claude/ammo-eval
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]


def _git_short_hash() -> Optional[str]:
    """Get the short git commit hash of the current HEAD."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _git_skill_diff() -> Optional[str]:
    """Get the git diff of the ammo skill directory vs HEAD."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--", ".claude/skills/ammo/"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout if result.stdout.strip() else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _derive_version_id(git_hash: Optional[str], description: str) -> str:
    """Derive version_id from git hash + description."""
    slug = _slugify(description)
    if git_hash:
        return f"{git_hash}_{slug}"
    # Fallback: use timestamp
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{ts}_{slug}"


def _next_run_number(target_dir: Path) -> int:
    """Find the next available run number for a target."""
    if not target_dir.exists():
        return 1
    existing = [
        d.name for d in target_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    numbers = []
    for name in existing:
        try:
            numbers.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers) + 1 if numbers else 1


def _derive_target_slug(snapshot: Dict[str, Any]) -> str:
    """Derive a filesystem-safe target slug from the snapshot."""
    target = snapshot.get("target") or {}
    model = target.get("model_id", "unknown")
    # Take last part of model path, lowercase
    model_short = model.split("/")[-1].lower()
    model_short = re.sub(r"[^a-z0-9]", "-", model_short)
    model_short = re.sub(r"-+", "-", model_short).strip("-")[:30]

    hw = (target.get("hardware") or "unknown").lower()
    dtype = (target.get("dtype") or "unknown").lower()
    tp = target.get("tp", 1)
    return f"{model_short}_{hw}_{dtype}_tp{tp}"


def archive_run(
    scorecard_path: Path,
    snapshot_path: Path,
    description: str,
    repository: Path,
    run_number: Optional[int] = None,
    transcript_grading_path: Optional[Path] = None,
    git_hash: Optional[str] = None,
    changes_snapshot_path: Optional[Path] = None,
    causal_dag_path: Optional[Path] = None,
    postmortem_narrative_path: Optional[Path] = None,
    causal_viz_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Archive a scored run into the repository. Returns paths created."""
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    if git_hash is None:
        git_hash = _git_short_hash()
    version_id = _derive_version_id(git_hash, description)
    target_slug = _derive_target_slug(snapshot)

    # Create directory structure
    version_dir = repository / "versions" / version_id
    target_dir = version_dir / "runs" / target_slug

    if run_number is None:
        run_number = _next_run_number(target_dir)
    run_dir = target_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy scorecard and snapshot
    shutil.copy2(scorecard_path, run_dir / "scorecard.json")
    shutil.copy2(snapshot_path, run_dir / "artifacts_snapshot.json")

    # Copy report.md if it exists alongside scorecard
    report_path = scorecard_path.parent / "report.md"
    if report_path.exists():
        shutil.copy2(report_path, run_dir / "report.md")

    # Copy transcript grading if provided
    if transcript_grading_path and transcript_grading_path.exists():
        shutil.copy2(transcript_grading_path, run_dir / "transcript_grading.json")

    # Store reference to original artifact directory
    artifact_dir = snapshot.get("artifact_dir", "unknown")
    (run_dir / "artifact_dir_ref.txt").write_text(artifact_dir + "\n", encoding="utf-8")

    # Copy changes snapshot (worktree patches, artifact copies, manifest)
    if changes_snapshot_path and changes_snapshot_path.is_dir():
        dest = run_dir / "changes_snapshot"
        shutil.copytree(changes_snapshot_path, dest, dirs_exist_ok=True)

    # Copy causal engine outputs if provided
    if causal_dag_path and causal_dag_path.exists():
        shutil.copy2(causal_dag_path, run_dir / "causal_dag.json")
    if postmortem_narrative_path and postmortem_narrative_path.exists():
        shutil.copy2(postmortem_narrative_path, run_dir / "postmortem.md")
    if causal_viz_path and causal_viz_path.exists():
        shutil.copy2(causal_viz_path, run_dir / "causal_viz.html")

    # Create/update meta.json for the version
    meta_path = version_dir / "meta.json"
    if not meta_path.exists():
        skill_diff = _git_skill_diff()
        meta = {
            "version_id": version_id,
            "git_commit": git_hash,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "skill_diff": skill_diff,
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return {
        "version_id": version_id,
        "target_slug": target_slug,
        "run_number": run_number,
        "run_dir": str(run_dir),
        "version_dir": str(version_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scorecard", type=str, required=True)
    parser.add_argument("--snapshot", type=str, required=True)
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument(
        "--repository", type=str,
        default=str(Path.home() / ".claude" / "ammo-eval"),
    )
    parser.add_argument("--run-number", type=int, default=None)
    parser.add_argument("--transcript-grading", type=str, default=None)
    parser.add_argument("--git-hash", type=str, default=None)
    parser.add_argument("--changes-snapshot", type=str, default=None,
                        help="Path to changes snapshot directory from snapshot_changes.py")
    parser.add_argument("--causal-dag", type=str, default=None,
                        help="Path to causal_dag_final.json from generate_postmortem.py")
    parser.add_argument("--postmortem-narrative", type=str, default=None,
                        help="Path to postmortem.md from generate_postmortem.py")
    parser.add_argument("--causal-viz", type=str, default=None,
                        help="Path to causal_viz.html from generate_postmortem.py")

    args = parser.parse_args()

    result = archive_run(
        scorecard_path=Path(args.scorecard).expanduser().resolve(),
        snapshot_path=Path(args.snapshot).expanduser().resolve(),
        description=args.description,
        repository=Path(args.repository).expanduser().resolve(),
        run_number=args.run_number,
        transcript_grading_path=(
            Path(args.transcript_grading).expanduser().resolve()
            if args.transcript_grading else None
        ),
        git_hash=args.git_hash,
        changes_snapshot_path=(
            Path(args.changes_snapshot).expanduser().resolve()
            if args.changes_snapshot else None
        ),
        causal_dag_path=(
            Path(args.causal_dag).expanduser().resolve()
            if args.causal_dag else None
        ),
        postmortem_narrative_path=(
            Path(args.postmortem_narrative).expanduser().resolve()
            if args.postmortem_narrative else None
        ),
        causal_viz_path=(
            Path(args.causal_viz).expanduser().resolve()
            if args.causal_viz else None
        ),
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
