#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Re-score all archived AMMO eval runs using the current scoring logic.

Reads each archived run's snapshot (artifacts_snapshot.json) and optional
transcript grading, re-runs score_campaign.compute_scorecard(), and overwrites
the scorecard.json and report.md in place.

This is used after scoring logic changes (e.g., adding accuracy verification
penalties) to retroactively update all historical scores.

Usage:
  python rescore_archived.py --repository ~/.codex/ammo-eval
  python rescore_archived.py --repository ~/.codex/ammo-eval --dry-run
  python rescore_archived.py --repository ~/.codex/ammo-eval --target qwen3-5-4b_l40s_bf16_tp1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from score_campaign import compute_scorecard, generate_report


def rescore_run(run_dir: Path, dry_run: bool = False) -> Optional[dict]:
    """Re-score a single archived run. Returns summary dict or None on skip."""
    snapshot_path = run_dir / "artifacts_snapshot.json"
    if not snapshot_path.exists():
        # Some older archives use different names
        for alt in ["snapshot.json"]:
            alt_path = run_dir / alt
            if alt_path.exists():
                snapshot_path = alt_path
                break
        else:
            return None

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    # Load transcript grading if available
    transcript_grading = None
    tg_path = run_dir / "transcript_grading.json"
    if tg_path.exists():
        transcript_grading = json.loads(tg_path.read_text(encoding="utf-8"))

    # Read old score for comparison
    old_scorecard_path = run_dir / "scorecard.json"
    old_score = None
    if old_scorecard_path.exists():
        old_sc = json.loads(old_scorecard_path.read_text(encoding="utf-8"))
        old_score = old_sc.get("overall_score")

    # Re-score
    new_scorecard = compute_scorecard(snapshot, transcript_grading)
    new_report = generate_report(new_scorecard, snapshot)
    new_score = new_scorecard.get("overall_score")

    result = {
        "run_dir": str(run_dir),
        "old_score": old_score,
        "new_score": new_score,
        "delta": round(new_score - old_score, 2) if old_score is not None and new_score is not None else None,
        "accuracy_verification": new_scorecard.get("accuracy_verification"),
    }

    if not dry_run:
        old_scorecard_path.write_text(
            json.dumps(new_scorecard, indent=2) + "\n", encoding="utf-8"
        )
        report_path = run_dir / "report.md"
        report_path.write_text(new_report, encoding="utf-8")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repository", type=str, default="~/.codex/ammo-eval",
        help="Path to eval repository (default: ~/.codex/ammo-eval)",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Only re-score runs for this target slug (e.g., qwen3-5-4b_l40s_bf16_tp1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute new scores but don't write files",
    )

    args = parser.parse_args()
    repo = Path(args.repository).expanduser().resolve()

    if not repo.exists():
        print(f"ERROR: Repository not found: {repo}", file=sys.stderr)
        return 1

    versions_dir = repo / "versions"
    if not versions_dir.exists():
        print(f"ERROR: No versions directory: {versions_dir}", file=sys.stderr)
        return 1

    results = []
    for version_dir in sorted(versions_dir.iterdir()):
        if not version_dir.is_dir():
            continue
        runs_dir = version_dir / "runs"
        if not runs_dir.exists():
            continue

        for target_dir in sorted(runs_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            if args.target and target_dir.name != args.target:
                continue

            for run_dir in sorted(target_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                result = rescore_run(run_dir, dry_run=args.dry_run)
                if result:
                    results.append(result)

    # Print summary
    action = "Would re-score" if args.dry_run else "Re-scored"
    print(f"\n{action} {len(results)} run(s):\n")
    print(f"{'Old':>7}  {'New':>7}  {'Delta':>7}  {'Unverified':>10}  Path")
    print("-" * 90)
    for r in results:
        old = f"{r['old_score']:.2f}" if r['old_score'] is not None else "  N/A"
        new = f"{r['new_score']:.2f}" if r['new_score'] is not None else "  N/A"
        delta = f"{r['delta']:+.2f}" if r['delta'] is not None else "  N/A"
        av = r.get("accuracy_verification") or {}
        unverified = av.get("lossy_unverified", 0)
        unv_str = f"{unverified}" if unverified > 0 else "-"
        # Shorten path for display
        short_path = str(r['run_dir']).replace(str(repo) + "/", "")
        print(f"{old:>7}  {new:>7}  {delta:>7}  {unv_str:>10}  {short_path}")

    # Summary stats
    adjusted = [r for r in results if (r.get("accuracy_verification") or {}).get("has_unverified_lossy")]
    print(f"\n{len(adjusted)} run(s) had unverified lossy ops and were adjusted.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
