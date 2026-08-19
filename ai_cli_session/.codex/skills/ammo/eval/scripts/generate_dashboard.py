#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Generate a static HTML dashboard from the AMMO eval repository.

Reads index.json and version summaries, embeds all data into a
self-contained HTML file with trend charts, leaderboards, and
run detail views.

Usage:
  python generate_dashboard.py --repository ~/.codex/ammo-eval
  python generate_dashboard.py --repository ~/.codex/ammo-eval --output /tmp/dashboard.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _collect_dashboard_data(repository: Path) -> Dict[str, Any]:
    """Collect all data needed for the dashboard."""
    index = _load_json(repository / "index.json")
    if not index:
        return {"error": "No index.json found", "versions": [], "targets": []}

    # Enrich versions with full summary data
    enriched_versions = []
    for v in index.get("versions", []):
        version_dir = repository / "versions" / v.get("version_id", "")
        summary = _load_json(version_dir / "summary.json")

        # Collect per-run detail for drill-down
        run_details = {}
        runs_dir = version_dir / "runs"
        if runs_dir.exists():
            for target_dir in runs_dir.iterdir():
                if not target_dir.is_dir():
                    continue
                target_runs = []
                for run_dir in sorted(target_dir.iterdir()):
                    if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                        continue
                    sc = _load_json(run_dir / "scorecard.json")
                    if sc:
                        target_runs.append({
                            "run_name": run_dir.name,
                            "scorecard": sc,
                        })
                if target_runs:
                    run_details[target_dir.name] = target_runs

        enriched_versions.append({
            **v,
            "summary_full": summary.get("summary", {}) if summary else {},
            "run_details": run_details,
        })

    # Compute deltas (change vs most recent previous version with same target)
    for i in range(1, len(enriched_versions)):
        curr = enriched_versions[i]
        deltas = {}
        for target in curr.get("summary", {}):
            curr_data = curr["summary"].get(target, {})
            # Scan backward to find the most recent version with this target
            prev_data = {}
            for j in range(i - 1, -1, -1):
                candidate = enriched_versions[j].get("summary", {}).get(target)
                if candidate:
                    prev_data = candidate
                    break

            curr_score = (curr_data.get("overall_score") or {}).get("mean")
            prev_score = (prev_data.get("overall_score") or {}).get("mean")
            curr_speedup = (curr_data.get("e2e_speedup") or {}).get("mean")
            prev_speedup = (prev_data.get("e2e_speedup") or {}).get("mean")

            deltas[target] = {
                "delta_score": round(curr_score - prev_score, 3) if curr_score and prev_score else None,
                "delta_speedup": round(curr_speedup - prev_speedup, 4) if curr_speedup and prev_speedup else None,
            }
        curr["deltas"] = deltas

    return {
        "repository_version": index.get("repository_version"),
        "last_updated": index.get("last_updated"),
        "reference_targets": index.get("reference_targets", []),
        "leaderboard": index.get("leaderboard", {}),
        "versions": enriched_versions,
    }


def generate_dashboard(repository: Path, output: Path) -> None:
    """Generate the static HTML dashboard."""
    data = _collect_dashboard_data(repository)

    # Load template
    template_path = Path(__file__).parent.parent / "viewer" / "dashboard_template.html"
    if not template_path.exists():
        print(f"ERROR: Template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    template = template_path.read_text(encoding="utf-8")

    # Inject data
    data_json = json.dumps(data, indent=2)
    html = template.replace("/*__EMBEDDED_DATA__*/", f"const EMBEDDED_DATA = {data_json};")

    output.write_text(html, encoding="utf-8")
    print(f"Wrote: {output}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repository", type=str,
        default=str(Path.home() / ".codex" / "ammo-eval"),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for dashboard.html (default: {repository}/dashboard.html)",
    )

    args = parser.parse_args()
    repository = Path(args.repository).expanduser().resolve()

    if not repository.exists():
        print(f"Repository does not exist: {repository}", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else repository / "dashboard.html"
    generate_dashboard(repository, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
