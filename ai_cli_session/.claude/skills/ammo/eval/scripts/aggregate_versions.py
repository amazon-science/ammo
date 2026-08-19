#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Aggregate multi-run eval results and rebuild the repository index.

For each version+target combination with multiple runs, computes
mean/stddev/min/max across runs. Rebuilds the master index.json
with leaderboards.

Usage:
  python aggregate_versions.py --repository ~/.claude/ammo-eval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_slug(slug: str) -> str:
    """Normalize dtype aliases in target slugs.

    Maps common dtype aliases to canonical short forms:
      bfloat16 -> bf16, float16 -> fp16, float32 -> fp32.

    Idempotent: applying twice gives the same result.
    """
    replacements = [
        ("bfloat16", "bf16"),
        ("float16", "fp16"),
        ("float32", "fp32"),
    ]
    for long, short in replacements:
        slug = slug.replace(long, short)
    return slug


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def calculate_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute mean, stddev, min, max for a list of values."""
    if not values:
        return {"mean": None, "stddev": None, "min": None, "max": None, "n": 0}

    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0

    return {
        "mean": round(mean, 4),
        "stddev": round(stddev, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": n,
    }


def _collect_metric(scorecards: List[Dict], *path: str) -> List[float]:
    """Extract a numeric metric from multiple scorecards by dot-path."""
    values = []
    for sc in scorecards:
        val = sc
        for key in path:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                val = None
                break
        if isinstance(val, (int, float)) and val is not None:
            values.append(float(val))
    return values


def _aggregate_scorecards(target_slug: str, scorecards: List[Dict]) -> Optional[Dict[str, Any]]:
    """Build an aggregate dict from a list of scorecards for one target slug."""
    if not scorecards:
        return None

    aggregate: Dict[str, Any] = {
        "target_slug": target_slug,
        "runs_count": len(scorecards),
        "overall_score": calculate_stats(
            _collect_metric(scorecards, "overall_score")
        ),
        "overall_score_without_transcript": calculate_stats(
            _collect_metric(scorecards, "overall_score_without_transcript")
        ),
        "e2e_speedup": calculate_stats(
            _collect_metric(scorecards, "raw_metrics", "cumulative_speedup_vs_round1")
        ),
        "shipped_count": calculate_stats(
            _collect_metric(scorecards, "raw_metrics", "shipped_optimizations")
        ),
        "total_rounds": calculate_stats(
            _collect_metric(scorecards, "raw_metrics", "total_rounds")
        ),
        "dimensions": {},
    }

    for dim in ["e2e_outcome", "gate_pass_rates", "debate_quality",
                "campaign_efficiency", "transcript_quality"]:
        aggregate["dimensions"][dim] = calculate_stats(
            _collect_metric(scorecards, "dimensions", dim, "score")
        )

    total_times = _collect_metric(scorecards, "timing", "total_tracked_seconds")
    if total_times:
        aggregate["total_time_seconds"] = calculate_stats(total_times)
        stage_times: Dict[str, List[float]] = {}
        for sc in scorecards:
            per_stage = _safe_get(sc, "timing", "per_stage_seconds") or {}
            for stage, secs in per_stage.items():
                if isinstance(secs, (int, float)):
                    stage_times.setdefault(stage, []).append(float(secs))
        aggregate["per_stage_time_seconds"] = {
            stage: calculate_stats(vals) for stage, vals in stage_times.items()
        }

    total_tokens = _collect_metric(scorecards, "agent_costs", "total_tokens")
    if total_tokens:
        aggregate["agent_costs"] = {
            "total_tokens": calculate_stats(total_tokens),
            "total_duration_ms": calculate_stats(
                _collect_metric(scorecards, "agent_costs", "total_duration_ms")
            ),
            "total_agent_invocations": calculate_stats(
                _collect_metric(scorecards, "agent_costs", "total_agent_invocations")
            ),
        }

    deleg_enabled = any(
        _safe_get(sc, "delegation", "enabled") for sc in scorecards
    )
    if deleg_enabled:
        aggregate["delegation"] = {
            "enabled": True,
            "delegate_citation_rate": calculate_stats(
                _collect_metric(scorecards, "delegation", "delegate_citation_rate")
            ),
            "delegate_artifacts_count": calculate_stats(
                _collect_metric(scorecards, "delegation", "delegate_artifacts_count")
            ),
        }

    return aggregate


def aggregate_target_runs(target_dir: Path) -> Optional[Dict[str, Any]]:
    """Aggregate runs for a single version+target combination."""
    run_dirs = sorted(
        [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )
    if not run_dirs:
        return None

    scorecards = []
    for rd in run_dirs:
        sc = _load_json(rd / "scorecard.json")
        if sc:
            scorecards.append(sc)

    if not scorecards:
        return None

    aggregate = {
        "target_slug": target_dir.name,
        "runs_count": len(scorecards),
        "overall_score": calculate_stats(
            _collect_metric(scorecards, "overall_score")
        ),
        "overall_score_without_transcript": calculate_stats(
            _collect_metric(scorecards, "overall_score_without_transcript")
        ),
        "e2e_speedup": calculate_stats(
            _collect_metric(scorecards, "raw_metrics", "cumulative_speedup_vs_round1")
        ),
        "shipped_count": calculate_stats(
            _collect_metric(scorecards, "raw_metrics", "shipped_optimizations")
        ),
        "total_rounds": calculate_stats(
            _collect_metric(scorecards, "raw_metrics", "total_rounds")
        ),
        "dimensions": {},
    }

    # Per-dimension aggregation
    for dim in ["e2e_outcome", "gate_pass_rates", "debate_quality",
                "campaign_efficiency", "transcript_quality"]:
        aggregate["dimensions"][dim] = calculate_stats(
            _collect_metric(scorecards, "dimensions", dim, "score")
        )

    # Timing aggregation
    total_times = _collect_metric(scorecards, "timing", "total_tracked_seconds")
    if total_times:
        aggregate["total_time_seconds"] = calculate_stats(total_times)
        # Per-stage timing
        stage_times: Dict[str, List[float]] = {}
        for sc in scorecards:
            per_stage = _safe_get(sc, "timing", "per_stage_seconds") or {}
            for stage, secs in per_stage.items():
                if isinstance(secs, (int, float)):
                    stage_times.setdefault(stage, []).append(float(secs))
        aggregate["per_stage_time_seconds"] = {
            stage: calculate_stats(vals) for stage, vals in stage_times.items()
        }

    # Agent cost aggregation
    total_tokens = _collect_metric(scorecards, "agent_costs", "total_tokens")
    if total_tokens:
        aggregate["agent_costs"] = {
            "total_tokens": calculate_stats(total_tokens),
            "total_duration_ms": calculate_stats(
                _collect_metric(scorecards, "agent_costs", "total_duration_ms")
            ),
            "total_agent_invocations": calculate_stats(
                _collect_metric(scorecards, "agent_costs", "total_agent_invocations")
            ),
        }

    # Delegation aggregation
    deleg_enabled = any(
        _safe_get(sc, "delegation", "enabled") for sc in scorecards
    )
    if deleg_enabled:
        aggregate["delegation"] = {
            "enabled": True,
            "delegate_citation_rate": calculate_stats(
                _collect_metric(scorecards, "delegation", "delegate_citation_rate")
            ),
            "delegate_artifacts_count": calculate_stats(
                _collect_metric(scorecards, "delegation", "delegate_artifacts_count")
            ),
        }

    return aggregate


def aggregate_version(version_dir: Path) -> Optional[Dict[str, Any]]:
    """Aggregate all targets for a single version."""
    runs_dir = version_dir / "runs"
    if not runs_dir.exists():
        return None

    target_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not target_dirs:
        return None

    meta = _load_json(version_dir / "meta.json") or {}
    targets = {}

    # Group target dirs by normalized slug so e.g. bfloat16 and bf16 merge
    slug_groups: Dict[str, List[Path]] = {}
    for td in sorted(target_dirs, key=lambda d: d.name):
        canonical = _normalize_slug(td.name)
        slug_groups.setdefault(canonical, []).append(td)

    for canonical_slug, dirs in sorted(slug_groups.items()):
        # Collect all run dirs across slug variants
        all_run_dirs = []
        for td in dirs:
            all_run_dirs.extend(
                d for d in td.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            )
        if not all_run_dirs:
            continue

        # Load scorecards from all runs
        scorecards = []
        for rd in sorted(all_run_dirs, key=lambda d: d.name):
            sc = _load_json(rd / "scorecard.json")
            if sc:
                scorecards.append(sc)
        if not scorecards:
            continue

        # Build aggregate using the same logic as aggregate_target_runs
        # but across merged slug variants
        agg = _aggregate_scorecards(canonical_slug, scorecards)
        if agg:
            targets[canonical_slug] = agg
            # Write per-target aggregate into first dir
            (dirs[0] / "aggregate.json").write_text(
                json.dumps(agg, indent=2) + "\n", encoding="utf-8"
            )

    if not targets:
        return None

    summary = {
        "version_id": meta.get("version_id", version_dir.name),
        "git_commit": meta.get("git_commit"),
        "description": meta.get("description"),
        "created_at": meta.get("created_at"),
        "targets_evaluated": list(targets.keys()),
        "summary": targets,
    }

    # Write version summary
    (version_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    return summary


def rebuild_index(repository: Path) -> Dict[str, Any]:
    """Rebuild the master index.json from all versions."""
    versions_dir = repository / "versions"
    if not versions_dir.exists():
        versions_dir.mkdir(parents=True, exist_ok=True)

    def _version_sort_key(d: Path) -> str:
        """Sort version dirs by created_at from meta.json, falling back to name."""
        meta = _load_json(d / "meta.json")
        if meta and meta.get("created_at"):
            return meta["created_at"]
        return d.name

    version_dirs = sorted(
        [d for d in versions_dir.iterdir() if d.is_dir()],
        key=_version_sort_key,
    )

    versions = []
    all_targets = set()

    for vd in version_dirs:
        summary = aggregate_version(vd)
        if summary:
            versions.append(summary)
            all_targets.update(summary.get("targets_evaluated", []))

    # Build leaderboard: best version per target by e2e_speedup mean
    leaderboard = {}
    for target_slug in sorted(all_targets):
        best_version = None
        best_speedup = 0.0
        best_score = 0.0

        for v in versions:
            target_data = v.get("summary", {}).get(target_slug)
            if not target_data:
                continue
            speedup_mean = (target_data.get("e2e_speedup") or {}).get("mean", 0)
            score_mean = (target_data.get("overall_score") or {}).get("mean", 0)
            if speedup_mean > best_speedup:
                best_speedup = speedup_mean
                best_score = score_mean
                best_version = v.get("version_id")

        if best_version:
            leaderboard[target_slug] = {
                "best_version": best_version,
                "best_e2e_speedup_mean": best_speedup,
                "best_overall_score_mean": best_score,
            }

    # Discover reference targets (all unique targets seen)
    reference_targets = []
    for slug in sorted(all_targets):
        # Try to extract model type from slug
        model_type = "moe" if any(kw in slug for kw in ["a3b", "moe", "mixtral"]) else "dense"
        parts = slug.rsplit("_tp", 1)
        tp = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        reference_targets.append({
            "slug": slug,
            "model_type": model_type,
            "tp": tp,
        })

    index = {
        "repository_version": 1,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "reference_targets": reference_targets,
        "versions": [
            {
                "version_id": v.get("version_id"),
                "git_commit": v.get("git_commit"),
                "description": v.get("description"),
                "created_at": v.get("created_at"),
                "targets_evaluated": v.get("targets_evaluated", []),
                "summary": {
                    target: {
                        "overall_score": data.get("overall_score"),
                        "e2e_speedup": data.get("e2e_speedup"),
                        "shipped_count": data.get("shipped_count"),
                    }
                    for target, data in v.get("summary", {}).items()
                },
            }
            for v in versions
        ],
        "leaderboard": leaderboard,
    }

    index_path = repository / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    return index


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repository", type=str,
        default=str(Path.home() / ".claude" / "ammo-eval"),
    )

    args = parser.parse_args()
    repository = Path(args.repository).expanduser().resolve()

    if not repository.exists():
        print(f"Repository does not exist: {repository}", file=sys.stderr)
        return 1

    index = rebuild_index(repository)
    n_versions = len(index.get("versions", []))
    n_targets = len(index.get("reference_targets", []))
    print(f"Rebuilt index: {n_versions} version(s), {n_targets} target(s)", file=sys.stderr)
    print(f"Wrote: {repository / 'index.json'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
