#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Verify structured citations in AMMO eval agent outputs.

Checks that every citation in an agent's JSON output:
  1. References a file that exists (relative to artifact dir or absolute)
  2. Has a line range within the file's actual line count
  3. Has quoted_content that fuzzy-matches the actual content (>70% similarity)

Usage:
  python verify_citations.py --input /tmp/ammo_eval_deep_analysis.json --artifact-dir kernel_opt_artifacts/...
  python verify_citations.py --input /tmp/ammo_eval_transcript_grading.json --artifact-dir kernel_opt_artifacts/...
  python verify_citations.py --input /tmp/ammo_eval_deep_analysis.json --artifact-dir kernel_opt_artifacts/... --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _fuzzy_match(quoted: str, actual: str) -> float:
    """Return similarity ratio between quoted and actual content (0.0–1.0).

    The rubric tells agents to truncate quoted_content to ~200 chars, so the
    quote is often a prefix/substring of the full line range content. We handle
    this by: (1) if the normalized quote appears verbatim in the normalized
    actual, return 1.0; (2) otherwise, compare the quote against a window of
    the actual text of equal length (sliding window best match); (3) fall back
    to SequenceMatcher on the full strings.
    """
    quoted_norm = " ".join(quoted.split())
    actual_norm = " ".join(actual.split())

    if not quoted_norm:
        return 0.0

    # Exact substring match
    if quoted_norm in actual_norm:
        return 1.0

    # Sliding window: find the best-matching window in actual of same length
    # as quoted (handles truncation + minor reformatting)
    qlen = len(quoted_norm)
    if len(actual_norm) > qlen:
        best = 0.0
        # Sample windows at stride of max(1, qlen//4) for performance
        stride = max(1, qlen // 4)
        for i in range(0, len(actual_norm) - qlen + 1, stride):
            window = actual_norm[i : i + qlen]
            score = SequenceMatcher(None, quoted_norm, window).ratio()
            if score > best:
                best = score
                if best > 0.95:
                    break  # close enough
        return best

    # Same length or quoted is longer — direct comparison
    return SequenceMatcher(None, quoted_norm, actual_norm).ratio()


def _resolve_file(file_path: str, artifact_dir: Path) -> Path | None:
    """Resolve a citation file_path to an actual filesystem path."""
    p = Path(file_path)

    # Absolute path — use directly
    if p.is_absolute():
        return p if p.exists() else None

    # Relative to artifact dir
    candidate = artifact_dir / p
    if candidate.exists():
        return candidate

    # Try with investigation/ prefix (common for bottleneck_analysis.md)
    candidate = artifact_dir / "investigation" / p
    if candidate.exists():
        return candidate

    return None


def _extract_citations(obj: Any) -> list[dict]:
    """Recursively extract all citation objects from a JSON structure.

    A citation object has all of: file_path, line_start, line_end, claim,
    quoted_content. This walks dicts and lists to find them regardless of
    nesting depth.
    """
    REQUIRED = {"file_path", "line_start", "line_end", "claim", "quoted_content"}
    results = []

    if isinstance(obj, dict):
        if REQUIRED.issubset(obj.keys()):
            results.append(obj)
        for v in obj.values():
            results.extend(_extract_citations(v))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_extract_citations(item))

    return results


def verify_citations(
    data: Any, artifact_dir: Path, similarity_threshold: float = 0.70
) -> dict:
    """Verify all citations in a parsed JSON structure.

    Returns a summary dict with per-citation results and aggregate stats.
    """
    citations = _extract_citations(data)

    if not citations:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "failure_rate": 0.0,
            "results": [],
        }

    results = []
    passed = 0
    failed = 0

    for cit in citations:
        file_path = cit["file_path"]
        line_start = cit["line_start"]
        line_end = cit["line_end"]
        claim = cit["claim"]
        quoted = cit["quoted_content"]

        errors = []

        # Check 1: File exists
        resolved = _resolve_file(file_path, artifact_dir)
        if resolved is None:
            errors.append(f"file_not_found: {file_path}")
        else:
            # Check 2: Line range valid
            try:
                lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception as e:
                errors.append(f"file_read_error: {e}")
                lines = []

            if lines:
                max_line = len(lines)
                if line_start < 1 or line_end < line_start:
                    errors.append(
                        f"invalid_line_range: {line_start}-{line_end}"
                    )
                elif line_end > max_line:
                    errors.append(
                        f"line_range_exceeds_file: {line_start}-{line_end} "
                        f"but file has {max_line} lines"
                    )
                else:
                    # Check 3: Content similarity
                    actual_lines = lines[line_start - 1 : line_end]
                    actual_content = "\n".join(actual_lines)
                    similarity = _fuzzy_match(quoted, actual_content)

                    if similarity < similarity_threshold:
                        errors.append(
                            f"content_mismatch: similarity={similarity:.2f} "
                            f"(threshold={similarity_threshold:.2f})"
                        )

        result = {
            "file_path": file_path,
            "line_range": f"{line_start}-{line_end}",
            "claim": claim,
            "verified": len(errors) == 0,
        }
        if errors:
            result["errors"] = errors
            failed += 1
        else:
            passed += 1
        results.append(result)

    total = len(results)
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "failure_rate": round(failed / total, 3) if total > 0 else 0.0,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file containing agent output with citations",
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Path to the campaign artifact directory (citations are relative to this)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum fuzzy-match similarity for quoted_content (default: 0.70)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if any citation fails (for pipeline gating)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write verification results JSON to this path (default: stdout summary only)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return 1

    artifact_dir = Path(args.artifact_dir).resolve()
    if not artifact_dir.exists():
        print(f"ERROR: Artifact directory not found: {artifact_dir}", file=sys.stderr)
        return 1

    data = json.loads(input_path.read_text(encoding="utf-8"))
    report = verify_citations(data, artifact_dir, args.threshold)

    # Print summary
    print(f"\nCitation Verification: {report['passed']}/{report['total']} passed "
          f"({report['failure_rate']:.1%} failure rate)")

    if report["failed"] > 0:
        print(f"\nFailed citations:")
        for r in report["results"]:
            if not r["verified"]:
                print(f"  - {r['file_path']}:{r['line_range']} — {r['claim']}")
                for err in r.get("errors", []):
                    print(f"    {err}")

    # Write output if requested
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nDetailed results written to: {out_path}")

    if args.strict and report["failed"] > 0:
        print(f"\n--strict mode: failing due to {report['failed']} broken citation(s)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
