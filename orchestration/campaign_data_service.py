# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# E2E Latency Normalizer (v4.0 schema backfill + cleanup)
#
# Retained for the L2/L3 endpoints — circuit-board.js relies on the
# post-normalization state shape. The other read-path normalizers
# (_normalize_fail_reasons, _normalize_shipped_ops, _normalize_speedup_field)
# and the path-based sidecar-stamping helpers (_BARE_STAGE_MAP,
# _stamp_from_state, _normalize_stage_field, _round_for_track,
# _dedupe_debate_entries + helpers) were removed in the v2 cleanup.
# ---------------------------------------------------------------------------

_KERN_META_KEYS = {'target', 'floor', 'ship_gate', 'threshold', 'ceiling'}
_HETEROGENEOUS_BUCKET_RE = re.compile(r"^il(?P<il>\d+)_ol(?P<ol>\d+)_bs(?P<bs>\d+)$")
_HOMOGENEOUS_BUCKET_RE = re.compile(r"^bs(?P<bs>\d+)$")
_LEGACY_BUCKET_RE = re.compile(r"^(?P<bs>\d+)$")


def _parse_bucket_tag(raw_tag: Any) -> Optional[dict[str, Any]]:
    """Parse a canonical bucket tag while preserving its exact lookup key.

    Homogeneous maps use ``bs{BS}``, heterogeneous maps use the complete
    ``il{IL}_ol{OL}_bs{BS}`` tuple, and archived maps may use a bare numeric
    key. The returned ``tag`` is never normalized, so same-BS heterogeneous
    rows cannot alias during lookup.
    """
    if raw_tag is None:
        return None
    tag = str(raw_tag)
    match = _HETEROGENEOUS_BUCKET_RE.fullmatch(tag)
    if match:
        input_len = int(match.group("il"))
        output_len = int(match.group("ol"))
        batch_size = int(match.group("bs"))
        if input_len < 1 or output_len < 1 or batch_size < 1:
            return None
        return {
            "tag": tag,
            "input_len": input_len,
            "output_len": output_len,
            "batch_size": batch_size,
            "heterogeneous": True,
            "legacy_numeric": False,
        }
    match = _HOMOGENEOUS_BUCKET_RE.fullmatch(tag)
    legacy_numeric = False
    if match is None:
        match = _LEGACY_BUCKET_RE.fullmatch(tag)
        legacy_numeric = match is not None
    if match is None:
        return None
    batch_size = int(match.group("bs"))
    if batch_size < 1:
        return None
    return {
        "tag": tag,
        "input_len": None,
        "output_len": None,
        "batch_size": batch_size,
        "heterogeneous": False,
        "legacy_numeric": legacy_numeric,
    }


def _bucket_records(bucket_map: Any) -> list[dict[str, Any]]:
    """Return valid bucket records sorted by the complete workload tuple."""
    if not isinstance(bucket_map, dict):
        return []
    records = [record for key in bucket_map if (record := _parse_bucket_tag(key))]
    return sorted(
        records,
        key=lambda record: (
            record["input_len"] if record["input_len"] is not None else -1,
            record["output_len"] if record["output_len"] is not None else -1,
            record["batch_size"],
            record["tag"],
        ),
    )


def _strip_s_suffix_entry(entry: dict) -> dict:
    """Strip '_s' suffix from e2e_latency entry keys (avg_s -> avg, p50_s -> p50)."""
    if not isinstance(entry, dict):
        return entry
    cleaned = {}
    for k, v in entry.items():
        if k.endswith("_s"):
            cleaned[k[:-2]] = v
        else:
            cleaned[k] = v
    return cleaned


def _strip_s_suffix_map(latency_map: dict) -> None:
    """In-place strip _s suffix from all entries in a latency map."""
    if not isinstance(latency_map, dict):
        return
    for bucket_tag in list(latency_map.keys()):
        entry = latency_map[bucket_tag]
        if isinstance(entry, dict):
            # Check if any key has _s suffix
            if any(k.endswith("_s") for k in entry):
                latency_map[bucket_tag] = _strip_s_suffix_entry(entry)


def _normalize_round_e2e_latency(rnd: dict) -> None:
    """Normalize a single round's e2e latency fields in place."""
    baseline = rnd.get("baseline")
    if not isinstance(baseline, dict):
        baseline = {}
        rnd["baseline"] = baseline

    integration = rnd.get("integration")
    if not isinstance(integration, dict):
        integration = {}
        rnd["integration"] = integration

    # Source: combined_e2e_result (legacy field)
    cer = integration.get("combined_e2e_result")
    cer_dict = cer if isinstance(cer, dict) else {}

    # Preserve the first deterministic verdict tag. A missing tag means this is
    # a pre-bucket-tag scalar payload, so retain its historical numeric key.
    per_bs_verdict = cer_dict.get("per_bs_verdict")
    verdict_buckets = _bucket_records(per_bs_verdict)
    bucket_tag = verdict_buckets[0]["tag"] if verdict_buckets else "1"

    # --- Baseline backfill ---
    # 1. e2e_latency backfill from combined_e2e_result.latency_baseline_s
    existing_bl_latency = baseline.get("e2e_latency")
    if not existing_bl_latency:
        lat_baseline_s = cer_dict.get("latency_baseline_s")
        if isinstance(lat_baseline_s, (int, float)) and lat_baseline_s > 0:
            baseline["e2e_latency"] = {bucket_tag: {"avg": lat_baseline_s, "p50": lat_baseline_s}}

    # 2. per_bs_verdict backfill from combined_e2e_result.per_bs_verdict
    existing_bl_verdict = baseline.get("per_bs_verdict")
    if not existing_bl_verdict:
        if isinstance(per_bs_verdict, dict) and per_bs_verdict:
            baseline["per_bs_verdict"] = per_bs_verdict

    # --- Integration backfill ---
    # 3. e2e_latency_combined backfill from combined_e2e_result or integration.opt_s
    existing_integ_combined = integration.get("e2e_latency_combined")
    if not existing_integ_combined:
        # Try opt_s from combined_e2e_result first, then standalone integration.opt_s
        opt_s_val = cer_dict.get("opt_s")
        if not isinstance(opt_s_val, (int, float)) or opt_s_val <= 0:
            opt_s_val = integration.get("opt_s")
        if isinstance(opt_s_val, (int, float)) and opt_s_val > 0:
            integration["e2e_latency_combined"] = {bucket_tag: {"avg": opt_s_val, "p50": opt_s_val}}
        elif not opt_s_val:
            # Fall back to latency_baseline_s if no opt_s available (single-source legacy)
            lat_baseline_s = cer_dict.get("latency_baseline_s")
            if isinstance(lat_baseline_s, (int, float)) and lat_baseline_s > 0 and cer_dict.get("speedup_x"):
                # Only use baseline as fallback when there's evidence of an integration
                # (speedup_x present means integration happened)
                pass  # Don't backfill with baseline value for integration

    # --- Track normalizations ---
    pt = rnd.get("parallel_tracks")
    if isinstance(pt, dict):
        tracks = pt.get("tracks")
        if isinstance(tracks, dict):
            for track in tracks.values():
                if not isinstance(track, dict):
                    continue
                _normalize_track_speedups(track)

    # --- passing_candidates normalization ---
    candidates = integration.get("passing_candidates")
    if isinstance(candidates, list):
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            es = cand.get("e2e_speedup")
            if isinstance(es, dict):
                cand["e2e_speedup"] = es.get("speedup_x") or es.get("measured") or es.get("value")

    # --- _s suffix strip ---
    bl_latency = baseline.get("e2e_latency")
    if isinstance(bl_latency, dict):
        _strip_s_suffix_map(bl_latency)

    integ_latency = integration.get("e2e_latency_combined")
    if isinstance(integ_latency, dict):
        _strip_s_suffix_map(integ_latency)


def _normalize_track_speedups(track: dict) -> None:
    """Normalize kernel_speedup and e2e_speedup on a single track."""
    # kernel_speedup split: dict -> scalar + variants
    ks = track.get("kernel_speedup")
    if isinstance(ks, dict):
        # Extract max non-meta value
        non_meta = [(k, v) for k, v in ks.items()
                    if k not in _KERN_META_KEYS and isinstance(v, (int, float)) and v > 0]
        if non_meta:
            max_val = max(v for _, v in non_meta)
            track["kernel_speedup"] = max_val
        else:
            track["kernel_speedup"] = None
        track["kernel_speedup_variants"] = ks

    # e2e_speedup flatten: dict with speedup_x -> number
    es = track.get("e2e_speedup")
    if isinstance(es, dict):
        track["e2e_speedup"] = es.get("speedup_x") or es.get("measured") or es.get("value")


def _speedup_at_first_shared_bucket(baseline_map: Any, combined_map: Any) -> Optional[float]:
    """Compare maps at the first shared canonical workload identity.

    A bare numeric legacy key aliases only the homogeneous ``bsN`` form.
    Heterogeneous buckets still require the exact IL/OL/BS tuple, so two rows
    that happen to share a batch size can never be collapsed.
    """
    if not isinstance(baseline_map, dict) or not isinstance(combined_map, dict):
        return None
    combined_by_identity = {
        (
            bucket["input_len"],
            bucket["output_len"],
            bucket["batch_size"],
        ): bucket
        for bucket in _bucket_records(combined_map)
    }
    for bucket in _bucket_records(baseline_map):
        identity = (
            bucket["input_len"],
            bucket["output_len"],
            bucket["batch_size"],
        )
        combined_bucket = combined_by_identity.get(identity)
        if combined_bucket is None:
            continue
        baseline_entry = baseline_map.get(bucket["tag"])
        combined_entry = combined_map.get(combined_bucket["tag"])
        if not isinstance(baseline_entry, dict) or not isinstance(combined_entry, dict):
            continue
        baseline_avg = baseline_entry.get("avg")
        combined_avg = combined_entry.get("avg")
        if not isinstance(baseline_avg, (int, float)) or baseline_avg <= 0:
            continue
        if not isinstance(combined_avg, (int, float)) or combined_avg <= 0:
            continue
        return baseline_avg / combined_avg
    return None


def _compute_cumulative_speedup(campaign: dict) -> float:
    """Compute R1-to-latest speedup at a shared canonical bucket identity."""
    rounds = campaign.get("rounds")
    if not rounds or not isinstance(rounds, list):
        return 1.0
    # Round 1 baseline (anchor)
    r1 = rounds[0] if rounds else None
    if not isinstance(r1, dict):
        return 1.0
    r1_baseline = r1.get("baseline", {})
    if not isinstance(r1_baseline, dict):
        return 1.0
    r1_latency = r1_baseline.get("e2e_latency")
    if not r1_latency or not isinstance(r1_latency, dict):
        return 1.0
    if not _bucket_records(r1_latency):
        return 1.0
    # Latest completed integration (iterate from last round backwards)
    for rnd in reversed(rounds):
        if not isinstance(rnd, dict):
            continue
        integ = rnd.get("integration")
        if not isinstance(integ, dict):
            continue
        # Try e2e_latency_combined first (new schema). Exact tag intersection
        # is required; comparing by BS alone would collapse heterogeneous rows.
        elc = integ.get("e2e_latency_combined")
        speedup = _speedup_at_first_shared_bucket(r1_latency, elc)
        if speedup is not None:
            return round(speedup, 4)
        # Fallback: combined_e2e_result (legacy)
        cer = integ.get("combined_e2e_result")
        if isinstance(cer, dict):
            speedup = _speedup_at_first_shared_bucket(r1_latency, cer)
            if speedup is not None:
                return round(speedup, 4)
    return 1.0


def _normalize_e2e_latency(state: dict) -> None:
    """Read-path normalizer: backfill new map-based e2e_latency fields from legacy.

    Walks campaign.rounds[*] and for each round:
    1. Backfills baseline.e2e_latency from combined_e2e_result.latency_baseline_s
    2. Backfills baseline.per_bs_verdict from combined_e2e_result.per_bs_verdict
    3. Backfills integration.e2e_latency_combined from combined_e2e_result.opt_s
    4. Splits kernel_speedup dict into scalar + kernel_speedup_variants
    5. Flattens e2e_speedup dict to scalar
    6. Flattens passing_candidates[*].e2e_speedup to scalar
    7. Strips _s suffix from e2e_latency entry keys

    Then computes and injects campaign.cumulative_e2e_speedup.

    Idempotent: running twice produces identical output.
    Must handle null/absent gracefully (no crashes on missing data).
    """
    if not isinstance(state, dict):
        return
    campaign = state.get("campaign")
    if not isinstance(campaign, dict):
        return
    rounds = campaign.get("rounds")
    if not isinstance(rounds, list):
        return
    for rnd in rounds:
        if not isinstance(rnd, dict):
            continue
        _normalize_round_e2e_latency(rnd)

    # Compute and inject cumulative speedup (replaces old _normalize_speedup_field)
    campaign["cumulative_e2e_speedup"] = _compute_cumulative_speedup(campaign)


MIME_MAP = {".md": "text/markdown", ".json": "application/json",
            ".py": "text/x-python", ".txt": "text/plain", ".csv": "text/csv",
            ".html": "text/html", ".log": "text/plain", ".yaml": "text/yaml",
            ".yml": "text/yaml", ".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
            ".parquet": "application/octet-stream",
            ".nsys-rep": "application/octet-stream",
            ".ncu-rep": "application/octet-stream"}

BINARY_EXTENSIONS = {".nsys-rep", ".ncu-rep", ".png", ".jpg", ".jpeg", ".svg", ".parquet"}

ARTIFACT_EXCLUDED_DIRS = {
    "_archive",
    "__pycache__",
    ".git",
    "cache",
    "triton_cache",
    "torch_compile_cache",
}


class CampaignDataService:

    def __init__(self):
        self._artifact_dir_cache: dict[str, str] = {}
        # mtime-keyed state.json content cache: {artifact_dir: (mtime, parsed_data)}
        self._state_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    def clear_state_cache(self, artifact_dir: str) -> None:
        self._state_cache.pop(artifact_dir, None)

    async def find_artifact_dir(self, worktree_path: str) -> Optional[str]:
        """Find the active artifact dir (the one whose state.json is most recently written)."""
        if worktree_path in self._artifact_dir_cache:
            cached = self._artifact_dir_cache[worktree_path]
            if os.path.exists(os.path.join(cached, "state.json")):
                return cached
            self._artifact_dir_cache.pop(worktree_path, None)
        dirs = await self.find_all_artifact_dirs(worktree_path)
        if not dirs:
            return None
        self._artifact_dir_cache[worktree_path] = dirs[0]
        return dirs[0]

    async def find_all_artifact_dirs(self, worktree_path: str) -> list[str]:
        """Return all artifact dirs, ordered by state.json mtime (newest first).

        Multi-round campaigns create _r2, _r3 suffixed dirs. The orchestrator
        only updates the active round's state.json, so mtime-newest = live.
        """
        def _glob() -> list[str]:
            wt = Path(worktree_path)
            candidates = list(wt.glob("kernel_opt_artifacts/*/state.json"))
            if not candidates:
                candidates = list(wt.glob("state.json"))
            if candidates:
                candidates.sort(key=lambda c: c.stat().st_mtime, reverse=True)
                return [str(c.parent) for c in candidates]
            return []

        return await asyncio.to_thread(_glob)

    async def list_artifact_tree(self, artifact_dir: str) -> dict[str, Any]:
        """Recursively list artifact files under `artifact_dir`.

        Walks the directory tree and returns a flat list of relative paths
        (POSIX-style separators). Excludes:
          * `_archive/` subdirectories (legacy round-rollover snapshots)
          * `cache/`, `triton_cache/`, and `torch_compile_cache/`
            (ephemeral compiler caches)
          * `*.metrics.json` sidecar files (deprecated)
          * `state.json.lock` (empty flock file, carries no campaign data)
          * `__pycache__/` (Python bytecode)
          * `.git/` (git internals)

        Returns `{"root": <basename of artifact_dir>, "files": [<sorted rels>]}`.
        Missing/non-directory paths return an empty `files` list.
        """
        def _walk() -> dict[str, Any]:
            base = Path(artifact_dir)
            root_name = base.name
            files: list[str] = []
            if not base.is_dir():
                return {"root": root_name, "files": []}
            base_str = str(base)
            for dirpath, dirnames, filenames in os.walk(base_str):
                # Prune excluded directories in-place so os.walk skips them.
                dirnames[:] = [
                    d for d in dirnames
                    if d not in ARTIFACT_EXCLUDED_DIRS
                ]
                for fname in filenames:
                    if fname.endswith(".metrics.json") or fname == "state.json.lock":
                        continue
                    abs_path = os.path.join(dirpath, fname)
                    rel = os.path.relpath(abs_path, base_str).replace(os.sep, "/")
                    files.append(rel)
            files.sort()
            return {"root": root_name, "files": files}

        return await asyncio.to_thread(_walk)

    async def list_artifact_children(
        self, artifact_dir: str, rel_path: str = ""
    ) -> dict[str, Any]:
        """List only the immediate visible children of an artifact directory.

        ``rel_path`` is resolved beneath the real artifact root. Paths which
        escape that root (including through symlinks) raise ``ValueError``;
        symlink entries whose targets escape the root are omitted. Missing or
        non-directory targets return ``exists: false``.
        """
        def _list() -> dict[str, Any]:
            root = os.path.realpath(artifact_dir)
            requested = os.path.abspath(os.path.join(root, rel_path or ""))
            try:
                if os.path.commonpath((root, requested)) != root:
                    raise ValueError("Artifact path escapes artifact root")
            except ValueError as exc:
                raise ValueError("Artifact path escapes artifact root") from exc

            resolved = os.path.realpath(requested)
            try:
                if os.path.commonpath((root, resolved)) != root:
                    raise ValueError("Artifact path escapes artifact root")
            except ValueError as exc:
                raise ValueError("Artifact path escapes artifact root") from exc

            normalized = os.path.relpath(requested, root).replace(os.sep, "/")
            if normalized == ".":
                normalized = ""
            response: dict[str, Any] = {
                "path": normalized,
                "exists": False,
                "entries": [],
            }
            if any(part in ARTIFACT_EXCLUDED_DIRS for part in normalized.split("/") if part):
                return response
            if not os.path.isdir(resolved):
                return response

            entries: list[dict[str, Any]] = []
            try:
                children = list(os.scandir(resolved))
            except OSError:
                return response

            response["exists"] = True
            for child in children:
                child_real = os.path.realpath(child.path)
                try:
                    if os.path.commonpath((root, child_real)) != root:
                        continue
                except ValueError:
                    continue

                try:
                    is_dir = child.is_dir(follow_symlinks=True)
                    is_file = child.is_file(follow_symlinks=True)
                except OSError:
                    continue
                if is_dir and child.name in ARTIFACT_EXCLUDED_DIRS:
                    continue
                if is_file and child.name.endswith(".metrics.json"):
                    continue
                if not is_dir and not is_file:
                    continue

                child_rel = f"{normalized}/{child.name}" if normalized else child.name
                item: dict[str, Any] = {
                    "name": child.name,
                    "path": child_rel,
                    "type": "directory" if is_dir else "file",
                }
                if is_file:
                    try:
                        item["size"] = child.stat(follow_symlinks=True).st_size
                    except OSError:
                        continue
                    item["mime"] = MIME_MAP.get(
                        Path(child.name).suffix.lower(), "text/plain"
                    )
                entries.append(item)

            entries.sort(key=lambda item: (item["type"] != "directory", item["name"]))
            response["entries"] = entries
            return response

        return await asyncio.to_thread(_list)

    async def read_state(self, artifact_dir: str) -> Optional[dict[str, Any]]:
        state_path = os.path.join(artifact_dir, "state.json")

        def _read() -> Optional[dict[str, Any]]:
            try:
                mtime = os.path.getmtime(state_path)
            except OSError:
                self._state_cache.pop(artifact_dir, None)
                return None

            cached = self._state_cache.get(artifact_dir)
            if cached and cached[0] == mtime:
                return cached[1]

            try:
                with open(state_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read state.json: {e}")
                self._state_cache.pop(artifact_dir, None)
                return None
            # Read path is now pure: callers that need a normalized state
            # (L2/L3 endpoints, build_l1_projection's v3-field fallback)
            # apply normalizers explicitly. Normalizing here would mutate
            # the cached dict in place and bake one-shot transformations
            # into every subsequent reader.
            self._state_cache[artifact_dir] = (mtime, data)
            if len(self._state_cache) > 50:
                oldest_key = next(iter(self._state_cache))
                self._state_cache.pop(oldest_key, None)
            return data

        return await asyncio.to_thread(_read)

    def build_l1_projection(
        self,
        session_id: str,
        state: dict[str, Any],
        created_at: Optional[str] = None,
    ) -> dict[str, Any]:
        """Reconciled L1 projection consumed by FE campaign cards.

        Returns a plain dict (no Pydantic model) shaped like:

            {
              session_id, created_at,
              target: {model_id, hardware, dtype, tp},
              campaign: {
                status, current_round, current_stage, cumulative_e2e_speedup,
                shipped_optimizations,
                shipped_count, failed_count, active_count,
                pipeline_progress: [{stage, status}],
                rounds: [{round_id, status, shipped,
                          parallel_tracks: {tracks: {[op_id]: {...}}}}]
              }
            }

        `cumulative_e2e_speedup` falls back from `cumulative_speedup_vs_round1`
        when the canonical field is absent or at its default of 1.0. Round
        track details retain only the fields the FE renders directly
        (status, verdict, kernel_speedup, classification, fail_reason).
        """
        if not isinstance(state, dict):
            state = {}
        raw_target = state.get("target") or {}
        if not isinstance(raw_target, dict):
            raw_target = {}
        campaign = state.get("campaign") or {}
        if not isinstance(campaign, dict):
            campaign = {}

        shipped, failed, active = self._count_track_statuses(state)
        diluted_count = self._count_diluted_tracks(state)
        current_stage = campaign.get("current_stage") or state.get("stage") or "unknown"

        canonical_speedup = campaign.get("cumulative_e2e_speedup")
        v3_speedup = campaign.get("cumulative_speedup_vs_round1")
        if canonical_speedup not in (None, 1.0):
            cumulative_e2e_speedup = canonical_speedup
        elif v3_speedup not in (None, 1.0):
            cumulative_e2e_speedup = v3_speedup
        else:
            cumulative_e2e_speedup = canonical_speedup or v3_speedup or 1.0

        rounds_out: list[dict[str, Any]] = []
        raw_rounds = campaign.get("rounds") or []
        if isinstance(raw_rounds, list):
            for rnd in raw_rounds:
                if not isinstance(rnd, dict):
                    continue
                rounds_out.append(self._project_round(rnd))

        return {
            "session_id": session_id,
            "target": {
                "model_id": raw_target.get("model_id", "Unknown"),
                "hardware": raw_target.get("hardware", "Unknown"),
                "dtype": raw_target.get("dtype", "Unknown"),
                "tp": raw_target.get("tp", 1),
            },
            "campaign": {
                "status": campaign.get("status", "unknown"),
                "current_round": campaign.get("current_round", 0),
                "current_stage": current_stage,
                "cumulative_e2e_speedup": cumulative_e2e_speedup,
                "shipped_optimizations": campaign.get("shipped_optimizations", []) or [],
                "shipped_count": shipped,
                "diluted_count": diluted_count,
                "failed_count": failed,
                "active_count": active,
                "pipeline_progress": self._build_pipeline_progress(state),
                "rounds": rounds_out,
            },
            "created_at": created_at,
        }

    @staticmethod
    def _project_round(rnd: dict[str, Any]) -> dict[str, Any]:
        """Trim a campaign round to the fields the L1 cards render.

        Keeps round_id/status/shipped and a slim parallel_tracks.tracks map
        with just the per-track columns the FE binds to. Drops debug-only
        and large nested fields (implementation_results, integration, etc.).
        """
        track_keys = ("status", "verdict", "kernel_speedup", "classification", "fail_reason")
        tracks_in = {}
        pt = rnd.get("parallel_tracks")
        if isinstance(pt, dict):
            raw_tracks = pt.get("tracks")
            if isinstance(raw_tracks, dict):
                tracks_in = raw_tracks
        tracks_out: dict[str, dict[str, Any]] = {}
        for op_id, track in tracks_in.items():
            if not isinstance(track, dict):
                continue
            proj = {k: track.get(k) for k in track_keys}
            if proj.get("fail_reason") is None:
                proj["fail_reason"] = track.get("failure_reason") or track.get("reason")
            tracks_out[op_id] = proj
        return {
            "round_id": rnd.get("round_id"),
            "status": rnd.get("status"),
            "shipped": list(rnd.get("shipped") or []),
            "parallel_tracks": {"tracks": tracks_out},
        }

    async def read_artifact(self, artifact_dir: str, rel_path: str) -> tuple[Optional[str | bytes], str]:
        resolved = os.path.realpath(os.path.join(artifact_dir, rel_path))
        real_base = os.path.realpath(artifact_dir)
        if not (resolved.startswith(real_base + os.sep) or resolved == real_base):
            return None, ""  # path traversal blocked
        if not os.path.exists(resolved):
            return None, ""
        ext = os.path.splitext(resolved)[1].lower()
        mime = MIME_MAP.get(ext, "text/plain")
        is_binary = ext in BINARY_EXTENSIONS

        def _read() -> tuple[Optional[str | bytes], str]:
            try:
                if is_binary:
                    with open(resolved, "rb") as f:
                        return f.read(), mime
                else:
                    with open(resolved) as f:
                        return f.read(), mime
            except (OSError, UnicodeDecodeError):
                return None, ""

        return await asyncio.to_thread(_read)

    async def read_artifact_from_any(self, artifact_dirs: list[str], rel_path: str) -> tuple[Optional[str | bytes], str]:
        """Search for an artifact across all round dirs, returning the first hit."""
        for d in artifact_dirs:
            content, mime = await self.read_artifact(d, rel_path)
            if content is not None:
                return content, mime
        return None, ""

    async def stat_artifact(self, artifact_dir: str, rel_path: str) -> tuple[Optional[int], str]:
        """Stat an artifact within artifact_dir and return (size_bytes, mime).

        Mirrors read_artifact's security posture (realpath traversal guard),
        but only issues a single os.stat so the L3 viewer can cheaply probe
        file size via a HEAD request before deciding whether to inline-render
        an image. Returns (None, "") on traversal, missing file, or stat error.

        The sync os.stat is wrapped in asyncio.to_thread to keep the event
        loop responsive when the artifact lives on a slow filesystem.
        """
        resolved = os.path.realpath(os.path.join(artifact_dir, rel_path))
        real_base = os.path.realpath(artifact_dir)
        if not (resolved.startswith(real_base + os.sep) or resolved == real_base):
            return None, ""  # path traversal blocked
        if not os.path.exists(resolved):
            return None, ""
        ext = os.path.splitext(resolved)[1].lower()
        mime = MIME_MAP.get(ext, "text/plain")

        def _stat() -> tuple[Optional[int], str]:
            try:
                return os.stat(resolved).st_size, mime
            except OSError:
                return None, ""

        return await asyncio.to_thread(_stat)

    async def stat_artifact_from_any(self, artifact_dirs: list[str], rel_path: str) -> tuple[Optional[int], str]:
        """Search for an artifact across all round dirs, returning the first stat hit."""
        for d in artifact_dirs:
            size, mime = await self.stat_artifact(d, rel_path)
            if size is not None:
                return size, mime
        return None, ""

    def _count_track_statuses(self, state: dict[str, Any]) -> tuple[int, int, int]:
        """Count across ALL campaign rounds' parallel_tracks.tracks."""
        campaign = state.get("campaign", {}) or {}
        raw_shipped = campaign.get("shipped_optimizations", []) or []
        shipped_ops = set()
        for item in raw_shipped:
            if isinstance(item, str):
                shipped_ops.add(item)
            elif isinstance(item, dict) and item.get("op_id"):
                shipped_ops.add(item["op_id"])
        rounds = campaign.get("rounds", []) or []
        current_round_id = campaign.get("current_round") or 1
        counted_ops: set[str] = set()
        shipped = failed = active = 0
        fail_statuses = {"FAIL", "FAILED", "GPU_BLOCKED"}
        pass_statuses = {"PASS", "PASSED", "GATED_PASS", "GATED-PASS", "SHIPPED"}

        def _track_token(track: Any) -> str:
            if not isinstance(track, dict):
                return ""
            return str(track.get("verdict") or track.get("status") or "").upper()

        if not rounds and isinstance(state.get("parallel_tracks"), dict):
            rounds = [
                {
                    "round_id": current_round_id,
                    "parallel_tracks": {"tracks": state.get("parallel_tracks") or {}},
                    "shipped": [],
                }
            ]

        for rnd in rounds:
            if not isinstance(rnd, dict):
                continue
            rid = rnd.get("round_id")
            is_current = rid == current_round_id
            for op_id in rnd.get("shipped", []) or []:
                if op_id not in counted_ops:
                    shipped += 1
                    counted_ops.add(op_id)
            tracks = (rnd.get("parallel_tracks") or {}).get("tracks") or {}
            for op_id, track in tracks.items():
                if op_id in counted_ops:
                    continue
                status = _track_token(track)
                if op_id in shipped_ops:
                    shipped += 1
                    counted_ops.add(op_id)
                elif status in pass_statuses:
                    shipped += 1
                    counted_ops.add(op_id)
                elif status in fail_statuses:
                    failed += 1
                    counted_ops.add(op_id)
                elif is_current:
                    active += 1
                    counted_ops.add(op_id)
        return shipped, failed, active

    def _count_diluted_tracks(self, state: dict[str, Any]) -> int:
        """Count tracks with diluted=True across ALL campaign rounds (dashboard-only, additive)."""
        campaign = state.get("campaign", {}) or {}
        rounds = campaign.get("rounds", []) or []
        count = 0
        for rnd in rounds:
            if not isinstance(rnd, dict):
                continue
            tracks = (rnd.get("parallel_tracks") or {}).get("tracks") or {}
            for track in tracks.values():
                if isinstance(track, dict) and track.get("diluted") is True:
                    count += 1
        return count

    def _build_pipeline_progress(self, state: dict[str, Any]) -> list[dict[str, str]]:
        """Derives pipeline progress from campaign.current_stage.

        Terminal detection uses campaign.status (campaign_complete /
        campaign_exhausted), not current_stage — terminal pseudo-stages
        no longer live in the enum.
        """
        stages = ["baseline", "mining", "debate", "implementation", "validation", "integration"]
        stage_order = {"1_baseline": 0, "2_bottleneck_mining": 1, "3_debate": 2,
                       "4_5_parallel_tracks": 3, "6_integration": 5, "7_campaign_eval": 6,
                       "7b_report": 7}
        campaign = state.get("campaign", {}) or {}
        current_stage = campaign.get("current_stage") or state.get("stage") or ""
        current_idx = stage_order.get(current_stage, -1)
        campaign_status = campaign.get("status", "")
        terminal = campaign_status in ("campaign_complete", "campaign_exhausted") or current_stage == "campaign_complete"
        progress = []
        for i, name in enumerate(stages):
            if terminal or current_idx >= 6:
                progress.append({"stage": name, "status": "completed"})
            elif i < current_idx:
                progress.append({"stage": name, "status": "completed"})
            elif i == current_idx or (i == 4 and current_idx == 3):
                progress.append({"stage": name, "status": "active"})
            else:
                progress.append({"stage": name, "status": "pending"})
        return progress
