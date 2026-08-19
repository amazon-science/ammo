#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""GPU reservation module for AMMO benchmark orchestration.

Provides a file-locked, JSON-backed reservation system for NVIDIA GPUs.
State lives under AMMO_GPU_RES_DIR when set. Otherwise it falls back to
/tmp/ammo_gpu_res_{sha256(CUDA_VISIBLE_DEVICES)[:12]}/state.json so sessions
with different visible GPU pools do not share reservation state.

Agents dynamically request N GPUs from the pool:
    CVD=$(python gpu_reservation.py reserve --num-gpus 2 \
        --session-id {gpu_session_id} --no-auto-release) && \
        CUDA_VISIBLE_DEVICES=$CVD python benchmark.py

Subcommands
-----------
reserve         allocate N GPUs from the session pool
release-session release a session's GPUs (optionally its children)
status          dump raw state as JSON; --table for a human-readable view
force-clear     crash recovery: break stale holds (see force_clear())

Public API
----------
read_state()                              -> dict
reserve(num_gpus, session_id, ...)        -> list[int]
release_by_session(session_id, include_children=False) -> list[int]
write_reservation(gpu_ids, session_id, ...) -> None   (internal helper)
force_clear(...)                          -> None
check_and_reclaim_expired()               -> list[int]
print_table(state)                        -> None    (status --table renderer)

No GPU discovery or file creation occurs on import.  All state is initialised
lazily on the first read_state() call.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (module-level constants patchable in tests)
# ---------------------------------------------------------------------------

# STATE_DIR is assigned after _compute_state_dir() is defined. It remains a
# module-level constant so tests and emergency tooling can monkeypatch it.


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class LockTimeoutError(Exception):
    """Raised when the exclusive file lock cannot be acquired after all retries."""


class ReservationError(Exception):
    """Raised when a GPU reservation cannot be created because the GPU is held."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lock_path() -> Path:
    return STATE_DIR / "state.lock"


def _state_path() -> Path:
    return STATE_DIR / "state.json"


def _discover_gpu_count() -> int:
    """Return the number of NVIDIA GPUs visible to nvidia-smi.

    Returns 0 on any failure (nvidia-smi not found, no GPUs, etc.).
    NOTE: This function sees ALL host GPUs, ignoring session boundaries.
    Use _discover_session_gpus() for session-scoped reservation pools.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        return len(lines)
    except Exception:
        return 0


def _discover_session_gpus() -> list[int]:
    """Discover which physical GPU IDs this session owns via CUDA_VISIBLE_DEVICES.

    Parses the CUDA_VISIBLE_DEVICES environment variable to get the exact
    physical GPU IDs allocated to this session by the server.

    Critical for multi-session isolation: nvidia-smi sees all host GPUs,
    but CUDA_VISIBLE_DEVICES reflects only THIS session's allocation.
    Since CVD is non-composable (child CVD overrides parent entirely),
    returned IDs are physical host IDs that must be used directly.

    Returns:
        Sorted list of integer GPU IDs, e.g. [4, 5, 6, 7].
        Empty list if CUDA_VISIBLE_DEVICES is unset, empty, or "-1".
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd or cvd == "-1":
        return []
    gpu_ids = []
    for x in cvd.split(","):
        x = x.strip()
        if x.isdigit():
            gpu_ids.append(int(x))
        elif x.startswith("GPU-") or x.startswith("MIG-"):
            logger.warning(
                f"Non-integer GPU ID in CUDA_VISIBLE_DEVICES: {x} — skipping. "
                "gpu_reservation.py requires integer GPU IDs."
            )
    if gpu_ids:
        logger.info(f"GPU reservation pool initialized with physical IDs: {gpu_ids}")
    return sorted(gpu_ids)


def _compute_state_dir() -> Path:
    """Compute the state directory path at call time.

    Uses AMMO_GPU_RES_DIR env var if set (injected by the server at session
    creation for per-session isolation).

    Falls back to /tmp/ammo_gpu_res_{sha256(CVD)[:12]}/ using a SHA256 hash
    of CUDA_VISIBLE_DEVICES. This is deterministic across processes (unlike
    hash() which is randomized by PYTHONHASHSEED) and provides isolation
    between sessions with different GPU allocations.
    """
    env_dir = os.environ.get("AMMO_GPU_RES_DIR")
    if env_dir:
        return Path(env_dir)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    hash_suffix = hashlib.sha256(cvd.encode()).hexdigest()[:12]
    return Path(f"/tmp/ammo_gpu_res_{hash_suffix}")


# Delays between successive flock attempts (seconds).  4 delays -> 5 attempts.
STATE_DIR: Path = _compute_state_dir()
BACKOFF_DELAYS: list[float] = [0.1, 0.2, 0.4, 0.8]


def _init_state() -> dict:
    """Create and persist an initial state.json based on session's GPU pool.

    Uses CUDA_VISIBLE_DEVICES to discover the physical GPU IDs owned by this
    session (not nvidia-smi, which would see all host GPUs).
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    gpu_ids = _discover_session_gpus()
    state: dict = {
        "gpus": {str(g): None for g in gpu_ids},
        "gpu_count": len(gpu_ids),
        "audit": [],
    }
    _write_state(state)
    return state


def _acquire_flock() -> IO:
    """Acquire an exclusive advisory lock on the lock file.

    Attempts up to len(BACKOFF_DELAYS)+1 times with exponential backoff.
    Returns the open file handle (caller must close / release it).
    Raises LockTimeoutError if all attempts fail.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = _lock_path()
    attempts = len(BACKOFF_DELAYS) + 1  # e.g. 5 with default 4-element list

    fh = None
    for attempt in range(attempts):
        try:
            fh = open(lock_file, "w")
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh  # success
        except BlockingIOError:
            if fh is not None:
                fh.close()
                fh = None
            if attempt < len(BACKOFF_DELAYS):
                time.sleep(BACKOFF_DELAYS[attempt])
            # else: last attempt failed — fall through to raise below

    raise LockTimeoutError(
        f"Could not acquire GPU reservation lock after {attempts} attempts"
    )


def _release_flock(fh: IO) -> None:
    """Release and close the lock file handle."""
    try:
        fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def _write_state(state: dict) -> None:
    """Atomically write state to state.json via a temp-file + rename."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    target = _state_path()
    fd, tmp_path = tempfile.mkstemp(dir=STATE_DIR, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_state() -> dict:
    """Read and return the state dict, initialising state.json if absent."""
    state_file = _state_path()
    if not state_file.exists():
        return _init_state()
    return json.loads(state_file.read_text())


def write_reservation(
    gpu_ids: list[int],
    session_id: str,
    command_snippet: str = "",
    lease_hours: float = 0.25,
) -> None:
    """Standalone low-level reservation function that manages its own locking.

    For pool-based dynamic allocation, use reserve() instead. Does NOT
    auto-release existing session reservations or reclaim expired leases.

    Acquires the exclusive lock, checks that each GPU is free (or expired),
    writes reservation entries, and releases the lock.

    Raises ReservationError if any GPU is currently held by a non-expired lease.
    """
    fh = _acquire_flock()
    try:
        state = read_state()
        now = datetime.now(tz=timezone.utc)

        # Validate that all requested GPU IDs exist in state
        known = set(state["gpus"].keys())
        unknown = [gid for gid in gpu_ids if str(gid) not in known]
        if unknown:
            raise ReservationError(f"GPU IDs not found in state: {unknown}")

        # Check all GPUs are free before writing any
        held: list[str] = []
        for gid in gpu_ids:
            key = str(gid)
            entry = state["gpus"].get(key)
            if entry is not None:
                # Check if the lease has expired
                expires = datetime.fromisoformat(entry["lease_expires"])
                if expires > now:
                    held.append(
                        f"GPU {gid} held by session={entry['session_id']} "
                        f"until {entry['lease_expires']}"
                    )
        if held:
            raise ReservationError(
                "Cannot reserve GPUs — already held:\n" + "\n".join(held)
            )

        # Write entries
        reserved_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        lease_expires = (now + timedelta(hours=lease_hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for gid in gpu_ids:
            state["gpus"][str(gid)] = {
                "session_id": session_id,
                "reserved_at": reserved_at,
                "lease_expires": lease_expires,
                "command_snippet": command_snippet[:80],
            }

        _write_state(state)
    finally:
        _release_flock(fh)


def reserve(
    num_gpus: int,
    session_id: str,
    command_snippet: str = "",
    lease_hours: float = 0.25,
    auto_release: bool = True,
) -> list[int]:
    """Dynamically reserve *num_gpus* GPUs from the pool.

    Uses contiguous-first allocation:
      - For num_gpus=1, takes the first free GPU.
      - For num_gpus>1, requires a contiguous block of free GPUs.

    When *auto_release* is True (default), releases any existing reservations
    for *session_id* before allocating, so retries don't exhaust the pool.
    Set *auto_release* to False when multiple agents share a session_id and
    should not evict each other.

    Returns the list of allocated GPU IDs on success.
    Raises ReservationError if the request cannot be satisfied.
    """
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")

    fh = _acquire_flock()
    try:
        state = read_state()
        now = datetime.now(tz=timezone.utc)
        now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # --- Reclaim expired leases (inline) ---
        for key, entry in list(state["gpus"].items()):
            if entry is None:
                continue
            expires = datetime.fromisoformat(entry["lease_expires"])
            if expires <= now:
                state["audit"].append(
                    {
                        "action": "reclaim_expired",
                        "gpu_id": key,
                        "reclaimed_at": now_str,
                        "was_session": entry.get("session_id"),
                        "lease_expired_at": entry["lease_expires"],
                    }
                )
                state["gpus"][key] = None

        # --- Auto-release existing reservations for this session_id ---
        if auto_release:
            for key, entry in state["gpus"].items():
                if entry is not None and entry.get("session_id") == session_id:
                    state["gpus"][key] = None

        # --- Find free GPUs ---
        # Iterate the actual pool keys (physical IDs), not range(gpu_count).
        # This is critical for multi-session isolation: physical IDs may be
        # non-contiguous (e.g. [4,5,6,7]), and range(gpu_count) would wrongly
        # return 0-based indices that map to different physical GPUs.
        all_pool_ids = sorted(int(k) for k in state["gpus"].keys())
        free_set = set()
        for gpu_id in all_pool_ids:
            if state["gpus"].get(str(gpu_id)) is None:
                free_set.add(gpu_id)

        if len(free_set) < num_gpus:
            raise ReservationError(
                f"Not enough free GPUs: requested {num_gpus}, "
                f"available {len(free_set)}"
            )

        # --- Contiguous-first allocation ---
        allocated: list[int] | None = None

        if num_gpus == 1:
            # Just take the first free GPU
            for i in all_pool_ids:
                if i in free_set:
                    allocated = [i]
                    break
        else:
            # Scan for a contiguous block of num_gpus consecutive free GPUs
            run_start = None
            run_len = 0
            prev_i = None
            for i in all_pool_ids:
                if i in free_set and (prev_i is None or i == prev_i + 1):
                    if run_start is None:
                        run_start = i
                        run_len = 1
                    else:
                        run_len += 1
                    if run_len >= num_gpus:
                        allocated = list(range(run_start, run_start + num_gpus))
                        break
                    prev_i = i
                elif i in free_set:
                    run_start = i
                    run_len = 1
                    prev_i = i
                    if run_len >= num_gpus:
                        allocated = [i]
                        break
                else:
                    run_start = None
                    run_len = 0
                    prev_i = None

        if allocated is None:
            raise ReservationError(
                f"No contiguous block of {num_gpus} free GPUs available. "
                f"Free GPUs: {sorted(free_set)}"
            )

        # --- Write reservation entries ---
        reserved_at = now_str
        lease_expires = (now + timedelta(hours=lease_hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for gid in allocated:
            state["gpus"][str(gid)] = {
                "session_id": session_id,
                "reserved_at": reserved_at,
                "lease_expires": lease_expires,
                "command_snippet": command_snippet[:80],
            }

        _write_state(state)
        return allocated
    finally:
        _release_flock(fh)


def _default_session_id() -> str:
    """Return the portable default reservation owner.

    The server injects AMMO_SESSION_ID for every CLI tool. CLAUDE_SESSION_ID is
    the Claude-runtime fallback. 'cli' is the last resort for manual use.
    """
    return (
        os.environ.get("AMMO_SESSION_ID")
        or os.environ.get("CLAUDE_SESSION_ID")
        or "cli"
    )


def release_by_session(session_id: str, include_children: bool = False) -> list[int]:
    """Clear all GPU entries owned by *session_id*.

    If include_children is true, also clear entries whose session_id starts
    with ``<session_id>:``. Subagent reservations use that prefix.

    Returns the list of released GPU IDs (as integers).
    """
    fh = _acquire_flock()
    try:
        state = read_state()
        released: list[int] = []
        for key, entry in state["gpus"].items():
            owner = entry.get("session_id") if entry is not None else None
            owned = owner == session_id or (
                include_children
                and isinstance(owner, str)
                and owner.startswith(f"{session_id}:")
            )
            if owned:
                state["gpus"][key] = None
                released.append(int(key))
        if released:
            _write_state(state)
        return released
    finally:
        _release_flock(fh)


def force_clear(
    gpu_ids: list[int] | None = None,
    session_id: str | None = None,
    force_no_session: bool = False,
) -> None:
    """Forcibly clear reservation entries.

    Must provide either session_id or force_no_session=True.

    If session_id is given, clears only entries owned by that session.
    If gpu_ids is given together with session_id, further restricts to those GPUs.
    If force_no_session=True, clears all entries (or only gpu_ids if given).
    An audit log entry is written for each cleared GPU.

    Raises ValueError if neither session_id nor force_no_session=True is supplied.
    """
    if session_id is None and not force_no_session:
        raise ValueError(
            "force_clear requires either session_id or force_no_session=True"
        )

    fh = _acquire_flock()
    try:
        state = read_state()
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        gpu_keys: list[str]
        if gpu_ids is not None:
            gpu_keys = [str(g) for g in gpu_ids]
        else:
            gpu_keys = list(state["gpus"].keys())

        for key in gpu_keys:
            entry = state["gpus"].get(key)
            if entry is None:
                continue
            # Filter by session_id if provided
            if session_id is not None and entry.get("session_id") != session_id:
                continue
            # Log to audit before clearing
            state["audit"].append(
                {
                    "action": "force_clear",
                    "gpu_id": key,
                    "cleared_at": now_str,
                    "was_session": entry.get("session_id"),
                }
            )
            state["gpus"][key] = None

        _write_state(state)
    finally:
        _release_flock(fh)


def check_and_reclaim_expired() -> list[int]:
    """Check for expired leases, clear them, write state, and return reclaimed IDs.

    Acquires the exclusive lock so it is safe to call standalone from hooks.
    Reads fresh state under the lock to avoid stale-state races.
    Returns the list of reclaimed GPU IDs (as integers).
    """
    fh = _acquire_flock()
    try:
        state = read_state()
        now = datetime.now(tz=timezone.utc)
        reclaimed: list[int] = []
        now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        for key, entry in state["gpus"].items():
            if entry is None:
                continue
            expires = datetime.fromisoformat(entry["lease_expires"])
            if expires <= now:
                state["audit"].append(
                    {
                        "action": "reclaim_expired",
                        "gpu_id": key,
                        "reclaimed_at": now_str,
                        "was_session": entry.get("session_id"),
                        "lease_expired_at": entry["lease_expires"],
                    }
                )
                state["gpus"][key] = None
                reclaimed.append(int(key))

        if reclaimed:
            _write_state(state)

        return reclaimed
    finally:
        _release_flock(fh)


# ---------------------------------------------------------------------------
# Display helpers (`status --table`)
# ---------------------------------------------------------------------------

def _iso_to_display(iso: str | None) -> str:
    """Convert ISO timestamp to a more readable local-ish form."""
    if iso is None:
        return "-"
    # Input is always UTC Z-suffix; strip and reformat for display
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        # Display in UTC without timezone suffix for brevity
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso


def _truncate(s: str | None, width: int) -> str:
    if s is None:
        return "-"
    return s if len(s) <= width else s[: width - 1] + "…"


def print_table(state: dict) -> None:
    """Print one fixed-width row per pool GPU — output is bounded by GPU count."""
    gpus = state.get("gpus", {})
    if not gpus:
        print("No GPUs tracked in state.")
        return

    # Column widths
    COL_GPU = 4
    COL_STATUS = 9
    COL_CMD = 42
    COL_SINCE = 19
    COL_EXPIRES = 19
    COL_SESSION = 16

    header = (
        f"{'GPU':<{COL_GPU}}  "
        f"{'Status':<{COL_STATUS}}  "
        f"{'Command':<{COL_CMD}}  "
        f"{'Since':<{COL_SINCE}}  "
        f"{'Lease Expires':<{COL_EXPIRES}}  "
        f"{'Session':<{COL_SESSION}}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    for gpu_id in sorted(gpus.keys(), key=lambda x: int(x)):
        entry = gpus[gpu_id]
        if entry is None:
            row = (
                f"{gpu_id:<{COL_GPU}}  "
                f"{'FREE':<{COL_STATUS}}  "
                f"{'-':<{COL_CMD}}  "
                f"{'-':<{COL_SINCE}}  "
                f"{'-':<{COL_EXPIRES}}  "
                f"{'-':<{COL_SESSION}}"
            )
        else:
            cmd = _truncate(entry.get("command_snippet"), COL_CMD)
            since = _iso_to_display(entry.get("reserved_at"))
            expires = _iso_to_display(entry.get("lease_expires"))
            session = _truncate(entry.get("session_id"), COL_SESSION)
            row = (
                f"{gpu_id:<{COL_GPU}}  "
                f"{'HELD':<{COL_STATUS}}  "
                f"{cmd:<{COL_CMD}}  "
                f"{since:<{COL_SINCE}}  "
                f"{expires:<{COL_EXPIRES}}  "
                f"{session:<{COL_SESSION}}"
            )
        print(row)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AMMO GPU reservation pool manager"
    )
    sub = parser.add_subparsers(dest="command")

    # --- reserve ---
    p_reserve = sub.add_parser("reserve", help="Reserve N GPUs from the pool")
    p_reserve.add_argument(
        "--num-gpus", type=int, required=True, help="Number of GPUs to reserve"
    )
    p_reserve.add_argument(
        "--session-id",
        type=str,
        default=_default_session_id(),
        help="Session identifier (default: $AMMO_SESSION_ID, $CLAUDE_SESSION_ID, or 'cli')",
    )
    p_reserve.add_argument(
        "--lease-hours",
        type=float,
        default=0.25,
        help="Lease duration in hours (default: 0.25 = 15 min; pass 2.0 for long sweeps/nsys)",
    )
    p_reserve.add_argument(
        "--command-snippet", type=str, default="", help="Command description"
    )
    p_reserve.add_argument(
        "--no-auto-release",
        action="store_true",
        default=False,
        help="Skip auto-release of existing reservations for this session_id",
    )

    # --- release-session ---
    p_release = sub.add_parser(
        "release-session", help="Release all GPUs held by a session"
    )
    p_release.add_argument(
        "--session-id", type=str, required=True, help="Session identifier"
    )
    p_release.add_argument(
        "--include-children",
        action="store_true",
        help="Also release subagent sessions prefixed with '<session-id>:'",
    )

    # --- status ---
    p_status = sub.add_parser("status", help="Print current reservation state as JSON")
    p_status.add_argument(
        "--table",
        action="store_true",
        help="Print a human-readable table instead of JSON (one row per pool GPU)",
    )

    # --- force-clear ---
    p_force = sub.add_parser(
        "force-clear", help="Forcibly clear reservations after a crash"
    )
    force_target = p_force.add_mutually_exclusive_group(required=True)
    force_target.add_argument(
        "--gpu-ids",
        metavar="IDS",
        help="Comma-separated GPU IDs to clear (e.g. 0,1).",
    )
    force_target.add_argument(
        "--all",
        action="store_true",
        help="Clear reservations on all GPUs.",
    )
    force_session = p_force.add_mutually_exclusive_group(required=True)
    force_session.add_argument(
        "--session-id",
        metavar="ID",
        help="Only clear reservations owned by this session ID.",
    )
    force_session.add_argument(
        "--force-no-session",
        action="store_true",
        help="Emergency override: clear without session check.",
    )

    args = parser.parse_args()

    if args.command == "reserve":
        try:
            gpu_ids = reserve(
                num_gpus=args.num_gpus,
                session_id=args.session_id,
                command_snippet=args.command_snippet,
                lease_hours=args.lease_hours,
                auto_release=not args.no_auto_release,
            )
            # Print comma-separated GPU IDs to stdout
            print(",".join(str(g) for g in gpu_ids))
            sys.exit(0)
        except (ReservationError, LockTimeoutError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

    elif args.command == "release-session":
        try:
            released = release_by_session(
                args.session_id,
                include_children=args.include_children,
            )
            print(
                f"Released GPUs: {','.join(str(g) for g in released)}",
                file=sys.stderr,
            )
            sys.exit(0)
        except (ReservationError, LockTimeoutError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

    elif args.command == "status":
        state = read_state()
        if args.table:
            print_table(state)
        else:
            print(json.dumps(state, indent=2))

    elif args.command == "force-clear":
        # Resolve GPU IDs (None means all)
        if args.all:
            gpu_ids = None
        else:
            try:
                gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
            except ValueError:
                print(
                    f"ERROR: --gpu-ids must be comma-separated integers, got: {args.gpu_ids!r}",
                    file=sys.stderr,
                )
                sys.exit(1)

        if args.force_no_session:
            print(
                "WARNING: --force-no-session is an emergency override. "
                "All matched reservations will be cleared regardless of session ownership.",
                file=sys.stderr,
            )

        try:
            force_clear(
                gpu_ids=gpu_ids,
                session_id=args.session_id if not args.force_no_session else None,
                force_no_session=args.force_no_session,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        print("GPU reservations cleared.")

    else:
        parser.print_help()
        sys.exit(1)
