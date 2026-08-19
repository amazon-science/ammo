#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
GPU Lock Wrapper - Execute commands with automatic GPU file lock acquisition.

This utility allows any command to be executed with exclusive GPU access.
It's designed to be used by Claude Code and other processes that need
to run GPU commands without manually managing locks.

Usage:
    # Run a Python script with GPU 0 locked
    python -m shared.gpu_lock_wrapper 0 python benchmark.py --validate

    # Run with any command
    python -m shared.gpu_lock_wrapper 0 ./my_cuda_program

    # Specify timeout (default 300s)
    python -m shared.gpu_lock_wrapper 0 --timeout 60 python quick_test.py

    # List current GPU lock status (reports all GPUs discovered from lock files,
    # not just CUDA-visible ones)
    python -m shared.gpu_lock_wrapper --status

The wrapper will:
1. Acquire the file lock for the specified GPU
2. Set CUDA_VISIBLE_DEVICES environment variable
3. Execute the command
4. Release the lock when the command completes (or fails)

Environment:
    GPU_LOCK_DIR: Override default lock directory (default: /tmp/gpu_locks)
"""

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional

# Import from the same package
try:
    from shared.gpu_file_lock import get_gpu_lock_manager, GPUFileLockManager
except ImportError:
    # Handle case when running as __main__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gpu_file_lock",
        os.path.join(os.path.dirname(__file__), "gpu_file_lock.py")
    )
    gpu_file_lock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu_file_lock)
    get_gpu_lock_manager = gpu_file_lock.get_gpu_lock_manager
    GPUFileLockManager = gpu_file_lock.GPUFileLockManager


def show_status():
    """Display current GPU lock status."""
    lock_dir = os.environ.get("GPU_LOCK_DIR", "/tmp/gpu_locks")
    manager = GPUFileLockManager(lock_dir=lock_dir)

    print(f"GPU Lock Status (lock_dir={lock_dir})")
    print(f"Total GPUs: {manager.get_gpu_count()}")
    print(f"Available: {manager.get_available_gpu_count()}")
    print()

    for gpu_id in manager.get_gpu_ids():
        is_locked = manager.is_gpu_locked(gpu_id)
        info = manager.get_lock_info(gpu_id)

        status = "LOCKED" if is_locked else "available"
        print(f"GPU {gpu_id}: {status}")

        if info and is_locked:
            print(f"    PID: {info.get('pid', 'unknown')}")
            print(f"    Job: {info.get('job_id', 'unknown')}")
            acquired_at = info.get('acquired_at')
            if acquired_at:
                elapsed = time.time() - acquired_at
                print(f"    Held for: {elapsed:.1f}s")


def run_with_lock(
    gpu_id: int,
    command: List[str],
    timeout: int = 300,
    job_id: Optional[str] = None
) -> int:
    """
    Run a command with GPU lock acquired.

    Args:
        gpu_id: GPU device ID to lock
        command: Command and arguments to execute
        timeout: Max seconds to wait for lock acquisition
        job_id: Optional job identifier for logging

    Returns:
        Exit code from the executed command
    """
    lock_dir = os.environ.get("GPU_LOCK_DIR", "/tmp/gpu_locks")
    manager = GPUFileLockManager(lock_dir=lock_dir)

    # Generate job_id if not provided
    if job_id is None:
        job_id = f"wrapper-{os.getpid()}"

    print(f"[gpu-lock] Acquiring GPU {gpu_id} lock (timeout={timeout}s)...")
    start_time = time.time()

    try:
        with manager.acquire(gpu_id=gpu_id, timeout=timeout, job_id=job_id):
            wait_time = time.time() - start_time
            print(f"[gpu-lock] GPU {gpu_id} acquired after {wait_time:.2f}s")

            # Set CUDA_VISIBLE_DEVICES for the subprocess
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Log the command
            cmd_str = " ".join(command)
            print(f"[gpu-lock] Executing: {cmd_str}")
            print(f"[gpu-lock] CUDA_VISIBLE_DEVICES={gpu_id}")
            print("-" * 60)

            # Execute the command
            result = subprocess.run(
                command,
                env=env,
                # Pass through stdin/stdout/stderr
            )

            print("-" * 60)
            print(f"[gpu-lock] Command completed with exit code {result.returncode}")
            print(f"[gpu-lock] Releasing GPU {gpu_id} lock")

            return result.returncode

    except TimeoutError as e:
        print(f"[gpu-lock] ERROR: {e}")
        print(f"[gpu-lock] GPU {gpu_id} is busy. Current holder:")
        info = manager.get_lock_info(gpu_id)
        if info:
            print(f"    PID: {info.get('pid', 'unknown')}")
            print(f"    Job: {info.get('job_id', 'unknown')}")
        return 1

    except Exception as e:
        print(f"[gpu-lock] ERROR: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Execute commands with automatic GPU lock acquisition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Python benchmark with GPU 0 locked
  python -m shared.gpu_lock_wrapper 0 python benchmark.py --validate

  # Run with custom timeout
  python -m shared.gpu_lock_wrapper 0 --timeout 60 python quick_test.py

  # Show GPU lock status
  python -m shared.gpu_lock_wrapper --status

Environment Variables:
  GPU_LOCK_DIR    Override lock directory (default: /tmp/gpu_locks)
"""
    )

    parser.add_argument(
        "gpu_id",
        type=int,
        nargs="?",
        help="GPU device ID to lock (0, 1, 2, ...)"
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command and arguments to execute"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Max seconds to wait for lock (default: 300)"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job identifier for logging"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current GPU lock status and exit"
    )

    args = parser.parse_args()

    if args.status:
        show_status()
        return 0

    if args.gpu_id is None:
        parser.error("GPU ID is required (unless using --status)")

    if not args.command:
        parser.error("Command is required")

    return run_with_lock(
        gpu_id=args.gpu_id,
        command=args.command,
        timeout=args.timeout,
        job_id=args.job_id
    )


if __name__ == "__main__":
    sys.exit(main())
