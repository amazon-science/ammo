#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Patch CMakeUserPresets.json with GPU-specific build settings.

Designed to run at session init time (not Docker build time) so it can
auto-detect the actual GPU architecture and CPU count of the host.

Patches applied:
  - TORCH_CUDA_ARCH_LIST: compile only for the target GPU arch
  - NVCC_THREADS: parallel threads per nvcc invocation
  - CMAKE_JOB_POOLS: parallel compile job pool (matched to CPU count)
  - buildPresets.jobs: parallel build jobs
  - CCACHE_BASEDIR / CCACHE_NOHASHDIR: ccache portability across worktrees

The script is idempotent -- safe to run multiple times on the same file.
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def detect_cuda_arch() -> str:
    """Auto-detect GPU compute capability.

    Tries (in order):
      1. torch.cuda.get_device_capability(0)
      2. nvidia-smi --query-gpu=compute_cap
      3. Raises RuntimeError with a clear message
    """
    # Method 1: PyTorch
    try:
        import torch  # noqa: F811
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"{major}.{minor}"
            print(f"[detect] GPU arch from PyTorch: {arch}")
            return arch
    except Exception:
        pass

    # Method 2: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            arch = result.stdout.strip().splitlines()[0].strip()
            if arch:
                print(f"[detect] GPU arch from nvidia-smi: {arch}")
                return arch
    except Exception:
        pass

    raise RuntimeError(
        "Could not auto-detect GPU architecture. "
        "Neither PyTorch CUDA nor nvidia-smi are available. "
        "Please pass --cuda-arch explicitly."
    )


def patch_presets(
    presets_path: Path,
    cuda_arch: str,
    nvcc_threads: int,
    jobs: int,
    python_path: Optional[str] = None,
) -> dict:
    """Patch CMakeUserPresets.json and return a summary of changes."""
    with open(presets_path) as f:
        presets = json.load(f)

    changes = {}

    # --- Configure presets ---
    config_preset = presets["configurePresets"][0]

    # Environment variables
    env = config_preset.setdefault("environment", {})

    old_ccache_basedir = env.get("CCACHE_BASEDIR")
    env["CCACHE_BASEDIR"] = "${sourceDir}"
    if old_ccache_basedir != "${sourceDir}":
        changes["CCACHE_BASEDIR"] = "${sourceDir}"

    old_ccache_nohashdir = env.get("CCACHE_NOHASHDIR")
    env["CCACHE_NOHASHDIR"] = "1"
    if old_ccache_nohashdir != "1":
        changes["CCACHE_NOHASHDIR"] = "1"

    # Cache variables
    cv = config_preset["cacheVariables"]

    old_arch = cv.get("TORCH_CUDA_ARCH_LIST")
    cv["TORCH_CUDA_ARCH_LIST"] = cuda_arch
    if old_arch != cuda_arch:
        changes["TORCH_CUDA_ARCH_LIST"] = f"{old_arch!r} -> {cuda_arch!r}"

    old_nvcc = cv.get("NVCC_THREADS")
    nvcc_str = str(nvcc_threads)
    cv["NVCC_THREADS"] = nvcc_str
    if old_nvcc != nvcc_str:
        changes["NVCC_THREADS"] = f"{old_nvcc!r} -> {nvcc_str!r}"

    job_pool = f"compile={jobs}"
    old_pool = cv.get("CMAKE_JOB_POOLS")
    cv["CMAKE_JOB_POOLS"] = job_pool
    if old_pool != job_pool:
        changes["CMAKE_JOB_POOLS"] = f"{old_pool!r} -> {job_pool!r}"

    # --- Build presets ---
    old_jobs = presets["buildPresets"][0].get("jobs")
    presets["buildPresets"][0]["jobs"] = jobs
    if old_jobs != jobs:
        changes["buildPresets.jobs"] = f"{old_jobs!r} -> {jobs!r}"

    # --- Python paths (Bug 2 fix) ---
    if python_path is not None:
        # Derive venv root from /path/to/venv/bin/python -> /path/to/venv
        venv_root = str(Path(python_path).parent.parent)
        for key in list(cv.keys()):
            # Case-insensitive check: matches Python_*, VLLM_PYTHON_EXECUTABLE, etc.
            if "python" not in key.lower():
                continue
            old_val = cv[key]
            if key.endswith("_EXECUTABLE"):
                new_val = python_path
            elif key.endswith("_ROOT_DIR"):
                new_val = venv_root
            else:
                continue
            if old_val != new_val:
                cv[key] = new_val
                changes[key] = f"{old_val!r} -> {new_val!r}"

    # Write back
    with open(presets_path, "w") as f:
        json.dump(presets, f, indent=4)
        f.write("\n")

    return changes


def main():
    parser = argparse.ArgumentParser(
        description="Patch CMakeUserPresets.json with GPU-specific build settings."
    )
    parser.add_argument(
        "--presets-path",
        type=Path,
        default=Path("CMakeUserPresets.json"),
        help="Path to CMakeUserPresets.json (default: ./CMakeUserPresets.json)",
    )

    arch_group = parser.add_mutually_exclusive_group(required=True)
    arch_group.add_argument(
        "--cuda-arch",
        type=str,
        help="Explicit CUDA architecture (e.g., 8.9, 9.0)",
    )
    arch_group.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect GPU architecture from hardware",
    )

    parser.add_argument(
        "--nvcc-threads",
        type=int,
        default=16,
        help="Parallel threads per nvcc invocation (default: 16)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel compile jobs (default: nproc)",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="Only patch; don't run generate_cmake_presets.py first",
    )
    parser.add_argument(
        "--python-path",
        type=str,
        default=None,
        help="Path to the session venv's python binary; patches Python_* CMake cache variables",
    )

    args = parser.parse_args()

    # Resolve jobs default
    if args.jobs is None:
        args.jobs = multiprocessing.cpu_count()

    # Resolve CUDA arch
    if args.auto_detect:
        cuda_arch = detect_cuda_arch()
    else:
        cuda_arch = args.cuda_arch

    presets_path = args.presets_path.resolve()

    # Optionally regenerate base presets first
    if not args.patch_only:
        print("[generate] Running generate_cmake_presets.py --force-overwrite ...")
        result = subprocess.run(
            [sys.executable, "tools/generate_cmake_presets.py", "--force-overwrite"],
            cwd=presets_path.parent,
        )
        if result.returncode != 0:
            print("[error] generate_cmake_presets.py failed", file=sys.stderr)
            sys.exit(1)

    if not presets_path.exists():
        print(f"[error] {presets_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Apply patches
    changes = patch_presets(presets_path, cuda_arch, args.nvcc_threads, args.jobs,
                            python_path=args.python_path)

    # Summary
    if changes:
        print(f"\n[patched] {presets_path}")
        for key, value in changes.items():
            print(f"  {key}: {value}")
    else:
        print(f"\n[no-op] {presets_path} already up to date")

    print(f"\nSettings: arch={cuda_arch}, nvcc_threads={args.nvcc_threads}, jobs={args.jobs}")


if __name__ == "__main__":
    main()
