# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Inject an in-range identity into vLLM CUDA-profiler NVTX annotations.

Loaded only for AMMO Nsight children via a prepended PYTHONPATH. The hook is
installed before vLLM imports, including in spawned executor workers.
"""

from __future__ import annotations

import builtins
import json
import os
import sys


_ORIGINAL_IMPORT = builtins.__import__
_PATCHED = False


def _install_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    module = sys.modules.get("vllm.profiler.wrapper")
    if module is None or not hasattr(module, "WorkerProfiler") or not hasattr(
        module, "CudaProfilerWrapper"
    ):
        return
    raw = os.environ.get("AMMO_NSYS_CAPTURE_IDENTITIES_JSON")
    if not raw:
        return
    identities = json.loads(raw)
    if not isinstance(identities, list) or not identities or not all(
        isinstance(item, str) and item.startswith("AMMO_CAPTURE_V1|")
        for item in identities
    ):
        raise RuntimeError("invalid AMMO_NSYS_CAPTURE_IDENTITIES_JSON")

    worker_profiler = module.WorkerProfiler
    cuda_profiler = module.CudaProfilerWrapper
    original_call_start = worker_profiler._call_start
    original_annotate = cuda_profiler.annotate_context_manager
    original_cuda_start = cuda_profiler._start
    original_cuda_stop = cuda_profiler._stop

    def _distributed_controller() -> tuple[bool, object | None]:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0, dist
        except Exception:
            pass
        rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        return int(rank) == 0, None

    def _start_once_globally(self):
        controller, dist = _distributed_controller()
        if dist is not None:
            dist.barrier()
        if controller:
            original_cuda_start(self)
        if dist is not None:
            dist.barrier()

    def _stop_once_globally(self):
        controller, dist = _distributed_controller()
        if dist is not None:
            dist.barrier()
        if controller:
            original_cuda_stop(self)
        if dist is not None:
            dist.barrier()

    def _call_start_with_identity(self):
        original_call_start(self)
        if not getattr(self, "_running", False):
            return
        index = int(getattr(self, "_ammo_capture_identity_index", 0))
        if index >= len(identities):
            raise RuntimeError("AMMO capture identity sequence exhausted")
        self._ammo_capture_identity = identities[index]
        self._ammo_capture_identity_index = index + 1

    def _annotate_with_identity(self, name: str):
        identity = getattr(self, "_ammo_capture_identity", None)
        if getattr(self, "_running", False) and isinstance(identity, str):
            name = f"{name}|{identity}"
        return original_annotate(self, name)

    worker_profiler._call_start = _call_start_with_identity
    cuda_profiler._start = _start_once_globally
    cuda_profiler._stop = _stop_once_globally
    cuda_profiler.annotate_context_manager = _annotate_with_identity
    _PATCHED = True


def _import_and_patch(name, globals=None, locals=None, fromlist=(), level=0):
    result = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if not _PATCHED and (
        name == "vllm.profiler.wrapper" or "vllm.profiler.wrapper" in sys.modules
    ):
        _install_patch()
    return result


if os.environ.get("AMMO_NSYS_CAPTURE_IDENTITIES_JSON"):
    builtins.__import__ = _import_and_patch
