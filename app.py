# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
FastAPI frontend for the AMMO Session Service
Provides async HTTP endpoints for AI-CLI GPU optimization sessions
"""

import logging
import time
import hmac
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Header, Cookie, Depends
from fastapi.responses import JSONResponse, HTMLResponse, Response, FileResponse
import re
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketState
from contextlib import asynccontextmanager
import asyncio
import os
import subprocess
from typing import Dict, Optional
from pydantic import ValidationError
import httpx
import websockets

from shared.utils import create_error_response, create_no_cache_headers
from shared.gpu_resource_manager import GPUResourceManager
from shared.session_models import (
    CreateSessionRequest,
    SessionStatus,
)
from shared.fork_url_validator import validate_fork_url, ForkUrlError
from shared.fork_token_crypto import fork_token_key_configured
from orchestration.session_manager import SessionManager, SessionError as SessionMgrError, SessionLimitError as SessionLimitMgrError, get_session_manager
from orchestration.terminal_manager import get_terminal_manager, TerminalManager
from orchestration.inactivity_monitor import get_inactivity_monitor, InactivityMonitor
from orchestration.session_state import get_session_storage, SessionS3Storage
from orchestration.checkpoint_manager import get_checkpoint_manager, CheckpointManager
from orchestration.campaign_data_service import CampaignDataService, _normalize_e2e_latency

logger = logging.getLogger(__name__)

# ============================================================================
# API Key Authentication Middleware
# ============================================================================

AMMO_API_KEY = os.getenv("AMMO_API_KEY", "")

# Paths that require auth when AMMO_API_KEY is set
PROTECTED_PATH_PREFIXES = ["/sessions", "/api/", "/docs", "/redoc", "/openapi.json"]
# Exact paths that are open even though they match a prefix
OPEN_EXACT_PATHS = {"/ui", "/api/changelog"}


SLOW_REQUEST_THRESHOLD = float(os.getenv("SLOW_REQUEST_THRESHOLD_SECONDS", "1.0"))


class SlowRequestMiddleware(BaseHTTPMiddleware):
    """Log warnings for requests exceeding a latency threshold.

    Catches performance regressions at runtime — any endpoint taking longer
    than SLOW_REQUEST_THRESHOLD_SECONDS (default 1s) is logged as a warning.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        if duration > SLOW_REQUEST_THRESHOLD:
            logger.warning(
                f"SLOW REQUEST: {request.method} {request.url.path} "
                f"took {duration:.2f}s (threshold: {SLOW_REQUEST_THRESHOLD}s)"
            )

        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to gate protected endpoints behind an API key.

    When AMMO_API_KEY env var is set and non-empty, requests to protected
    paths must provide a valid key via one of:
      1. Authorization: Bearer <key>
      2. X-API-Key: <key> header
      3. ammo_api_key cookie
      4. ?token=<key> query parameter

    /health remains open.
    """

    async def dispatch(self, request: Request, call_next):
        # If no API key configured, pass everything through (dev mode)
        if not AMMO_API_KEY:
            return await call_next(request)

        path = request.url.path

        # Exact open paths bypass auth (e.g. /ui serves the login page).
        if path in OPEN_EXACT_PATHS:
            return await call_next(request)

        # Check if path matches any protected prefix
        is_protected = any(path.startswith(prefix) for prefix in PROTECTED_PATH_PREFIXES)

        if not is_protected:
            return await call_next(request)

        # Extract key from request (priority order)
        provided_key = self._extract_key(request)

        if provided_key and hmac.compare_digest(provided_key, AMMO_API_KEY):
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
        )

    @staticmethod
    def _extract_key(request: Request) -> str:
        """Extract API key from request using priority order."""
        # 1. Authorization: Bearer <key>
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            # Return even if empty — caller will reject via compare_digest
            return token if token else ""

        # 2. X-API-Key header
        x_api_key = request.headers.get("x-api-key", "")
        if x_api_key:
            return x_api_key

        # 3. ammo_api_key cookie
        cookie_key = request.cookies.get("ammo_api_key", "")
        if cookie_key:
            return cookie_key

        # 4. ?token= query parameter
        token_param = request.query_params.get("token", "")
        if token_param:
            return token_param

        return ""


# Global GPU resource manager - will be initialized in lifespan
gpu_manager: GPUResourceManager = None
# Global GPU type - detected at startup
gpu_type: str = None
# Global session manager - will be initialized in lifespan
session_manager: SessionManager = None
# Global terminal manager - will be initialized in lifespan
terminal_manager: TerminalManager = None
# Global inactivity monitor - will be initialized in lifespan
inactivity_monitor: InactivityMonitor = None
# Global session S3 storage - will be initialized in lifespan
session_s3_storage: SessionS3Storage = None
# Global checkpoint manager - will be initialized in lifespan
checkpoint_manager: CheckpointManager = None
# Base repo readiness state (surfaced in /health). Set True by lifespan when
# WorktreeManager.ensure_base_repos() succeeds; stays False if the clone fails
# so operators can see that S3 restore onto this server is degraded.
_base_repos_ready: bool = False
# Track pending checkpoint tasks by session_id (for cancellation on reconnect)
_pending_checkpoint_tasks: Dict[str, asyncio.Task] = {}
# Track active WebSocket proxy tasks for graceful shutdown
_active_websocket_tasks: Dict[str, asyncio.Task] = {}


# ============================================================================
# Client ID Extraction for Multi-User Session Isolation
# ============================================================================

CLIENT_ID_HEADER = "X-Client-ID"
CLIENT_ID_COOKIE = "ammo_client_id"
# UUID v4 pattern: 8-4-4-4-12 hex digits with version 4 marker
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.I)


async def get_client_id(
    x_client_id: Optional[str] = Header(None, alias=CLIENT_ID_HEADER),
    client_id_cookie: Optional[str] = Cookie(None, alias=CLIENT_ID_COOKIE),
) -> Optional[str]:
    """
    Extract and validate client ID from header or cookie.

    The client ID is used for multi-user session isolation. Each browser
    generates a UUID on first visit and sends it with all requests.

    Priority:
    1. X-Client-ID header
    2. ammo_client_id cookie

    Returns:
        Valid UUID v4 string or None if not provided/invalid
    """
    client_id = x_client_id or client_id_cookie
    if client_id and UUID_PATTERN.match(client_id):
        return client_id
    return None


def detect_gpu_type() -> str:
    """Detect GPU type from environment or hardware"""
    # Supported GPU types (order matters - check more specific first)
    SUPPORTED_GPUS = ['b300', 'b200', 'h200', 'h100', 'a100', 'l40s', 'l40', 'a10g', 'a10', 'a30', 'a40', 'v100']

    # First check environment variable
    env_gpu = os.environ.get('GPU_TYPE', '').lower()
    if env_gpu in SUPPORTED_GPUS:
        return env_gpu

    # Fallback to hardware detection
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_name = result.stdout.strip().lower()
            for gpu in SUPPORTED_GPUS:
                if gpu in gpu_name:
                    return gpu
    except Exception as e:
        logger.warning(f"GPU detection via nvidia-smi failed: {e}")

    return 'unknown'


def detect_gpu_memory_gb() -> float:
    """Detect per-GPU memory in GB from hardware.

    Assumes homogeneous GPUs (all same model on a given host).
    Fallback is 16 GB (smallest supported GPU: V100-16GB) to avoid
    over-provisioning on unknown hardware.
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0]) / 1024  # MiB → GB
    except Exception:
        pass
    logger.warning("GPU memory detection failed, using conservative 16 GB fallback")
    return 16.0


gpu_memory_gb: float = 48.0


# ============================================================================
# Startup Pre-Warm Functions (eliminates cold-start latency after rolling deploy)
# ============================================================================


async def _prewarm_campaign_caches(sess_mgr, cds) -> None:
    """Pre-warm campaign data caches for all local sessions.

    Iterates all sessions (bypassing owner filter) and calls find_artifact_dir +
    read_state to populate _artifact_dir_cache and _state_cache before traffic arrives.
    """
    try:
        sessions = list(sess_mgr._sessions.values())
        for state in sessions:
            try:
                if not state.worktree_path:
                    continue
                artifact_dir = await cds.find_artifact_dir(state.worktree_path)
                if artifact_dir:
                    await cds.read_state(artifact_dir)
            except Exception:
                continue
        logger.info(f"Campaign pre-warm complete: {len(sessions)} sessions processed")
    except Exception as e:
        logger.warning(f"Campaign cache pre-warm failed (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for FastAPI lifespan events
    Initialize services on startup and cleanup on shutdown
    Combines FastAPI lifespan with MCP app lifespan for proper session manager initialization
    """
    global gpu_manager, gpu_type, gpu_memory_gb, session_manager, terminal_manager
    global inactivity_monitor, session_s3_storage, checkpoint_manager, logger
    global _base_repos_ready

    # Startup
    logger.info("Starting AMMO Session Service...")

    # Detect GPU type and memory
    gpu_type = detect_gpu_type()
    gpu_memory_gb = detect_gpu_memory_gb()
    logger.info(f"Detected GPU type: {gpu_type}, memory: {gpu_memory_gb:.1f} GB")

    # Initialize the standalone GPU resource manager (single source of truth
    # for GPU allocation; shared with the session manager so /health sees
    # session GPU locks).
    gpu_manager = GPUResourceManager()

    # Initialize session manager with shared GPU manager
    # IMPORTANT: Must use the same GPU manager so /health endpoint
    # sees session GPU locks.
    session_manager = get_session_manager(gpu_manager=gpu_manager)
    logger.info("✅ Session Manager initialized (sharing GPU manager)")

    # Clean up orphaned worktrees from crashed sessions
    try:
        await session_manager.cleanup_orphaned_worktrees()
        logger.info("✅ Orphaned worktree cleanup completed")
    except Exception as e:
        logger.warning(f"Failed to clean up orphaned worktrees (non-fatal): {e}")

    # Ensure base repositories are cloned (critical for S3 restore).
    # repair_worktree_linkage() is a no-op if the base repo doesn't exist, so a
    # session restored from S3 onto a fresh host would silently fail without this.
    # Run BEFORE discover_s3_sessions (which can trigger session restore) and
    # off-load to a thread (clone_base_repo is sync subprocess.run). The failure
    # is non-fatal — the server can still serve eval/profiling traffic — but it is
    # surfaced via /health's `base_repos_ready` so operators see the degradation.
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, session_manager.worktree_manager.ensure_base_repos
        )
        _base_repos_ready = all(results.values())
        for repo, success in results.items():
            if success:
                logger.info(f"✅ Base repo '{repo}' ready")
            else:
                logger.error(
                    f"❌ Base repo '{repo}' clone FAILED — S3 restore will not work"
                )
    except Exception as e:
        logger.error(
            f"❌ Base repo bootstrap failed: {e} — S3 restore will not work"
        )
        _base_repos_ready = False

    # Discover sessions from S3 (for local server restart or host recovery)
    if session_manager.session_storage and session_manager.session_storage.enabled:
        try:
            discovered_count = await session_manager.discover_s3_sessions()
            if discovered_count > 0:
                logger.info(f"✅ Discovered {discovered_count} sessions from S3")
        except Exception as e:
            logger.warning(f"Failed to discover S3 sessions (non-fatal): {e}")

    # Initialize terminal manager
    terminal_manager = get_terminal_manager()
    logger.info(f"✅ Terminal Manager initialized (ttyd available: {terminal_manager.is_available()})")

    # Initialize session S3 storage
    session_s3_storage = get_session_storage()
    logger.info(f"✅ Session S3 Storage initialized (available: {session_s3_storage.enabled})")

    # Initialize and start inactivity monitor
    inactivity_monitor = get_inactivity_monitor()

    # Set up pause callback - when timeout triggers, pause session with S3 sync
    async def pause_callback(session_id: str):
        try:
            logger.info(f"Inactivity timeout triggered for session {session_id}")
            await session_manager.pause_session(session_id, sync_to_s3=True)
        except Exception as e:
            logger.error(f"Failed to auto-pause session {session_id}: {e}")

    inactivity_monitor.set_pause_callback(pause_callback)
    inactivity_monitor.start()
    logger.info("✅ Inactivity Monitor started")

    # Initialize checkpoint manager (for S3 sync on WebSocket disconnect)
    checkpoint_manager = get_checkpoint_manager()
    checkpoint_manager.set_session_manager(session_manager)
    logger.info("✅ Checkpoint Manager initialized")

    # Pre-warm local campaign caches in background.
    asyncio.create_task(_prewarm_campaign_caches(session_manager, campaign_data_service))
    logger.info("✅ Campaign cache pre-warm task launched")

    # Start background cleanup tasks
    session_cleanup_task = asyncio.create_task(cleanup_sessions_periodically())

    logger.info("✅ AMMO Sessions Server started successfully")
    logger.info("   FastAPI REST endpoints: /health")
    logger.info("   Session endpoints: /sessions")
    logger.info("   UI: /ui")

    yield

    # Shutdown
    logger.info("Shutting down AMMO Session Service...")

    # 1. Stop terminal manager FIRST - kills ttyd processes
    #    This closes ttyd WebSocket connections, breaking the forward_to_client() loops
    if terminal_manager:
        await terminal_manager.cleanup()
        logger.info("✅ Terminal Manager cleaned up (ttyd processes killed)")

    # 2. Cancel any remaining WebSocket proxy tasks (should exit quickly now)
    for task_id, task in list(_active_websocket_tasks.items()):
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled WebSocket proxy task: {task_id}")

    if _active_websocket_tasks:
        await asyncio.wait(
            list(_active_websocket_tasks.values()),
            timeout=5.0
        )
        _active_websocket_tasks.clear()

    # 3. Cancel all pending checkpoint tasks
    for session_id, task in list(_pending_checkpoint_tasks.items()):
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled pending checkpoint task for session {session_id}")

    if _pending_checkpoint_tasks:
        await asyncio.gather(*_pending_checkpoint_tasks.values(), return_exceptions=True)
        _pending_checkpoint_tasks.clear()

    # 4. Cancel background cleanup tasks and await them
    session_cleanup_task.cancel()
    await asyncio.gather(
        session_cleanup_task,
        return_exceptions=True
    )

    # 4b. Cancel in-flight fork build tasks so they mark sessions FAILED and
    #     release GPUs (otherwise they strand as BUILDING after restart).
    try:
        sm = get_session_manager()
        fork_tasks = getattr(sm, "_fork_build_tasks", None)
        if isinstance(fork_tasks, dict) and fork_tasks:
            tasks = [t for t in fork_tasks.values() if not t.done()]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"Cancelled {len(tasks)} in-flight fork build task(s)")
    except Exception as e:
        logger.warning(f"Fork build task cancellation on shutdown failed: {e}")

    # 5. Stop inactivity monitor
    if inactivity_monitor:
        await inactivity_monitor.stop()
        logger.info("✅ Inactivity Monitor stopped")

    logger.info("✅ AMMO Session Service shutdown complete")


app = FastAPI(
    title="AMMO Session Service",
    description="Creates and manages AI-CLI GPU optimization sessions in isolated git worktrees",
    version="2.0.0",
    lifespan=lifespan
)

# Register middleware (outermost = first registered, so SlowRequest wraps APIKey)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(SlowRequestMiddleware)


# Mount frontend static files
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


async def cleanup_sessions_periodically():
    """Background task to clean up old sessions (7-day TTL for terminated/failed)"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            if session_manager:
                await session_manager.cleanup_old_sessions(max_age_days=7)

            # Also clean up stale S3 sessions (based on SESSION_S3_TTL_DAYS, default 30)
            if session_s3_storage and session_s3_storage.enabled:
                try:
                    deleted = await session_s3_storage.cleanup_stale_sessions()
                    if deleted:
                        logger.info(f"Cleaned up {deleted} stale sessions from S3")
                except Exception as e:
                    logger.warning(f"Failed to cleanup stale S3 sessions: {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in session cleanup task: {e}")


def _read_vllm_artifact(filename: str, max_len: Optional[int] = None) -> Optional[str]:
    """Read a small artifact file written by the Dockerfile into /workspace/vllm/.

    Returns the stripped contents (truncated to max_len chars when given),
    or None when the file is absent. Used to expose .docker_commit
    (40-char SHA) and .docker_version (release tag string) via /health.
    """
    artifact_path = Path(f"/workspace/vllm/{filename}")
    if not artifact_path.is_file():
        return None
    value = artifact_path.read_text().strip()
    if max_len is not None:
        value = value[:max_len]
    return value


@app.get("/health")
async def health_check():
    """Health check endpoint — maintains compatibility with original API.

    Composes the response from a locally computed `status` plus `gpu` and
    `vllm` blocks.
    """
    from shared.constants import GPU_DTYPE_MAP

    try:
        detected_gpu = gpu_type or "unknown"
        allowed_dtypes = GPU_DTYPE_MAP.get(detected_gpu, GPU_DTYPE_MAP["unknown"])

        total_gpus: int = 0
        available_gpus: int = 0
        if gpu_manager is not None:
            try:
                total_gpus = gpu_manager.get_gpu_count()
                available_gpus = gpu_manager.get_available_gpu_count()
            except Exception as e:  # pragma: no cover — never break /health
                logger.warning(f"Failed to read GPU counts for /health: {e}")

        health_data = {
            "status": "healthy",
            "gpu_manager": {
                "total_gpus": total_gpus,
                "available_gpus": available_gpus,
            },
            "gpu": {
                "type": detected_gpu,
                "allowed_dtypes": allowed_dtypes,
                "total_gpus": total_gpus,
                "available_gpus": available_gpus,
            },
        }

        def _read_vllm_block():
            return {
                "docker_commit": _read_vllm_artifact(".docker_commit", max_len=40),
                "version": _read_vllm_artifact(".docker_version"),
            }
        health_data["vllm"] = await asyncio.to_thread(_read_vllm_block)
        health_data["base_repos_ready"] = _base_repos_ready
        return JSONResponse(content=health_data, status_code=200)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=500,
        )


# ============================================================================
# UI Endpoints
# ============================================================================

@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    """
    Serve the session management UI.

    Returns the single-page application for managing MoE optimizer sessions.
    """
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend not found. Create frontend/index.html")


# ---- HuggingFace model search cache ----
_hf_cache: Dict[str, dict] = {}
_hf_cache_ts: Dict[str, float] = {}
_HF_CACHE_TTL = 60  # seconds
_HF_CACHE_MAX_SIZE = 200  # max entries before evicting oldest


@app.get("/api/hf-models")
async def search_hf_models(q: str = "", limit: int = 20):
    """
    Search HuggingFace for vLLM-compatible models.

    Proxies the HuggingFace API with a 60s TTL cache per query.
    Returns a list of model metadata (id, downloads, likes, pipeline_tag).
    """
    q = q.strip()
    if not q:
        return {"models": [], "source": "huggingface"}

    cache_key = f"{q}:{limit}"
    now = time.time()

    # Check cache
    if cache_key in _hf_cache and (now - _hf_cache_ts.get(cache_key, 0)) < _HF_CACHE_TTL:
        return _hf_cache[cache_key]

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://huggingface.co/api/models",
                params={"search": q, "limit": limit, "sort": "downloads", "direction": "-1"},
            )
            resp.raise_for_status()
            raw_models = resp.json()

        models = [
            {
                "id": m.get("id", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "tags": m.get("tags", []),
            }
            for m in raw_models
            if m.get("id")
        ]
        result = {"models": models, "source": "huggingface"}
    except Exception as e:
        logger.warning(f"HuggingFace API search failed: {e}")
        return {"models": [], "source": "error"}

    # Evict oldest entry if cache is full (before adding new entry)
    if len(_hf_cache) >= _HF_CACHE_MAX_SIZE:
        oldest_key = min(_hf_cache_ts, key=_hf_cache_ts.get)
        _hf_cache.pop(oldest_key, None)
        _hf_cache_ts.pop(oldest_key, None)

    _hf_cache[cache_key] = result
    _hf_cache_ts[cache_key] = now
    return result


@app.get("/api/hf-model-config/{model_id:path}")
async def get_hf_model_config(model_id: str):
    """Return HuggingFace model metadata + TP/DP/dtype auto-suggestions.

    Protected under the /api/ prefix (API-key gated when AMMO_API_KEY is set).
    2-call flow: HF metadata (expanded) + config.json. Per-server LRU cache
    with 24h TTL. Returns graceful reason codes for gated / network /
    config_missing_fields errors (no 5xx for upstream HF failures).
    """
    from shared.constants import GPU_DTYPE_MAP
    from shared.hf_model_config import get_hf_model_config_service

    detected_gpu = gpu_type or "unknown"
    allowed_dtypes = GPU_DTYPE_MAP.get(detected_gpu, GPU_DTYPE_MAP["unknown"])

    total_gpus = 8
    try:
        from shared.gpu_file_lock import GPUFileLockManager
        total_gpus = len(GPUFileLockManager().get_gpu_ids())
    except Exception:
        pass

    service = get_hf_model_config_service(
        gpu_memory_gb=gpu_memory_gb,
        total_gpus=total_gpus,
    )
    try:
        return await service.get_config(model_id, allowed_dtypes=allowed_dtypes)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning(f"hf-model-config unexpected failure for {model_id}: {e}")
        return {
            "model_id": model_id,
            "is_moe": False,
            "suggested_tp": None,
            "suggested_dp": None,
            "suggested_dtype": None,
            "reason": "network_error",
            "config": None,
        }


@app.get("/api/changelog")
async def get_changelog():
    """Return current AMMO version and changelog entries parsed from VERSION file."""
    # Primary: in-container template path populated by the Dockerfile cp.
    # Fallback: in-repo source-of-truth, used for native/dev runs and any
    # case where the Docker template dir is missing or stale (host edits to
    # ai_cli_session/.claude/VERSION are not bind-mounted at runtime).
    version_file = Path(
        os.getenv("SESSION_TEMPLATES_DIR", "/data/templates")
    ) / "claude/.claude/VERSION"
    if not version_file.exists():
        version_file = Path(__file__).parent / "ai_cli_session/.claude/VERSION"

    result = {"version": None, "entries": []}

    try:
        text = version_file.read_text()
    except Exception as exc:
        logger.warning("Failed to read VERSION file at %s: %s", version_file, exc)
        return result

    lines = text.splitlines()
    if lines:
        m = re.match(r"^version:\s+(\S+)", lines[0])
        if m:
            result["version"] = m.group(1)

    current_entry = None
    for line in lines:
        header_match = re.match(r"^###\s+(\S+)\s+\(([^)]+)\)", line)
        if header_match:
            current_entry = {
                "version": header_match.group(1),
                "date": header_match.group(2),
                "changes": [],
            }
            result["entries"].append(current_entry)
        elif current_entry is not None and line.startswith("- "):
            current_entry["changes"].append(line[2:])

    return result


campaign_data_service = CampaignDataService()


@app.get("/api/campaigns")
async def get_campaigns_overview(request: Request, client_id: Optional[str] = Depends(get_client_id)):
    """List all sessions that have campaign state.json data (L1 overview)."""
    result = await session_manager.list_sessions(owner_id=client_id)
    _sem = asyncio.Semaphore(16)

    async def _process(s):
        if not s.worktree_path:
            return None
        async with _sem:
            artifact_dir = await campaign_data_service.find_artifact_dir(s.worktree_path)
            if not artifact_dir:
                return None
            state = await campaign_data_service.read_state(artifact_dir)
            if not state:
                return None
            created_str = str(s.created_at) if hasattr(s, 'created_at') and s.created_at else None
            return campaign_data_service.build_l1_projection(s.session_id, state, created_at=created_str)

    results = await asyncio.gather(*[_process(s) for s in result.sessions], return_exceptions=True)
    campaigns = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"Failed to build campaign overview: {r}")
        elif r is not None:
            campaigns.append(r)
    return {"campaigns": campaigns}


@app.get("/api/campaigns/{session_id}")
async def get_campaign_detail(
    session_id: str,
    client_id: Optional[str] = Depends(get_client_id),
):
    """Return full campaign state.json for a session (L2 circuit board data)."""
    if session_id == "all":
        raise HTTPException(404, "Campaign not found")
    session = await session_manager.get_session(session_id, owner_id=client_id)
    if not session or not session.worktree_path:
        raise HTTPException(404, "Session not found")
    artifact_dir = await campaign_data_service.find_artifact_dir(session.worktree_path)
    if not artifact_dir:
        raise HTTPException(404, "No campaign data")
    state = await campaign_data_service.read_state(artifact_dir)
    if not state:
        raise HTTPException(404, "Campaign state not found")
    # read_state caches the raw dict; normalize a per-request copy so the
    # cache stays raw and L1's projection-side v3-field fallback isn't
    # short-circuited by an early _normalize_e2e_latency mutation.
    import copy as _copy
    state = _copy.deepcopy(state)
    _normalize_e2e_latency(state)
    return state


@app.get("/api/campaign-data/{session_id}")
async def get_campaign_data(
    session_id: str,
    client_id: Optional[str] = Depends(get_client_id),
):
    """Return normalized state.json for a session.

    Sidecar aggregation has been removed (post-sidecar architecture): state.json
    is now the sole source of structured campaign metrics, and the L3 artifact
    viewer reads the file tree via `GET /api/campaigns/{id}/tree`. Multi-round
    campaigns: state comes from the active dir (most-recently-written state.json).
    """
    session = await session_manager.get_session(session_id, owner_id=client_id)
    if not session or not session.worktree_path:
        raise HTTPException(404, "Session not found")
    all_dirs = await campaign_data_service.find_all_artifact_dirs(session.worktree_path)
    if not all_dirs:
        raise HTTPException(404, "No campaign data")
    state = await campaign_data_service.read_state(all_dirs[0])
    if not state:
        raise HTTPException(404, "Campaign state not found")
    # Same pattern as the L2 endpoint — normalize a per-request copy so the
    # read_state cache stays raw. circuit-board.js relies on the post-
    # normalization shape (cumulative_e2e_speedup, _s-stripped latency map).
    import copy as _copy
    state = _copy.deepcopy(state)
    _normalize_e2e_latency(state)
    return {"state": state}


@app.get("/api/campaigns/{session_id}/tree")
async def get_campaign_tree(
    session_id: str,
    client_id: Optional[str] = Depends(get_client_id),
):
    """Return the artifact-directory file tree for a session.

    Replaces sidecar/catalog aggregation: the FE walks this list to drive the
    L3 artifact viewer and to filter artifacts by path conventions
    (`rounds/{N}/{stage}/...`). Excludes `_archive/`, `__pycache__/`, `.git/`,
    `cache/`, `triton_cache/`, `torch_compile_cache/`, and
    `*.metrics.json`. For multi-round campaigns, the tree of the active
    (most-recent) round dir is returned.

    Response shape: ``{"root": "<artifact_dir_name>", "files": [<rel paths>]}``.
    """
    session = await session_manager.get_session(session_id, owner_id=client_id)
    if not session or not session.worktree_path:
        raise HTTPException(404, "Session not found")
    artifact_dir = await campaign_data_service.find_artifact_dir(session.worktree_path)
    if not artifact_dir:
        raise HTTPException(404, "No campaign data")
    return await campaign_data_service.list_artifact_tree(artifact_dir)


@app.get("/api/campaigns/{session_id}/artifact-children")
async def get_campaign_artifact_children(
    session_id: str,
    path: str = "",
    client_id: Optional[str] = Depends(get_client_id),
):
    """Return one directory level from the session's active artifact root."""
    session = await session_manager.get_session(session_id, owner_id=client_id)
    if not session or not session.worktree_path:
        raise HTTPException(404, "Session not found")

    artifact_dir = await campaign_data_service.find_artifact_dir(session.worktree_path)
    if not artifact_dir:
        raise HTTPException(404, "No campaign data")

    try:
        return await campaign_data_service.list_artifact_children(artifact_dir, path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.api_route(
    "/api/campaigns/{session_id}/artifacts/{path:path}",
    methods=["GET", "HEAD"],
)
async def get_campaign_artifact_route(
    request: Request,
    session_id: str,
    path: str,
    client_id: Optional[str] = Depends(get_client_id),
):
    return await get_campaign_artifact(
        session_id, path, client_id=client_id, request=request
    )


async def get_campaign_artifact(
    *args,
    client_id: Optional[str] = None,
    request: Optional[Request] = None,
):
    """Return a campaign artifact file (GET) or its metadata (HEAD).

    HEAD is required by the L3 artifact viewer: the frontend probes
    Content-Length before deciding whether to inline-render an image
    (<= 5 MB) or fall back to a binary-download card.

    Ownership gate, realpath traversal check, and API-key middleware
    inheritance are identical across both methods.
    """
    if len(args) == 3 and hasattr(args[0], "method"):
        request, session_id, path = args
    elif len(args) == 2:
        session_id, path = args
    else:
        raise TypeError("get_campaign_artifact expects (session_id, path) or (request, session_id, path)")

    session = await session_manager.get_session(session_id, owner_id=client_id)
    if not session or not session.worktree_path:
        raise HTTPException(404, "Session not found")
    all_dirs = await campaign_data_service.find_all_artifact_dirs(session.worktree_path)
    if not all_dirs:
        raise HTTPException(404, "No artifact directory")

    if request is not None and request.method == "HEAD":
        size, mime = await campaign_data_service.stat_artifact_from_any(all_dirs, path)
        if size is None:
            raise HTTPException(404, "Artifact not found")
        return Response(
            status_code=200,
            media_type=mime or "application/octet-stream",
            headers={"Content-Length": str(size)},
        )

    content, mime = await campaign_data_service.read_artifact_from_any(all_dirs, path)
    if content is None:
        raise HTTPException(404, "Artifact not found")
    if isinstance(content, bytes):
        return Response(content=content, media_type=mime)
    return Response(content=content, media_type=mime)


# ============================================================================
# AI CLI Session Endpoints
# ============================================================================

@app.post("/sessions")
async def create_session(request: dict, client_id: Optional[str] = Depends(get_client_id)):
    """
    Create a new AI CLI session.

    Creates an isolated git worktree from a base repository and prepares
    it for AI CLI tool interaction (Claude Code or Codex CLI).

    Request body:
    - repo_name: Repository identifier (default: "vllm")
    - cli_tool: "claude" or "codex" (default: "claude")
    - branch: Git branch to base worktree on (default: "main")
    - initial_prompt: Optional initial prompt for AI CLI
    - gpu_count: Number of GPUs to allocate (default: 0)
    - session_id: Optional - resume existing session by ID
    - inactivity_timeout_mins: Minutes before auto-pause (default: 30)

    Headers:
    - X-Client-ID: Optional client identifier for session isolation

    Returns:
    - session_id: Unique session identifier
    - status: Session status
    - terminal_url: URL to access web terminal (Phase 3)
    - gpu_ids: Allocated GPU IDs
    """
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session service not initialized"
        )

    try:
        # Parse and validate request
        session_request = CreateSessionRequest(**request)

        # Custom fork: validate URL allowlist + token-key gate before any
        # session/GPU allocation work.
        if session_request.vllm_fork_url:
            try:
                session_request.vllm_fork_url = validate_fork_url(
                    session_request.vllm_fork_url
                )
            except ForkUrlError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if session_request.vllm_fork_token and not fork_token_key_configured():
                raise HTTPException(
                    status_code=400,
                    detail="Private fork tokens require AMMO_FORK_TOKEN_KEY to be "
                           "configured on the server.",
                )

        # Pre-creation GPU availability check.
        if session_request.gpu_count > 0:
            available = gpu_manager.get_available_gpu_count()
            if available < session_request.gpu_count:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "insufficient_gpus",
                        "available": available,
                        "requested": session_request.gpu_count,
                        "message": f"This server has {available} GPUs available, "
                                   f"but {session_request.gpu_count} requested. "
                                   f"Pause or terminate a local session, then try again."
                    }
                )

        # Create session with owner_id for isolation
        response = await session_manager.create_session(session_request, owner_id=client_id)

        return response.model_dump()

    except SessionLimitMgrError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except SessionMgrError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError:
        raise  # Let validation error handler deal with it
    except HTTPException:
        raise  # Fork URL/token-gate 400s (and any other explicit HTTP errors) pass through
    except Exception as e:
        logger.exception(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions(status: Optional[str] = None, client_id: Optional[str] = Depends(get_client_id)):
    """
    List all sessions.

    Query parameters:
    - status: Filter by status (creating, active, paused, terminated, failed)

    Headers:
    - X-Client-ID: Optional client identifier for session isolation

    Returns:
    - sessions: List of session info objects (filtered by owner)
    - total: Total number of sessions
    """
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session service not initialized"
        )

    try:
        status_filter = None
        if status:
            try:
                status_filter = SessionStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: {[s.value for s in SessionStatus]}"
                )

        response = await session_manager.list_sessions(status_filter=status_filter, owner_id=client_id)
        return response.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, client_id: Optional[str] = Depends(get_client_id)):
    """
    Get information about a specific session.

    Headers:
    - X-Client-ID: Optional client identifier for session isolation

    Returns session details including:
    - session_id: Session identifier
    - status: Current status
    - cli_tool: AI CLI tool type
    - repo_name: Repository name
    - gpu_ids: Allocated GPUs
    - terminal_url: Terminal URL (if active)
    - worktree_path: Path to session worktree
    """
    if session_id == "all":
        raise HTTPException(status_code=404, detail="Session not found")

    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session service not initialized"
        )

    try:
        session_info = await session_manager.get_session(session_id, owner_id=client_id)

        if not session_info:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return session_info.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# NOTE: This route MUST be defined before /sessions/{session_id} to avoid
# "terminated" being matched as a session_id
@app.delete("/sessions/terminated")
async def delete_terminated_sessions(client_id: Optional[str] = Depends(get_client_id)):
    """
    Delete all terminated sessions.

    Cleans up session data for all sessions with status 'terminated'.
    Only deletes sessions owned by the requesting client.

    Headers:
    - X-Client-ID: Optional client identifier for session isolation

    Returns the count of deleted sessions.
    """
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session service not initialized"
        )

    try:
        # Get all terminated sessions (filtered by owner)
        all_sessions = await session_manager.list_sessions(status_filter=SessionStatus.TERMINATED, owner_id=client_id)
        deleted_count = 0

        for session in all_sessions.sessions:
            try:
                await session_manager.terminate_session(session.session_id, owner_id=client_id)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete session {session.session_id}: {e}")

        return {
            "deleted_count": deleted_count,
            "message": f"Deleted {deleted_count} terminated sessions"
        }

    except Exception as e:
        logger.exception(f"Failed to delete terminated sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def terminate_session(
    session_id: str,
    client_id: Optional[str] = Depends(get_client_id),
):
    """
    Terminate a session and clean up all resources.

    Headers:
    - X-Client-ID: Optional client identifier for session isolation

    This will:
    - Stop any running CLI tool and terminal processes
    - Release allocated GPUs
    - Remove the git worktree
    - Clean up session directory
    """
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session service not initialized"
        )

    try:
        # Cancel any pending checkpoint task for this session
        pending_task = _pending_checkpoint_tasks.pop(session_id, None)
        if pending_task and not pending_task.done():
            pending_task.cancel()
            logger.debug(f"Session {session_id}: Cancelled pending checkpoint (session terminated)")

        # Clean up checkpoint manager tracking
        if checkpoint_manager:
            checkpoint_manager.cleanup_session(session_id)

        response = await session_manager.terminate_session(session_id, owner_id=client_id)
        return response.model_dump()

    except SessionMgrError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to terminate session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/pause")
async def pause_session(session_id: str, client_id: Optional[str] = Depends(get_client_id)):
    """
    Pause an active session.

    Headers:
    - X-Client-ID: Optional client identifier for session isolation

    This will:
    - Stop CLI tool and terminal processes
    - Release GPUs
    - Preserve worktree and session state

    The session can be resumed later with POST /sessions/{session_id}/resume.
    """
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session service not initialized"
        )

    try:
        # Cancel any pending checkpoint task for this session
        pending_task = _pending_checkpoint_tasks.pop(session_id, None)
        if pending_task and not pending_task.done():
            pending_task.cancel()
            logger.debug(f"Session {session_id}: Cancelled pending checkpoint (session paused manually)")

        # Clean up checkpoint manager tracking
        if checkpoint_manager:
            checkpoint_manager.cleanup_session(session_id)

        response = await session_manager.pause_session(session_id, sync_to_s3=True, owner_id=client_id)
        return response.model_dump()

    except SessionMgrError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to pause session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/resume")
async def resume_session(
    session_id: str,
    request: dict = None,
    client_id: Optional[str] = Depends(get_client_id),
):
    """
    Resume a paused session.

    Headers:
    - X-Client-ID: Optional client identifier for session isolation

    This will:
    - Restart CLI tool process (Phase 2)
    - Restart terminal process (Phase 3)
    - Re-acquire GPUs if needed

    Request body (optional):
    - initial_prompt: Optional prompt to send after resuming
    """
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session service not initialized"
        )

    try:
        initial_prompt = None
        if request and "initial_prompt" in request:
            initial_prompt = request["initial_prompt"]

        response = await session_manager.resume_session(
            session_id,
            initial_prompt=initial_prompt,
            owner_id=client_id
        )
        return response.model_dump()

    except SessionMgrError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to resume session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/report")
async def get_session_report(session_id: str, client_id: Optional[str] = Depends(get_client_id)):
    """
    Get the optimization report for a session.

    Reads the REPORT.md from the session's worktree, replaces image references
    with base64 data URIs, and returns the processed markdown.

    Only available for active and paused sessions (worktree must exist).

    Headers:
    - X-Client-ID: Optional client identifier for ownership validation
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session service not initialized")

    session_info = await session_manager.get_session(session_id, owner_id=client_id)
    if session_info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_info.status not in ("active", "paused"):
        raise HTTPException(status_code=404, detail="Report only available for active or paused sessions")

    if not session_info.worktree_path:
        raise HTTPException(status_code=404, detail="Session worktree not available")

    worktree = Path(session_info.worktree_path)
    if not worktree.is_dir():
        raise HTTPException(status_code=404, detail="Session worktree not found on disk")

    # Find REPORT.md in the worktree
    report_path = next(worktree.glob("**/REPORT.md"), None)
    if report_path is None:
        raise HTTPException(status_code=404, detail="No optimization report found for this session")

    import base64
    import mimetypes

    markdown = report_path.read_text(encoding="utf-8")
    report_dir = report_path.parent

    # Replace image references with base64 data URIs
    def _replace_image(match):
        alt_text = match.group(1)
        img_rel_path = match.group(2)
        img_path = (report_dir / img_rel_path).resolve()

        # Security: ensure resolved path stays within the worktree
        try:
            img_path.relative_to(worktree.resolve())
        except ValueError:
            logger.warning(f"Report image path traversal blocked: {img_rel_path}")
            return match.group(0)  # Leave original reference

        if not img_path.is_file():
            return match.group(0)  # Leave original reference for missing files

        mime_type = mimetypes.guess_type(str(img_path))[0] or "image/png"
        img_data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f"![{alt_text}](data:{mime_type};base64,{img_data})"

    markdown = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _replace_image, markdown)

    return {
        "session_id": session_id,
        "markdown": markdown,
        "report_path": str(report_path.relative_to(worktree)),
    }


@app.post("/sessions/{session_id}/prepare-download")
async def prepare_session_download(session_id: str, client_id: Optional[str] = Depends(get_client_id)):
    """
    Prepare session data for download.

    Headers:
    - X-Client-ID: Optional client identifier for session isolation

    - Syncs current session state to S3 (if needed)
    - Creates ZIP archive
    - Returns download info with presigned URL

    For active sessions, the caller should pause the session first to ensure
    all data is synced to S3. For paused sessions, data should already be in S3.

    Returns:
    - session_id: Session identifier
    - download_url: Presigned S3 URL for direct download
    - download_size_bytes: Size of the archive
    - archive_ready: Whether archive is ready
    - expires_at: Unix timestamp when URL expires
    - error: Error message if preparation failed
    """
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session service not initialized"
        )

    try:
        download_info = await session_manager.prepare_download(session_id, owner_id=client_id)
        return download_info.model_dump()

    except SessionMgrError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to prepare download for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/download")
async def download_session(session_id: str, client_id: Optional[str] = Depends(get_client_id)):
    """
    Download session archive.

    Headers:
    - X-Client-ID: Optional client identifier for ownership validation

    Returns a redirect to the presigned S3 URL for the session archive.
    The archive must first be prepared using POST /sessions/{id}/prepare-download.

    If the archive has not been prepared or has expired, returns 404.
    """
    from fastapi.responses import RedirectResponse

    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session service not initialized"
        )

    if not session_s3_storage or not session_s3_storage.enabled:
        raise HTTPException(
            status_code=400,
            detail="S3 storage not configured - downloads require S3"
        )

    # Validate ownership — only allow download if session is local and owned by caller
    session = await session_manager.get_session(session_id, owner_id=client_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found on this server. Use POST /sessions/{id}/prepare-download first."
        )

    try:
        # Get presigned URL for existing archive
        download_url = await session_s3_storage.get_download_url(session_id)

        if not download_url:
            raise HTTPException(
                status_code=404,
                detail="Download archive not found. Call POST /sessions/{id}/prepare-download first."
            )

        return RedirectResponse(url=download_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to download session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Terminal Proxy Endpoints
# ============================================================================

# Activity recording throttle - avoid excessive calls during high terminal output
_last_activity_recorded: Dict[str, float] = {}
ACTIVITY_THROTTLE_SECONDS = 1.0


def _record_activity_throttled(session_id: str) -> None:
    """Record activity for a session, throttled to once per second."""
    global inactivity_monitor, _last_activity_recorded
    now = time.time()
    last = _last_activity_recorded.get(session_id, 0)
    if now - last >= ACTIVITY_THROTTLE_SECONDS:
        if inactivity_monitor:
            inactivity_monitor.record_activity(session_id)
        _last_activity_recorded[session_id] = now


@app.get("/sessions/{session_id}/tmux-mouse-mode")
async def get_tmux_mouse_mode(
    session_id: str,
    client_id: Optional[str] = Depends(get_client_id),
):
    """Get current tmux mouse mode for a session's terminal."""
    if not session_manager or not terminal_manager:
        raise HTTPException(status_code=503, detail="Session service not initialized")

    session = await session_manager.get_session(session_id, owner_id=client_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    mode = await terminal_manager.get_tmux_mouse_mode(session_id)
    return {"mouse_mode": mode}


@app.post("/sessions/{session_id}/tmux-mouse-mode")
async def toggle_tmux_mouse_mode(
    session_id: str,
    request: Request,
    client_id: Optional[str] = Depends(get_client_id),
):
    """Toggle tmux mouse mode for a session's terminal (used by Copy Mode UI toggle)."""
    if not session_manager or not terminal_manager:
        raise HTTPException(status_code=503, detail="Session service not initialized")

    session = await session_manager.get_session(session_id, owner_id=client_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if session.status != SessionStatus.ACTIVE.value:
        raise HTTPException(status_code=400, detail=f"Session is {session.status.value}")

    try:
        body = await request.json()
        mode = body.get("mode", "toggle")
    except Exception:
        mode = "toggle"

    try:
        result_mode = await terminal_manager.set_tmux_mouse_mode(session_id, mode)
        return {"status": "ok", "mouse_mode": result_mode}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _extract_client_id_from_connection(conn) -> Optional[str]:
    """Extract and validate client_id from a Starlette connection.

    Works with both :class:`Request` and :class:`WebSocket` objects since
    both expose ``.cookies`` and ``.query_params``.

    Checks (in order):
    1. ``ammo_client_id`` cookie
    2. ``client_id`` query parameter

    Returns a valid UUID v4 string or ``None``.
    """
    client_id = (
        conn.cookies.get(CLIENT_ID_COOKIE)
        or conn.query_params.get("client_id")
    )
    if client_id and UUID_PATTERN.match(client_id):
        return client_id
    return None


def _validate_terminal_ownership(session_id: str, conn) -> None:
    """Check session ownership for terminal proxy endpoints.

    Extracts client_id from the connection and delegates to
    ``session_manager._validate_ownership``.  Raises
    :class:`SessionMgrError` if the caller is not the session owner.

    No-op when ``session_manager`` is ``None`` (dev / no-session mode).
    """
    if session_manager is None:
        return
    client_id = _extract_client_id_from_connection(conn)
    session_manager._validate_ownership(session_id, client_id)


@app.websocket("/sessions/{session_id}/terminal/ws")
async def terminal_websocket_proxy(websocket: WebSocket, session_id: str):
    """
    WebSocket proxy to ttyd terminal.

    Forwards WebSocket connections from /sessions/{session_id}/terminal/ws
    to the internal ttyd WebSocket server.
    """
    if terminal_manager is None:
        await websocket.close(code=1011, reason="Terminal service not initialized")
        return

    # --- Ownership validation ---
    try:
        _validate_terminal_ownership(session_id, websocket)
    except SessionMgrError:
        await websocket.close(code=4403, reason="Access denied")
        return

    # Get terminal port for this session, with auto-recovery if dead
    port = terminal_manager.get_terminal_port(session_id)
    if port is None or not terminal_manager.is_terminal_running(session_id):
        if session_manager:
            recovered_port = await session_manager.ensure_terminal_healthy(session_id)
            if recovered_port:
                port = recovered_port
            else:
                await websocket.close(code=1011, reason=f"Terminal recovery failed for session {session_id}")
                return
        else:
            await websocket.close(code=1008, reason=f"No terminal found for session {session_id}")
            return

    # Cancel any pending checkpoint task (client reconnected)
    pending_task = _pending_checkpoint_tasks.pop(session_id, None)
    if pending_task and not pending_task.done():
        pending_task.cancel()
        logger.info(f"Session {session_id}: Cancelled pending checkpoint (client reconnected)")

    # ttyd uses the "tty" subprotocol - pass through what the client requests
    requested_subprotocols = websocket.scope.get("subprotocols", [])
    subprotocol = "tty" if "tty" in requested_subprotocols else (requested_subprotocols[0] if requested_subprotocols else "tty")
    await websocket.accept(subprotocol=subprotocol)

    # Connect to ttyd WebSocket (ttyd uses /ws endpoint with tty subprotocol)
    ttyd_ws_url = f"ws://127.0.0.1:{port}/ws"

    try:
        logger.info(f"Connecting to ttyd WebSocket at {ttyd_ws_url} with '{subprotocol}' subprotocol")
        # Enable ping/pong keep-alive to prevent idle disconnects
        # ping_interval=20s sends pings, ping_timeout=20s waits for pong
        # close_timeout=2s reduces stall when ttyd doesn't complete close handshake
        async with websockets.connect(
            ttyd_ws_url,
            subprotocols=[subprotocol],
            ping_interval=20,
            ping_timeout=20,
            close_timeout=2,
        ) as ttyd_ws:
            logger.info(f"Connected to ttyd WebSocket at {ttyd_ws_url}")

            async def forward_to_ttyd():
                """Forward messages from client to ttyd."""
                try:
                    while True:
                        # Handle both text and binary messages from the client
                        message = await websocket.receive()
                        msg_type = message.get("type", "unknown")
                        if msg_type == "websocket.disconnect":
                            logger.info(f"Session {session_id}: Client sent disconnect")
                            break
                        if "bytes" in message and message["bytes"]:
                            data = message["bytes"]
                            # Log message type (first byte for ttyd protocol)
                            msg_code = chr(data[0]) if data else '?'
                            logger.debug(f"Session {session_id}: Client->ttyd BINARY type={msg_code} len={len(data)}")
                            await ttyd_ws.send(data)
                            _record_activity_throttled(session_id)
                        elif "text" in message and message["text"]:
                            data = message["text"]
                            # Log message type (first char for ttyd protocol)
                            msg_code = data[0] if data else '?'
                            logger.debug(f"Session {session_id}: Client->ttyd TEXT type={msg_code} len={len(data)}")
                            await ttyd_ws.send(data)
                            _record_activity_throttled(session_id)
                except WebSocketDisconnect:
                    logger.info(f"Session {session_id}: Client WebSocketDisconnect")
                except Exception as e:
                    logger.error(f"Session {session_id}: Error forwarding to ttyd: {e}")

            async def forward_to_client():
                """Forward messages from ttyd to client."""
                try:
                    async for message in ttyd_ws:
                        if websocket.client_state == WebSocketState.CONNECTED:
                            if isinstance(message, bytes):
                                msg_code = chr(message[0]) if message else '?'
                                logger.debug(f"Session {session_id}: ttyd->Client BINARY type={msg_code} len={len(message)}")
                                await websocket.send_bytes(message)
                                _record_activity_throttled(session_id)
                            else:
                                msg_code = message[0] if message else '?'
                                logger.debug(f"Session {session_id}: ttyd->Client TEXT type={msg_code} len={len(message)}")
                                await websocket.send_text(message)
                                _record_activity_throttled(session_id)
                        else:
                            logger.info(f"Session {session_id}: Client not connected, stopping forward")
                            break
                except asyncio.CancelledError:
                    logger.info(f"Session {session_id}: forward_to_client cancelled")
                    raise
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Session {session_id}: ttyd connection closed")
                except Exception as e:
                    logger.error(f"Session {session_id}: Error forwarding from ttyd: {e}")

            # Register this task for shutdown cancellation
            task_id = f"ws_proxy_{session_id}"
            current_task = asyncio.current_task()
            _active_websocket_tasks[task_id] = current_task

            try:
                # Run both forwarding tasks; cancel the other when one side disconnects
                client_to_ttyd_task = asyncio.create_task(forward_to_ttyd())
                ttyd_to_client_task = asyncio.create_task(forward_to_client())

                done, pending = await asyncio.wait(
                    [client_to_ttyd_task, ttyd_to_client_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check for exceptions in completed tasks
                for task in done:
                    if task.exception():
                        logger.error(f"Session {session_id}: Proxy task error: {task.exception()}")
                    else:
                        logger.info(f"Session {session_id}: Proxy task completed normally")
            except asyncio.CancelledError:
                logger.info(f"Session {session_id}: WebSocket proxy cancelled during shutdown")
            finally:
                _active_websocket_tasks.pop(task_id, None)

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"ttyd WebSocket closed for session {session_id}: {e}")
    except ConnectionRefusedError:
        logger.warning(f"Cannot connect to ttyd for session {session_id} on port {port}, attempting recovery")
        if session_manager:
            recovered_port = await session_manager.ensure_terminal_healthy(session_id)
            if recovered_port:
                logger.info(f"Session {session_id}: Recovered terminal on port {recovered_port}, client should reconnect")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Terminal restarting, please reconnect")
    except Exception as e:
        logger.error(f"WebSocket proxy error for session {session_id}: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason=str(e))
    finally:
        # Clean up activity throttle state
        _last_activity_recorded.pop(session_id, None)
        # Schedule checkpoint after WebSocket disconnect with grace period
        await _schedule_checkpoint_on_disconnect(session_id)


async def _schedule_checkpoint_on_disconnect(session_id: str, grace_seconds: int = 60) -> None:
    """
    Schedule a checkpoint after WebSocket disconnect with grace period.

    The checkpoint syncs session state to S3 but keeps the session active.
    If the client reconnects before the grace period, the checkpoint is cancelled.

    Args:
        session_id: Session identifier
        grace_seconds: Seconds to wait before triggering checkpoint
    """
    if checkpoint_manager is None:
        logger.debug(f"Session {session_id}: No checkpoint manager, skipping checkpoint scheduling")
        return

    logger.info(f"Session {session_id}: WebSocket disconnected, scheduling checkpoint in {grace_seconds}s")

    async def delayed_checkpoint():
        try:
            await asyncio.sleep(grace_seconds)

            # Check if session is still active (not terminated/paused)
            if session_manager:
                session_info = await session_manager.get_session(session_id)
                # Handle both enum and string status
                status = session_info.status if session_info else None
                status_str = status.value if hasattr(status, 'value') else str(status) if status else None
                if not session_info or status_str not in ["active"]:
                    logger.debug(f"Session {session_id}: No longer active (status={status_str}), skipping checkpoint")
                    return

            # Trigger checkpoint
            logger.info(f"Session {session_id}: Grace period elapsed, triggering checkpoint")
            await checkpoint_manager.checkpoint_session(session_id)

        except asyncio.CancelledError:
            logger.debug(f"Session {session_id}: Checkpoint cancelled (reconnected or shutdown)")
            raise
        except Exception as e:
            logger.error(f"Session {session_id}: Checkpoint scheduling failed: {e}")
        finally:
            _pending_checkpoint_tasks.pop(session_id, None)

    # Create and track background task for delayed checkpoint
    task = asyncio.create_task(delayed_checkpoint())
    _pending_checkpoint_tasks[session_id] = task


@app.get("/sessions/{session_id}/terminal/")
@app.get("/sessions/{session_id}/terminal")
async def terminal_http_proxy(session_id: str, request: Request):
    """
    HTTP proxy to ttyd web interface.

    Serves the ttyd HTML page through the main FastAPI server,
    allowing access to terminals via /sessions/{session_id}/terminal/
    """
    if terminal_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Terminal service not initialized"
        )

    # --- Ownership validation ---
    try:
        _validate_terminal_ownership(session_id, request)
    except SessionMgrError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Get terminal port for this session, with auto-recovery if dead
    port = terminal_manager.get_terminal_port(session_id)
    if port is None or not terminal_manager.is_terminal_running(session_id):
        if session_manager:
            recovered_port = await session_manager.ensure_terminal_healthy(session_id)
            if recovered_port:
                port = recovered_port
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Terminal not available for session {session_id}. "
                           f"Try resuming the session."
                )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No terminal found for session {session_id}"
            )

    # Proxy the request to ttyd
    ttyd_url = f"http://127.0.0.1:{port}/"

    try:
        async with httpx.AsyncClient() as client:
            # Retry with backoff — ttyd may still be starting up
            last_exc = None
            for attempt in range(3):
                try:
                    response = await client.get(ttyd_url, timeout=10.0)
                    break
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    last_exc = e
                    if attempt < 2:
                        await asyncio.sleep(1.0 * (attempt + 1))
            else:
                if isinstance(last_exc, httpx.ConnectError):
                    raise httpx.ConnectError(str(last_exc))
                raise httpx.TimeoutException(str(last_exc))

            # Modify the HTML to update WebSocket URL to go through our proxy
            content = response.text

            # Inject a script to override WebSocket URL
            # This intercepts WebSocket connections and routes them through our proxy
            ws_override_script = f'''
<script>
    // Override WebSocket to route through proxy
    (function() {{
        var OrigWebSocket = window.WebSocket;
        var ProxyWebSocket = function(url, protocols) {{
            // Rewrite ttyd's relative WebSocket URL to our proxy
            if (url === './ws' || url === '/ws' || url.endsWith('/ws')) {{
                var proxyUrl = (location.protocol === "https:" ? "wss://" : "ws://") +
                              location.host + "/sessions/{session_id}/terminal/ws";
                console.log("ttyd WebSocket routing to proxy:", proxyUrl);
                url = proxyUrl;
            }}
            // Call original constructor with possibly modified URL
            if (protocols) {{
                return new OrigWebSocket(url, protocols);
            }} else {{
                return new OrigWebSocket(url);
            }}
        }};
        // Properly inherit from WebSocket
        ProxyWebSocket.prototype = OrigWebSocket.prototype;
        ProxyWebSocket.CONNECTING = OrigWebSocket.CONNECTING;
        ProxyWebSocket.OPEN = OrigWebSocket.OPEN;
        ProxyWebSocket.CLOSING = OrigWebSocket.CLOSING;
        ProxyWebSocket.CLOSED = OrigWebSocket.CLOSED;
        window.WebSocket = ProxyWebSocket;
        console.log("WebSocket proxy override installed for session {session_id}");
    }})();
</script>
'''
            # Insert the script at the beginning of the head
            content = content.replace('<head>', '<head>' + ws_override_script, 1)

            return HTMLResponse(content=content, status_code=response.status_code)

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Terminal not responding for session {session_id}"
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Terminal request timeout for session {session_id}"
        )
    except Exception as e:
        logger.error(f"Terminal proxy error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/terminal/{path:path}")
async def terminal_static_proxy(session_id: str, path: str, request: Request):
    """
    Proxy static assets from ttyd (JS, CSS, etc).
    """
    if terminal_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Terminal service not initialized"
        )

    # --- Ownership validation ---
    try:
        _validate_terminal_ownership(session_id, request)
    except SessionMgrError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Get terminal port for this session
    port = terminal_manager.get_terminal_port(session_id)
    if port is None:
        raise HTTPException(
            status_code=404,
            detail=f"No terminal found for session {session_id}"
        )

    # Proxy the request to ttyd
    ttyd_url = f"http://127.0.0.1:{port}/{path}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(ttyd_url, timeout=5.0)

            # Return with same content type
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type=response.headers.get('content-type', 'application/octet-stream')
            )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Terminal not responding for session {session_id}"
        )
    except Exception as e:
        logger.error(f"Terminal static proxy error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors as HTTP 400 Bad Request"""
    headers = create_no_cache_headers()
    
    # Format validation errors nicely
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error['loc'])
        errors.append({
            "field": field_path,
            "message": error['msg'],
            "type": error['type']
        })
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid request format",
            "validation_errors": errors,
            "status": "client_error"
        },
        headers=headers
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    headers = create_no_cache_headers()
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": "error"},
        headers=headers
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    headers = create_no_cache_headers()
    return JSONResponse(
        status_code=500,
        content=create_error_response(f"Internal server error: {str(exc)}"),
        headers=headers
    )


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
