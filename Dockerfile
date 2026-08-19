# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# syntax=docker/dockerfile:1
# Base image with CUDA 13.0.2 and development tools
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONUNBUFFERED=1

# ============================================================
# RARE LAYERS — base OS, toolchain, GPU tooling
# ============================================================

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    git \
    wget \
    curl \
    jq \
    vim \
    sudo \
    unzip \
    pigz \
    libevent-dev \
    libncurses-dev \
    bison \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Build tmux 3.5a from source (Ubuntu 22.04 ships 3.2a which lacks
# allow-passthrough and reliable OSC 52 clipboard support needed for
# copy/paste in ttyd/xterm.js browser terminals)
RUN cd /tmp \
    && wget -q https://github.com/tmux/tmux/releases/download/3.5a/tmux-3.5a.tar.gz \
    && tar xzf tmux-3.5a.tar.gz \
    && cd tmux-3.5a \
    && ./configure --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && cd / && rm -rf /tmp/tmux-3.5a*

# Install AWS CLI v2 (required for S3 session sync)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws

# Install ttyd for web terminal support (AI CLI sessions)
RUN wget -q https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 -O /usr/local/bin/ttyd \
    && chmod +x /usr/local/bin/ttyd

# Install Node.js (required for Claude Code CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda for per-job environments (vLLM benchmark generation)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Accept conda Terms of Service and configure conda
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --set auto_update_conda false

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Upgrade pip (already installed via get-pip.py)
RUN python3.12 -m pip install --upgrade setuptools wheel cmake>=3.26

# Install uv for fast Python package management
RUN pip3 install uv

# ============================================================
# OCCASIONAL LAYERS — vLLM
# ============================================================

# --- vLLM Development Environment Setup ---
# Provides Python-ready vLLM environment. C++ builds run on-demand in sessions.
# Pin vLLM to a release tag so the precompiled wheel URL is deterministic and
# reproducible (no drift from moving nightly).
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_VERSION=0.24.0
# VLLM_BRANCH must be a static literal — Docker ARG does not support reliable
# cross-ARG interpolation (v${VLLM_VERSION} would leak the literal string).
ARG VLLM_BRANCH=v0.24.0

# Configure ccache for on-demand builds within sessions
ENV CCACHE_DIR=/home/session_user/.ccache
ENV CCACHE_MAXSIZE=20G

# Clone vLLM repository at the pinned release tag.
RUN git clone -b ${VLLM_BRANCH} ${VLLM_REPO} /workspace/vllm

# Allow any user (including session_user uid 1000) to operate on any repo
# without "dubious ownership" errors when repo is owned by a different user
RUN git config --system --add safe.directory '*'

WORKDIR /workspace/vllm

# Create virtual environment for vLLM
RUN uv venv --python 3.12 .venv

# Save the wheel commit SHA for image build and session-time reuse.
# worktree_manager.py depends on this being a 40-char SHA, so keep it as
# `git rev-parse HEAD` (which at the pinned tag resolves to the tag's commit).
RUN git rev-parse HEAD > /workspace/vllm/.docker_commit

# Install vLLM as editable package (Python-only, uses precompiled binaries).
# Use the cloned release commit plus the cu130 variant index so CUDA 13 images
# pull CUDA 13-compatible precompiled artifacts instead of a hardcoded cu129
# release wheel URL. The metadata probe fails early if the release commit has
# no CUDA 13 wheel index.
RUN . .venv/bin/activate && \
    VLLM_WHEEL_COMMIT="$(cat /workspace/vllm/.docker_commit)" && \
    curl -fsSL "https://wheels.vllm.ai/${VLLM_WHEEL_COMMIT}/cu130/vllm/metadata.json" >/tmp/vllm-cu130-metadata.json && \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_WHEEL_COMMIT} \
    VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    uv pip install -e . --torch-backend=cu130

# Verify precompiled .so files are present (can't import at build time — no libcuda.so)
RUN test $(find vllm -name '*.abi3.so' | wc -l) -ge 5 && \
    echo "Found $(find vllm -name '*.abi3.so' | wc -l) .so files"

# Save the release version string for display in the AMMO UI (e.g. v0.20.0).
# This is a NEW companion file to .docker_commit — read by /api/supported-models
# to expose vllm_version. Legacy (nightly) images do not have this file and
# the backend returns null, which the UI falls back on to showing the short SHA.
RUN echo "v${VLLM_VERSION}" > /workspace/vllm/.docker_version

# Generate base CMake presets (patched with GPU-specific settings at session init)
RUN . .venv/bin/activate && \
    python tools/generate_cmake_presets.py --force-overwrite

# Clean up uv and pip caches to reduce image size
RUN rm -rf /root/.cache/pip /tmp/*

# NOTE: We intentionally skip cmake configure and build here to keep image small.
# Users can run these on-demand within sessions if they need C++ changes:
#   cmake --preset release
#   cmake --build --preset release --target install
# First build takes ~15-20 min, subsequent builds use ccache (~1-3 min).

# Set vLLM environment for session service
ENV VLLM_BASE_REPO=/workspace/vllm
ENV VLLM_VENV=/workspace/vllm/.venv
ENV HF_HUB_ENABLE_HF_TRANSFER=1
# Add vLLM venv to PATH so 'vllm' CLI and python packages are accessible
ENV PATH=/workspace/vllm/.venv/bin:${PATH}

# --- End vLLM Development Environment Setup ---

# ============================================================
# RARE-ISH LAYERS — Python requirements
# ============================================================

# Add the app root to the Python path for imports
ENV PYTHONPATH=/app:${PYTHONPATH}

# Create working directory
WORKDIR /app

# Copy both requirements files
COPY requirements.txt /app/requirements.main.txt
COPY requirements.server.txt /app/requirements.eval_server.txt

# Install Python dependencies from both requirements files
# Use uv pip to install into the venv (pip3 falls back to conda, venv has no pip)
# Install the main requirements first
RUN uv pip install --no-cache-dir -r requirements.main.txt

# Then install eval server specific requirements
RUN uv pip install --no-cache-dir -r requirements.eval_server.txt

# ============================================================
# FREQUENT LAYERS — CLI tools (after all heavy layers)
# ============================================================

# Install Agents CLI globally
# Placed here so bumping versions doesn't invalidate vLLM/requirements layers
RUN npm install -g @anthropic-ai/claude-code@2.1.202 @openai/codex@0.144.1

# ============================================================
# SETUP LAYERS — dirs, users, config (rarely change)
# ============================================================

# Create cache directories with proper permissions
RUN mkdir -p /tmp/torch_extensions \
    && mkdir -p /tmp/nvidia \
    && mkdir -p /root/.cache \
    && mkdir -p /root/.aws \
    && chmod -R 777 /tmp/torch_extensions \
    && chmod 1777 /tmp/nvidia \
    && chmod -R 777 /root/.cache

# Configure AWS CLI to use CRT (Common Runtime) for faster S3 transfers
RUN echo "[default]" > /root/.aws/config \
    && echo "s3 =" >> /root/.aws/config \
    && echo "  preferred_transfer_client = crt" >> /root/.aws/config \
    && echo "  target_bandwidth = 10Gb/s" >> /root/.aws/config

# Create session data directories for AI CLI sessions
RUN mkdir -p /data/sessions \
    && mkdir -p /data/repos \
    && mkdir -p /data/templates/claude/.claude \
    && mkdir -p /data/templates/codex/.codex \
    && chmod -R 777 /data

# Deploy Claude Code managed settings (highest precedence, automatically trusted)
# This provides pre-approved permissions without needing --dangerously-skip-permissions
RUN mkdir -p /etc/claude-code

# Security: Create non-root user for session terminals
RUN useradd -m -s /bin/bash -u 1000 session_user

# Set session environment variables
ENV SESSION_DATA_DIR=/data/sessions
ENV SESSION_REPOS_DIR=/data/repos
ENV SESSION_TEMPLATES_DIR=/data/templates

# AMMO_FORK_TOKEN_KEY (a urlsafe-base64 Fernet key) enables PRIVATE vLLM fork
# tokens. No default is baked in — generate one and pass it at deploy time so
# every server shares it (required for S3 restore to decrypt stored fork
# tokens):
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Leaving it unset disables private-fork tokens; public forks still work.

# ============================================================
# FREQUENT LAYERS — server source code + templates
# These layers rebuild on every code change but are now small
# thanks to .dockerignore excluding test_data/ (472MB)
# ============================================================

# Copy the entire server package (with permissions set in one shot)
COPY --chmod=750 . /app

# Codex AMMO hooks are an enforcement boundary, not agent-editable project
# files. Install the complete hook/verifier/schema dependency closure under a
# root-owned managed directory and require managed hooks exclusively. The
# readable worktree copy remains documentation/template material only.
RUN mkdir -p /opt/codex-managed-hooks /etc/codex \
    && cp -a /app/ai_cli_session/.codex/. /opt/codex-managed-hooks/ \
    && chown -R root:root /opt/codex-managed-hooks \
    && find /opt/codex-managed-hooks -type d -exec chmod 0555 {} + \
    && find /opt/codex-managed-hooks -type f -exec chmod 0444 {} + \
    && cp /app/ai_cli_session/.codex/requirements.toml /etc/codex/requirements.toml \
    && chmod 0444 /etc/codex/requirements.toml \
    && /workspace/vllm/.venv/bin/python -c 'import jsonschema' \
    && /workspace/vllm/.venv/bin/python /opt/codex-managed-hooks/skills/ammo/scripts/ammo_state.py --help >/dev/null

# Ensure fork-support helper scripts are executable.
RUN chmod 755 /app/scripts/fork_build_console.sh \
              /app/scripts/git_askpass_helper.sh || true

# Copy AI CLI session templates + managed settings + gpu_lock_wrapper + chown in one layer.
# The trailing `/.` copies the CONTENTS into the pre-created dest dir. Without
# it, cp -r nests one level deeper (.claude/.claude) and every
# SESSION_TEMPLATES_DIR consumer silently falls back to the repo path.
# The two `test -f` lines below fail the build if the layout moves again.
RUN cp -r /app/ai_cli_session/.claude/. /data/templates/claude/.claude/ \
    && cp -r /app/ai_cli_session/.codex/. /data/templates/codex/.codex/ \
    && test -f /data/templates/claude/.claude/VERSION \
    && test -f /data/templates/codex/.codex/config.toml \
    && cp /app/ai_cli_session/managed-settings.json /etc/claude-code/managed-settings.json \
    && chmod 0755 /etc/claude-code \
    && chmod 0644 /etc/claude-code/managed-settings.json \
    && mkdir -p /usr/local/lib/ammo \
    && cp /app/shared/gpu_lock_wrapper.py /usr/local/lib/ammo/gpu_lock_wrapper.py \
    && chmod 755 /usr/local/lib/ammo/gpu_lock_wrapper.py \
    && cp /app/scripts/teammate-cmd-wrapper.sh /usr/local/lib/ammo/teammate-cmd-wrapper.sh \
    && chmod 755 /usr/local/lib/ammo/teammate-cmd-wrapper.sh \
    && chown session_user:session_user /data/sessions /data/repos

# Copy entrypoint script
COPY --chmod=755 scripts/entrypoint.sh /usr/local/bin/entrypoint.sh

# Expose the default port and terminal port range
EXPOSE 8000
EXPOSE 8001-8100

# Use tini as PID 1 so orphaned hook/subprocess children are reaped.
# docker-run.sh passes --init locally, and this keeps direct container runs safe.
# -g forwards signals to the entire process group, which matters for the
# 5-layer graceful shutdown in app.py (terminal → WS → checkpoint → cleanup).
ENTRYPOINT ["/usr/bin/tini", "-g", "--"]

# Default command
CMD ["/usr/local/bin/entrypoint.sh", "main.py", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
