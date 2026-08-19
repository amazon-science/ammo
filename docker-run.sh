#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Run script for the AMMO session server Docker container
# This script starts the Docker container with proper GPU support

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
IMAGE_NAME="ammo-server"
IMAGE_TAG="latest"
CONTAINER_NAME="ammo-server"
HOST_PORT=8000
CONTAINER_PORT=8000
GPU_DEVICE="0"
LOG_LEVEL="info"
DETACHED=false
DEVELOPMENT_MODE=false

# Session service configuration
SESSION_DATA_DIR="${SESSION_DATA_DIR:-$HOME/.ammo-server/sessions}"
SESSION_REPOS_DIR="${SESSION_REPOS_DIR:-$HOME/.ammo-server/repos}"
TERMINAL_PORT_START=8001
TERMINAL_PORT_END=8100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            HOST_PORT="$2"
            shift 2
            ;;
        --gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --detach|-d)
            DETACHED=true
            shift
            ;;
        --dev)
            DEVELOPMENT_MODE=true
            shift
            ;;
        --session-data-dir)
            SESSION_DATA_DIR="$2"
            shift 2
            ;;
        --session-repos-dir)
            SESSION_REPOS_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Run the AMMO session server in Docker"
            echo ""
            echo "Options:"
            echo "  --port PORT        Host port to bind (default: 8000)"
            echo "  --gpu DEVICE       GPU device to use (default: 0, use 'all' for all GPUs)"
            echo "  --tag TAG          Image tag to use (default: latest)"
            echo "  --name NAME        Container name (default: ammo-server)"
            echo "  --log-level LEVEL  Log level: debug|info|warning|error (default: info)"
            echo "  --detach, -d       Run container in background"
            echo "  --dev              Development mode (mount current directory)"
            echo "  --session-data-dir DIR   Session data directory (default: ~/.ammo-server/sessions)"
            echo "  --session-repos-dir DIR  Base repos directory (default: ~/.ammo-server/repos)"
            echo "  --help             Show this help message"
            echo ""
            echo "Endpoints:"
            echo "  Health:    http://localhost:PORT/health"
            echo "  Sessions:  http://localhost:PORT/sessions"
            echo "  Terminal:  http://localhost:8001-8100/ (per-session terminals)"
            echo ""
            echo "Examples:"
            echo "  $0                        # Start the server (default)"
            echo "  $0 --port 8080            # Custom port"
            echo "  $0 --gpu all              # Use all GPUs"
            echo "  $0 -d                     # Run in background"
            echo "  $0 --session-data-dir /mnt/sessions  # Custom session storage"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  AMMO Session Server - Docker Run${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if the image exists
if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
    echo -e "${YELLOW}Warning: Image ${IMAGE_NAME}:${IMAGE_TAG} not found${NC}"
    echo "Building image first..."
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    "$SCRIPT_DIR/docker-build.sh" --tag "$IMAGE_TAG"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build image${NC}"
        exit 1
    fi
fi

# Check if nvidia-docker/nvidia-container-runtime is available
if ! docker info 2>/dev/null | grep -q nvidia; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected${NC}"
    echo "The container may not have GPU access"
    echo "Install nvidia-docker2 or nvidia-container-toolkit for GPU support"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Stop existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping existing container: ${CONTAINER_NAME}${NC}"
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Prepare Docker run command
DOCKER_CMD="docker run"

# Determine interactive mode based on detached flag
if [ "$DETACHED" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -d"
else
    DOCKER_CMD="$DOCKER_CMD -it --rm"
fi

# Add basic options
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
DOCKER_CMD="$DOCKER_CMD --user root"
DOCKER_CMD="$DOCKER_CMD --ipc=host"
DOCKER_CMD="$DOCKER_CMD --init"  # Use tini for proper process management (required for ttyd)
DOCKER_CMD="$DOCKER_CMD --cap-add=SYS_ADMIN --security-opt seccomp=unconfined"
DOCKER_CMD="$DOCKER_CMD --cgroupns=host"  # Prevent systemd daemon-reload from evicting GPU cgroup eBPF rules

# Add port mapping (unified server always needs ports)
DOCKER_CMD="$DOCKER_CMD -p ${HOST_PORT}:${CONTAINER_PORT}"

# Add terminal port range for session service
DOCKER_CMD="$DOCKER_CMD -p ${TERMINAL_PORT_START}-${TERMINAL_PORT_END}:${TERMINAL_PORT_START}-${TERMINAL_PORT_END}"

# Add GPU configuration
if [ "$GPU_DEVICE" = "all" ]; then
    DOCKER_CMD="$DOCKER_CMD --gpus all"
else
    DOCKER_CMD="$DOCKER_CMD --gpus '\"device=$GPU_DEVICE\"'"
fi

# Add environment variables
DOCKER_CMD="$DOCKER_CMD -e NVIDIA_DRIVER_CAPABILITIES=compute,utility"
DOCKER_CMD="$DOCKER_CMD -e PYTHONUNBUFFERED=1"
DOCKER_CMD="$DOCKER_CMD -e LOG_LEVEL=$LOG_LEVEL"

# Forward Anthropic auth so Claude Code sessions can authenticate. Only
# forwarded when set in the environment.
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    export ANTHROPIC_API_KEY
    DOCKER_CMD="$DOCKER_CMD -e ANTHROPIC_API_KEY"
fi

# Mount the host Codex login so new AMMO Codex sessions can seed their
# per-session CODEX_HOME without requiring another interactive login.
if [ -f "$HOME/.codex/auth.json" ]; then
    DOCKER_CMD="$DOCKER_CMD -v $HOME/.codex/auth.json:/run/secrets/codex-auth.json:ro"
    DOCKER_CMD="$DOCKER_CMD -e CODEX_AUTH_JSON_PATH=/run/secrets/codex-auth.json"
fi

# Fallback Codex auth: the server seeds per-session Codex homes from
# OPENAI_API_KEY when no auth.json is available.
if [ -n "${OPENAI_API_KEY:-}" ]; then
    export OPENAI_API_KEY
    DOCKER_CMD="$DOCKER_CMD -e OPENAI_API_KEY"
fi

# Mount AWS credentials so `aws s3` works for session state persistence
# (SESSION_S3_BUCKET) and benchmark artifact upload. Optional — the server runs
# without S3 if no credentials are present.
# Mount to both /root/.aws (for server process) and /etc/aws (for session_user
# who can't traverse /root due to 700 permissions)
if [ -d "$HOME/.aws" ]; then
    DOCKER_CMD="$DOCKER_CMD -v $HOME/.aws:/root/.aws:ro"
    DOCKER_CMD="$DOCKER_CMD -v $HOME/.aws:/etc/aws:ro"
    DOCKER_CMD="$DOCKER_CMD -e AWS_SHARED_CREDENTIALS_FILE=/etc/aws/credentials"
    DOCKER_CMD="$DOCKER_CMD -e AWS_CONFIG_FILE=/etc/aws/config"
fi

# Create and mount session directories
mkdir -p "$SESSION_DATA_DIR"
mkdir -p "$SESSION_REPOS_DIR"
DOCKER_CMD="$DOCKER_CMD -v ${SESSION_DATA_DIR}:/data/sessions"
DOCKER_CMD="$DOCKER_CMD -v ${SESSION_REPOS_DIR}:/data/repos"

# Mount ccache volume for on-demand vLLM builds within sessions
# Named volume persists across container restarts
DOCKER_CMD="$DOCKER_CMD -v ammo-ccache:/home/session_user/.ccache"

# Add session service environment variables
DOCKER_CMD="$DOCKER_CMD -e SESSION_DATA_DIR=/data/sessions"
DOCKER_CMD="$DOCKER_CMD -e SESSION_REPOS_DIR=/data/repos"
DOCKER_CMD="$DOCKER_CMD -e SESSION_TERMINAL_BASE_PORT=${TERMINAL_PORT_START}"
DOCKER_CMD="$DOCKER_CMD -e SESSION_MAX_TERMINAL_PORTS=$((TERMINAL_PORT_END - TERMINAL_PORT_START))"
# S3 bucket for session persistence (pause/resume + cross-host restore). Optional:
# without it, sessions live only on this host's local disk. Set SESSION_S3_BUCKET
# in the environment to enable.
if [ -n "${SESSION_S3_BUCKET:-}" ]; then
    export SESSION_S3_BUCKET
    DOCKER_CMD="$DOCKER_CMD -e SESSION_S3_BUCKET"
fi

# Custom-fork token encryption key (Fernet). No default is baked into the image,
# so PRIVATE vLLM fork tokens require you to supply one (public forks work
# without it). Generate with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Only forward it if one is explicitly set in the environment.
if [ -n "${AMMO_FORK_TOKEN_KEY:-}" ]; then
    export AMMO_FORK_TOKEN_KEY
    DOCKER_CMD="$DOCKER_CMD -e AMMO_FORK_TOKEN_KEY"
fi

# Add the image
DOCKER_CMD="$DOCKER_CMD ${IMAGE_NAME}:${IMAGE_TAG}"

# Override command
DOCKER_CMD="$DOCKER_CMD /usr/local/bin/entrypoint.sh main.py --host 0.0.0.0 --port $CONTAINER_PORT --log-level $LOG_LEVEL"

# Add reload flag in development mode
if [ "$DEVELOPMENT_MODE" = true ]; then
    DOCKER_CMD="$DOCKER_CMD --reload"
fi

# Display configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Container: $CONTAINER_NAME"
echo "  Port: $HOST_PORT -> $CONTAINER_PORT"
echo "  Terminal Ports: ${TERMINAL_PORT_START}-${TERMINAL_PORT_END}"
echo "  GPU: $GPU_DEVICE"
echo "  Log Level: $LOG_LEVEL"
echo "  Detached: $DETACHED"
echo "  Development Mode: $DEVELOPMENT_MODE"
echo "  Session Data: $SESSION_DATA_DIR"
echo "  Session Repos: $SESSION_REPOS_DIR"
echo ""

# Run the container
echo -e "${GREEN}Starting container...${NC}"
echo "Run CMD: ${DOCKER_CMD}"
eval $DOCKER_CMD

# Check if container started successfully
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Container started successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    if [ "$DETACHED" = true ]; then
        echo "Container is running in background"
        echo ""
        echo "Commands:"
        echo "  View logs:    docker logs -f $CONTAINER_NAME"
        echo "  Stop:         docker stop $CONTAINER_NAME"
        echo "  Shell access: docker exec -it $CONTAINER_NAME bash"
        echo ""
    fi

    echo -e "${GREEN}REST API Endpoints:${NC}"
    echo "  Health:   http://localhost:${HOST_PORT}/health"
    echo "  Docs:     http://localhost:${HOST_PORT}/docs"
    echo ""
    echo -e "${GREEN}Session Service Endpoints:${NC}"
    echo "  List:     http://localhost:${HOST_PORT}/sessions"
    echo "  Create:   POST http://localhost:${HOST_PORT}/sessions"
    echo "  Terminal: http://localhost:${TERMINAL_PORT_START}-${TERMINAL_PORT_END}/"
    echo ""

    if [ "$DETACHED" = false ]; then
        echo "Press Ctrl+C to stop the server"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Failed to start container!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
