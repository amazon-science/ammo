#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Build script for the AMMO session server Docker container
# This script builds the Docker image with all necessary dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="ammo-server"
IMAGE_TAG="latest"
DOCKERFILE_PATH="Dockerfile"
CONTEXT_PATH="."

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Build the Docker image for the AMMO session server"
            echo ""
            echo "Options:"
            echo "  --tag TAG        Specify image tag (default: latest)"
            echo "  --no-cache       Build without using cache"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Build with the default tag"
            echo "  $0 --tag v2.0         # Build with custom tag"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  AMMO Session Server - Docker Build${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if nvidia-docker/nvidia-container-runtime is available
if ! docker info 2>/dev/null | grep -q nvidia; then
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime not detected${NC}"
    echo "You may need to install nvidia-docker2 or nvidia-container-toolkit"
    echo "Continuing anyway..."
fi

# Navigate to the repo root (build context)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Building from directory: $(pwd)${NC}"
echo -e "${GREEN}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo ""

# Check if requirements file exists
if [ ! -f "requirements.server.txt" ]; then
    echo -e "${RED}Error: requirements.server.txt not found${NC}"
    echo "Expected location: $(pwd)/requirements.server.txt"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo -e "${RED}Error: Dockerfile not found at $DOCKERFILE_PATH${NC}"
    exit 1
fi

# Build the Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    ${NO_CACHE} \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "$DOCKERFILE_PATH" \
    --progress=plain \
    "$CONTEXT_PATH"

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Build completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Image created: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    echo "To run the container:"
    echo "    ./docker-run.sh"
    echo "    ./docker-run.sh --port 8000 --gpu all"
    echo ""
    echo "Or use docker-compose:"
    echo "  docker-compose up"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Build failed!${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi