#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
set -e

# Set UV_CACHE_DIR at runtime only (not build time — /data doesn't exist during build)
export UV_CACHE_DIR=/data/.uv_cache

# Bootstrap uv cache from Docker image layer to /data (same FS for hardlinks)
if [ ! -d /data/.uv_cache ] && [ -d /root/.cache/uv ]; then
    cp -a /root/.cache/uv /data/.uv_cache
fi

exec python3 "$@"
