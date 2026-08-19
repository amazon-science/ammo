#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Git askpass helper for private fork clone/fetch. Git calls this with the
# prompt text as $1; we answer the username with a fixed value and the
# password with the token from the environment. The token NEVER appears in
# argv or on disk — only in the GIT_FORK_TOKEN env var of the git subprocess.
case "$1" in
    Username*) echo "x-access-token" ;;
    *)         echo "${GIT_FORK_TOKEN}" ;;
esac
