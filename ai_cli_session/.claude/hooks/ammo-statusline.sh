#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Claude Code statusLine command

input=$(cat)

# AMMO Lightgrid palette — neon cyberpunk
DIM='\033[2m'
BOLD='\033[1m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
GREEN='\033[1;32m'
MAGENTA='\033[1;35m'
BLUE='\033[1;34m'
WHITE='\033[1;37m'
RESET='\033[0m'

SEP="${DIM} ┃ ${RESET}"

# --- session_id ---
session_id=$(echo "$input" | jq -r '.session_id // empty')

# --- model ---
model=$(echo "$input" | jq -r '.model.display_name // empty')

# --- cwd ---
cwd=$(echo "$input" | jq -r '.workspace.current_dir // .cwd // empty')
[ -z "$cwd" ] && cwd=$(pwd)
cwd=$(echo "$cwd" | sed "s|$HOME|~|")

# --- k-formatter (shared) ---
fmt_k() { awk -v n="$1" 'BEGIN{ if(n>=1000) printf "%.1fk", n/1000; else printf "%d", n }'; }

# --- context used percentage ---
used_pct=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
if [ -n "$used_pct" ]; then
  ctx_used="Used: $(printf '%.0f' "$used_pct")%"
else
  ctx_used=""
fi

# --- transcript path ---
transcript_path=$(echo "$input" | jq -r '.transcript_path // empty')

# --- version ---
version=$(echo "$input" | jq -r '.version // empty')

# --- assemble fields left-to-right ---
parts=()
[ -n "$session_id"      ] && parts+=("${CYAN}◈ ${session_id}${RESET}")
[ -n "$model"           ] && parts+=("${YELLOW}${model}${RESET}")
[ -n "$cwd"             ] && parts+=("${GREEN}${cwd}${RESET}")
[ -n "$ctx_used"        ] && parts+=("${MAGENTA}${ctx_used}${RESET}")
[ -n "$transcript_path" ] && parts+=("${BLUE}${transcript_path}${RESET}")
[ -n "$version"         ] && parts+=("${DIM}${version}${RESET}")

# join with separator
result=""
for part in "${parts[@]}"; do
  if [ -z "$result" ]; then
    result="$part"
  else
    result="${result}${SEP}${part}"
  fi
done

printf "%b" "$result"
