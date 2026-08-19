#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <worktree-name> [branch-name] [main-repo]" >&2
  exit 2
fi

WORKTREE_NAME="$1"
BRANCH_NAME="${2:-ammo/${WORKTREE_NAME}}"
MAIN_REPO="${3:-$(git rev-parse --show-toplevel)}"
MAIN_REPO="$(cd "$MAIN_REPO" && pwd)"
WORKTREE_ROOT="$MAIN_REPO/.codex/worktrees"
WORKTREE_DIR="$WORKTREE_ROOT/$WORKTREE_NAME"
LOCKFILE="$WORKTREE_ROOT/.create-lock"
MAIN_VENV="$MAIN_REPO/.venv"

if [[ ! -x "$MAIN_VENV/bin/python" ]]; then
  echo "Missing main repo Python environment: $MAIN_VENV/bin/python" >&2
  exit 1
fi

install_codex_template() {
  local template_dir="$MAIN_REPO/.codex"
  [[ -d "$template_dir" ]] || return 0
  mkdir -p "$WORKTREE_DIR/.codex"
  for entry_name in AGENTS.md README.md config.toml hooks.json hooks agents schemas skills; do
    [[ -e "$template_dir/$entry_name" ]] || continue
    rm -rf "$WORKTREE_DIR/.codex/$entry_name"
    cp -a "$template_dir/$entry_name" "$WORKTREE_DIR/.codex/"
  done
  rm -rf "$WORKTREE_DIR/.codex/worktrees"
}

install_worktree_venv() {
  local py_version site_packages
  if [[ ! -x "$WORKTREE_DIR/.venv/bin/python" ]]; then
    rm -rf "$WORKTREE_DIR/.venv"
    "$MAIN_VENV/bin/python" -m venv --without-pip "$WORKTREE_DIR/.venv" >/dev/null 2>&1
  fi
  py_version="$($MAIN_VENV/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  site_packages="$WORKTREE_DIR/.venv/lib/python${py_version}/site-packages"
  mkdir -p "$site_packages"

  local main_sp pth line resolved
  main_sp="$MAIN_REPO/.venv/lib/python${py_version}/site-packages"
  printf '%s\n' "$main_sp" > "$site_packages/main-venv.pth"

  # Materialize AMMO-editable optional GPU/runtime packages into the track venv
  # instead of editing them through the shared session venv. This curated list is
  # intentionally version-sensitive: update it as vLLM adds/removes optional
  # package-backed kernel runtimes.
  #
  # NOTE: precompiled-output packages (flashinfer_cubin ~1.1GB, flashinfer_jit_cache)
  # are intentionally EXCLUDED. They are build artifacts, not an authoring surface,
  # and the materialized `flashinfer` python package still resolves them at runtime
  # via the main-venv.pth fallback (the whole main site-packages is on sys.path).
  # Copying them only burned ~1.1GB of disk + thousands of file-creates per track.
  local materialized_roots_file materialized_src materialized_dst
  local -a ammo_track_local_runtime_packages=(
      flashinfer
      nvidia_cutlass_dsl
      deep_gemm
      deepgemm
      flash_mla
      flashmla
      flash_attn
      mamba_ssm
      causal_conv1d
  )
  local -a ammo_track_local_runtime_extra_globs=(
      flashinfer_python-*.dist-info
      flash_attn_*.so
      flash_attn*cuda*.so
      selective_scan*.so
      causal_conv1d*.so
  )
  materialized_roots_file="$site_packages/.ammo-materialized-runtime-roots"
  : > "$materialized_roots_file"

  # Copy src -> dst atomically (tmp + mv) so an existing dst ALWAYS means a
  # complete copy. A mid-copy abort (ENOSPC / EACCES) leaves only a .ammo-tmp
  # leftover, never a half-populated dst that the [[ ! -e dst ]] guard would then
  # skip forever on a repair re-run. Every failure is non-fatal: this hook runs at
  # top level under `set -euo pipefail`, so an unguarded `cp` failure would abort
  # worktree creation BEFORE the _worktree_finder.py/worktree.pth write, yielding a
  # venv with no `import vllm` redirect (broken track). A failed copy must only warn;
  # the package then falls back to the shared copy via main-venv.pth.
  materialize_runtime_path() {
      local src="$1" rel dst tmp
      [[ -e "$src" ]] || return 0
      rel="${src#$main_sp/}"
      [[ "$rel" == "$src" ]] && return 0
      dst="$site_packages/$rel"
      mkdir -p "$(dirname "$dst")" || { echo "  WARN: mkdir failed for $rel" >&2; return 0; }
      if [[ ! -e "$dst" ]]; then
          tmp="${dst}.ammo-tmp.$$"
          rm -rf "$tmp" 2>/dev/null || true
          if cp -a "$src" "$tmp" 2>/dev/null && mv "$tmp" "$dst" 2>/dev/null; then
              : # complete copy now visible at dst
          else
              echo "  WARN: failed to materialize $rel (using shared copy)" >&2
              rm -rf "$tmp" 2>/dev/null || true
              return 0
          fi
      fi
      if [[ -d "$src" && -d "$dst" ]]; then
          printf '%s\t%s\n' "$src" "$dst" >> "$materialized_roots_file" || true
      fi
      return 0
  }

  shopt -s nullglob
  for package_name in "${ammo_track_local_runtime_packages[@]}"; do
      materialize_runtime_path "$main_sp/$package_name"
      materialize_runtime_path "$main_sp/${package_name}.py"
      for candidate in "$main_sp/${package_name}"*.so \
                       "$main_sp/${package_name}"*.dist-info \
                       "$main_sp/${package_name//_/-}"*.dist-info; do
          materialize_runtime_path "$candidate"
      done
  done
  for pattern in "${ammo_track_local_runtime_extra_globs[@]}"; do
      for candidate in "$main_sp"/$pattern; do
          materialize_runtime_path "$candidate"
      done
  done
  shopt -u nullglob

  # Replay path-style .pth redirects generically (content-driven; e.g.
  # nvidia_cutlass_dsl.pth -> python_packages) so `import cutlass` and any future
  # redirect-style lib works in the worktree. Skip `import `/`import\t` directives and
  # `;` shim lines so the editable-vllm finder stays unloaded and WorktreeFinder keeps
  # precedence.
  for pth in "$main_sp"/*.pth; do
      [[ -e "$pth" ]] || continue
      # newline-safe loop: nvidia_cutlass_dsl.pth has no trailing newline.
      while IFS= read -r line || [[ -n "$line" ]]; do
          [[ -z "${line//[[:space:]]/}" ]] && continue
          case "$line" in import\ *|import$'\t'*|*\;*) continue ;; esac
          case "$line" in
              /*) resolved="$line" ;;
              *)  resolved="$main_sp/$line" ;;
          esac
          [[ "$resolved" == "$main_sp" ]] && continue
          materialized_dst="$resolved"
          while IFS=$'\t' read -r materialized_src materialized_dst_candidate; do
              [[ -z "${materialized_src:-}" ]] && continue
              if [[ "$resolved" == "$materialized_src" || "$resolved" == "$materialized_src/"* ]]; then
                  materialized_dst="$materialized_dst_candidate${resolved#$materialized_src}"
                  break
              fi
          done < "$materialized_roots_file"
          if [[ -d "$materialized_dst" ]]; then
              printf '%s\n' "$materialized_dst" >> "$site_packages/main-venv.pth"
          elif [[ -d "$resolved" ]]; then
              printf '%s\n' "$resolved" >> "$site_packages/main-venv.pth"
          fi
      done < "$pth"
  done

  # Redirect `import vllm` to this worktree before the main venv editable
  # install can resolve it to the outer checkout.
  cat > "$site_packages/_worktree_finder.py" <<'FINDER'
import sys
from importlib.util import spec_from_file_location
from pathlib import Path

WORKTREE_ROOT = "__WORKTREE_DIR__"
WORKTREE_PACKAGES = {"vllm"}


class WorktreeFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        top_level = fullname.split(".")[0]
        if top_level not in WORKTREE_PACKAGES:
            return None
        if fullname in WORKTREE_PACKAGES:
            pkg_path = Path(WORKTREE_ROOT) / fullname
            init = pkg_path / "__init__.py"
            if init.exists():
                return spec_from_file_location(
                    fullname,
                    str(init),
                    submodule_search_locations=[str(pkg_path)],
                )
        return None


def install():
    if not any(getattr(f, "__name__", "") == "WorktreeFinder" for f in sys.meta_path):
        for i, finder in enumerate(sys.meta_path):
            if getattr(finder, "__name__", "") == "PathFinder":
                sys.meta_path.insert(i, WorktreeFinder)
                return
        sys.meta_path.append(WorktreeFinder)
FINDER
  sed -i "s|__WORKTREE_DIR__|$WORKTREE_DIR|g" "$site_packages/_worktree_finder.py"
  printf '%s\n' 'import _worktree_finder; _worktree_finder.install()' > "$site_packages/worktree.pth"
  printf '%s\n' "$WORKTREE_DIR" > "$site_packages/worktree-path.pth"

  for cmd in pytest vllm; do
    cat > "$WORKTREE_DIR/.venv/bin/$cmd" <<'WRAPPER'
#!/usr/bin/env bash
exec "$(dirname "$0")/python" -m CMD_PLACEHOLDER "$@"
WRAPPER
    sed -i "s|CMD_PLACEHOLDER|$cmd|g" "$WORKTREE_DIR/.venv/bin/$cmd"
    chmod +x "$WORKTREE_DIR/.venv/bin/$cmd"
  done
}

repair_worktree_setup() {
  install_codex_template
  install_worktree_venv
}

mkdir -p "$WORKTREE_ROOT"

if [[ ! -d "$WORKTREE_DIR" ]]; then
  (
    flock -x 200
    git -C "$MAIN_REPO" worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" HEAD >/dev/null 2>&1 || \
    git -C "$MAIN_REPO" worktree add "$WORKTREE_DIR" "$BRANCH_NAME" >/dev/null 2>&1 || {
      echo "Failed to create worktree $WORKTREE_NAME" >&2
      exit 1
    }
  ) 200>"$LOCKFILE"

  if [[ ! -d "$WORKTREE_DIR" ]]; then
    echo "Worktree creation failed: $WORKTREE_DIR" >&2
    exit 1
  fi
fi

repair_worktree_setup

if [[ -f "$MAIN_REPO/CMakeUserPresets.json" ]]; then
  cp "$MAIN_REPO/CMakeUserPresets.json" "$WORKTREE_DIR/"
  if command -v jq >/dev/null 2>&1; then
    jq --arg pm "$WORKTREE_DIR:$MAIN_REPO" \
      '.configurePresets[0].environment.CCACHE_PATH_MAP = $pm' \
      "$WORKTREE_DIR/CMakeUserPresets.json" > "$WORKTREE_DIR/CMakeUserPresets.json.tmp"
    mv "$WORKTREE_DIR/CMakeUserPresets.json.tmp" "$WORKTREE_DIR/CMakeUserPresets.json"
  fi
fi

while IFS= read -r so_path; do
  rel_path="${so_path#$MAIN_REPO/}"
  mkdir -p "$(dirname "$WORKTREE_DIR/$rel_path")"
  cp "$so_path" "$WORKTREE_DIR/$rel_path"
done < <(find "$MAIN_REPO/vllm" -name '*.so' -type f 2>/dev/null | sort)

if [[ -f "$MAIN_REPO/vllm/_version.py" ]]; then
  mkdir -p "$WORKTREE_DIR/vllm"
  cp "$MAIN_REPO/vllm/_version.py" "$WORKTREE_DIR/vllm/_version.py"
fi

while IFS= read -r ignored_file; do
  [[ -z "$ignored_file" ]] && continue
  mkdir -p "$(dirname "$WORKTREE_DIR/$ignored_file")"
  cp "$MAIN_REPO/$ignored_file" "$WORKTREE_DIR/$ignored_file"
done < <(
  git -C "$MAIN_REPO" ls-files --others --ignored --exclude-standard -- \
    'vllm/vllm_flash_attn' 'vllm/third_party' 'vllm/grpc' 2>/dev/null | \
    grep -v '__pycache__' || true
)

echo "$WORKTREE_DIR"
