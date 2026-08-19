#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import hashlib
import subprocess
import sys
from pathlib import Path

from common import (
    active_artifact_dir,
    additional_context,
    command_from_payload,
    command_invokes_gpu_heavy_tool,
    emit,
    expected_monitor_name,
    find_repo_root,
    format_monitor_records,
    is_static_inspection_command,
    pending_monitor_pairs,
    pending_monitor_records,
    record_monitor_pair,
    clear_monitor_pair,
    complete_monitor_pair,
    read_stdin_json,
    session_id_from_payload,
    spawn_role_and_name,
    start_monitor_pair,
    task_names_equivalent,
    tool_name_from_payload,
)


payload = read_stdin_json()
cmd = command_from_payload(payload)


def _tool_input() -> dict:
    tool_input = payload.get("tool_input") or payload.get("toolInput") or {}
    return tool_input if isinstance(tool_input, dict) else {}


def _spawn_role_and_name() -> tuple[str, str]:
    """Infer current Codex role bootstrap inputs while retaining legacy fields."""
    inferred_role, inferred_name = spawn_role_and_name(payload)
    if inferred_role:
        return inferred_role, inferred_name
    tool_input = _tool_input()
    explicit_role = str(
        tool_input.get("agent_type")
        or tool_input.get("subagent_type")
        or tool_input.get("agentType")
        or payload.get("agent_type")
        or payload.get("subagent_type")
        or ""
    ).strip()
    task_name = str(
        tool_input.get("task_name")
        or tool_input.get("taskName")
        or payload.get("task_name")
        or payload.get("taskName")
        or tool_input.get("name")
        or ""
    ).strip()
    message = "\n".join(
        str(value)
        for value in (
            tool_input.get("message"),
            payload.get("message"),
            payload.get("prompt"),
        )
        if value
    )
    agent_path = str(
        payload.get("agent_path")
        or payload.get("agentPath")
        or tool_input.get("agent_path")
        or tool_input.get("agentPath")
        or ""
    )
    role_markers = {
        "ammo-implementer": "agents/ammo-implementer.md",
        "ammo-auditor": "agents/ammo-auditor.md",
        "ammo-transcript-monitor": "agents/ammo-transcript-monitor.md",
    }
    for role, marker in role_markers.items():
        if marker in message or Path(agent_path).name in {
            f"{role}.toml", f"{role}.md"
        }:
            return role, task_name
    lowered = task_name.lower().replace("-", "_")
    if lowered.startswith(("impl_", "implementer_", "ammo_implementer_")):
        return "ammo-implementer", task_name
    if lowered.startswith(("audit_", "auditor_", "ammo_auditor_")):
        return "ammo-auditor", task_name
    if lowered.startswith(("monitor_", "ammo_transcript_monitor_")):
        return "ammo-transcript-monitor", task_name
    return explicit_role, task_name


def _monitor_spawn_reminder() -> str | None:
    tool_input = _tool_input()
    agent_type, task_name = _spawn_role_and_name()
    if agent_type not in {"ammo-implementer", "ammo-impl-champion"}:
        return None

    direct_agent_name = str(payload.get("agentName") or payload.get("agent_name") or "").strip()
    lifecycle_event = str(payload.get("hook_event_name") or payload.get("hookEventName") or "")
    if lifecycle_event != "SubagentStart" and direct_agent_name and direct_agent_name != "team-lead":
        return None

    transcript_path = payload.get("transcript_path") or payload.get("transcriptPath")
    if transcript_path:
        try:
            path = Path(str(transcript_path))
            if path.exists():
                with path.open(encoding="utf-8") as handle:
                    for _, line in zip(range(5), handle):
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            continue
                        transcript_agent_name = str(
                            record.get("agentName") or record.get("agent_name") or ""
                        ).strip()
                        if transcript_agent_name == "team-lead":
                            break
                        if transcript_agent_name:
                            return None
        except Exception:
            pass

    agent_name = str(task_name or tool_input.get("name") or tool_input.get("target") or "").strip()
    if agent_name:
        monitor_name = expected_monitor_name(agent_name)
        detail = (
            f"You just spawned {agent_name} (type: {agent_type}). You MUST now spawn a "
            f"corresponding ammo-transcript-monitor agent named {monitor_name} to monitor this thread."
        )
    else:
        detail = (
            f"You just spawned an unnamed {agent_type} agent. You MUST now spawn a corresponding "
            "ammo-transcript-monitor to monitor this thread. Consider re-spawning with a name for "
            "proper coordination."
        )
    return f"AMMO MONITOR REMINDER: {detail} Do this before spawning any other agents or doing other work."


def _subagent_start_policy_messages(
    repo: Path, spawned_role: str, spawned_name: str
) -> list[str]:
    """Apply spawn policy on Codex's supported SubagentStart lifecycle.

    Codex does not expose native spawn_agent calls to PreToolUse. SubagentStart
    cannot cancel a launch, so violations are surfaced to both the parent UI and
    the child as fail-closed instructions; the monitor ledger/state engine still
    prevents stage completion until the required pairing exists.
    """
    issues: list[str] = []
    cwd = str(payload.get("cwd") or "").rstrip("/")
    root = str(
        os.environ.get("CODEX_PROJECT_DIR")
        or os.environ.get("AMMO_PROJECT_DIR")
        or repo
    ).rstrip("/")
    if cwd and root and cwd != root:
        issues.append(
            f"subagent launched from cwd {cwd}, outside the AMMO project root {root}; "
            "do not perform campaign work until the parent corrects the session root"
        )

    pending = [
        record
        for record in pending_monitor_pairs(repo, payload)
        if record.get("status") == "pending"
    ]
    expected = {
        str(record.get("expected_monitor_name") or "") for record in pending
    }
    if pending and not (
        spawned_role == "ammo-transcript-monitor"
        and any(task_names_equivalent(spawned_name, value) for value in expected)
    ):
        owed = ", ".join(sorted(value for value in expected if value)) or "<matching monitor>"
        issues.append(
            "subagent launched while an implementer monitor was still owed; "
            f"do not begin campaign work in this child. The parent must spawn {owed} first, "
            "then explicitly resume this thread"
        )
    if not issues:
        return []
    return ["AMMO SUBAGENT START POLICY: " + "; ".join(issues) + "."]


def _payload_texts() -> list[str]:
    """Command/patch-bearing fields only; arbitrary edit content is not a path target."""
    tool_input = _tool_input()
    texts = [cmd, str(payload.get("command") or ""), str(payload.get("haystack") or "")]
    for key in (
        "command",
        "haystack",
        "patch",
    ):
        if key in tool_input:
            texts.append(str(tool_input.get(key) or ""))
    return [text for text in texts if text]


def _resolve_payload_path(raw: str) -> Path:
    path = Path(os.path.expandvars(raw.strip().strip("\"'")))
    if path.is_absolute():
        return Path(os.path.abspath(path))
    cwd = Path(payload.get("cwd") or os.getcwd()).resolve()
    return Path(os.path.abspath(cwd / path))


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    tool_input = _tool_input()
    for key in (
        "file_path",
        "filePath",
        "path",
        "target_file",
        "targetFile",
        "target_path",
        "targetPath",
        "destination",
        "destination_path",
        "destinationPath",
        "dest",
        "new_path",
        "newPath",
        "source",
        "source_path",
        "sourcePath",
        "old_path",
        "oldPath",
    ):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            paths.append(_resolve_payload_path(value))
    command_path_patterns = [
        r"(?:^|\s)(kernel_opt_artifacts/[^\s'\"`]+)",
        r"([A-Za-z0-9_./-]*kernel_opt_artifacts/[A-Za-z0-9_./-]+/state\.json)",
        r"^[+-]{3}\s+(?:a/|b/)?(kernel_opt_artifacts/[^\s]+)",
        r"(?<!\S)((?:\.\.?/)+[^\s'\"`;|&]*state\.json)",
    ]
    for original_text in _payload_texts():
        for text in dict.fromkeys((original_text, os.path.expandvars(original_text))):
            for pattern in command_path_patterns:
                for match in re.findall(pattern, text, flags=re.MULTILINE):
                    raw = match[0] if isinstance(match, tuple) else match
                    paths.append(_resolve_payload_path(raw))
            if re.search(r"(^|[\s;&|])(?:state\.json|state\.json\.tmp)(?:\s|$|[;&|])", text):
                paths.append(_resolve_payload_path("state.json"))
    patch_text = _tool_input().get("patch")
    if isinstance(patch_text, str):
        for match in re.findall(
            r"^\*\*\* (?:Add|Update|Delete) File:\s+(.+)$",
            patch_text,
            flags=re.MULTILINE,
        ):
            paths.append(_resolve_payload_path(match))
    if _command_may_mutate_state(cmd):
        paths.extend(_first_party_state_paths(cmd))
        has_concrete_state = any(path.name == "state.json" for path in paths)
        if not has_concrete_state:
            artifact = active_artifact_dir(find_repo_root(payload.get("cwd")), payload)
            if artifact is not None:
                paths.append((artifact / "state.json").resolve())
    elif cmd and not is_static_inspection_command(cmd) and not any(
        path.name == "state.json" for path in paths
    ):
        # Unknown helpers may mutate the active campaign internally without
        # spelling state.json on their command line. Validate the active state
        # after every non-static command when no concrete state target exists.
        artifact = active_artifact_dir(find_repo_root(payload.get("cwd")), payload)
        if artifact is not None:
            paths.append((artifact / "state.json").resolve())
    seen: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def _first_party_state_paths(command: str) -> list[Path]:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = []
    paths: list[Path] = []
    for idx, token in enumerate(parts):
        script = Path(token.strip("\"'")).name
        tail = parts[idx + 1 :]
        if script == "new_target.py":
            for arg_idx, arg in enumerate(tail):
                if arg == "--artifact-dir" and arg_idx + 1 < len(tail):
                    paths.append(_resolve_payload_path(tail[arg_idx + 1]) / "state.json")
                elif arg.startswith("--artifact-dir="):
                    paths.append(_resolve_payload_path(arg.split("=", 1)[1]) / "state.json")
        elif script == "ammo_state.py" and any(
            verb in tail for verb in {"set", "advance", "enrich", "backfill"}
        ):
            for arg_idx, arg in enumerate(tail):
                if arg == "--state" and arg_idx + 1 < len(tail):
                    paths.append(_resolve_payload_path(tail[arg_idx + 1]))
                elif arg.startswith("--state="):
                    paths.append(_resolve_payload_path(arg.split("=", 1)[1]))
        elif script == "reconcile_track_state.py" and "--write" in tail:
            for arg_idx, arg in enumerate(tail):
                if arg == "--artifact-dir" and arg_idx + 1 < len(tail):
                    paths.append(
                        _resolve_payload_path(tail[arg_idx + 1]) / "state.json"
                    )
                elif arg.startswith("--artifact-dir="):
                    paths.append(
                        _resolve_payload_path(arg.split("=", 1)[1]) / "state.json"
                    )
    return paths


def _invokes_first_party_state_writer(command: str) -> bool:
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    for idx, token in enumerate(parts):
        script = Path(token.strip("\"'")).name
        tail = parts[idx + 1 :]
        if script == "new_target.py" and any(
            arg == "--artifact-dir" or arg.startswith("--artifact-dir=") for arg in tail
        ):
            return True
        if script == "ammo_state.py" and any(
            verb in tail for verb in {"set", "advance", "enrich", "backfill"}
        ):
            return True
        if script == "reconcile_track_state.py" and "--write" in tail and any(
            arg == "--artifact-dir" or arg.startswith("--artifact-dir=")
            for arg in tail
        ):
            return True
    return False


def _command_may_mutate_state(command: str) -> bool:
    if not command:
        return False
    mentions_state = "state.json" in command
    first_party_writer = _invokes_first_party_state_writer(command)
    if not mentions_state and not first_party_writer:
        return False
    if first_party_writer:
        return True
    # A PostToolUse guard cannot safely enumerate every possible writer
    # (perl -pi, a one-off helper, compiled tools, etc.). Once a non-read-only
    # command names state.json, treat it as a possible mutation and validate
    # the concrete target after the call. Read-only rg/cat/jq/python-json
    # inspection remains exempt through the shared shell classifier.
    if mentions_state and not is_static_inspection_command(command):
        return True
    write_markers = (
        "json.dump",
        "os.replace(",
        "os.rename(",
        "NamedTemporaryFile",
        "mkstemp",
        "shutil.move",
        "shutil.copy",
        "os.remove(",
        "os.unlink(",
        ".unlink(",
        "unlink ",
        "rm ",
        "write_text",
        "write_bytes",
        "sed -i",
        "tee ",
        "mv ",
        "cp ",
        "truncate ",
        "sponge ",
        "install ",
    )
    if any(marker in command for marker in write_markers):
        return True
    if re.search(r"\bdd\b[^\n;&|]*\bof=(?:[^\s;&|]*state\.json)", command):
        return True
    if re.search(r"(?:\bopen|\.open)\s*\([^)]*,\s*['\"][^'\"]*[wax+]", command):
        return True
    return bool(re.search(r"(?:>|>>)\s*[^;&|\n]*state\.json", command))


def _block_posttool(reason: str, event_name: str = "PostToolUse") -> None:
    if event_name == "SubagentStart":
        # Codex accepts developer context/systemMessage here but explicitly
        # cannot cancel a subagent start. Make the child fail closed and rely on
        # the lifecycle ledger/state gate for the authoritative block.
        emit(
            {
                "systemMessage": reason,
                "hookSpecificOutput": {
                    "hookEventName": "SubagentStart",
                    "additionalContext": "AMMO BLOCKING START POLICY: " + reason
                    + " Do not perform campaign work; return control to the parent.",
                },
            }
        )
        return
    emit(
        {
            "decision": "block",
            "reason": reason,
            "hookSpecificOutput": {
                "hookEventName": event_name,
                "additionalContext": reason,
            },
        }
    )


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown").strip("_")
    return (token or "unknown")[:80]


def _phase2_root() -> Path:
    return Path(os.environ.get("AMMO_AUDIT_PHASE2_DIR") or "/tmp/ammo_codex_audit_phase2").expanduser()


def _phase2_sentinel_path(repo: Path, session_id: str, agent_id: str, verdict_file: Path) -> Path:
    repo_hash = hashlib.sha256(str(repo.resolve()).encode("utf-8")).hexdigest()[:12]
    session_hash = hashlib.sha256((session_id or "unknown").encode("utf-8")).hexdigest()[:12]
    file_hash = hashlib.sha256(str(verdict_file.resolve()).encode("utf-8")).hexdigest()[:16]
    return _phase2_root() / repo_hash / session_hash / f"{_safe_token(agent_id)}_{file_hash}.json"


def _agent_id_for_phase2() -> str:
    tool_input = _tool_input()
    for key in ("agent_id", "agentId", "agent_name", "agentName", "task_name", "taskName"):
        value = payload.get(key)
        if value:
            return str(value)
    for key in ("agent_id", "agentId", "agent_name", "agentName", "task_name", "taskName", "name"):
        value = tool_input.get(key)
        if value:
            return str(value)
    return "unknown"


def _is_auditor_payload() -> bool:
    tool_input = _tool_input()
    values = []
    inferred_role, _ = _spawn_role_and_name()
    values.append(inferred_role)
    for key in ("agent_type", "agentType", "subagent_type", "subagentType", "agent_name", "agentName", "task_name", "taskName", "name"):
        if payload.get(key):
            values.append(str(payload.get(key)))
        if tool_input.get(key):
            values.append(str(tool_input.get(key)))
    messages = [tool_input.get("message"), payload.get("message"), payload.get("prompt")]
    if any("agents/ammo-auditor.md" in str(message) for message in messages if message):
        return True
    return any(
        value == "ammo-auditor"
        or value.startswith("ammo-auditor-")
        or value.lower().replace("-", "_").startswith(("audit_", "auditor_", "ammo_auditor_"))
        for value in values
    )


def _is_audit_verdict_path(path: Path) -> bool:
    return path.name.startswith("stage_") and path.suffix == ".md" and "rounds" in path.parts and "audits" in path.parts


def _has_phase2_section(path: Path) -> bool:
    try:
        return bool(re.search(r"^## Phase 2\b", path.read_text(encoding="utf-8"), flags=re.MULTILINE))
    except OSError:
        return False


def _audit_phase2_messages(repo: Path, paths: list[Path]) -> list[str]:
    messages: list[str] = []
    if not _is_auditor_payload():
        return messages
    invariants = repo / ".codex" / "skills" / "ammo" / "references" / "audit-invariants.md"
    if not invariants.exists():
        return messages

    for path in paths:
        if not _is_audit_verdict_path(path) or _has_phase2_section(path):
            continue
        session_id = session_id_from_payload(payload, repo) or "unknown"
        agent_id = _agent_id_for_phase2()
        sentinel = _phase2_sentinel_path(repo, session_id, agent_id, path)
        if sentinel.exists():
            continue
        try:
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "verdict_file": str(path),
                        "invariants": str(invariants),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        messages.append(
            "IMPORTANT: Your Write/Edit SUCCEEDED - the auditor verdict file is on disk. "
            "This is NOT an error.\n\n"
            "Phase 1 written successfully. Now complete Phase 2: Checklist Verification.\n\n"
            "Instructions:\n"
            f"1. Read the audit invariants file at: {invariants}\n"
            "2. Apply Pre-Check, the section matching your stage, and Cross-Artifact Checks.\n"
            "3. Gather non-inline evidence with independent ammo-delegate fanout where useful.\n"
            "4. Mark each non-applicable property with its reason.\n"
            "5. Reconcile Phase 1 findings with Phase 2 findings. Deduplicate. Assign blocker categories.\n"
            "6. Write residual risks: what could still be wrong even if every finding is resolved?\n"
            "7. Render final verdict (PASS / BLOCKED / NEEDS_INVESTIGATION) and append to this same file using Edit.\n\n"
            "The worst severity across both phases determines the verdict. Phase 2 cannot downgrade Phase 1 BLOCKING findings."
        )
    return messages


def _is_kernel_artifact_path(path: Path) -> bool:
    return "kernel_opt_artifacts" in path.parts


_ERE_SPECIAL = set(r".^$*+?()[]{}|\\")


def _layout_manifest_path() -> Path:
    """Resolve skills/ammo/scripts/artifact_layout.json next to this hook.

    `hooks/../skills/...` holds in the repo tree AND in the managed bundle,
    because the Dockerfile copies `.codex/.` wholesale into
    /opt/codex-managed-hooks/ and so keeps hooks/ and skills/ siblings.
    """
    return Path(__file__).resolve().parents[1] / "skills" / "ammo" / "scripts" / "artifact_layout.json"


def _layout_template_to_regex(template: str, placeholders: dict[str, str]) -> str:
    tokens = sorted(placeholders, key=len, reverse=True)
    out: list[str] = []
    index = 0
    while index < len(template):
        for token in tokens:
            if template.startswith(token, index):
                out.append(placeholders[token])
                index += len(token)
                break
        else:
            char = template[index]
            out.append("\\" + char if char in _ERE_SPECIAL else char)
            index += 1
    return "".join(out)


def _load_layout_patterns() -> list[re.Pattern[str]] | None:
    """Build the layout allowlist from the shared manifest.

    The manifest owns every path fact; this hook owns detection only, so the
    Claude twin and this one cannot disagree. Returns None when the manifest is
    absent or unusable, which disables layout checking (advisory, fail-open) —
    never an empty allowlist, which would warn on every correct write.
    """
    path = _layout_manifest_path()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        placeholders = manifest["placeholders"]
        templates = [slot["path_template"] for slot in manifest["slots"].values()]
        templates += manifest.get("open_dirs", [])
        if not templates:
            raise ValueError("manifest declares no slots")
    except Exception as exc:  # noqa: BLE001 - advisory layer, report and continue
        print(
            f"LAYOUT WARN HOOK DISABLED: cannot load layout manifest {path}: {exc}. "
            "Layout drift is unchecked until it is restored.",
            file=sys.stderr,
        )
        return None

    patterns: list[str] = []
    ancestors: set[str] = set()
    for template in templates:
        body = _layout_template_to_regex(template.rstrip("/"), placeholders)
        patterns.append("^" + body + ("(/|$)" if template.endswith("/") else "$"))
        segments = template.strip("/").split("/")
        for cut in range(1, len(segments)):
            prefix = "/".join(segments[:cut])
            ancestors.add("^" + _layout_template_to_regex(prefix, placeholders) + "$")
    return [re.compile(pattern) for pattern in patterns + sorted(ancestors)]


_LAYOUT_ALLOWED_PATTERNS = _load_layout_patterns()


def _artifact_relative_path(path: Path) -> str | None:
    parts = path.parts
    if "kernel_opt_artifacts" not in parts:
        return None
    idx = parts.index("kernel_opt_artifacts")
    if len(parts) <= idx + 2:
        return None
    return "/".join(parts[idx + 2 :])


def _layout_warning_messages(paths: list[Path]) -> list[str]:
    if _LAYOUT_ALLOWED_PATTERNS is None:
        return []
    messages: list[str] = []
    for path in paths:
        rel = _artifact_relative_path(path)
        if not rel:
            continue
        rel = rel.rstrip("/")
        if any(pattern.search(rel) for pattern in _LAYOUT_ALLOWED_PATTERNS):
            continue
        messages.append(
            "LAYOUT WARN: "
            + rel
            + " is outside the canonical AMMO V2 layout. Expected: "
            + "rounds/{N}/{profiling|sweeps|mining|debate|tracks|audits|_archive}/... "
            + "See .codex/skills/ammo/references/artifact-layout.md § Prohibited Patterns."
        )
    return messages


def _infer_target_dp_from_target_json(state_path: Path) -> int:
    target_path = state_path.with_name("target.json")
    try:
        target = json.loads(target_path.read_text(encoding="utf-8"))
    except Exception:
        return 1
    extra_args = target.get("bench", {}).get("extra_args", [])
    if not isinstance(extra_args, list):
        return 1
    for idx, token in enumerate(extra_args):
        if token == "--data-parallel-size" and idx + 1 < len(extra_args):
            try:
                return max(1, int(extra_args[idx + 1]))
            except Exception:
                return 1
        if isinstance(token, str) and token.startswith("--data-parallel-size="):
            try:
                return max(1, int(token.split("=", 1)[1]))
            except Exception:
                return 1
    return 1


def _state_dp_backfill_message(path: Path) -> list[str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    target = document.get("target") if isinstance(document, dict) else None
    if not isinstance(target, dict) or "dp" in target:
        return []
    inferred = _infer_target_dp_from_target_json(path)
    return [
        "state.json compatibility issue: target.dp is missing. "
        f"Backfill target.dp={inferred} from target.json bench.extra_args "
        "(or 1 for legacy single-DP artifacts) before deriving TP×DP GPU counts."
    ]


def _state_engine_path() -> Path:
    # Resolve the engine (and the schemas it owns) from the managed-hook
    # installation only. The artifact dir and the worktree are writable by the
    # campaign agent, so they must never supply enforcement code or schemas.
    override = os.environ.get("AMMO_STATE_ENGINE")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "skills" / "ammo" / "scripts" / "ammo_state.py"


def _state_engine_failure(reason: str) -> list[str]:
    if os.environ.get("AMMO_VALIDATE_FAIL_OPEN") == "1":
        return []
    return [
        "state.json validation hard issue: state.json validation could not run "
        f"({reason}) — restore the Codex AMMO state engine/validator or set "
        "AMMO_VALIDATE_FAIL_OPEN=1 to bypass (degraded)."
    ]


def _validate_state_with_engine(path: Path) -> list[str]:
    engine = _state_engine_path()
    if not engine.is_file():
        return _state_engine_failure(f"state engine unavailable at {engine}")
    trusted_server_session = str(os.environ.get("AMMO_SESSION_ID") or "")
    trusted_codex_thread = str(
        payload.get("session_id") or payload.get("sessionId")
        or payload.get("conversation_id") or payload.get("conversationId") or ""
    )
    try:
        state_doc = json.loads(path.read_text(encoding="utf-8"))
        existing_session = state_doc.get("session_id") or state_doc.get("sessionId")
        existing_codex = state_doc.get("codex_thread_id")
        changed = False
        if not existing_session and trusted_server_session:
            state_doc["session_id"] = trusted_server_session
            changed = True
        elif trusted_server_session and str(existing_session) != trusted_server_session:
            return _state_engine_failure("state.session_id differs from trusted AMMO_SESSION_ID")
        if not existing_codex and trusted_codex_thread:
            state_doc["codex_thread_id"] = trusted_codex_thread
            changed = True
        elif trusted_codex_thread and str(existing_codex) != trusted_codex_thread:
            return _state_engine_failure("state.codex_thread_id differs from trusted hook payload session_id")
        if changed:
            temp = path.with_name(f".{path.name}.{os.getpid()}.session-bind.tmp")
            temp.write_text(json.dumps(state_doc, indent=2) + "\n", encoding="utf-8")
            os.replace(temp, path)
    except (OSError, json.JSONDecodeError) as exc:
        return _state_engine_failure(f"trusted session binding failed: {exc}")
    argv = [
        sys.executable,
        str(engine),
        "validate",
        "--state",
        str(path),
        "--emit",
        "hook",
        "--fail-closed",
    ]
    # Feed the pre-write snapshot as --prev so the scope/tier gates enforce only
    # NEW or MODIFIED scoreboard/candidate entries. _next_step_with_engine writes
    # this snapshot AFTER validate runs, so at validate time it still holds the
    # pre-write state. When it is absent, cmd_validate treats prev as None and
    # grandfathers every pre-existing entry (never a false block).
    snapshot = _next_step_snapshot(path)
    if snapshot.is_file():
        argv.extend(["--prev", str(snapshot)])
    try:
        # Stage-7 source verification hashes nsys reports and queries durable
        # SQLite attribution. Large real captures can legitimately exceed the
        # former 20-second structural-only budget.
        child_env = os.environ.copy()
        if trusted_codex_thread:
            child_env["CODEX_SESSION_ID"] = trusted_codex_thread
        result = subprocess.run(
            argv, text=True, capture_output=True, timeout=7000, check=False, env=child_env
        )
    except Exception as exc:
        return _state_engine_failure(f"engine invocation failed: {type(exc).__name__}: {exc}")
    if result.returncode != 0:
        return _state_engine_failure(
            f"engine exited {result.returncode}: {(result.stderr or result.stdout).strip()[:300]}"
        )
    output = result.stdout.strip()
    if not output:
        return []
    try:
        body = json.loads(output)
    except json.JSONDecodeError:
        return _state_engine_failure(f"engine returned malformed hook output: {output[:300]}")
    if not isinstance(body, dict):
        return _state_engine_failure(f"engine returned non-object hook output: {output[:300]}")
    if body.get("decision") == "block":
        return ["state.json validation hard issue: " + str(body.get("reason") or output)]
    return _state_engine_failure(f"engine returned unexpected non-empty hook output: {output[:300]}")


def _is_child_agent_payload() -> bool:
    if os.environ.get("CODEX_SUBAGENT") == "1" or os.environ.get("CLAUDE_SUBAGENT") == "1":
        return True
    if payload.get("agent_type") or payload.get("agentType"):
        return True
    name = str(payload.get("agentName") or payload.get("agent_name") or "").strip()
    return bool(name and name != "team-lead")


def _dispatch_audit_gate_stages() -> tuple[str, ...]:
    """Read the dispatchable gate list from the state engine, never re-list it.

    ammo_state.py owns AUDIT_GATE_STAGES. The literal tuple here is only the
    degraded fallback for a missing/unloadable engine, and it holds the same
    four stages, so a load failure cannot widen what the hook accepts.
    """
    fallback = ("stage_1", "stage_2", "stage_45", "stage_67")
    engine = _state_engine_path()
    if not engine.is_file():
        return fallback
    try:
        spec = importlib.util.spec_from_file_location("_ammo_state_stages", engine)
        if spec is None or spec.loader is None:
            return fallback
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stages = tuple(str(stage) for stage in getattr(module, "AUDIT_GATE_STAGES", ()))
        return stages or fallback
    except (Exception, SystemExit):
        return fallback


def _spawn_prompt_text() -> str:
    """The spawn message a parent hands a child, from every field it may use."""
    tool_input = _tool_input()
    return "\n".join(
        str(value)
        for value in (
            tool_input.get("message"),
            tool_input.get("prompt"),
            payload.get("message"),
            payload.get("prompt"),
        )
        if value
    )


def _parse_audit_dispatch(text: str) -> dict | None:
    """Parse the canonical `task: audit_gate` dispatch block.

    Tolerates leading whitespace and trailing `# ...` comments, per the format
    in orchestration/audit-protocol.md. Returns None unless every field the
    stamp needs is present and well formed.
    """
    fields: dict[str, str] = {}
    for line in str(text or "").splitlines():
        body = line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z_]+)\s*:\s*(.*)$", body)
        if match:
            fields.setdefault(match.group(1).lower(), match.group(2).strip())
    if fields.get("task") != "audit_gate":
        return None
    stage = fields.get("stage") or ""
    if stage not in _dispatch_audit_gate_stages():
        return None
    artifact_dir = fields.get("artifact_dir") or ""
    if not artifact_dir:
        return None
    try:
        round_id = int(fields.get("round") or "")
        cycle = int(fields.get("cycle") or "")
    except ValueError:
        return None
    if round_id < 1 or cycle < 1:
        return None
    return {
        "artifact_dir": str(_resolve_payload_path(artifact_dir)),
        "stage": stage,
        "round": round_id,
        "cycle": cycle,
    }


def _spawn_call_failed() -> bool:
    """True when the spawn tool reported an error, so nothing was launched."""
    response = payload.get("tool_response")
    if response is None:
        response = payload.get("toolResponse")
    if isinstance(response, dict):
        return bool(response.get("error") or response.get("is_error") or response.get("isError"))
    return False


def _audit_gate_is_stamped(dispatch: dict) -> bool:
    """Re-read the gate and confirm the stamp is really on disk.

    Exit 0 from `audit-started` does NOT mean a write happened: the engine exits
    0 on its two fail-open no-ops (round absent from campaign.rounds, round with
    no audit key). Claiming "stamped" on a no-op sends the lead to record
    passed_at, which the schema-4.2 provenance backstop then rejects with no
    trace of the false claim. Reading the gate back also covers any future
    silent no-op.
    """
    try:
        state_path = Path(dispatch["artifact_dir"]) / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        rounds = ((state.get("campaign") or {}).get("rounds")) or []
        if not isinstance(rounds, list) or dispatch["round"] > len(rounds):
            return False
        round_state = rounds[dispatch["round"] - 1]
        gate = ((round_state or {}).get("audit") or {}).get(dispatch["stage"])
        if not isinstance(gate, dict):
            return False
        return bool(gate.get("started_at")) and gate.get("cycle") == dispatch["cycle"]
    except Exception:
        return False


def _stamp_audit_started(repo: Path) -> str | None:
    """Record audit.{stage}.started_at + cycle for a dispatched ammo-auditor.

    Entirely fail-open: any parse, path, or engine failure returns None and adds
    no message. The lead writes passed_at after verdict review; started_at and
    cycle are only ever stamped here.
    """
    try:
        if _spawn_call_failed():
            return None
        dispatch = _parse_audit_dispatch(_spawn_prompt_text())
        if dispatch is None:
            return None
        engine = _state_engine_path()
        if not engine.is_file():
            engine = repo / ".codex" / "skills" / "ammo" / "scripts" / "ammo_state.py"
        if not engine.is_file():
            return None
        result = subprocess.run(
            [
                sys.executable,
                str(engine),
                "audit-started",
                "--artifact-dir", dispatch["artifact_dir"],
                "--stage", dispatch["stage"],
                "--round", str(dispatch["round"]),
                "--cycle", str(dispatch["cycle"]),
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return None
        if not _audit_gate_is_stamped(dispatch):
            return None
        return (
            "AMMO AUDIT STARTED: stamped round %d audit.%s started_at (cycle %d)."
            % (dispatch["round"], dispatch["stage"], dispatch["cycle"])
        )
    except Exception:
        return None


def _next_step_snapshot(path: Path) -> Path:
    root = Path(os.environ.get("AMMO_REMINDER_STATE_DIR") or "/tmp/ammo_codex_reminder").expanduser()
    session = session_id_from_payload(payload, find_repo_root(payload.get("cwd"))) or "default"
    token = hashlib.sha256(f"{session}\0{path.resolve()}".encode("utf-8")).hexdigest()[:24]
    return root / f"{token}.json"


def _next_step_with_engine(path: Path) -> list[str]:
    if _is_child_agent_payload():
        return []
    engine = _state_engine_path()
    if not engine.is_file():
        return []  # advisory path is intentionally fail-open
    snapshot = _next_step_snapshot(path)
    argv = [sys.executable, str(engine), "next-step", "--state", str(path), "--emit", "hook"]
    if snapshot.is_file():
        argv.extend(["--prev", str(snapshot)])
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=20, check=False)
        output = result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        output = ""
    try:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        tmp = snapshot.with_suffix(".tmp")
        tmp.write_bytes(path.read_bytes())
        tmp.replace(snapshot)
    except OSError:
        pass
    if not output:
        return []
    try:
        body = json.loads(output)
        context = body.get("hookSpecificOutput", {}).get("additionalContext")
    except (json.JSONDecodeError, AttributeError):
        return []
    return [str(context)] if context else []


def _state_validation_messages(paths: list[Path]) -> list[str]:
    messages: list[str] = []
    tool_name = tool_name_from_payload(payload)
    if cmd:
        state_mutation = _command_may_mutate_state(cmd) or not is_static_inspection_command(cmd)
    elif tool_name.lower() in {
        "apply_patch",
        "functions.apply_patch",
        "edit",
        "write",
        "multiedit",
        "notebookedit",
    }:
        state_mutation = True
    elif tool_name and re.search(
        r"(?:^|[_:.])(write|edit|create|delete|patch|replace|move|rename|copy|remove)(?:[_:.]|$)",
        tool_name.lower(),
    ):
        state_mutation = True
    elif not tool_name:
        # Codex test/compat payloads for Write/Edit may omit tool_name but carry
        # file_path. The live hook matcher does not invoke this hook for Read.
        tool_input = _tool_input()
        state_mutation = any(
            isinstance(tool_input.get(key), str) and bool(tool_input.get(key))
            for key in (
                "file_path",
                "filePath",
                "path",
                "target_file",
                "targetFile",
                "target_path",
                "targetPath",
                "destination",
                "destination_path",
                "destinationPath",
                "dest",
                "new_path",
                "newPath",
            )
        )
    else:
        state_mutation = False
    if not state_mutation:
        return messages
    for path in paths:
        if path.name == "state.json" and _is_kernel_artifact_path(path):
            if path.is_symlink() or not path.is_file():
                messages.extend(
                    _state_engine_failure(
                        f"mutated state path is not a regular file after the tool call: {path}"
                    )
                )
                continue
            validation = _validate_state_with_engine(path)
            messages.extend(validation)
            if validation:
                continue
            if _command_may_mutate_state(cmd) or not cmd:
                messages.extend(_state_dp_backfill_message(path))
                messages.extend(_next_step_with_engine(path))
                messages.append(
                    "state.json updated. Treat state.json plus the round-scoped artifact layout as the "
                    "authoritative metadata source before relying on dashboard/API artifact views."
                )
    return messages


def _reservation_session_ids(command: str) -> list[str]:
    if not re.search(r"gpu_reservation\.py\s+reserve(?:\s|$)", command):
        return []
    matches = re.findall(r"--session-id(?:=|\s+)([A-Za-z0-9_.:@/+-]+)", command)
    if matches:
        return list(dict.fromkeys(matches))
    return [os.environ.get("AMMO_SESSION_ID", "cli")]


def _release_reserved_gpus(command: str) -> None:
    if "--no-auto-release" in command:
        return
    session_ids = _reservation_session_ids(command)
    if not session_ids:
        return
    repo = find_repo_root(payload.get("cwd"))
    script = repo / ".codex" / "skills" / "ammo" / "scripts" / "gpu_reservation.py"
    if not script.exists():
        return
    log_dir = Path(os.environ.get("AMMO_GPU_RES_DIR", "/tmp/ammo_gpu_res")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "release.log").open("a", encoding="utf-8") as log:
        for session_id in session_ids:
            subprocess.run(
                ["python3", str(script), "release-session", "--session-id", session_id],
                stdout=log,
                stderr=log,
                text=True,
                check=False,
            )


def _lifecycle_gpu_owner() -> str | None:
    """Return the official Codex child identity for lifecycle-bound leases."""
    session_id = payload.get("session_id")
    agent_id = payload.get("agent_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    if not isinstance(agent_id, str) or not agent_id.strip():
        return None
    return f"{session_id.strip()}:{agent_id.strip()}"


def _release_lifecycle_gpus(owner: str) -> None:
    repo = find_repo_root(payload.get("cwd"))
    script = repo / ".codex" / "skills" / "ammo" / "scripts" / "gpu_reservation.py"
    if not script.is_file():
        return
    log_dir = Path(os.environ.get("AMMO_GPU_RES_DIR", "/tmp/ammo_gpu_res")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "release.log").open("a", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(script), "release-session", "--session-id", owner],
            stdout=log,
            stderr=log,
            text=True,
            check=False,
        )


_release_reserved_gpus(cmd)

messages = []
repo = find_repo_root(payload.get("cwd"))
is_static_command = is_static_inspection_command(cmd)
tool_name = tool_name_from_payload(payload)
event_name = str(payload.get("hook_event_name") or payload.get("hookEventName") or "")
if event_name == "SubagentStop" and _is_auditor_payload():
    artifact = active_artifact_dir(repo, payload)
    missing_phase2 = []
    if artifact is not None:
        try:
            state = json.loads((artifact / "state.json").read_text(encoding="utf-8"))
            current_round = int(state.get("campaign", {}).get("current_round"))
            audit_dir = artifact / "rounds" / str(current_round) / "audits"
            missing_phase2 = [
                path for path in audit_dir.glob("stage_*.md")
                if path.is_file()
                and "_partial_" not in path.name
                and not _has_phase2_section(path)
            ]
        except Exception:
            missing_phase2 = []
    if missing_phase2:
        _block_posttool(
            "Auditor SubagentStop blocked: append the required Phase 2 checklist/reconciliation "
            "to: " + ", ".join(str(path) for path in missing_phase2[:4]),
            "SubagentStop",
        )
        raise SystemExit(0)
if event_name == "SubagentStop":
    lifecycle_owner = _lifecycle_gpu_owner()
    if lifecycle_owner:
        _release_lifecycle_gpus(lifecycle_owner)
if event_name == "SubagentStart":
    lifecycle_owner = _lifecycle_gpu_owner()
    if lifecycle_owner:
        messages.append(
            "AMMO GPU OWNER: use --session-id " + lifecycle_owner
            + " --no-auto-release for this subagent's persistent GPU reservations. "
            "The SubagentStop lifecycle hook releases that exact owner."
        )
if event_name == "SubagentStart" or tool_name in {"Agent", "spawn_agent", "functions.spawn_agent", "collaboration.spawn_agent"} \
        or tool_name.endswith("__spawn_agent"):
    spawned_role, spawned_name = _spawn_role_and_name()
    if event_name == "SubagentStart":
        messages.extend(_subagent_start_policy_messages(repo, spawned_role, spawned_name))
    try:
        if spawned_role in {"ammo-implementer", "ammo-impl-champion"}:
            if not record_monitor_pair(repo, payload, spawned_name):
                _block_posttool(
                    "Implementer spawn could not record its mandatory paired-monitor obligation.",
                    event_name or "PostToolUse",
                )
                raise SystemExit(0)
        elif spawned_role == "ammo-transcript-monitor":
            start_monitor_pair(repo, payload, spawned_name)
    except OSError as exc:
        _block_posttool(
            f"Mandatory paired-monitor ledger update failed: {exc}",
            event_name or "PostToolUse",
        )
        raise SystemExit(0)
    reminder = _monitor_spawn_reminder()
    if reminder:
        messages.append(reminder)
    # Stamp audit provenance only on the spawn tool call. SubagentStart carries
    # the child's identity and resolves its name through the Codex thread index,
    # not a dispatch prompt, so it cannot supply artifact_dir/stage/round/cycle.
    # _is_child_agent_payload() keeps the stamp with the orchestrator: on this
    # path payload.agent_type is the caller, tool_input.agent_type the child.
    if (
        event_name != "SubagentStart"
        and spawned_role == "ammo-auditor"
        and not _is_child_agent_payload()
    ):
        stamped = _stamp_audit_started(repo)
        if stamped:
            messages.append(stamped)
if event_name == "SubagentStop":
    stopped_role, stopped_name = _spawn_role_and_name()
    if stopped_role == "ammo-transcript-monitor":
        artifact = active_artifact_dir(repo, payload)
        if artifact is None or not complete_monitor_pair(repo, payload, stopped_name, artifact):
            _block_posttool(
                "Transcript monitor completion blocked: require a nonempty observation log and "
                "matching final INFO queue record with target rollout and poll summary.",
                "SubagentStop",
            )
            raise SystemExit(0)
try:
    touched_paths = _candidate_paths()
except Exception as exc:
    touched_paths = []
    state_hint = "state.json" in cmd or any(
        isinstance(value, str) and Path(value.strip().strip("\"'")).name == "state.json"
        for value in _tool_input().values()
    )
    if state_hint and os.environ.get("AMMO_VALIDATE_FAIL_OPEN") != "1":
        _block_posttool(
            "state.json validation hard issue: state mutation target detection failed "
            f"({type(exc).__name__}: {exc}). Restore the hook or set "
            "AMMO_VALIDATE_FAIL_OPEN=1 to bypass (degraded)."
        )
        raise SystemExit(0)
    if os.environ.get("AMMO_DEBUG_HOOKS") == "1":
        messages.append(f"PostToolUse AMMO path detection skipped due to hook error: {exc}")

has_artifact_write = any(_is_kernel_artifact_path(path) for path in touched_paths)
is_ammo_command = (
    command_invokes_gpu_heavy_tool(cmd)
    or "gpu_reservation.py" in cmd
    or ".codex/skills/ammo" in cmd
    or "kernel_opt_artifacts" in cmd
)

pending = pending_monitor_records(repo, payload, {"WARNING", "CRITICAL", "HARD_GATE"})
if pending and (has_artifact_write or (is_ammo_command and not is_static_command)):
    messages.append(
        "Pending AMMO monitor intervention(s). Triage before continuing:\n"
        + format_monitor_records(pending, limit=3)
    )

messages.extend(_audit_phase2_messages(repo, touched_paths))
messages.extend(_layout_warning_messages(touched_paths))

if command_invokes_gpu_heavy_tool(cmd) and "run_vllm_bench_latency_sweep.py" in cmd:
    messages.append("Review run_purpose, production-parity metadata, profiler contamination, and fast-path proof before using sweep evidence.")

try:
    before_state = len(messages)
    messages.extend(_state_validation_messages(touched_paths))
    state_hard_issues = [msg for msg in messages[before_state:] if "state.json validation hard issue" in msg]
    if state_hard_issues:
        _block_posttool("\n".join(state_hard_issues))
        raise SystemExit(0)
except Exception as exc:
    failure = _state_engine_failure(
        f"PostToolUse validation raised {type(exc).__name__}: {exc}"
    )
    if failure:
        _block_posttool("\n".join(failure))
        raise SystemExit(0)
    if os.environ.get("AMMO_DEBUG_HOOKS") == "1":
        messages.append(f"PostToolUse AMMO validation bypassed due to hook error: {exc}")

if messages:
    context = "\n\n".join(messages)
    if event_name == "SubagentStart":
        emit(
            {
                "systemMessage": context,
                "hookSpecificOutput": {
                    "hookEventName": "SubagentStart",
                    "additionalContext": context,
                },
            }
        )
    else:
        additional_context("PostToolUse", context)
