# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Every Codex hook must still emit a verdict when hook_cmd_classify.py is gone.

`.codex/hooks/common.py` imports `skills/ammo/scripts/hook_cmd_classify.py` at
module scope, and all five Codex hooks import `common`. An unguarded import
would turn one missing file into a total enforcement outage: every PreToolUse
deny (pip install, session-identity mutation, VLLM_OP default-on, orchestrator-
owned writes, worktree venv, sweep venv, monitor pairing) and the Stop gate
would stop firing with only a traceback on stderr.

That is reachable in production. The Dockerfile ships `hooks/` and
`skills/ammo/scripts/` as one `cp -a` tree into `/opt/codex-managed-hooks`, so
any future change that filters what lands under `skills/`, or a partial S3
worktree restore on cross-host S3 restore, removes the classifier while leaving the
hooks registered.

The same restore can also leave the classifier PRESENT but incomplete -- a
zero-byte file, or a file truncated mid-way -- and both still import cleanly, so
the failure lands on the attribute reads instead of the import. That must degrade
the same way: the attribute reads live inside the same guard, and one missing
attribute degrades all five re-exports together.

Each test runs the hooks against a COPY of the tree, so the real tree is never
mutated. The declared fail directions mirror the shell hooks that guard the same
file: `ammo-pip-guard.sh` is fail-OPEN on the install deny, and
`ammo-pretool-guard.sh` is fail-CLOSED on the inspection fast path.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CODEX_SRC = ROOT / "ai_cli_session" / ".codex"
HOOK_NAMES = (
    "pre_tool_use_guard.py",
    "post_tool_use_guard.py",
    "stop_gate_guard.py",
    "session_start.py",
    "pre_compact.py",
)
# Hooks that emit a document unconditionally, so "still works" is observable as
# stdout rather than only as "did not crash".
ALWAYS_EMITTING = ("pre_compact.py",)


def _clean_env(**overrides):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AMMO_") and key != "CODEX_SESSION_ID"
    }
    env.update(overrides)
    return env


@pytest.fixture(scope="module")
def codex_trees(tmp_path_factory):
    """(intact, degraded) copies of the .codex tree; degraded lacks the classifier."""
    base = tmp_path_factory.mktemp("codex_trees")
    trees = {}
    for name in ("intact", "degraded"):
        dest = base / name
        shutil.copytree(
            CODEX_SRC,
            dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "worktrees"),
        )
        trees[name] = dest
    classifier = trees["degraded"] / "skills" / "ammo" / "scripts" / "hook_cmd_classify.py"
    assert classifier.is_file(), f"fixture precondition: {classifier} must exist to remove"
    classifier.unlink()
    return trees


def _run(tree, hook_name, payload, env=None):
    return subprocess.run(
        [sys.executable, str(tree / "hooks" / hook_name)],
        input=json.dumps(payload),
        env=env if env is not None else _clean_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _pretool(command, cwd):
    return {
        "cwd": str(cwd),
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "hook_event_name": "PreToolUse",
    }


def _artifact_dir(tmp_path, stage="0_preflight"):
    artifact = tmp_path / "kernel_opt_artifacts" / "cuda_vllm_kernel_target"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "state.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "active",
                    "current_round": 1,
                    "current_stage": stage,
                    "rounds": [{"round_id": 1, "debate": {}}],
                }
            }
        ),
        encoding="utf-8",
    )
    return artifact


@pytest.mark.parametrize("hook_name", HOOK_NAMES)
def test_hook_does_not_traceback_without_the_classifier(codex_trees, tmp_path, hook_name):
    """A missing classifier must not replace the verdict with a traceback."""
    result = _run(
        codex_trees["degraded"],
        hook_name,
        _pretool("pip install numpy", tmp_path),
    )
    assert "Traceback" not in result.stderr, (
        f"{hook_name} crashed without hook_cmd_classify.py:\n{result.stderr}"
    )
    assert "ModuleNotFoundError" not in result.stderr
    assert result.returncode == 0, f"{hook_name} exited {result.returncode}"


@pytest.mark.parametrize("hook_name", HOOK_NAMES)
def test_degradation_is_announced_on_stderr_not_stdout(codex_trees, tmp_path, hook_name):
    """Degraded must be loud, and loud must not corrupt the verdict document."""
    result = _run(
        codex_trees["degraded"],
        hook_name,
        _pretool("pip install numpy", tmp_path),
    )
    assert "AMMO hooks DEGRADED" in result.stderr, (
        f"{hook_name} degraded silently; stderr was {result.stderr!r}"
    )
    assert "AMMO hooks DEGRADED" not in result.stdout, (
        f"{hook_name} wrote the warning to stdout, breaking the verdict document"
    )
    if result.stdout.strip():
        json.loads(result.stdout)


@pytest.mark.parametrize("hook_name", ALWAYS_EMITTING)
def test_always_emitting_hook_still_emits_its_document(codex_trees, tmp_path, hook_name):
    result = _run(codex_trees["degraded"], hook_name, {"cwd": str(tmp_path)})
    assert result.stdout.strip(), f"{hook_name} emitted nothing without the classifier"
    json.loads(result.stdout)


def test_stop_gate_still_blocks_an_active_campaign(codex_trees, tmp_path):
    """The Stop gate carries no classifier dependency, so it must be unaffected."""
    artifact = _artifact_dir(tmp_path)
    env = _clean_env(AMMO_ARTIFACT_DIR=str(artifact))
    verdicts = {}
    for name, tree in codex_trees.items():
        result = _run(tree, "stop_gate_guard.py", {"cwd": str(tmp_path)}, env=env)
        assert result.stdout.strip(), f"stop gate emitted nothing in the {name} tree"
        verdicts[name] = json.loads(result.stdout)
    assert verdicts["degraded"]["decision"] == "block"
    assert "still active at 0_preflight" in verdicts["degraded"]["reason"]
    assert verdicts["degraded"] == verdicts["intact"]


@pytest.mark.parametrize(
    "payload_kind,command_or_edit,expected_fragment",
    [
        (
            "bash",
            "rm -f /tmp/x/codex_hook_session.json",
            "AMMO trusted-session identity",
        ),
        ("edit", "VLLM_OP1: bool = True", "must default off"),
    ],
)
def test_non_install_pretool_denies_still_fire_without_the_classifier(
    codex_trees, tmp_path, payload_kind, command_or_edit, expected_fragment
):
    """Only the install deny is fail-open; the other denies must stay live."""
    if payload_kind == "bash":
        payload = _pretool(command_or_edit, tmp_path)
    else:
        payload = {
            "cwd": str(tmp_path),
            "tool_name": "Edit",
            "tool_input": {"file_path": "vllm/envs.py", "new_string": command_or_edit},
            "hook_event_name": "PreToolUse",
        }
    result = _run(codex_trees["degraded"], "pre_tool_use_guard.py", payload)
    assert result.stdout.strip(), (
        f"pre_tool_use_guard.py emitted no verdict for {payload_kind}; "
        f"stderr was {result.stderr!r}"
    )
    verdict = json.loads(result.stdout)
    assert verdict.get("decision") == "block"
    assert expected_fragment in json.dumps(verdict)


def test_install_deny_fires_with_the_classifier_and_fails_open_without_it(
    codex_trees, tmp_path
):
    """Pin BOTH sides of the declared fail-open direction for the install deny.

    Fail-open matches `ammo-pip-guard.sh`, which declares the same direction for
    the same file, so the two runtimes agree even when degraded. The pip policy
    is also carried in AGENTS.md prose and the .venv provisioning path.
    """
    payload = _pretool("pip install numpy", tmp_path)

    intact = _run(codex_trees["intact"], "pre_tool_use_guard.py", payload)
    assert intact.stdout.strip(), "install deny must fire when the classifier is present"
    assert json.loads(intact.stdout)["decision"] == "block"
    assert "package install/uninstall is forbidden" in intact.stdout

    degraded = _run(codex_trees["degraded"], "pre_tool_use_guard.py", payload)
    assert "package install/uninstall is forbidden" not in degraded.stdout
    assert "install classification unavailable" in degraded.stderr


def test_inspection_fast_path_fails_closed_without_the_classifier(codex_trees, tmp_path):
    """`is_static_inspection_command` must degrade to False, never to True.

    True would hand every command the inspection fast path and skip the guards
    outright. False costs the guards only their early exit, which is the
    direction `ammo-pretool-guard.sh` declares for the same file.
    """
    hooks_dir = str(codex_trees["degraded"] / "hooks")
    probe = (
        "import sys; sys.path.insert(0, %r); import common; "
        "print(common.is_static_inspection_command('grep -rn kernel .'))" % hooks_dir
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=_clean_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "degraded is_static_inspection_command returned "
        f"{result.stdout.strip()!r}; True would skip every guard"
    )


def test_split_shell_command_keeps_real_semantics_when_degraded(codex_trees):
    """`split_shell_command` carries no policy, so degrade it to real shlex.split.

    `command_invokes_vllm_bench_latency` and the monitor-ack path tokenize
    through it; returning `[]` would silently mute them.
    """
    hooks_dir = str(codex_trees["degraded"] / "hooks")
    probe = (
        "import sys; sys.path.insert(0, %r); import common; "
        "print(common.split_shell_command('vllm bench latency --model m')); "
        "print(common.split_shell_command('cat \\\"unterminated'))" % hooks_dir
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=_clean_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "['vllm', 'bench', 'latency', '--model', 'm']"
    assert lines[1] == "[]", "an unbalanced quote must still yield []"


def test_common_guards_the_classifier_import(codex_trees):
    """Keep the guard itself from being refactored away."""
    source = (CODEX_SRC / "hooks" / "common.py").read_text(encoding="utf-8")
    assert "import hook_cmd_classify as _cmd_classify" in source
    head = source[: source.index("import hook_cmd_classify as _cmd_classify")]
    assert head.rstrip().endswith("try:"), (
        "common.py must import hook_cmd_classify inside a try/except -- an "
        "unguarded import disables all five Codex hooks"
    )


# ---------------------------------------------------------------------------
# Present-but-incomplete classifier. The file imports, so the guarded import
# succeeds and the failure lands on the attribute reads instead -- which must
# degrade exactly like a missing file, never raise AttributeError at the module
# scope of common.py.
# ---------------------------------------------------------------------------

INCOMPLETE_FLAVORS = ("zero_byte", "truncated")


def _truncated_classifier_source(source):
    """Source cut before the later definitions, still parseable Python."""
    marker = "def _is_read_only_python_segment("
    cut = source.index(marker)
    head = source[:cut]
    compile(head, "hook_cmd_classify.py", "exec")  # fixture precondition
    assert "def split_shell_command(" in head
    assert "def is_static_inspection_command(" not in head
    assert "def command_installs(" not in head
    return head


@pytest.fixture(scope="module", params=INCOMPLETE_FLAVORS)
def incomplete_tree(request, tmp_path_factory):
    """A .codex tree whose classifier imports cleanly but lacks attributes."""
    dest = tmp_path_factory.mktemp("codex_incomplete_" + request.param) / "codex"
    shutil.copytree(
        CODEX_SRC,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "worktrees"),
    )
    classifier = dest / "skills" / "ammo" / "scripts" / "hook_cmd_classify.py"
    assert classifier.is_file(), f"fixture precondition: {classifier} must exist to damage"
    if request.param == "zero_byte":
        classifier.write_text("", encoding="utf-8")
    else:
        classifier.write_text(
            _truncated_classifier_source(classifier.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    return dest


def test_incomplete_classifier_still_imports(incomplete_tree):
    """Pin the scenario: import succeeds, the expected attributes are absent."""
    scripts_dir = str(incomplete_tree / "skills" / "ammo" / "scripts")
    probe = (
        "import sys; sys.path.insert(0, %r); import hook_cmd_classify as c; "
        "print(hasattr(c, 'is_static_inspection_command'), "
        "hasattr(c, '_is_read_only_python_segment'), "
        "hasattr(c, 'command_installs'))" % scripts_dir
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=_clean_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"the damaged classifier must still import; stderr was {result.stderr!r}"
    )
    assert result.stdout.strip() == "False False False"


@pytest.mark.parametrize("hook_name", HOOK_NAMES)
def test_hook_survives_an_incomplete_classifier(incomplete_tree, tmp_path, hook_name):
    """A missing attribute must not replace the verdict with a traceback."""
    result = _run(incomplete_tree, hook_name, _pretool("pip install numpy", tmp_path))
    assert "Traceback" not in result.stderr, (
        f"{hook_name} crashed on an incomplete hook_cmd_classify.py:\n{result.stderr}"
    )
    assert "AttributeError" not in result.stderr
    assert result.returncode == 0, f"{hook_name} exited {result.returncode}"
    assert "AMMO hooks DEGRADED" in result.stderr, (
        f"{hook_name} degraded silently; stderr was {result.stderr!r}"
    )
    assert "AMMO hooks DEGRADED" not in result.stdout
    if result.stdout.strip():
        json.loads(result.stdout)


@pytest.mark.parametrize("hook_name", ALWAYS_EMITTING)
def test_always_emitting_hook_still_emits_when_incomplete(
    incomplete_tree, tmp_path, hook_name
):
    result = _run(incomplete_tree, hook_name, {"cwd": str(tmp_path)})
    assert result.stdout.strip(), f"{hook_name} emitted nothing on an incomplete classifier"
    json.loads(result.stdout)


def test_incomplete_classifier_keeps_the_stop_gate_blocking(incomplete_tree, tmp_path):
    artifact = _artifact_dir(tmp_path)
    result = _run(
        incomplete_tree,
        "stop_gate_guard.py",
        {"cwd": str(tmp_path)},
        env=_clean_env(AMMO_ARTIFACT_DIR=str(artifact)),
    )
    assert result.stdout.strip(), "stop gate emitted nothing on an incomplete classifier"
    verdict = json.loads(result.stdout)
    assert verdict["decision"] == "block"
    assert "still active at 0_preflight" in verdict["reason"]


@pytest.mark.parametrize(
    "payload_kind,command_or_edit,expected_fragment",
    [
        (
            "bash",
            "rm -f /tmp/x/codex_hook_session.json",
            "AMMO trusted-session identity",
        ),
        ("edit", "VLLM_OP1: bool = True", "must default off"),
    ],
)
def test_non_install_pretool_denies_still_fire_when_incomplete(
    incomplete_tree, tmp_path, payload_kind, command_or_edit, expected_fragment
):
    """Only the declared fail-open guards go quiet; the other denies stay live."""
    if payload_kind == "bash":
        payload = _pretool(command_or_edit, tmp_path)
    else:
        payload = {
            "cwd": str(tmp_path),
            "tool_name": "Edit",
            "tool_input": {"file_path": "vllm/envs.py", "new_string": command_or_edit},
            "hook_event_name": "PreToolUse",
        }
    result = _run(incomplete_tree, "pre_tool_use_guard.py", payload)
    assert result.stdout.strip(), (
        f"pre_tool_use_guard.py emitted no verdict for {payload_kind}; "
        f"stderr was {result.stderr!r}"
    )
    verdict = json.loads(result.stdout)
    assert verdict.get("decision") == "block"
    assert expected_fragment in json.dumps(verdict)


def test_incomplete_classifier_degrades_to_the_same_verdicts(incomplete_tree):
    """Same fail directions as a missing file: fail-closed path, real shlex."""
    hooks_dir = str(incomplete_tree / "hooks")
    probe = (
        "import sys; sys.path.insert(0, %r); import common; "
        "print(common.is_static_inspection_command('grep -rn kernel .')); "
        "print(common.command_invokes_gpu_heavy_tool('ncu --set full python x.py')); "
        "print(common.split_shell_command('vllm bench latency --model m')); "
        "print(common.split_shell_command('cat \\\"unterminated'))" % hooks_dir
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env=_clean_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "False", "the inspection fast path must fail closed"
    assert lines[1] == "False", "GPU-heavy detection degrades to False (disclosed)"
    assert lines[2] == "['vllm', 'bench', 'latency', '--model', 'm']"
    assert lines[3] == "[]", "an unbalanced quote must still yield []"
    assert "AMMO hooks DEGRADED" in result.stderr


def test_common_guards_the_classifier_attribute_reads():
    """The re-exports must sit inside the guard, not after it.

    An `else:` branch that reads the attributes is unguarded: a classifier that
    imports but lacks one name then raises AttributeError at the module scope of
    common.py, which every Codex hook imports -- the total outage the guard
    exists to prevent.
    """
    source = (CODEX_SRC / "hooks" / "common.py").read_text(encoding="utf-8")
    body = source[source.index("import hook_cmd_classify as _cmd_classify") :]
    guarded = body[: body.index("except Exception as _classify_exc:")]
    for name in (
        "split_shell_command",
        "_executable_segments",
        "_is_python_executable_name",
        "_is_read_only_python_segment",
        "is_static_inspection_command",
    ):
        assert f"{name}=_cmd_classify.{name}" in guarded, (
            f"{name} is re-exported outside the try/except -- a classifier that "
            "imports without it would traceback in all five hooks"
        )
