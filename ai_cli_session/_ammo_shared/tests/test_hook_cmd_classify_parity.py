# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Verdict pins and internal parity for hook_cmd_classify.py.

`scripts/hook_cmd_classify.py` is the ONE shell-command classifier. The shell
hooks call its CLI; `.codex/hooks/common.py` and
`.codex/hooks/pre_tool_use_guard.py` import it directly. There is no second
implementation left to compare against, so this module pins the verdicts
themselves over a command table that covers each bypass the older anchored
`grep -P` regexes used to allow.

Three relations are asserted:

  * every table row pins the expected `readonly` and `install` verdict;
  * `is_inspection_only` may only ever be STRICTER than
    `is_static_inspection_command` -- a stricter verdict costs a hook its fast
    path, a looser one would hand a bypass back, so the relation is
    one-directional by contract;
  * `is_static_inspection_command` carries its OWN verdict pins, not only that
    inequality. The inequality relates two functions in this one file, so a
    loosening applied to the shared helpers they both call satisfies it
    trivially. `is_static_inspection_command` now feeds BOTH runtimes -- the
    shell hooks through the CLI and the Codex hooks through a direct import --
    so `test_static_rejects_shell_plumbing` pins the rejections that a
    self-referential relation cannot see.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _TESTS_DIR.parent / "scripts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import hook_cmd_classify as classify  # noqa: E402


# --------------------------------------------------------------------------
# The command table. `readonly` / `install` pin the expected verdict for every
# row: a row left at None asserts only the one-directional relation, which a
# loosening of the shared helpers satisfies trivially, so None is reserved for
# rows where no verdict is defensible.
# --------------------------------------------------------------------------

# (command, expected_readonly, expected_install)
TABLE: tuple[tuple[str, bool | None, bool | None], ...] = (
    # --- inspection prefixes that used to disable the whole guard ---
    ("grep -rn kernel .", True, False),
    ("rg --files kernel_opt_artifacts", True, False),
    ("cat kernel_opt_artifacts/t/state.json", True, False),
    ("head -20 REPORT.md", True, False),
    ("ls -la kernel_opt_artifacts", True, False),
    ("jq .campaign.status state.json", True, False),
    ("wc -l envs.py", True, False),
    ("git log --oneline -10", True, False),
    ("git status", True, False),
    ("git commit -m x", False, False),
    ("sed -n 1,5p envs.py", True, False),
    ("sed -i s/a/b/ envs.py", False, False),
    # --- prefix bypasses: an inspection head must not license the tail ---
    ("cat notes.md && vllm bench latency --model m", False, False),
    ("true && nvidia-smi --query-compute-apps=pid --format=csv", False, False),
    ("echo hi; nsys profile python bench.py", False, False),
    ("grep -q x f && python run_vllm_bench_latency_sweep.py", False, False),
    ("ls | grep state.json", True, False),
    # --- shell plumbing makes a read-only executable non-static ---
    ("cat state.json > /tmp/copy.json", False, False),
    ("echo $(rm -rf /tmp/x)", False, False),
    # --- python inspection vs python work ---
    ("python --version", True, False),
    ("python -c 'import json; json.load(open(\"state.json\"))'", True, False),
    ("python -c 'import torch; torch.zeros(1)'", False, False),
    ("python bench.py", False, False),
    # --- install detection: command position ---
    # `readonly` is False on every row below: pip/uv/pip3 are not inspection
    # executables, so admitting one would hand back the fast path that silences
    # the guards. The pin is what catches that loosening.
    ("pip install numpy", False, True),
    ("pip -q install numpy", False, True),
    ("pip3 --quiet install numpy", False, True),
    ("python -m pip -q install numpy", False, True),
    ("python3 -m pip install -e .", False, True),
    ("uv pip -q install numpy", False, True),
    ("pip uninstall -y numpy", False, True),
    # --- install detection: path-qualified interpreters and wrappers ---
    ("/usr/bin/pip install numpy", False, True),
    (".venv/bin/pip install -e .", False, True),
    ("/opt/py/bin/python3 -m pip install numpy", False, True),
    ("sudo pip install numpy", False, True),
    ("env VLLM_USE_PRECOMPILED=1 uv pip install -e .", False, True),
    ("VLLM_USE_PRECOMPILED=1 uv pip install -e .", False, True),
    # --- install detection: nested shells ---
    ('bash -c "pip install numpy"', False, True),
    ("sh -c 'python -m pip install numpy'", False, True),
    ('bash -lc "cd /w && uv pip install -e ."', False, True),
    # --- benign: pip mentioned but not invoked, or read-only pip ---
    ('grep "pip install" requirements.txt', True, False),
    ('echo "do not pip install things"', True, False),
    ('rg "uv pip install" docs/', True, False),
    # Read-only pip verbs are still NOT_READONLY: `pip` never earns the fast
    # path, whatever its verb. Only `command_installs` distinguishes the verbs.
    ("pip list", False, False),
    ("pip freeze", False, False),
    ("pip --version", False, False),
    ("pip show numpy", False, False),
    ("echo running pipeline", True, False),
    # --- degenerate input must not crash either mode ---
    ("", False, False),
    ("   ", False, False),
    ('cat "unterminated', False, False),
)


@pytest.mark.parametrize("command,expected,_ignored", TABLE, ids=range(len(TABLE)))
def test_inspection_only_is_conservative(command, expected, _ignored):
    """`--mode readonly` may only ever be stricter than the static classifier.

    A stricter verdict costs a hook its fast path; a looser one would hand a
    bypass back. The relation is therefore one-directional by contract.
    """
    mine = classify.is_inspection_only(command)
    looser = classify.is_static_inspection_command(command)
    if mine:
        assert looser, (
            f"is_inspection_only is LOOSER than is_static_inspection_command for "
            f"{command!r} -- that direction reopens a bypass"
        )
    if expected is not None:
        assert mine is expected, f"readonly verdict wrong for {command!r}"


def test_inspection_only_closes_the_glued_separator_hole():
    """The case that motivates the per-segment pass over the static classifier.

    `shlex.split` does not treat `;` as a token boundary, so the static form
    lexes `hi;` as one word and sees a single `echo` segment. The
    punctuation-aware lexer splits the command properly.
    """
    command = "echo hi; nsys profile python bench.py"
    assert classify.is_static_inspection_command(command) is True
    assert classify.is_inspection_only(command) is False


# `is_static_inspection_command` is the function BOTH runtimes call: the shell
# hooks reach it through `--mode readonly`, `.codex/hooks/common.py` re-exports
# it, and `post_tool_use_guard.py` gates state.json validation on it. The rows
# below pin its verdict DIRECTLY, because the table walk above reads
# `is_inspection_only` and the two-function inequality cannot see a loosening
# that both functions inherit from a shared helper.
#
# (command, expected_static, what a loosening would let through)
_SHELL_PLUMBING: tuple[tuple[str, bool, str], ...] = (
    # command substitution: the substituted command runs, whatever the head is
    ("echo $(rm -rf /tmp/x)", False, "rm inside $( )"),
    ("echo $(nsys profile -o t python bench.py)", False, "nsys run inside $( )"),
    (
        "cat state.json $(perl -pi -e s/a/b/ state.json)",
        False,
        "state.json rewrite inside $( ) skipping post-hoc state validation",
    ),
    # backticks: the older form of the same hole
    ("cat `whoami`.log", False, "command substitution via backticks"),
    # redirection: a read-only executable still writes through the shell
    ("cat state.json > /tmp/copy.json", False, "> truncating a file"),
    ("cat notes.md >> /tmp/log", False, ">> appending to a file"),
    ("cat state.json 2>/dev/null", False, "fd-qualified redirection"),
    # process substitution and heredocs
    ("cat <(rm -rf /tmp/x)", False, "rm inside <( )"),
    ("grep -rn x . <<< payload", False, "here-string feeding a command"),
    # the positive control: plain inspection with no plumbing must stay READONLY,
    # so a blanket `return False` cannot pass this case list either
    ("grep -rn kernel .", True, "n/a - positive control"),
)


@pytest.mark.parametrize("command,expected,leak", _SHELL_PLUMBING)
def test_static_rejects_shell_plumbing(command, expected, leak):
    """Pin `is_static_inspection_command` itself, not only its relation."""
    assert classify.is_static_inspection_command(command) is expected, (
        f"is_static_inspection_command({command!r}) must be {expected} -- "
        f"otherwise {leak} takes the fast path in both runtimes"
    )


@pytest.mark.parametrize("command,expected,_ignored", TABLE, ids=range(len(TABLE)))
def test_static_verdicts_are_pinned(command, expected, _ignored):
    """Every table row pins `is_static_inspection_command` too.

    `is_inspection_only` is only ever stricter, and the sole row where the two
    differ is the glued-separator case pinned by its own test below.
    """
    if command == "echo hi; nsys profile python bench.py":
        pytest.skip("static form is deliberately looser here; see its own test")
    assert classify.is_static_inspection_command(command) is expected, (
        f"static readonly verdict wrong for {command!r}"
    )


@pytest.mark.parametrize("command,_ignored,expected", TABLE, ids=range(len(TABLE)))
def test_install_verdicts_are_pinned(command, _ignored, expected):
    mine = classify.command_installs(command)
    if expected is not None:
        assert mine is expected, f"install verdict wrong for {command!r}"


def test_table_covers_every_bypass_class():
    """The table is the regression record; keep it from shrinking silently."""
    assert len(TABLE) >= 25
    joined = " ".join(cmd for cmd, _r, _i in TABLE)
    for marker in ("&&", "bash -c", "/usr/bin/pip", "-m pip", "sudo ", "grep "):
        assert marker in joined, f"table lost coverage of {marker!r}"


# --------------------------------------------------------------------------
# CLI contract: the shell hooks consume exit codes, so pin them.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,command,code,token",
    [
        ("readonly", "grep -rn x .", 0, "READONLY"),
        ("readonly", "cat f && vllm bench latency", 1, "NOT_READONLY"),
        ("install", "pip -q install numpy", 0, "INSTALL"),
        ("install", "pip list", 1, "NO_INSTALL"),
    ],
)
def test_cli_exit_codes(mode, command, code, token, capsys):
    assert classify.main(["--mode", mode, command]) == code
    assert capsys.readouterr().out.strip() == token


def test_cli_rejects_unknown_mode_with_code_2(capsys):
    assert classify.main(["--mode", "nonsense", "ls"]) == 2
    capsys.readouterr()


def test_cli_reads_stdin_when_command_omitted(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("pip install numpy"))
    assert classify.main(["--mode", "install"]) == 0
    assert capsys.readouterr().out.strip() == "INSTALL"


def test_module_is_stdlib_only():
    """A hook runs under a bare system python3; a third-party import would break it."""
    source = (_SCRIPTS_DIR / "hook_cmd_classify.py").read_text(encoding="utf-8")
    imports = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_.]*)", source, re.M))
    allowed = {"__future__", "argparse", "re", "shlex", "sys", "pathlib"}
    assert imports <= allowed, f"non-stdlib imports: {sorted(imports - allowed)}"
