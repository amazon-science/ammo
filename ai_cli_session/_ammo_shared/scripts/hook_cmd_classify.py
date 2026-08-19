#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Shell-command classifier for the AMMO shell hooks. Stdlib only.

Why this file exists
--------------------
Two blocking shell hooks used anchored `grep -P` regexes to classify a Bash
command, and a single token defeated both:

  * the read-only fast path in the PreToolUse campaign guard anchored at the
    START of the command string, so `cat notes.md && vllm bench latency ...`
    silenced every warning AND both hard blocks behind it;
  * the package-install deny required the verb to sit IMMEDIATELY after the
    invoker, so `pip -q install`, `/usr/bin/pip install`, and
    `bash -c "pip install"` all passed.

A regex cannot express "every segment of this pipeline is inspection-class" or
"an install verb appears in command position after unwrapping env/sudo/nested
shells". A tokenizer can, and this module is it.

This module is the ONLY implementation. The shell hooks call its CLI;
`.codex/hooks/common.py` re-exports `is_static_inspection_command` and its
helpers from here, and `.codex/hooks/pre_tool_use_guard.py` imports
`command_installs`. Nothing carries a second copy, so the two runtimes cannot
drift apart. `test_hook_cmd_classify_parity.py` pins the verdicts.

Interface
---------
    hook_cmd_classify.py --mode readonly [COMMAND]
    hook_cmd_classify.py --mode install  [COMMAND]

COMMAND is read from argv when present, otherwise from stdin. Exit codes are
the verdict, so a shell hook can branch with a bare `if`:

    0  verdict TRUE   (readonly: every segment is inspection-class /
                       install: a package install-or-uninstall was found)
    1  verdict FALSE
    2  usage error or an internal fault -- the CALLER decides fail-open vs
       fail-closed, this module never decides for it.

The verdict token (`READONLY` / `NOT_READONLY` / `INSTALL` / `NO_INSTALL`) is
also printed on stdout, for harnesses and for hooks that prefer text.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Tokenizing. Two lexers, deliberately: `readonly` needs shlex.split's view of
# redirections and command substitutions as ordinary tokens so it can reject
# them; `install` needs punctuation_chars so `&&`/`;`/`|` split into segments.
# --------------------------------------------------------------------------

_SEGMENT_SEPARATORS = {"&&", ";", "||", "|"}
_PUNCTUATION_CHARS = ";&|()`"


def split_shell_command(command):
    """shlex.split, [] on an unbalanced quote."""
    try:
        return shlex.split(command or "")
    except ValueError:
        return []


def _package_guard_tokens(command):
    """Punctuation-aware lexer: `&&` / `;` / `|` become their own tokens."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=_PUNCTUATION_CHARS)
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return []


# --------------------------------------------------------------------------
# readonly: is_static_inspection_command and its helpers
# --------------------------------------------------------------------------

_INSPECTION_EXECUTABLES = {
    "rg", "grep", "cat", "head", "tail", "less", "find", "ag", "ack", "env",
    "printenv", "echo", "printf", "jq", "wc", "sqlite3", "ls", "stat", "file",
    "tree", "du", "pwd", "realpath", "readlink", "dirname", "basename",
}
_READ_ONLY_GIT_VERBS = {"log", "show", "diff", "blame", "status", "branch", "tag"}


def _command_segments(parts):
    segments = []
    current = []
    for token in parts:
        if token in _SEGMENT_SEPARATORS:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _first_executable_index(parts):
    skip_next_for = {"timeout"}
    idx = 0
    while idx < len(parts):
        token = parts[idx]
        if token in {"(", ")"}:
            return -1
        if token in {"env", "command", "time"} or re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*", token
        ):
            idx += 1
            continue
        if token in skip_next_for:
            idx += 2
            continue
        return idx
    return -1


def _executable_segments(parts):
    resolved = []
    for segment in _command_segments(parts):
        idx = _first_executable_index(segment)
        if idx >= 0:
            resolved.append((Path(segment[idx]).name, segment[idx + 1:]))
    return resolved


def _is_python_executable_name(name):
    return bool(re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", name))


def _is_read_only_python_snippet(text):
    lowered = text.lower()
    blocked = [
        r"\b(torch|vllm|triton|cupy|tensorflow|jax)\b",
        r"\b(subprocess|os\.system|popen|runpy|exec|eval)\b",
        r"\b(json\.dump|pickle\.dump)\b",
        r"\b(write_text|write_bytes|\.write\s*\(|mkdir|touch|unlink|remove|rename|replace|rmtree|copyfile)\b",
        r"\bopen\s*\([^)]*,\s*[\"'][wa+x]",
    ]
    if any(re.search(pattern, lowered) for pattern in blocked):
        return False
    read_markers = [
        r"\bjson\.load[s]?\b",
        r"\bread_text\s*\(",
        r"\bread_bytes\s*\(",
        r"\bopen\s*\(",
        r"\bstate\.json\b",
    ]
    return "json" in lowered and any(re.search(p, lowered) for p in read_markers)


def _is_read_only_python_segment(tail):
    if not tail:
        return False
    first = tail[0]
    if first in {"-V", "--version", "-VV", "--help", "-h"}:
        return True
    if len(tail) >= 2 and first == "-m" and tail[1] in {"json.tool"}:
        return True
    if len(tail) >= 2 and first == "-c":
        return _is_read_only_python_snippet(" ".join(tail[1:]))
    if first == "-":
        return _is_read_only_python_snippet(" ".join(tail[1:]))
    return False


def is_static_inspection_command(command):
    """True only when EVERY executable segment is inspection-class.

    A read-only executable can still mutate through shell plumbing, so every
    redirection, heredoc/process substitution, and command substitution makes
    the whole command non-static -- callers then take their fail-closed path.
    """
    parts = split_shell_command(command)
    if any(
        token in {">", ">>", "<>", "&>", "&>>", "<<", "<<<"}
        or ">" in token
        or token.startswith(("<(", ">("))
        or "$(" in token
        or "`" in token
        for token in parts
    ):
        return False
    segments = [(name, tail) for name, tail in _executable_segments(parts) if name != "cd"]
    if not segments:
        return False
    for name, tail in segments:
        if name == "sed":
            if any(token == "-i" or token.startswith("-i") for token in tail):
                return False
            continue
        if name == "git":
            if not tail or tail[0] not in _READ_ONLY_GIT_VERBS:
                return False
            continue
        if _is_python_executable_name(name) and _is_read_only_python_segment(tail):
            continue
        if name not in _INSPECTION_EXECUTABLES:
            return False
    return True


def is_inspection_only(command):
    """`is_static_inspection_command` per SEGMENT, using the punctuation lexer.

    Strictly more conservative than `is_static_inspection_command`, in one
    direction only: it can turn a READONLY verdict into NOT_READONLY, never the
    reverse. The parity test pins that one-directional relation.

    Why the extra pass: `shlex.split` does not treat `;` as a token boundary,
    so `echo hi; nsys profile python bench.py` lexes `hi;` as one word and the
    static form sees a single `echo` segment -- READONLY, with an nsys run
    hiding behind it. The punctuation-aware lexer splits that command properly,
    and each resulting segment is then judged by the static form itself.

    A NOT_READONLY verdict only costs the caller its fast path: the guard then
    evaluates the command normally. So being conservative here adds no block
    the guard would not otherwise have considered.
    """
    tokens = _package_guard_tokens(command)
    if not tokens:
        # An unbalanced quote yields no tokens; the static form reaches the
        # same verdict through its own empty-token path.
        return is_static_inspection_command(command)
    segments = list(_punct_segments(tokens))
    if not segments:
        return False
    return all(
        is_static_inspection_command(shlex.join(segment)) for segment in segments
    )


# --------------------------------------------------------------------------
# install: command_installs and its helpers
# --------------------------------------------------------------------------

_INSTALL_VERBS = {"install", "uninstall"}
_NESTED_SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}


def _punct_segments(tokens):
    current = []
    for token in tokens:
        if token and all(ch in _PUNCTUATION_CHARS for ch in token):
            if current:
                yield current
                current = []
        else:
            current.append(token)
    if current:
        yield current


def _drop_assignments(tokens):
    idx = 0
    while idx < len(tokens) and re.fullmatch(
        r"[A-Za-z_][A-Za-z_0-9]*=.*", tokens[idx], re.DOTALL
    ):
        idx += 1
    return tokens[idx:]


def _unwrap_command(tokens):
    """Strip env-var assignments and the env / command / sudo wrappers."""
    tokens = _drop_assignments(tokens)
    while tokens:
        name = Path(tokens[0]).name
        if name == "env":
            idx = 1
            while idx < len(tokens):
                token = tokens[idx]
                if token in {"-u", "--unset"} and idx + 1 < len(tokens):
                    idx += 2
                elif token.startswith("-") or re.fullmatch(
                    r"[A-Za-z_][A-Za-z_0-9]*=.*", token, re.DOTALL
                ):
                    idx += 1
                else:
                    break
            tokens = tokens[idx:]
            continue
        if name == "command":
            if any(token in {"-v", "-V"} for token in tokens[1:]):
                return []
            idx = 1
            while idx < len(tokens) and tokens[idx].startswith("-"):
                idx += 1
            tokens = tokens[idx:]
            continue
        if name == "sudo":
            idx = 1
            while idx < len(tokens) and tokens[idx].startswith("-"):
                idx += 1
            tokens = tokens[idx:]
            continue
        break
    return tokens


def _has_install_verb(tokens):
    return any(token in _INSTALL_VERBS for token in tokens)


def _segment_installs(tokens, depth=0):
    tokens = _unwrap_command(tokens)
    if not tokens:
        return False
    name = Path(tokens[0]).name
    args = tokens[1:]
    if re.fullmatch(r"pip\d*", name):
        return _has_install_verb(args)
    if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", name):
        for idx, token in enumerate(args[:-1]):
            if token == "-m" and Path(args[idx + 1]).name == "pip":
                return _has_install_verb(args[idx + 2:])
        return False
    if name == "uv":
        for idx, token in enumerate(args):
            if token == "pip":
                return _has_install_verb(args[idx + 1:])
        return False
    if name in _NESTED_SHELLS and depth < 3:
        for idx, token in enumerate(args[:-1]):
            if token == "-c" or (token.startswith("-") and "c" in token[1:]):
                return command_installs(args[idx + 1], depth + 1)
    return False


def command_installs(command, depth=0):
    """True when any segment invokes a package install/uninstall.

    Command position only: a quoted mention (`grep "pip install"`) is an
    argument, not an invoker, and does not match.
    """
    tokens = _package_guard_tokens(command)
    return any(_segment_installs(segment, depth) for segment in _punct_segments(tokens))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

MODES = {
    # readonly uses the per-segment form: a hook's fast path must require EVERY
    # segment to be inspection-class, not just the one the reference lexer saw.
    "readonly": (is_inspection_only, "READONLY", "NOT_READONLY"),
    "install": (command_installs, "INSTALL", "NO_INSTALL"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Classify a shell command for the AMMO hooks.",
        epilog="exit 0 = verdict true, 1 = verdict false, 2 = usage/internal fault",
    )
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument(
        "command",
        nargs="?",
        help="the command string; read from stdin when omitted",
    )
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2

    command = args.command
    if command is None:
        try:
            command = sys.stdin.read()
        except OSError:
            return 2

    predicate, yes, no = MODES[args.mode]
    try:
        verdict = predicate(command or "")
    except Exception:  # a classifier fault must not read as a verdict
        return 2
    print(yes if verdict else no)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
