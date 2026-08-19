# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for the detached CLI daemon teardown on pause/terminate (item A2).

Since Claude Code 2.1.x the CLI re-hosts under a background daemon detached from
the tmux pane tree, so killing ttyd + the session tmux server left the lead and
its teammates running headless — still launching GPU sweeps on a session whose
server-layer GPU locks were already released.

Coverage:
  (a) a daemon-shaped process carrying a session marker is killed on pause
  (b) another session's process is untouched
  (c) a reserved_detached_run.sh process survives (documented exemption)
  (d) SIGKILL escalation fires when the daemon survives SIGTERM
  (e) marker matching respects token boundaries: a spoofed marker (in a shell
      comment on argv, or inside another env var's value) does NOT exempt, and a
      sibling session id ("abc" vs "abc-2") does NOT match
  (f) a process in the server's own process group (the pause-time S3 upload
      pipeline, which carries the session dir on argv) is never swept

Process-matching markers under test (a later in-container stub test replicates
them): a pid is session-owned when its /proc/<pid>/environ holds the exact line
``AMMO_SESSION_ID={session_id}``, or when a whole argv element, the cwd, or an
environ line's value is the session dir (``/data/sessions/{session_id}``) or the
worktree path — or a path under either, at a ``/`` boundary.
"""

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Helpers
# ============================================================================

def _make_manager(tmp_path):
    """SessionManager with every collaborator mocked."""
    from orchestration.session_manager import SessionManager
    from orchestration.worktree_manager import reset_worktree_manager

    reset_worktree_manager()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    mock_worktree = MagicMock()
    mock_gpu = MagicMock()
    mock_gpu.get_gpu_count.return_value = 4
    mock_gpu.get_available_gpu_count.return_value = 4

    mock_terminal = MagicMock()
    mock_terminal.stop_terminal = AsyncMock()

    mock_storage = MagicMock()
    mock_storage.enabled = False

    manager = SessionManager(
        sessions_dir=str(sessions_dir),
        worktree_manager=mock_worktree,
        gpu_manager=mock_gpu,
        terminal_manager=mock_terminal,
        cli_tool_manager=MagicMock(),
        inactivity_monitor=MagicMock(),
        session_storage=mock_storage,
    )
    return manager, mock_gpu, mock_terminal


def _make_active_session(manager, session_id):
    from shared.session_models import SessionState, SessionStatus, CLIToolType

    session_dir = manager.sessions_dir / session_id
    (session_dir / "worktree").mkdir(parents=True, exist_ok=True)
    state = SessionState(
        session_id=session_id,
        status=SessionStatus.ACTIVE,
        cli_tool=CLIToolType.CLAUDE,
        repo_name="vllm",
        branch="main",
        gpu_ids=[0],
        created_at=time.time(),
        last_accessed=time.time(),
        terminal_port=8001,
        session_dir=str(session_dir),
        worktree_path=str(session_dir / "worktree"),
    )
    manager._sessions[session_id] = state
    return state


class _Proc:
    """A real detached child process used as a kill target."""

    def __init__(self, popen):
        self.popen = popen
        self.pid = popen.pid

    def alive(self) -> bool:
        """True while the process exists and is not a reaped/zombie corpse."""
        self.popen.poll()
        if self.popen.returncode is not None:
            return False
        try:
            with open(f"/proc/{self.pid}/stat", "rb") as f:
                fields = f.read().rsplit(b")", 1)[-1].split()
            return fields[0] != b"Z"
        except (OSError, IndexError):
            return False

    def wait_dead(self, timeout=6.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive():
                return True
            time.sleep(0.05)
        return False

    def cleanup(self):
        try:
            os.kill(self.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.popen.wait(timeout=5)
        except Exception:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            fields = f.read().rsplit(b")", 1)[-1].split()
        return fields[0] != b"Z"
    except (OSError, IndexError):
        return False


def _pids_in_pgid(pgid: int) -> list:
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.getpgid(int(entry)) == pgid:
                found.append(int(entry))
        except OSError:
            continue
    return found


@pytest.fixture
def spawned():
    procs = []
    yield procs
    for proc in procs:
        proc.cleanup()


def _spawn(
    procs,
    env_extra,
    script="trap 'exit 0' TERM; while :; do sleep 0.2; done",
    args=(),
    cwd=None,
    start_new_session=True,
):
    """
    Spawn a detached shell.

    ``args`` become real argv elements after the ``-c`` script (``sh -c SCRIPT
    sh ARG...``), which is how a marker reaches argv as a whole token — the only
    form the sweep may match. ``start_new_session=False`` keeps the child in the
    test process's own process group, the shape of a server-spawned helper.
    """
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp"}
    env.update(env_extra)
    popen = subprocess.Popen(
        ["/bin/sh", "-c", script, "sh", *args],
        env=env,
        cwd=cwd,
        start_new_session=start_new_session,  # own group, like the CLI daemon
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc = _Proc(popen)
    procs.append(proc)
    # Let the shell install its TERM handler before anyone signals it.
    time.sleep(0.3)
    return proc


# ============================================================================
# (a) daemon-shaped process with a session marker is killed on pause
# ============================================================================

@pytest.mark.unit
class TestDaemonKilledOnPause:

    @pytest.mark.asyncio
    async def test_daemon_with_session_env_marker_is_killed(self, tmp_path, spawned):
        manager, mock_gpu, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        daemon = _spawn(spawned, {"AMMO_SESSION_ID": session_id})
        assert daemon.alive()
        assert daemon.pid in manager._find_session_owned_pids(state)

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)

        assert daemon.wait_dead(), (
            "pause must kill the detached daemon carrying AMMO_SESSION_ID"
        )

    @pytest.mark.asyncio
    async def test_daemon_matched_by_session_dir_in_cmdline(self, tmp_path, spawned):
        """A path UNDER the session dir, as a whole argv element, still matches."""
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        daemon = _spawn(
            spawned, {}, args=(f"{state.session_dir}/claude-config",)
        )
        assert daemon.pid in manager._find_session_owned_pids(state)

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.terminate_session(session_id)
        assert daemon.wait_dead(), "terminate must kill a daemon matched by session dir"

    @pytest.mark.asyncio
    async def test_kill_runs_before_gpu_release(self, tmp_path, spawned):
        """A surviving agent must never outlive its GPU locks."""
        manager, mock_gpu, mock_terminal = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        _make_active_session(manager, session_id)
        _spawn(spawned, {"AMMO_SESSION_ID": session_id})

        order = []
        mock_gpu.release_gpus_for_session = MagicMock(
            side_effect=lambda sid: order.append("release_gpus")
        )
        original = manager._kill_session_owned_processes

        async def tracked(state, proc_root=None):
            order.append("kill_daemon")
            return await original(state, proc_root=proc_root)

        manager._kill_session_owned_processes = tracked
        manager._DAEMON_KILL_GRACE_SEC = 0.5

        await manager.pause_session(session_id)

        assert order.index("kill_daemon") < order.index("release_gpus"), (
            f"daemon kill must precede GPU release, got {order}"
        )


# ============================================================================
# (b) another session's process is untouched
# ============================================================================

@pytest.mark.unit
class TestOtherSessionUntouched:

    @pytest.mark.asyncio
    async def test_other_session_process_survives_pause(self, tmp_path, spawned):
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        other_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)
        other_state = _make_active_session(manager, other_id)

        mine = _spawn(spawned, {"AMMO_SESSION_ID": session_id})
        theirs = _spawn(spawned, {"AMMO_SESSION_ID": other_id})

        owned = manager._find_session_owned_pids(state)
        assert mine.pid in owned
        assert theirs.pid not in owned
        assert theirs.pid in manager._find_session_owned_pids(other_state)

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)

        assert mine.wait_dead()
        assert theirs.alive(), "another session's process must not be signalled"

    def test_server_and_init_are_never_targets(self, tmp_path):
        manager, _, _ = _make_manager(tmp_path)
        state = _make_active_session(manager, f"a2-{uuid.uuid4()}")
        protected = manager._protected_pids("/proc")
        assert 1 in protected
        assert os.getpid() in protected
        assert not set(manager._find_session_owned_pids(state)) & protected


# ============================================================================
# (c) reserved_detached_run.sh process survives
# ============================================================================

@pytest.mark.unit
class TestDetachedRunExemption:

    @pytest.mark.asyncio
    async def test_reserved_detached_run_survives_pause(self, tmp_path, spawned):
        """A detached run owns its own GPU reservation and outlives the session."""
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        detached = _spawn(
            spawned,
            {"AMMO_SESSION_ID": session_id},
            args=(f"{state.worktree_path}/scripts/reserved_detached_run.sh",),
        )
        daemon = _spawn(spawned, {"AMMO_SESSION_ID": session_id})

        owned = manager._find_session_owned_pids(state)
        assert daemon.pid in owned
        assert detached.pid not in owned, (
            "reserved_detached_run.sh must be exempt from the kill sweep"
        )

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)

        assert daemon.wait_dead()
        assert detached.alive(), "the detached run must survive pause by design"

    @pytest.mark.asyncio
    async def test_env_flagged_detached_run_survives_pause(self, tmp_path, spawned):
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        detached = _spawn(
            spawned,
            {"AMMO_SESSION_ID": session_id, "AMMO_DETACHED_RUN": "1"},
        )
        assert detached.pid not in manager._find_session_owned_pids(state)

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)
        assert detached.alive()

    @pytest.mark.asyncio
    async def test_detached_run_sharing_a_pgid_with_a_daemon_survives(self, tmp_path, spawned):
        """
        A group-wide SIGTERM would take the exempt run down with its sibling, so
        a group sheltering an exempt run is signalled per-pid instead.
        """
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        # One setsid group holding both an exempt detached run and a killable
        # daemon-shaped sibling. The exempt child lives in its own script file so
        # the marker appears on its argv only, never on the group leader's.
        exempt_script = tmp_path / "reserved_detached_run.sh"
        exempt_script.write_text("trap '' TERM\nwhile :; do sleep 0.2; done\n")
        # The launcher itself is a file too, so the exempt marker never lands on
        # the group leader's own argv.
        launcher = tmp_path / "group_leader.sh"
        launcher.write_text(
            f"/bin/sh {exempt_script} &\ntrap 'exit 0' TERM\nwhile :; do sleep 0.2; done\n"
        )
        leader = _spawn(spawned, {"AMMO_SESSION_ID": session_id}, script=f". {launcher}")
        time.sleep(0.4)
        pgid = os.getpgid(leader.pid)
        exempt_pids = [
            pid for pid in _pids_in_pgid(pgid)
            if (identity := manager._read_proc_identity(pid, "/proc"))
            and manager._is_exempt_detached_run(identity)
        ]
        assert exempt_pids, "test setup: no exempt process spawned in the group"

        owned, exempt_pgids = manager._scan_session_processes(state)
        assert leader.pid in owned
        assert pgid in exempt_pgids, "the sheltering group must be recorded as exempt"
        assert not set(exempt_pids) & set(owned)

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        try:
            await manager.pause_session(session_id)

            assert leader.wait_dead(), "the killable sibling must still die"
            for pid in exempt_pids:
                assert _pid_alive(pid), (
                    "an exempt detached run must survive even when it shares a "
                    "process group with a killed daemon"
                )
        finally:
            for pid in exempt_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

    def test_exemption_markers_are_documented_in_code(self):
        """The exemption is a contract, so it must be stated where it is coded."""
        import inspect
        from orchestration.session_manager import SessionManager

        source = inspect.getsource(SessionManager)
        assert "reserved_detached_run.sh" in SessionManager._DETACHED_RUN_EXEMPT_MARKERS
        assert "EXEMPTION" in source, (
            "the detached-run exemption must carry a code comment stating the constraint"
        )


# ============================================================================
# (e) marker matching respects token boundaries — spoofs do not exempt
# ============================================================================

@pytest.mark.unit
class TestExemptionSpoofsAreKilled:
    """
    The exemption is boundary-matched, so an in-session agent cannot make itself
    pause-immune by putting an exemption marker anywhere a substring search would
    have found it. Both spoofs below survived pause before this fix.
    """

    @pytest.mark.asyncio
    async def test_shell_comment_spoof_on_argv_is_killed(self, tmp_path, spawned):
        """AMMO_DETACHED_RUN=1 inside a shell comment is not an environ line."""
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        spoof = _spawn(
            spawned,
            {"AMMO_SESSION_ID": session_id},
            script=(
                "# keep alive AMMO_DETACHED_RUN=1 via reserved_detached_run.sh\n"
                "trap 'exit 0' TERM; while :; do sleep 0.2; done"
            ),
        )

        owned, exempt_pgids = manager._scan_session_processes(state)
        assert spoof.pid in owned, (
            "an exemption marker inside a shell comment must not exempt"
        )
        assert os.getpgid(spoof.pid) not in exempt_pgids

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)
        assert spoof.wait_dead(), "the shell-comment spoof must be killed on pause"

    @pytest.mark.asyncio
    async def test_env_var_quoting_spoof_is_killed(self, tmp_path, spawned):
        """A marker inside ANOTHER variable's value is not an exemption."""
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        spoof = _spawn(
            spawned,
            {
                "AMMO_SESSION_ID": session_id,
                "AMMO_RUN_CMD": "bash scripts/reserved_detached_run.sh --sweep",
                "AMMO_NOTE": "set AMMO_DETACHED_RUN=1 to detach",
            },
        )

        owned, exempt_pgids = manager._scan_session_processes(state)
        assert spoof.pid in owned, (
            "an exemption marker quoted inside another env var must not exempt"
        )
        assert os.getpgid(spoof.pid) not in exempt_pgids

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)
        assert spoof.wait_dead(), "the env-quoting spoof must be killed on pause"

    @pytest.mark.asyncio
    async def test_longer_word_containing_the_script_name_is_killed(self, tmp_path, spawned):
        """The script marker matches a whole argv element or a '/' basename only."""
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        spoof = _spawn(
            spawned,
            {"AMMO_SESSION_ID": session_id},
            args=("my_reserved_detached_run.sh.bak",),
        )
        assert spoof.pid in manager._find_session_owned_pids(state)

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)
        assert spoof.wait_dead()

    @pytest.mark.asyncio
    async def test_genuine_env_flagged_detached_run_still_survives(self, tmp_path, spawned):
        """The real env flag is a complete environ line, so it still exempts."""
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        detached = _spawn(
            spawned,
            {"AMMO_SESSION_ID": session_id, "AMMO_DETACHED_RUN": "1"},
        )
        owned, exempt_pgids = manager._scan_session_processes(state)
        assert detached.pid not in owned
        assert os.getpgid(detached.pid) in exempt_pgids

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)
        assert detached.alive(), "a genuine AMMO_DETACHED_RUN=1 run must survive"

    @pytest.mark.asyncio
    async def test_genuine_script_argv_element_still_survives(self, tmp_path, spawned):
        """A real invocation puts the script path on argv as one whole element."""
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        detached = _spawn(
            spawned,
            {"AMMO_SESSION_ID": session_id},
            args=(
                f"{state.worktree_path}/.claude/skills/ammo/scripts/"
                "reserved_detached_run.sh",
                "--sweep",
            ),
        )
        bare = _spawn(
            spawned,
            {"AMMO_SESSION_ID": session_id},
            args=("reserved_detached_run.sh",),
        )

        owned = manager._find_session_owned_pids(state)
        assert detached.pid not in owned, "a full script path on argv must exempt"
        assert bare.pid not in owned, "the bare script name on argv must exempt"

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)
        assert detached.alive()
        assert bare.alive()


@pytest.mark.unit
class TestSiblingSessionIdIsNotMatched:
    """
    Path markers match at a '/' boundary only, so one session id that PREFIXES
    another cannot pull the other's processes into its sweep.
    """

    def test_sibling_id_suffix_is_not_owned(self, tmp_path):
        manager, _, _ = _make_manager(tmp_path)
        state = _make_active_session(manager, "abc")
        sibling = _make_active_session(manager, "abc-2")

        from orchestration.session_manager import ProcIdentity

        markers = manager._session_process_markers(state)
        sibling_identity = ProcIdentity(
            environ=("AMMO_SESSION_ID=abc-2",),
            argv=("/bin/sh", f"{sibling.session_dir}/run.sh"),
            cwd=sibling.worktree_path,
        )
        assert not markers.matches(sibling_identity), (
            "session 'abc' must not match session 'abc-2'"
        )

        own_identity = ProcIdentity(
            environ=("AMMO_SESSION_ID=abc",),
            argv=("/bin/sh",),
            cwd=None,
        )
        assert markers.matches(own_identity)
        assert markers.matches(
            ProcIdentity(environ=(), argv=(), cwd=f"{state.worktree_path}/vllm")
        ), "a path UNDER the worktree must still match"

    @pytest.mark.asyncio
    async def test_sibling_session_process_survives_pause(self, tmp_path, spawned):
        manager, _, _ = _make_manager(tmp_path)
        _make_active_session(manager, "abc")
        sibling = _make_active_session(manager, "abc-2")

        theirs = _spawn(
            spawned,
            {"AMMO_SESSION_ID": "abc-2"},
            args=(f"{sibling.worktree_path}/run.sh",),
        )
        mine = _spawn(spawned, {"AMMO_SESSION_ID": "abc"})

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session("abc")

        assert mine.wait_dead()
        assert theirs.alive(), "session 'abc-2' must survive session 'abc' pausing"


# ============================================================================
# (f) the server's own process group is never swept (S3 upload pipeline)
# ============================================================================

@pytest.mark.unit
class TestServerOwnPgidIsSkipped:
    """
    session_state.py uploads the paused session's worktree with
    ``bash -c "tar cf - -C {session_dir} worktree | pigz | aws s3 cp - s3://..."``.
    That helper is server-spawned, so it inherits the server's pgid AND carries
    the session dir on its argv. Signalling it truncates the very upload the
    pause exists to produce, so the scan must exclude the server's own group
    outright.
    """

    @pytest.mark.asyncio
    async def test_s3_upload_shaped_helper_is_not_owned_and_survives(self, tmp_path, spawned):
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        # Same shape as the real upload: session_dir on argv, server's own pgid.
        helper = _spawn(
            spawned,
            {},
            args=("-C", state.session_dir, "worktree"),
            start_new_session=False,
        )
        assert os.getpgid(helper.pid) == os.getpgid(0), "test setup: not our pgid"

        owned = manager._find_session_owned_pids(state)
        assert helper.pid not in owned, (
            "a process in the server's own pgid must be excluded from the sweep"
        )

        daemon = _spawn(spawned, {"AMMO_SESSION_ID": session_id})
        assert daemon.pid in manager._find_session_owned_pids(state)

        manager._DAEMON_KILL_GRACE_SEC = 0.5
        await manager.pause_session(session_id)

        assert daemon.wait_dead(), "the real session daemon must still die"
        assert helper.alive(), (
            "the server's own S3 upload pipeline must survive the pause sweep"
        )

    def test_rationale_is_documented_in_code(self):
        """The skip is load-bearing for upload integrity, so it must be stated."""
        import inspect
        from orchestration.session_manager import SessionManager

        source = inspect.getsource(SessionManager._scan_session_processes)
        assert "start_new_session" in source, (
            "the skip must state why no session process inherits the server pgid"
        )


# ============================================================================
# (d) escalation when the daemon survives SIGTERM
# ============================================================================

@pytest.mark.unit
class TestSigkillEscalation:

    @pytest.mark.asyncio
    async def test_sigterm_ignoring_daemon_is_sigkilled(self, tmp_path, spawned):
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        state = _make_active_session(manager, session_id)

        stubborn = _spawn(
            spawned,
            {"AMMO_SESSION_ID": session_id},
            script="trap '' TERM; while :; do sleep 0.2; done",
        )
        manager._DAEMON_KILL_GRACE_SEC = 0.5

        with patch.object(
            manager, "_force_kill_session_processes", wraps=manager._force_kill_session_processes
        ) as spy:
            await manager.pause_session(session_id)

        assert spy.called, (
            "escalation must fire on the daemon-survival case, not only when "
            "stop_terminal() raises"
        )
        assert stubborn.wait_dead(), "a SIGTERM-ignoring daemon must be SIGKILLed"

    @pytest.mark.asyncio
    async def test_no_escalation_when_nothing_survives(self, tmp_path, spawned):
        manager, _, _ = _make_manager(tmp_path)
        session_id = f"a2-{uuid.uuid4()}"
        _make_active_session(manager, session_id)
        daemon = _spawn(spawned, {"AMMO_SESSION_ID": session_id})
        manager._DAEMON_KILL_GRACE_SEC = 0.5

        with patch.object(manager, "_force_kill_session_processes") as spy:
            await manager.pause_session(session_id)

        assert daemon.wait_dead()
        assert not spy.called, "no escalation when the SIGTERM sweep already worked"


# ============================================================================
# Server-restart PAUSED-by-load must not kill anything
# ============================================================================

@pytest.mark.unit
class TestLoadPausedPathDoesNotKill:

    def test_load_sessions_marks_paused_without_killing(self, tmp_path, spawned):
        """_load_sessions reattaches on resume, so it must never signal a pid."""
        import inspect
        from orchestration.session_manager import SessionManager

        source = inspect.getsource(SessionManager._load_sessions)
        for token in (
            "_kill_session_owned_processes",
            "_force_kill_session_processes",
            "_stop_session_processes",
            "os.kill",
        ):
            assert token not in source, (
                f"_load_sessions must not kill processes (found {token}); the "
                "PAUSED-by-load path reattaches to the live tmux session"
            )
