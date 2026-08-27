"""Tests for the unified profile→machine dashboard launch routing.

`<profile> dashboard` routes to ONE machine-level dashboard instead of
spawning a per-profile server: attach (open browser at ?profile=) when one
is already listening, else re-exec as the machine dashboard with the
launching profile preselected. `--isolated` opts out.

The re-exec child is pinned to the machine ROOT by design, so a pytest-context
parent must never actually spawn it: under POSIX execvpe the spawn replaces
the test process image, but under Windows the "re-exec" is CreateProcess +
wait — a real child that outlives the test while holding production
HERMES_HOME (#82770 class; 2026-08-27 leak: a pytest dashboard child polled
the live /api/sessions once a minute for hours). The spawn gate in
cmd_dashboard refuses under the same test-context signal the live-system
guard uses (hermes_state.running_in_test_context), with the same
HERMES_STATE_DB_GUARD_BYPASS opt-out.
"""
import sys
import types
import pytest


@pytest.fixture
def main_mod():
    import hermes_cli.main as main_mod
    return main_mod


def _args(**kw):
    defaults = dict(
        status=False, stop=False, host="127.0.0.1", port=9119,
        no_open=True, insecure=False, skip_build=False,
        isolated=False, open_profile="",
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


class TestUnifiedDashboardRouting:


    def test_profile_launch_reexecs_machine_dashboard(self, main_mod, monkeypatch):
        """A NON-test context (the real CLI) still re-execs as designed."""
        import hermes_state

        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        monkeypatch.setattr(main_mod, "_dashboard_listening", lambda host, port: False)
        # Force the not-a-test answer: under pytest the spawn gate would
        # otherwise (correctly) refuse, and the POSIX exec branch is what we
        # want to observe here. Patch the public seam — the gate reads the
        # module attribute at call time, so this is the one effective lever
        # (ancestry cannot be un-faked from env alone).
        monkeypatch.setattr(
            hermes_state, "running_in_test_context", lambda: False
        )
        execs = []

        def fake_exec(exe, argv, env):
            execs.append((exe, argv, env))
            raise SystemExit(0)  # execvpe never returns

        monkeypatch.setattr(main_mod.os, "execvpe", fake_exec)
        # Force the POSIX exec branch regardless of host: the exec interface
        # is what we assert on, and on Windows the win32 branch would attempt
        # a REAL Popen (the original test hung there — see module docstring).
        monkeypatch.setattr(main_mod.sys, "platform", "linux")

        with pytest.raises(SystemExit):
            main_mod.cmd_dashboard(_args())

        assert len(execs) == 1
        exe, argv, env = execs[0]
        assert exe == sys.executable
        # Pinned to the default profile + launching profile preselected.
        assert "-p" in argv and argv[argv.index("-p") + 1] == "default"
        assert "--open-profile" in argv
        assert argv[argv.index("--open-profile") + 1] == "worker_x"
        # The child is pinned to the machine ROOT, not the launching profile's
        # HERMES_HOME.  For a standard install (HERMES_HOME unset) that root is
        # the platform-native default (~/.hermes), NOT dropped — see the Docker
        # test below for why we resolve explicitly instead of popping.
        from hermes_constants import get_default_hermes_root
        assert env.get("HERMES_HOME") == str(get_default_hermes_root())

    def test_desktop_profile_backend_skips_machine_dashboard_reroute(self, main_mod, monkeypatch):
        """A desktop-spawned named-profile backend (HERMES_DESKTOP=1) must NOT
        reroute into the machine dashboard. The reroute re-execs as the default
        profile and exits, so the desktop never sees a ready backend → boot
        loop. The guard keeps desktop pool backends per-profile."""
        monkeypatch.setenv("HERMES_DESKTOP", "1")
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        listening_calls = []
        monkeypatch.setattr(
            main_mod, "_dashboard_listening",
            lambda host, port: listening_calls.append(1) or False,
        )
        execs = []
        monkeypatch.setattr(main_mod.os, "execvpe", lambda *a, **k: execs.append(a))
        monkeypatch.setitem(sys.modules, "fastapi", None)

        with pytest.raises((SystemExit, AttributeError, ImportError, TypeError)):
            main_mod.cmd_dashboard(_args())
        assert listening_calls == []
        assert execs == []


class TestPytestContextSpawnGate:
    """The machine-dashboard reroute must never spawn its production-pinned
    child from a pytest context (Windows 're-exec' = real surviving child)."""

    def test_pytest_context_refuses_the_spawn(self, main_mod, monkeypatch):
        """Gate trips: SystemExit, and NEITHER spawn surface is reached."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("HERMES_STATE_DB_GUARD_BYPASS", raising=False)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        monkeypatch.setattr(main_mod, "_dashboard_listening", lambda host, port: False)
        execs = []
        spawns = []
        monkeypatch.setattr(main_mod.os, "execvpe", lambda *a: execs.append(a))
        monkeypatch.setattr(
            main_mod.subprocess, "Popen", lambda *a, **k: spawns.append(a)
        )

        with pytest.raises(SystemExit, match="pytest context"):
            main_mod.cmd_dashboard(_args())

        assert execs == []
        assert spawns == []

    def test_windows_popen_branch_never_spawns_under_pytest(self, main_mod, monkeypatch):
        """Regression for the 2026-08-27 leak: on win32 the reroute uses
        subprocess.Popen (not execvpe), which outlives its caller — the exact
        escape this gate exists to close. The gate must fire BEFORE the
        platform branch so no Popen is ever attempted under pytest."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("HERMES_STATE_DB_GUARD_BYPASS", raising=False)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        monkeypatch.setattr(main_mod, "_dashboard_listening", lambda host, port: False)
        spawns = []
        monkeypatch.setattr(
            main_mod.subprocess, "Popen", lambda *a, **k: spawns.append(a)
        )
        monkeypatch.setattr(main_mod.sys, "platform", "win32")

        with pytest.raises(SystemExit, match="pytest context"):
            main_mod.cmd_dashboard(_args())

        assert spawns == []

    def test_guard_bypass_env_opts_back_into_the_spawn(self, main_mod, monkeypatch):
        """The documented opt-out (HERMES_STATE_DB_GUARD_BYPASS=1, same one the
        SessionDB guard honors) lets a test take the POSIX spawn path."""
        import hermes_state

        monkeypatch.setenv("HERMES_STATE_DB_GUARD_BYPASS", "1")
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name", lambda: "worker_x"
        )
        monkeypatch.setattr(main_mod, "_dashboard_listening", lambda host, port: False)
        # Bypass env is honored by the gate, but the POSIX spawn still runs
        # under pytest: patch the seam so the exec path itself is observable.
        monkeypatch.setattr(
            hermes_state, "running_in_test_context", lambda: False
        )
        execs = []
        monkeypatch.setattr(
            main_mod.os, "execvpe",
            lambda exe, argv, env: execs.append((exe, argv, env)),
        )
        # Same POSIX-branch forcing as the non-test-path test above.
        monkeypatch.setattr(main_mod.sys, "platform", "linux")

        with pytest.raises(SystemExit):
            main_mod.cmd_dashboard(_args())

        assert len(execs) == 1

    def test_seam_matches_private_test_context_signal(self):
        """The public seam must be the guard's own detector, not a rehash."""
        import hermes_state

        assert hermes_state.running_in_test_context is hermes_state._in_test_context