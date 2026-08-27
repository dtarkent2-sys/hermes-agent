"""Tests for the kanban-worker Job Object orphan containment (t_d3b27ff0).

2026-08-27 incident: a dispatcher-spawned worker crashed while running
``pytest tests/hermes_cli/``; on Windows ``start_new_session=True`` is a
no-op, so the worker's children were orphaned and an orphaned pytest
ballooned to ~20GB, and the memory pressure killed live workers on other
tasks (the engineer-lane "pid not alive" crash loop).

The fix: a dispatcher-spawned worker (``HERMES_KANBAN_TASK`` set) arms a
kill-on-close Win32 Job Object at CLI startup, so the kernel tears down the
worker's whole descendant tree the moment the worker process dies — no
dispatcher detection race, no orphan window.

The live containment behavior (child dies with a hard-killed job member)
is verified by scripts-level integration in the incident report; these
unit tests pin the decision surface: gating, idempotence, POSIX no-op,
diagnostics, and that non-worker processes are never contained.
"""

from __future__ import annotations

import sys

import pytest

from hermes_cli import worker_job_object as wjo


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    """Each test sees a fresh arm state (module globals are sticky)."""
    monkeypatch.setattr(wjo, "job_object_armed", False)
    monkeypatch.setattr(wjo, "_last_arm_result", "not attempted")
    yield


@pytest.mark.windows_only
class TestWindowsGating:
    """On Windows, only dispatcher-spawned workers arm containment."""

    def test_arms_when_kanban_task_env_set(self, monkeypatch):
        pytest.importorskip("ctypes.wintypes")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test1234")
        assert wjo.arm_kanban_worker_job_object() is True
        assert wjo.job_object_armed is True
        assert wjo.job_object_status() == "ok"

    def test_refuses_without_kanban_task_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        assert wjo.arm_kanban_worker_job_object() is False
        assert wjo.job_object_armed is False
        assert wjo.job_object_status() == "not-a-worker"

    def test_refuses_blank_kanban_task_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "   ")
        assert wjo.arm_kanban_worker_job_object() is False
        assert wjo.job_object_status() == "not-a-worker"

    def test_idempotent_second_call(self, monkeypatch):
        pytest.importorskip("ctypes.wintypes")
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test1234")
        assert wjo.arm_kanban_worker_job_object() is True
        # A second call must be a no-op returning True (already contained),
        # not attempt a second job/assignment (a process can join only one).
        assert wjo.arm_kanban_worker_job_object() is True
        assert wjo.job_object_status() == "ok"


@pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX no-op path; covered by CI lane"
)
class TestPosixNoOp:
    def test_never_arms_on_posix(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test1234")
        assert wjo.arm_kanban_worker_job_object() is False
        assert wjo.job_object_armed is False
        assert wjo.job_object_status() == "non-windows"


class TestCliWiring:
    """cli.main() must call the armer best-effort — never raise through it."""

    def test_main_wraps_arm_call(self):
        import inspect

        import cli

        src = inspect.getsource(cli.main)
        assert "arm_kanban_worker_job_object" in src
        # The call site must be exception-guarded: containment setup failing
        # must never break worker startup (dispatcher TTL is the backstop).
        assert "try:" in src.split("arm_kanban_worker_job_object")[0][-400:]