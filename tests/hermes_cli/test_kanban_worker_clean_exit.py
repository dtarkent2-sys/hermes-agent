"""Windows worker clean-exit observability (t_6db38d4a).

On Windows production the POSIX reap loop (``reap_worker_zombies``) is a
no-op and the ``os.WIF*`` helpers do not exist, so the wait-status registry
(``_recent_worker_exits``) never populates and ``_classify_worker_exit``
always fell through to ``("unknown", None)``: a clean worker exit (rc=0)
was indistinguishable from a crash, and clean-exit protocol violations were
mislabeled as generic "pid not alive" crashes — bypassing the
violation-only retry budget.

The fix extends the cross-platform worker-exit state-file channel: a kanban
worker on the normal-completion path persists ``exit_code=0`` to its per-run
state file right before ``sys.exit(0)`` (the sentinel codes 75/76 already
used this channel), and ``_classify_worker_exit`` reads a recorded 0 as a
clean exit. A genuine crash never reaches the write site, so a state file
can never mask one.

These tests drive the requeue through the cross-platform channel (no
monkeypatching of the POSIX wait-status helpers), so they pass natively on
every OS — and on Windows they exercise the ONLY clean-exit signal the
dispatcher has.
"""

import os
import time

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB and a pinned
    worker-exit state dir (mirrors the test_kanban_db.py fixture: hermetic
    and never writes run_<n>.json into a live production worker_exits dir
    inherited from the ambient environment)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Existing crash-detection tests pre-date the grace window; pin to 0
    # so they keep their immediate-reclaim semantics.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    monkeypatch.setenv(
        "HERMES_KANBAN_WORKER_EXIT_DIR", str(tmp_path / "worker_exits"),
    )
    kb.init_db()
    return home


def _claim_with_dead_pid(conn, tid, pid):
    """Claim ``tid`` under this host, pin ``pid`` as the (soon dead) worker,
    persist a CLEAN exit (exit_code=0) for the task's live run via the
    cross-platform channel, and return the run_id.

    Mirrors what a real worker does on the normal-completion path: reach
    the end of the single-query branch, write the per-run state file, then
    ``sys.exit(0)``.
    """
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w")
    conn.execute(
        "UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid),
    )
    conn.commit()
    run_id = kb.get_task(conn, tid).current_run_id
    assert run_id, "claim must open a run for the clean-exit channel to key on"
    kb.record_worker_exit_code(run_id, 0, None, pid=pid)
    return run_id


def _reap_with_dead_pid(conn):
    """Run one crash-detect pass with ``_pid_alive`` patched so every pid
    reports dead (the existing convention in test_kanban_core_functionality:
    the workers being reaped have exited, so nothing is alive).

    Uses a single module object for the patch and the reaper: earlier tests
    in a full-suite run can reload the module, and patching one module
    object while reaping through another (stale) one silently disables the
    patch.
    """
    import hermes_cli.kanban_db as _kb
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda p: False
    try:
        return _kb.detect_crashed_workers(conn)
    finally:
        _kb._pid_alive = original_alive


# ---------------------------------------------------------------------------
# Classification: the state file's recorded 0 IS the Windows clean-exit
# signal.
# ---------------------------------------------------------------------------


def test_state_file_zero_classifies_clean_exit_with_empty_registry(
    kanban_home,
):
    """A recorded exit_code=0 classifies as clean_exit even when the POSIX
    reap registry is empty — the exact production-Windows condition (no
    reaping, no os.WIF*)."""
    kb.record_worker_exit_code(4242, 0, None, pid=999)
    # Empty registry: pid 999 was never reaped (Windows production).
    assert kb._recent_worker_exits.get(999) is None
    assert kb._classify_worker_exit(999, run_id=4242) == ("clean_exit", 0)


def test_state_file_sentinels_still_win_over_zero(kanban_home):
    """The sentinel classifications are unchanged by the clean-exit branch,
    and a state file recording a code that is neither sentinel nor 0 falls
    through to the registry (never fabricates a clean exit)."""
    kb.record_worker_exit_code(
        4300, kb.KANBAN_RATE_LIMIT_EXIT_CODE, "rate_limit",
    )
    assert kb._classify_worker_exit(4300, run_id=4300) == (
        "rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE,
    )
    kb.record_worker_exit_code(
        4301, kb.KANBAN_PROVIDER_OUTAGE_EXIT_CODE, "server_error",
    )
    assert kb._classify_worker_exit(4301, run_id=4301) == (
        "provider_outage", kb.KANBAN_PROVIDER_OUTAGE_EXIT_CODE,
    )
    # A hypothetical crash-code file (exit 1 — workers never write this)
    # must NOT classify as clean: no registry entry -> unknown.
    kb.record_worker_exit_code(4302, 1, None)
    assert kb._classify_worker_exit(4302, run_id=4302) == ("unknown", None)


def test_stale_clean_exit_file_is_ignored(kanban_home):
    """A clean-exit file older than the TTL must not mask a later
    classification: the run is then an ordinary unknown crash, exactly like
    the pre-fix behavior for a missing file."""
    kb.record_worker_exit_code(4400, 0, None, pid=998)
    path = kb._worker_exit_state_path(4400)
    assert path.is_file()
    old = time.time() - (kb._WORKER_EXIT_STATE_TTL_SECONDS + 120)
    os.utime(path, (old, old))
    assert kb._classify_worker_exit(998, run_id=4400) == ("unknown", None)


# ---------------------------------------------------------------------------
# End-to-end through detect_crashed_workers: a Windows-shaped clean exit now
# lands in the protocol-violation budget instead of the crash counter.
# ---------------------------------------------------------------------------


def test_clean_exit_file_drives_protocol_violation_budget(kanban_home):
    """Full dispatcher path: worker exits rc=0 (state file, empty registry)
    while its task is still running -> protocol violation with the bounded
    violation-only retry budget; reaching the bound trips the breaker."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="clean-exit", assignee="w")
        limit = kb._PROTOCOL_VIOLATION_FAILURE_LIMIT

        # Below budget: limit-1 consecutive violations all retry, each
        # leaving the unified failure counter untouched (the two budgets
        # stay independent).
        for i in range(limit - 1):
            _claim_with_dead_pid(conn, tid, 80000 + i)
            crashed = _reap_with_dead_pid(conn)
            assert tid in crashed, (
                f"violation {i + 1} must be reclaimed as a crash-kind event"
            )
            task = kb.get_task(conn, tid)
            assert task.status == "ready", (
                f"violation {i + 1} below budget must retry, got {task.status}"
            )
            assert task.consecutive_failures == 0, (
                "below-budget violation must not tick consecutive_failures"
            )
            # The event says WHY: a clean-exit protocol violation, not a
            # bare "pid not alive".
            events = kb.list_events(conn, tid)
            kind = events[-1].kind if events else None
            assert kind == "protocol_violation", (
                f"expected protocol_violation event, got {kind!r}"
            )
            payload = events[-1].payload or {}
            assert payload.get("protocol_violation") is True
            assert payload.get("exit_code") == 0
            # The run metadata carries the durable violation marker the
            # streak counter reads back later.
            run_meta = conn.execute(
                "SELECT outcome, metadata FROM task_runs "
                "WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,),
            ).fetchone()
            assert run_meta["outcome"] == "crashed"
            import json as _json
            assert _json.loads(run_meta["metadata"]).get(
                "protocol_violation"
            ) is True

        # Streak reached the bound (violation #limit): breaker trips, task
        # blocked with a single gave_up event carrying the streak count.
        _claim_with_dead_pid(conn, tid, 80000 + limit)
        _reap_with_dead_pid(conn)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        gave_up = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]
        assert len(gave_up) == 1
        assert (gave_up[0].payload or {}).get("protocol_violations") == limit
    finally:
        conn.close()


def test_missing_state_file_still_counts_as_plain_crash(kanban_home):
    """The never-misclassify invariant: with NO state file (a real crash —
    exit 1/2, a signal, or a worker that died before writing), the
    classification falls to the empty registry -> unknown, and the crash
    counts against the unified failure counter exactly as before."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="real-crash", assignee="w")
        host = kb._claimer_id().split(":", 1)[0]

        # First crash: unified counter ticks to 1 (below the default limit
        # of 2) — task retries.
        kb.claim_task(conn, tid, claimer=f"{host}:w")
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (81000, tid))
        conn.commit()
        _reap_with_dead_pid(conn)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 1
        events = kb.list_events(conn, tid)
        assert (events[-1].kind if events else None) == "crashed"

        # Second crash: breaker trips at DEFAULT_FAILURE_LIMIT=2.
        kb.claim_task(conn, tid, claimer=f"{host}:w")
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (81001, tid))
        conn.commit()
        _reap_with_dead_pid(conn)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The prune sweep: every exit now writes a file, so expired ones must be
# reaped from disk each dispatch tick.
# ---------------------------------------------------------------------------


def test_prune_worker_exit_state_removes_only_expired_run_files(
    kanban_home, tmp_path,
):
    """Prune deletes run_*.json past the TTL, leaves fresh ones, foreign
    files, and .tmp leftovers untouched, and never raises on a missing
    dir."""
    exit_dir = tmp_path / "worker_exits"
    assert kb._worker_exit_dir() == exit_dir

    # No dir yet -> 0, no raise.
    assert kb.prune_worker_exit_state() == 0

    fresh = kb.record_worker_exit_code(5001, 0, None, pid=1)
    aged = kb.record_worker_exit_code(5002, 0, None, pid=2)
    assert fresh is not None and aged is not None
    old = time.time() - (kb._WORKER_EXIT_STATE_TTL_SECONDS + 60)
    os.utime(aged, (old, old))
    # Foreign / partial files must survive the sweep.
    (exit_dir / "run_5003.json.tmp").write_text("{", encoding="utf-8")
    (exit_dir / "notes.txt").write_text("x", encoding="utf-8")

    removed = kb.prune_worker_exit_state()
    assert removed == 1
    assert fresh.is_file(), "fresh state file must survive the sweep"
    assert not aged.exists(), "expired state file must be removed"
    assert (exit_dir / "run_5003.json.tmp").is_file()
    assert (exit_dir / "notes.txt").is_file()

    # An explicit max_age overrides the TTL.
    os.utime(fresh, (time.time() - 30, time.time() - 30))
    assert kb.prune_worker_exit_state(max_age_seconds=10) == 1
    assert not fresh.exists()


def test_dispatch_tick_prunes_expired_state_files(kanban_home, tmp_path):
    """The sweep is wired into the dispatcher tick: an expired state file
    that survived a full tick is gone afterward (and the tick itself is
    unaffected by the sweep)."""
    conn = kb.connect()
    try:
        aged = kb.record_worker_exit_code(5100, 0, None, pid=3)
        old = time.time() - (kb._WORKER_EXIT_STATE_TTL_SECONDS + 60)
        os.utime(aged, (old, old))
        result = kb.dispatch_once(conn)
        assert result.skipped_locked is False
        assert not aged.exists(), (
            "dispatch tick must sweep expired worker-exit state files"
        )
    finally:
        conn.close()