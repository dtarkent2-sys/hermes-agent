"""Tests for the background-review refusal circuit breaker + cost controls.

Covers (task t_c0a5ac1f, burn investigation 2026-08-27):
  • _is_curator_refusal — refusal payload detection (read-before-write marker,
    guard refusal wording, whitelist denial) and non-refusal negatives.
  • RefusalBreaker — threshold drain of the fork's IterationBudget, no abort
    below threshold, budget-drain → loop-exit contract, never raises.
  • Context isolation — two breakers on separate threads never see each
    other's refusals.
  • _resolve_review_max_iterations — default 4, config override, invalid
    fallback.
  • _parent_cache_effective / _review_history_for_fork — digest when routed OR
    when the provider reports zero cache reads; full replay only on a
    cache-effective same-model parent.

Pure in-process; no live model calls.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Dict
from unittest.mock import patch

import pytest

from agent import background_review as br
from agent.background_review_breaker import (
    RefusalBreaker,
    install_review_breaker,
    observe_tool_result,
    reset_review_breaker,
    _is_curator_refusal,
)
from agent.iteration_budget import IterationBudget


# ---------------------------------------------------------------------------
# _is_curator_refusal — payload detection
# ---------------------------------------------------------------------------

def test_refusal_detected_from_read_before_write_marker():
    payload = {"success": False, "error": "something", "_read_before_write_required": True}
    assert _is_curator_refusal(json.dumps(payload)) is True
    assert _is_curator_refusal(payload) is True


def test_refusal_detected_from_guard_wording():
    payload = {
        "success": False,
        "error": (
            "Refusing background curator edit for skill 'demo': the current "
            "SKILL.md content has not been loaded in this review turn."
        ),
    }
    assert _is_curator_refusal(json.dumps(payload)) is True


def test_refusal_detected_from_whitelist_denial():
    payload = {
        "error": "Background review denied non-whitelisted tool: terminal. "
                 "Only memory/skill tools are allowed."
    }
    assert _is_curator_refusal(json.dumps(payload)) is True


def test_ordinary_error_is_not_a_refusal():
    payload = {"success": False, "error": "file not found: skills/demo/SKILL.md"}
    assert _is_curator_refusal(json.dumps(payload)) is False


def test_success_result_and_garbage_are_not_refusals():
    assert _is_curator_refusal(json.dumps({"success": True})) is False
    assert _is_curator_refusal("not json at all") is False
    assert _is_curator_refusal(None) is False
    assert _is_curator_refusal({"error": 12345}) is False


# ---------------------------------------------------------------------------
# RefusalBreaker — budget drain at threshold
# ---------------------------------------------------------------------------

class _FakeBudget:
    def __init__(self, total: int):
        self.max_total = total
        self._used = 0

    def consume(self) -> bool:
        if self._used >= self.max_total:
            return False
        self._used += 1
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.max_total - self._used)


class _FakeReviewAgent:
    def __init__(self, iterations: int = 4):
        self.iteration_budget = _FakeBudget(iterations)
        self._budget_grace_call = False


def test_breaker_no_abort_below_threshold():
    agent = _FakeReviewAgent()
    breaker = RefusalBreaker(agent, threshold=3)
    refusal = json.dumps({"error": "Refusing background curator edit for skill 'x'"})
    breaker.observe("skill_manage", refusal)
    breaker.observe("skill_manage", refusal)
    assert breaker.refusals == 2
    assert breaker.aborted is False
    assert agent.iteration_budget.remaining == 4  # untouched


def test_breaker_aborts_at_threshold_and_drains_budget():
    agent = _FakeReviewAgent(iterations=4)
    breaker = RefusalBreaker(agent, threshold=2)
    refusal = json.dumps({"error": "Refusing background curator patch for skill 'x'"})
    breaker.observe("skill_manage", refusal)
    breaker.observe("skill_manage", refusal)  # second identical refusal → abort
    assert breaker.aborted is True
    assert breaker.refusals == 2
    assert "2 refused curator calls" in breaker.abort_reason
    assert agent.iteration_budget.remaining == 0, "budget must be drained"


def test_breaker_neutralizes_pending_grace_call():
    agent = _FakeReviewAgent(iterations=4)
    agent._budget_grace_call = True
    breaker = RefusalBreaker(agent, threshold=1)
    breaker.observe("skill_manage", json.dumps({"_read_before_write_required": True}))
    assert breaker.aborted is True
    assert agent._budget_grace_call is False
    assert agent.iteration_budget.remaining == 0


def test_breaker_counts_distinct_refusal_kinds_too():
    """Any curator refusal counts — the loop is stuck regardless of which
    guard refused. Two DIFFERENT refusal kinds at threshold=2 still abort."""
    agent = _FakeReviewAgent(iterations=4)
    breaker = RefusalBreaker(agent, threshold=2)
    breaker.observe("skill_manage", json.dumps(
        {"error": "Refusing background curator edit for skill 'a': pinned skills are off-limits"}))
    breaker.observe("skill_manage", json.dumps(
        {"error": "Refusing background curator delete for skill 'b': the skill is not curator-managed"}))
    assert breaker.aborted is True


def test_breaker_observe_never_raises():
    agent = _FakeReviewAgent()
    breaker = RefusalBreaker(agent, threshold=1)
    breaker.observe("skill_manage", object())  # unjsonable junk
    breaker.observe("skill_manage", {"error": "Refusing background curator edit"})
    assert breaker.aborted is True  # refusal still counted despite junk input


def test_breaker_ignores_non_refusal_failures():
    agent = _FakeReviewAgent(iterations=4)
    breaker = RefusalBreaker(agent, threshold=1)
    breaker.observe("skill_view", json.dumps({"success": False, "error": "skill not found"}))
    breaker.observe("terminal", json.dumps({"success": False, "error": "exit code 1"}))
    assert breaker.aborted is False
    assert breaker.refusals == 0


# ---------------------------------------------------------------------------
# Context isolation via the module-level observer
# ---------------------------------------------------------------------------

def test_observer_outside_review_context_is_noop():
    # Must not raise and must not create any state.
    observe_tool_result("skill_manage", json.dumps({"_read_before_write_required": True}))


def test_observer_feeds_installed_breaker_and_reset_unbinds():
    agent = _FakeReviewAgent(iterations=4)
    breaker = install_review_breaker(agent, threshold=2)
    try:
        refusal = json.dumps({"error": "Refusing background curator edit for skill 'x'"})
        observe_tool_result("skill_manage", refusal)
        observe_tool_result("skill_manage", refusal)
        assert breaker.aborted is True
        assert agent.iteration_budget.remaining == 0
    finally:
        reset_review_breaker(breaker)
    # After reset, further observations are inert.
    observe_tool_result("skill_manage", refusal)


def test_breaker_isolation_between_threads():
    """Two concurrent review forks (separate threads) must not trip each
    other's breakers: the aborting thread's drain must not leak into the
    healthy thread's breaker (or vice versa)."""
    results: Dict[str, Dict[str, Any]] = {}
    start_gate = threading.Barrier(2)

    def _run(key: str, refusals: int) -> None:
        agent = _FakeReviewAgent(iterations=4)
        breaker = install_review_breaker(agent, threshold=2)
        try:
            start_gate.wait()  # both breakers installed at the same time
            refusal = json.dumps({"error": "Refusing background curator edit"})
            for _ in range(refusals):
                observe_tool_result("skill_manage", refusal)
            results[key] = {"aborted": breaker.aborted, "agent": agent}
        finally:
            reset_review_breaker(breaker)

    t1 = threading.Thread(target=_run, args=("aborting", 2))
    t2 = threading.Thread(target=_run, args=("healthy", 1))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert results["aborting"]["aborted"] is True
    assert results["aborting"]["agent"].iteration_budget.remaining == 0
    assert results["healthy"]["aborted"] is False
    assert results["healthy"]["agent"].iteration_budget.remaining == 4


# ---------------------------------------------------------------------------
# _resolve_review_max_iterations — config knob
# ---------------------------------------------------------------------------

def test_max_iterations_default_is_4():
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert br._resolve_review_max_iterations() == 4


def test_max_iterations_config_override():
    # Contract: callers pass the ALREADY-EXTRACTED auxiliary.background_review
    # block (same as load_background_review_settings / spawn path).
    assert br._resolve_review_max_iterations({"max_iterations": 7}) == 7


def test_max_iterations_invalid_falls_back_to_legacy_16():
    assert br._resolve_review_max_iterations({"max_iterations": 0}) == 16
    assert br._resolve_review_max_iterations({"max_iterations": "not-a-number"}) == 16
    assert br._resolve_review_max_iterations({"max_iterations": -3}) == 16


def test_max_iterations_missing_key_uses_default_not_legacy():
    assert br._resolve_review_max_iterations({}) == 4


def test_fork_run_uses_config_budget_and_cache_adaptive_replay_end_to_end():
    """End-to-end through the REAL worker body (_run_review_in_thread), with
    only AIAgent replaced: the fork must be constructed with the
    config-resolved max_iterations, and on a no-cache parent the replayed
    conversation_history must be the bounded digest — not the full snapshot.
    Mirrors the harness in test_background_review.py."""
    import agent.background_review as br_mod

    init_kwargs = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            init_kwargs.update(kwargs)
            self._session_messages = []

        def run_conversation(self, **kwargs):
            init_kwargs["_replayed_history"] = kwargs.get("conversation_history")

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    from tests.run_agent.test_background_review import _bare_agent

    parent = _bare_agent()
    # The burn signature: real traffic, zero provider cache reads.
    parent.session_api_calls = 50
    parent.session_cache_read_tokens = 0
    parent._cached_system_prompt = None
    parent.prefill_messages = []
    parent.providers_allowed = None

    big_snapshot = []
    for i in range(80):
        big_snapshot.append({"role": "user", "content": f"turn {i}: " + "payload " * 60})
        big_snapshot.append({"role": "assistant", "content": f"reply {i}: " + "payload " * 50})

    cfg = {"auxiliary": {"background_review": {"max_iterations": 4}}}

    with patch("run_agent.AIAgent", FakeReviewAgent), \
         patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        br_mod._run_review_in_thread(
            parent, big_snapshot, "Review the conversation above.", cfg
        )

    assert init_kwargs["max_iterations"] == 4, "fork budget must come from config"
    replayed = init_kwargs["_replayed_history"]
    assert replayed is not big_snapshot, "no-cache parent must NOT get full replay"
    assert replayed[0]["role"] == "user"
    assert replayed[0]["content"].startswith("[Earlier conversation digest")
    assert "earlier turns omitted" in replayed[0]["content"] or len(replayed) <= 41


# ---------------------------------------------------------------------------
# Cache-adaptive replay policy
# ---------------------------------------------------------------------------

def _big_snapshot(n: int = 60) -> list:
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"u{i} " + "x" * 200})
        msgs.append({"role": "assistant", "content": f"a{i} " + "y" * 200})
    return msgs


def test_full_replay_when_cache_effective_same_model():
    snap = _big_snapshot()
    out = br._review_history_for_fork(snap, routed=False, parent_cache_effective=True)
    assert out is snap


def test_digest_when_routed_even_if_cache_effective():
    snap = _big_snapshot()
    out = br._review_history_for_fork(snap, routed=True, parent_cache_effective=True)
    assert out is not snap
    assert out[0]["role"] == "user" and out[0]["content"].startswith("[Earlier conversation digest")


def test_digest_when_parent_provider_has_no_cache_reads():
    """THE burn case: ollama-cloud-style provider reporting cache_read=0 on
    every response. Same-model fork must NOT replay the full transcript."""
    snap = _big_snapshot()
    out = br._review_history_for_fork(snap, routed=False, parent_cache_effective=False)
    assert out is not snap
    assert out[0]["content"].startswith("[Earlier conversation digest")


class _ParentProbe:
    """Minimal parent-attr shell for cache-probe tests."""

    def __init__(self, api_calls: int = 0, cache_read: int = 0):
        self.session_api_calls = api_calls
        self.session_cache_read_tokens = cache_read


def test_parent_cache_effective_zero_api_calls_is_unknown_true():
    assert br._parent_cache_effective(_ParentProbe(api_calls=0, cache_read=0)) is True


def test_parent_cache_effective_reads_parent_counters():
    warm = _ParentProbe(api_calls=50, cache_read=480_000)
    assert br._parent_cache_effective(warm) is True

    cold = _ParentProbe(api_calls=50, cache_read=0)
    assert br._parent_cache_effective(cold) is False


def test_parent_cache_probe_failure_fails_open_to_full_replay():
    parent = object()  # no attrs at all
    assert br._parent_cache_effective(parent) is True


# ---------------------------------------------------------------------------
# Integration: make_tool_result_message feeds the installed breaker
# ---------------------------------------------------------------------------

def test_make_tool_result_message_trips_breaker():
    """The real funnel: a tool result built inside an installed breaker's
    context must reach it — this is the seam the review fork relies on."""
    from agent.tool_dispatch_helpers import make_tool_result_message

    agent = _FakeReviewAgent(iterations=4)
    breaker = install_review_breaker(agent, threshold=2)
    try:
        refusal_payload = json.dumps({
            "error": "Refusing background curator edit for skill 'demo': "
                     "the current SKILL.md content has not been loaded in this "
                     "review turn. Call skill_view(name) ... then retry."
        })
        make_tool_result_message("skill_manage", refusal_payload, "call_1")
        make_tool_result_message("skill_manage", refusal_payload, "call_2")
        assert breaker.aborted is True
        assert agent.iteration_budget.remaining == 0
    finally:
        reset_review_breaker(breaker)


def test_make_tool_result_message_unaffected_outside_review():
    from agent.tool_dispatch_helpers import make_tool_result_message

    msg = make_tool_result_message(
        "skill_manage", json.dumps({"success": True}), "call_1"
    )
    assert msg["role"] == "tool"
    assert msg["tool_name"] == "skill_manage"


# ---------------------------------------------------------------------------
# Acceptance math for the task card
# ---------------------------------------------------------------------------

def test_acceptance_refusing_run_terminates_within_three_refused_calls():
    """Card acceptance: a refusing run terminates within 3 refused calls.

    threshold = max(2, max_iterations // 2). With the default budget of 4,
    threshold = 2 — the third refusal can never be SENT because the drain
    after the 2nd refusal leaves the loop with zero remaining iterations.
    """
    agent = _FakeReviewAgent(iterations=4)
    threshold = max(2, 4 // 2)
    breaker = RefusalBreaker(agent, threshold=threshold)
    refusal = json.dumps({"error": "Refusing background curator edit for skill 'x'"})
    breaker.observe("skill_manage", refusal)   # refusal 1 — sent to model
    breaker.observe("skill_manage", refusal)   # refusal 2 — sent, then drain
    assert agent.iteration_budget.remaining == 0
    assert breaker.aborted is True
    assert breaker.refusals <= 3


def test_acceptance_large_session_digest_stays_under_budget():
    """Card acceptance: a review fire on a large session stays under ~150k
    input tokens total. Simulate the burn session shape: ~60k tokens of
    history, digest caps the replay at the tail (24 msgs verbatim + digest).
    4 iterations × digest replay ≪ 150k even with zero provider caching."""
    snap = []
    for i in range(400):  # ~60k tokens of history at ~150 tok/msg
        snap.append({"role": "user", "content": f"turn {i}: " + "payload " * 60})
        snap.append({"role": "assistant", "content": f"reply {i}: " + "payload " * 50})

    replay = br._review_history_for_fork(
        snap, routed=False, parent_cache_effective=False
    )
    assert replay is not snap

    def _estimate_tokens(msgs):
        return sum(len(str(m.get("content", ""))) // 4 for m in msgs)

    per_call = _estimate_tokens(replay)
    total = per_call * 4  # max_iterations default
    assert total < 150_000, f"digest replay must stay under 150k total; got {total}"
    # And the OLD policy would have blown it (sanity: the test discriminates):
    full_total = _estimate_tokens(snap) * 4
    assert full_total > 150_000