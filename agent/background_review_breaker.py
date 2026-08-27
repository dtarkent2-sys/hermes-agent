"""Refusal-streak circuit breaker for the background skill-review fork.

The review fork's only tools are memory/skill management, and its curator
guards refuse disallowed writes with an EXPLICIT, actionable error ("call
skill_view first, then retry" / "pinned skills are off-limits"). A healthy
run absorbs one refusal and adapts. A stuck run retries the same refused
call until the iteration cap — each retry re-sends the fork's whole history
to the provider. On providers that report no prompt-cache activity
(cache_read=0 on every response) that multiplies into million-token burns
(2026-08-27: two fires, 2.24M input tokens, result=none both times).

This module gives the review run a cheap circuit breaker: every tool result
built inside the fork's context is observed once (see
``agent.tool_dispatch_helpers.make_tool_result_message``, the single funnel
all executor paths use), curator refusals are counted, and when the count
reaches the configured threshold the breaker drains the fork's
``IterationBudget``. The conversation loop's own budget check then exits the
run cleanly at the top of the next iteration — no exception, so the
existing finally-blocks (usage attribution, unregister, close) all still
run.

Thread-safety model: the breaker rides a ContextVar installed on the fork's
worker thread right before ``run_conversation``. Concurrent forks (different
sessions) run on different threads and never see each other's breaker; the
concurrent tool executor's workers inherit the ContextVar through
``tools.thread_context.propagate_context_to_thread``'s ``copy_context``.

The observer is a no-op (one ContextVar lookup) for every non-review tool
result in the process, and never raises into tool dispatch.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Substrings that identify a curator-guard / review-whitelist refusal in a
# tool-result payload. Kept in ONE place here rather than duplicated from the
# guard modules — the guards own their wording; this is a diagnostic tripwire,
# not a security boundary (the guards themselves fail closed regardless).
_REFUSAL_MARKERS = (
    "Refusing background curator",
    "Background review denied non-whitelisted tool",
)


def _is_curator_refusal(content: Any) -> bool:
    """Return True when a tool result is a curator-guard/whitelist refusal.

    Accepts the raw tool-result content (JSON string or already-parsed dict).
    Conservative by design: unparseable payloads and non-refusal errors never
    count, so ordinary tool failures cannot trip the breaker.
    """
    data = content
    if isinstance(data, (str, bytes, bytearray)):
        try:
            data = json.loads(data)
        except Exception:
            return False
    if not isinstance(data, dict):
        return False
    if data.get("_read_before_write_required"):
        return True
    err = data.get("error")
    if isinstance(err, str) and any(m in err for m in _REFUSAL_MARKERS):
        return True
    return False


class RefusalBreaker:
    """Counts curator refusals inside one review run; aborts at threshold.

    "Abort" = drain the review agent's :class:`~agent.iteration_budget.IterationBudget`
    so the conversation loop exits at its next budget check. Nothing is
    raised mid-tool: the current batch finishes, the model's response to the
    final refusal is generated, and the run tears down through the normal
    finally paths (usage attribution included).
    """

    def __init__(self, agent: Any, threshold: int):
        self._agent = agent
        self.threshold = max(1, int(threshold))
        self.refusals = 0
        self.aborted = False
        self.abort_reason: Optional[str] = None
        self._token: Optional[contextvars.Token] = None

    # -- lifecycle ---------------------------------------------------------

    def install(self) -> "RefusalBreaker":
        self._token = _breaker_var.set(self)
        return self

    def reset(self) -> None:
        """Clear this breaker from the current context (idempotent)."""
        if self._token is not None:
            try:
                _breaker_var.reset(self._token)
            except Exception:
                logger.debug(
                    "review refusal-breaker context reset failed", exc_info=True
                )
            self._token = None

    # -- observation -------------------------------------------------------

    def observe(self, tool_name: str, content: Any) -> None:
        """Record one tool result; drain the budget at the threshold.

        Never raises — a breaker fault must never break tool dispatch.
        """
        try:
            if not _is_curator_refusal(content):
                return
            self.refusals += 1
            if self.refusals < self.threshold:
                logger.info(
                    "Background review refusal %d/%d on tool '%s'",
                    self.refusals, self.threshold, tool_name,
                )
                return
            self._abort(tool_name)
        except Exception:
            logger.debug("refusal breaker observe failed", exc_info=True)

    def _abort(self, tool_name: str) -> None:
        self.aborted = True
        self.abort_reason = f"{self.refusals} refused curator calls (last tool: {tool_name})"
        agent = self._agent
        drained = 0
        try:
            budget = getattr(agent, "iteration_budget", None)
            if budget is not None:
                while budget.consume():
                    drained += 1
            # Neutralize a pending grace call so the drain cannot be
            # circumvented by the loop's one-call mercy path.
            try:
                agent._budget_grace_call = False
            except Exception:
                pass
        except Exception:
            logger.debug("refusal breaker drain failed", exc_info=True)
        logger.warning(
            "Background review aborted after %s — likely a curator refusal "
            "retry loop; iteration budget drained (%d remaining iterations "
            "revoked). The refusal messages themselves already told the "
            "model how to proceed.",
            self.abort_reason, drained,
        )


_breaker_var: contextvars.ContextVar[Optional[RefusalBreaker]] = contextvars.ContextVar(
    "background_review_refusal_breaker", default=None
)


def install_review_breaker(agent: Any, threshold: int) -> RefusalBreaker:
    """Create a breaker for one review run and bind it to this context."""
    return RefusalBreaker(agent, threshold).install()


def reset_review_breaker(breaker: Optional[RefusalBreaker]) -> None:
    """Unbind the run's breaker (call in the run's finally)."""
    if breaker is not None:
        breaker.reset()


def observe_tool_result(tool_name: str, content: Any) -> None:
    """Feed one tool result to the active breaker, if any.

    Called from ``make_tool_result_message`` — the single funnel every tool
    result passes through. One ContextVar lookup on the no-review hot path.
    """
    breaker = _breaker_var.get()
    if breaker is not None:
        breaker.observe(tool_name, content)


__all__ = [
    "RefusalBreaker",
    "install_review_breaker",
    "reset_review_breaker",
    "observe_tool_result",
    "_is_curator_refusal",
]