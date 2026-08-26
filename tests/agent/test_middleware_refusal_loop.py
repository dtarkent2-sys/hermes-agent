"""Regression: a request-middleware refusal must not spin the conversation loop.

Live incident (Aug 2026): a Telegram session switched to ``stealth/ox-alpha``
on a 65,536-token context budget (``safe_context config.default``). Starting
19:16:50 ET every turn logged ``preflight.compact`` then a middleware refusal,
but the conversation loop continued dispatching model/tool calls and the prompt
grew from 60,596 to >110,465 tokens (>168%) over minutes.

Root cause, in two layers:

1. The safe-context ``PreflightMiddleware`` signals that a request cannot fit by
   raising ``ContextBudgetExceeded`` — but ``PluginManager.invoke_middleware``
   swallowed *every* middleware exception as a plugin bug, logging a warning
   and returning ``[]``. The refusal never reached the loop.
2. Even when the loop's ``apply_llm_request_middleware`` call is reached, the
   bare ``except Exception`` there swallowed the refusal too.

The fix:
- ``MiddlewareRequestRefused`` (hermes_cli/middleware.py) is the explicit
  "do not dispatch this request" contract. ``PluginManager.invoke_middleware``
  and the loop's ``apply_llm_request_middleware`` call site propagate it.
- The loop's ``except Exception as api_error`` handler routes a refusal into
  the SAME bounded compression recovery as a provider 413: compress history
  and re-dispatch up to ``max_compression_attempts``; if compression cannot
  make the request fit, end the turn as a terminal ``compression_exhausted``
  failure instead of spinning.

These tests exercise the loop end-to-end against an in-process mock provider.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

# Repo root = three levels up from tests/agent/<file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hermes_cli.middleware import MiddlewareRequestRefused


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        req = json_loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        is_stream = req.get("stream") is True
        if type(self).response_queue:
            resp = type(self).response_queue.pop(0)
        else:
            resp = _text_resp("DONE")
        msg = resp["choices"][0]["message"]
        if is_stream:
            content = msg.get("content") or ""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
            if content:
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            for c in chunks:
                self.wfile.write(f"data: {json_dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json_dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):  # silence default stderr logging
        pass


def json_loads(s):
    import json
    return json.loads(s)


def json_dumps(o):
    import json
    return json.dumps(o)


@pytest.fixture()
def refusing_agent_env():
    """Mock provider + an agent wired with a refusing request middleware."""
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_refusal_loop_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    import hermes_cli.plugins as plugins_mod
    from run_agent import AIAgent

    # NOTE: we deliberately do NOT purge ``hermes_*`` modules from
    # ``sys.modules`` here. ``MiddlewareRequestRefused`` is compared by class
    # identity across the plugin manager and the conversation loop; reloading
    # ``hermes_cli.plugins`` after this test file imported the class would give
    # the middleware a *different* class object than the running
    # ``invoke_middleware`` checks, so the refusal would fall through to the
    # generic ``except Exception`` and be swallowed — masking the very bug this
    # test guards against.

    # Inject the refusing callback into the ACTIVE delivery manager's
    # llm_request middleware list. Changing HERMES_HOME above already gave us a
    # fresh per-home manager; we append to its live dict.
    manager = plugins_mod.get_plugin_manager()
    refusing, control = _build_refusing_preflight()
    manager._middleware.setdefault("llm_request", []).append(refusing)

    agent = AIAgent(
        api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
        provider="openai-compat", model="test-model",
        max_iterations=10, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
    )
    agent._compression_feasibility_checked = True

    try:
        yield agent, _MockHandler, manager, refusing, control
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home
        # Remove the injected callback so it does not leak into later tests.
        try:
            callbacks = manager._middleware.get("llm_request", [])
            if refusing in callbacks:
                callbacks.remove(refusing)
        except Exception:
            pass


def _build_refusing_preflight():
    """Build a controllable refusing request middleware + its control object.

    Returns ``(middleware, control)`` where ``control.refuse`` is a callable
    taking ``(request, **context)`` and returning True to refuse. Mirrors
    safe-context's ``PreflightMiddleware``, which raises ``ContextBudgetExceeded``
    (a subclass of ``MiddlewareRequestRefused``) to signal "do not dispatch".
    """

    class _Control:
        refuse = None  # set by the test; callable(request, **kw) -> bool

    control = _Control()

    def _middleware(request=None, **context):
        if control.refuse is not None and control.refuse(request, **context):
            raise MiddlewareRequestRefused(
                "safe_context refused: prompt exceeds limit 200 tokens"
            )
        return None

    return _middleware, control


def test_refusal_is_bounded_after_compression_exhausted(refusing_agent_env):
    """A persistent refusal ends the turn as a terminal failure, not a loop.

    The middleware refuses every dispatch (compression cannot bring the
    transcript below the tiny 200-token budget). The loop must compress up to
    max_compression_attempts (default 3) then STOP the turn with
    ``compression_exhausted=True`` — it must not dispatch the oversized
    request, and it must not spin forever.
    """
    agent, handler, _mgr, _refusing, control = refusing_agent_env
    control.refuse = lambda request=None, **kw: True  # refuse every dispatch

    # Compression can never succeed at this budget: stub _compress_context to
    # make a real attempt that returns a transcript still over the limit.
    original_compress = agent._compress_context
    compress_calls = []

    def _never_enough(msgs, sys_msg, **kw):
        compress_calls.append(1)
        # Return a transcript of the same size (still > 200 tokens) — i.e.
        # compression made no progress, as would happen for an incompressible
        # history.
        return list(msgs), sys_msg

    agent._compress_context = _never_enough

    try:
        result = agent.run_conversation(
            "This is a long user message that will push the prompt far over the 200 token budget. "
            "x" * 500,
            conversation_history=[],
            task_id="t",
        )
    finally:
        agent._compress_context = original_compress

    # The oversized conversation request must never reach the provider: a
    # refusal means "do not dispatch". The mock server would have recorded it
    # as a chat-completions POST with our user message. (A bare model-name
    # probe — the context-length lookup — is unrelated and may be present.)
    dispatched = [
        req
        for req in handler.captured_requests
        if isinstance(req, dict) and req.get("messages")
    ]
    assert dispatched == [], (
        "refused request must not be dispatched; captured=%d"
        % len(handler.captured_requests)
    )
    # Bounded: at least one compression pass ran and the loop STOPPED —
    # it did not spin re-dispatching the oversized request. (A stub that
    # makes no progress terminates after the first pass; a stub that
    # shrinks partway could take more, up to the max. The invariant is
    # bounded-and-terminating, not a specific count.)
    assert 1 <= len(compress_calls) <= 3, (
        f"expected bounded compressions, got {len(compress_calls)}"
    )
    # Terminal failure, not a hang.
    assert result.get("failed") is True
    assert result.get("compression_exhausted") is True
    assert "context limit" in (result.get("final_response") or "")


def test_refusal_recovers_when_compression_succeeds(refusing_agent_env):
    """When compression brings the prompt under budget, the loop re-dispatches.

    The refusal is a signal to compress, not an unconditional abort: if one
    compression pass is enough, the turn proceeds normally.
    """
    agent, handler, _mgr, _refusing, control = refusing_agent_env
    # Refuse exactly the first dispatch (the oversized prompt), then allow —
    # mirroring compression bringing the transcript under budget.
    refused_budget = {"remaining": 1}

    def _refuse_once(request=None, **kw):
        if refused_budget["remaining"] > 0:
            refused_budget["remaining"] -= 1
            return True
        return False

    control.refuse = _refuse_once

    original_compress = agent._compress_context
    compress_calls = []

    def _shrinks(msgs, sys_msg, **kw):
        compress_calls.append(1)
        # Compression succeeds: return a tiny transcript under the 200-token
        # budget so the next preflight no longer refuses.
        return [
            {"role": "system", "content": sys_msg or ""},
            {"role": "user", "content": "compacted"},
        ], sys_msg

    agent._compress_context = _shrinks

    # Provider answers the recovered request.
    handler.response_queue.append(_text_resp("recovered after compression"))

    try:
        result = agent.run_conversation(
            "This is a long user message that will push the prompt far over the 200 token budget. "
            "x" * 500,
            conversation_history=[],
            task_id="t",
        )
    finally:
        agent._compress_context = original_compress

    assert len(compress_calls) == 1, "one compression pass was enough"
    assert result.get("failed") is not True
    assert "recovered" in (result.get("final_response") or "")
