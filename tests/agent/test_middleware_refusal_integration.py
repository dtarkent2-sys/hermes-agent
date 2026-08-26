"""Integration regression: the real safe-context middleware bounds the refusal loop.

Live incident (Aug 2026): Telegram DM session ``20260825_191135_103417d0``
switched mid-session to ``stealth/ox-alpha`` — an *unconfigured* model. The
safe-context resolver fell back to ``config.default`` = 65,536 tokens, so
``PreflightMiddleware`` began refusing every oversized dispatch. But the
conversation loop kept dispatching the oversized prompt and the transcript
grew 60,596 -> >110,465 tokens.

Unlike the unit tests in this module (which inject a *generic*
``MiddlewareRequestRefused`` callback), this test drives the REAL
``safe_context`` plugin end-to-end through the host:

- the actual ``safe-context`` plugin is copied into a temp ``HERMES_HOME`` and
  loaded by host plugin discovery (``config.default.context_limit = 65,536``),
- the agent runs a long conversation turn against an in-process mock provider,
- mid-session the agent's model is switched to an *unconfigured* id, so the
  next dispatch re-resolves the budget via ``config.default`` = 65,536 and the
  real middleware refuses,
- we assert zero oversized requests reach the provider and recovery is bounded.

The core fix under test: ``PluginManager.invoke_middleware`` and the loop's
``apply_llm_request_middleware`` propagate the explicit refusal (now a
subclass of ``MiddlewareRequestRefused``), and the loop's error handler routes
it into bounded compression recovery (compress + re-dispatch up to
``max_compression_attempts``, then a terminal ``compression_exhausted`` stop)
instead of spinning.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

# Repo root = three levels up from tests/agent/<file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#: The real safe-context plugin lives under the SharkQuant component dir
#: (symlinked into the live ~/.hermes/plugins/). Resolve it from the live
#: plugin tree when present, else skip (the plugin is not part of this repo).
_LIVE_PLUGIN_DIR = Path(
    os.environ.get(
        "SAFE_CONTEXT_PLUGIN_DIR",
        r"C:/Users/dtark/AppData/Local/hermes/plugins/safe-context",
    )
)

#: An unconfigured model id: safe-context must resolve it via config.default.
UNCONFIGURED_MODEL = "stealth/ox-alpha"
_BASE_URL = "https://inference-api.nousresearch.com/v1"


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
        req = json.loads(self.rfile.read(length).decode())
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
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):  # silence default stderr logging
        pass


@pytest.fixture()
def real_safe_context_env():
    """A host with the REAL safe-context plugin loaded, driving a mock provider.

    Yields ``(agent, handler, manager)``. ``manager._middleware['llm_request']``
    contains the genuine ``PreflightMiddleware`` with ``config.default=65,536``.
    """
    if not _LIVE_PLUGIN_DIR.is_dir():
        pytest.skip(f"safe-context plugin not found at {_LIVE_PLUGIN_DIR}")

    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = Path(tempfile.mkdtemp(prefix="hermes_sc_integration_"))
    os.makedirs(test_home / "plugins", exist_ok=True)
    os.makedirs(test_home / "logs", exist_ok=True)
    # Copy the REAL plugin into the temp home's plugin tree.
    shutil.copytree(_LIVE_PLUGIN_DIR, test_home / "plugins" / "safe-context")

    config = {
        "safe_context": {
            "enabled": True,
            "default": {
                "context_limit": 65536,
                "max_output_tokens": 8192,
                "output_reserve": 4096,
                "safety_margin": 1024,
                "min_output_tokens": 512,
                "compact_threshold": 0.72,
            },
        },
        "plugins": {"enabled": ["safe-context"]},
        "model": {
            "provider": "openai-compat",
            "base_url": f"http://127.0.0.1:{port}/v1",
            "default": "some-model",
        },
    }
    (test_home / "config.yaml").write_text(json.dumps(config))
    (test_home / ".env").write_text("")

    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(test_home)

    # NOTE: we deliberately do NOT purge ``hermes_*`` modules from
    # ``sys.modules`` here. Changing HERMES_HOME already gives ``get_plugin_manager``
    # a fresh per-home manager (managers are cached keyed on the resolved home),
    # and purging ``hermes_cli.plugins`` would break module-level state that other
    # test files (e.g. ``test_plugins.py``) rely on. ``MiddlewareRequestRefused`` is
    # also compared by class identity, so reloading the module would mask the bug
    # under test.
    import hermes_cli.plugins as plugins_mod
    from run_agent import AIAgent

    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    llm_mw = manager._middleware.get("llm_request", [])
    assert any(type(cb).__name__ == "PreflightMiddleware" for cb in llm_mw), (
        "real safe-context PreflightMiddleware was not registered by host discovery"
    )

    agent = AIAgent(
        api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
        provider="openai-compat", model="some-model",
        max_iterations=10, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
    )
    agent._compression_feasibility_checked = True

    try:
        yield agent, _MockHandler, manager
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _oversized_dispatches(captured: list) -> list:
    """Chat-completions requests that carried a message set (not the model probe)."""
    return [req for req in captured if isinstance(req, dict) and req.get("messages")]


def test_real_middleware_refuses_after_switch_to_unconfigured_65k(
    real_safe_context_env,
):
    """A mid-session switch to an unconfigured 65,536 model must never dispatch
    an oversized request and must terminate (bounded), not spin."""
    agent, handler, _mgr = real_safe_context_env

    # The loop must call _compress_context. With a 65,536 budget and a prompt
    # pushed far over it, real compression cannot shrink below the refusal
    # threshold deterministically, so stub it to make no progress — mirroring
    # an incompressible transcript. Track the attempt count for the cap assert.
    original_compress = agent._compress_context
    compress_calls = []

    def _no_progress(msgs, sys_msg, **kw):
        compress_calls.append(1)
        return list(msgs), sys_msg

    agent._compress_context = _no_progress

    # Long transcript: large enough that the assembled prompt exceeds
    # config.default (65,536) minus the output reserve, so the real
    # PreflightMiddleware REFUSES (compaction signal) rather than clamps.
    # The estimator counts ~1 token / 4 chars; ~270k chars ≈ 67k tokens,
    # safely over the ~60.4k planning budget.
    long_turn = (
        "Analyze this large corpus and summarize the key findings. "
        + "word " * 67_000
    )

    try:
        # Mid-session switch to an UNCONFIGURED model. The safe-context
        # resolver must fall back to config.default = 65,536 on the next
        # dispatch, so PreflightMiddleware refuses the oversized prompt.
        agent.model = UNCONFIGURED_MODEL
        agent.base_url = _BASE_URL
        result = agent.run_conversation(
            long_turn,
            conversation_history=[],
            task_id="t",
        )
    finally:
        agent._compress_context = original_compress

    # The oversized request must NEVER reach the provider (mock records it).
    dispatched = _oversized_dispatches(handler.captured_requests)
    assert dispatched == [], (
        "refused oversized request must not be dispatched; captured=%d"
        % len(handler.captured_requests)
    )

    # Bounded recovery: at least one compression attempt, then terminal stop.
    assert 1 <= len(compress_calls) <= 3, (
        f"expected bounded compressions, got {len(compress_calls)}"
    )
    # Terminal failure (compression could not fit), not a hang.
    assert result.get("failed") is True
    assert result.get("compression_exhausted") is True
    assert "context limit" in (result.get("final_response") or "")


def test_real_middleware_recovers_when_compression_succeeds(real_safe_context_env):
    """One successful compact must re-dispatch normally (bounded recovery, not abort)."""
    agent, handler, _mgr = real_safe_context_env

    original_compress = agent._compress_context
    compress_calls = []

    def _shrinks(msgs, sys_msg, **kw):
        compress_calls.append(1)
        # Compression succeeds: tiny transcript far under 65,536.
        return [
            {"role": "system", "content": sys_msg or ""},
            {"role": "user", "content": "compacted summary"},
        ], sys_msg

    agent._compress_context = _shrinks
    handler.response_queue.append(_text_resp("recovered after compression"))

    try:
        agent.model = UNCONFIGURED_MODEL
        agent.base_url = _BASE_URL
        result = agent.run_conversation(
            "Analyze this large corpus and summarize the key findings. "
            + "word " * 67_000,
            conversation_history=[],
            task_id="t",
        )
    finally:
        agent._compress_context = original_compress

    # One compaction pass was enough to re-dispatch.
    assert len(compress_calls) == 1
    assert result.get("failed") is not True
    assert "recovered" in (result.get("final_response") or "")
    # The recovered (compacted) request DID reach the provider — bounded path.
    dispatched = _oversized_dispatches(handler.captured_requests)
    assert dispatched, "recovered request should be dispatched"
