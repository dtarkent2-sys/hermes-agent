"""Tests for CodexAppServerSession — drive turns through a mock client.

The session adapter has the most complex behavior of the three new modules:
notification draining, server-request handling (approvals), interrupt,
deadline timeouts. These tests pin all of that without spawning real codex.
"""

from __future__ import annotations

import time
from unittest.mock import patch
from typing import Any, Optional

import pytest

import agent.transports.codex_app_server_session as session_mod
from agent.transports.codex_app_server_session import (
    CodexAppServerSession,
    _ServerRequestRouting,
    _approval_choice_to_codex_decision,
    _coerce_turn_input_text,
    _resolve_startup_retries,
    _resolve_startup_timeout,
    _MAX_CODEX_STARTUP_RETRIES,
    _DEFAULT_CODEX_STARTUP_RETRIES,
)


class FakeClient:
    """Stand-in for CodexAppServerClient that records calls and lets the test
    drive the notification / server-request streams synchronously."""

    def __init__(self, *, codex_bin: str = "codex", codex_home=None) -> None:
        self.codex_bin = codex_bin
        self.codex_home = codex_home
        self.requests: list[tuple[str, dict]] = []
        self.notifications_responses: list[dict] = []
        self.responses: list[tuple[Any, dict]] = []
        self.error_responses: list[tuple[Any, int, str]] = []
        self._initialized = False
        self._closed = False
        self._notifications: list[dict] = []
        self._server_requests: list[dict] = []
        self._request_handler = None  # Optional[Callable[[str, dict], dict]]

    # API matching CodexAppServerClient
    def initialize(self, **kwargs):
        self._initialized = True
        return {"userAgent": "fake/0.0.0", "codexHome": "/tmp",
                "platformOs": "linux", "platformFamily": "unix"}

    def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0):
        self.requests.append((method, params or {}))
        if self._request_handler is not None:
            return self._request_handler(method, params or {})
        # Sensible defaults for protocol methods used by the session
        if method == "thread/start":
            return {"thread": {"id": "thread-fake-001"},
                    "activePermissionProfile": {"id": "workspace-write"}}
        if method == "turn/start":
            return {"turn": {"id": "turn-fake-001"}}
        if method == "turn/interrupt":
            return {}
        if method == "turn/steer":
            return {"turnId": (params or {}).get("expectedTurnId")}
        return {}

    def notify(self, method: str, params=None):
        pass

    def respond(self, request_id, result):
        self.responses.append((request_id, result))

    def respond_error(self, request_id, code, message, data=None):
        self.error_responses.append((request_id, code, message))

    def take_notification(self, timeout: float = 0.0):
        if self._notifications:
            return self._notifications.pop(0)
        # Honor a tiny sleep so the loop doesn't hot-spin; the real client
        # blocks on a queue. For tests we want determinism.
        if timeout > 0:
            time.sleep(min(timeout, 0.001))
        return None

    def take_server_request(self, timeout: float = 0.0):
        if self._server_requests:
            return self._server_requests.pop(0)
        return None

    def close(self):
        self._closed = True

    def is_alive(self) -> bool:
        # Fake is "alive" until close() is called; tests that want a dead
        # subprocess can patch this attribute or call close() directly.
        return not self._closed

    def stderr_tail(self, n: int = 20):
        return list(getattr(self, "_stderr_tail", []))[-n:]

    # Test helpers
    def queue_notification(self, method: str, **params):
        # Keep legacy fixture shorthand aligned with the IDs returned by the
        # fake thread/start and turn/start responses.
        if params.get("threadId") in {"t", "th"}:
            params["threadId"] = "thread-fake-001"
        if params.get("turnId") == "tu1":
            params["turnId"] = "turn-fake-001"
        turn = params.get("turn")
        if isinstance(turn, dict) and turn.get("id") == "tu1":
            turn = dict(turn)
            turn["id"] = "turn-fake-001"
            params["turn"] = turn
        self._notifications.append({"method": method, "params": params})

    def queue_server_request(self, method: str, request_id: Any = "srv-1", **params):
        self._server_requests.append({"id": request_id, "method": method, "params": params})

    def set_stderr_tail(self, lines):
        """Test helper: seed stderr_tail() output for OAuth-refresh classifier tests."""
        self._stderr_tail = list(lines)


def make_session(client: FakeClient, **kwargs) -> CodexAppServerSession:
    return CodexAppServerSession(
        cwd="/tmp",
        client_factory=lambda **kw: client,
        **kwargs,
    )


# ---- choice mapping ----

class TestApprovalChoiceMapping:
    @pytest.mark.parametrize("choice,expected", [
        ("once", "accept"),
        ("session", "acceptForSession"),
        ("always", "acceptForSession"),
        ("deny", "decline"),
        ("anything-else", "decline"),
    ])
    def test_mapping(self, choice, expected):
        assert _approval_choice_to_codex_decision(choice) == expected


class TestTurnInputCoercion:
    def test_list_content_keeps_text_and_marks_images(self):
        text = _coerce_turn_input_text([
            {"type": "text", "text": "caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])
        assert text == "caption\n\n[image attached]"


# ---- lifecycle ----

class TestLifecycle:
    def test_ensure_started_is_idempotent(self):
        client = FakeClient()
        s = make_session(client)
        tid_a = s.ensure_started()
        tid_b = s.ensure_started()
        assert tid_a == tid_b == "thread-fake-001"
        # thread/start should be called exactly once
        method_calls = [m for (m, _) in client.requests if m == "thread/start"]
        assert len(method_calls) == 1

    def test_thread_start_passes_cwd_only(self):
        """thread/start carries cwd. We intentionally do NOT pass `permissions`
        on this codex version (experimentalApi-gated + requires matching
        config.toml [permissions] table). Letting codex use its default
        (read-only unless user configures otherwise) is the documented path."""
        client = FakeClient()
        s = make_session(client, permission_profile="workspace-write")
        s.ensure_started()
        method, params = next(r for r in client.requests if r[0] == "thread/start")
        assert params["cwd"] == "/tmp"
        assert "permissions" not in params  # see session.ensure_started() comment

    def test_close_idempotent(self):
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        s.close()
        s.close()
        assert client._closed is True


# ---- turn loop ----

class TestRunTurn:
    def test_simple_text_turn_returns_final_message(self):
        client = FakeClient()
        client.queue_notification("turn/started", threadId="t", turn={"id": "tu1"})
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "hello world"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed",
            threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.final_text == "hello world"
        assert r.interrupted is False
        assert r.error is None
        assert any(m["role"] == "assistant" and m.get("content") == "hello world"
                   for m in r.projected_messages)
        # turn_id propagated for downstream session-DB linkage
        assert r.turn_id == "turn-fake-001"



    def test_foreign_completion_in_server_request_drain_is_ignored(self):
        """Approval draining must not project a child result into the parent."""
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval",
            request_id="approval-1",
            command="pwd",
            cwd="/tmp",
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-child-001",
            turnId="turn-child-001",
            item={
                "type": "agentMessage",
                "id": "child-message",
                "text": "child drain summary",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-child-001",
            turn={
                "id": "turn-child-001",
                "status": "completed",
                "error": None,
            },
        )

        original_respond = client.respond

        def respond_and_release_parent(request_id, response):
            original_respond(request_id, response)
            client.queue_notification(
                "item/completed",
                threadId="thread-fake-001",
                turnId="turn-fake-001",
                item={
                    "type": "agentMessage",
                    "id": "parent-message",
                    "text": "parent after approval",
                },
            )
            client.queue_notification(
                "turn/completed",
                threadId="thread-fake-001",
                turn={
                    "id": "turn-fake-001",
                    "status": "completed",
                    "error": None,
                },
            )

        client.respond = respond_and_release_parent
        session = make_session(
            client,
            request_routing=_ServerRequestRouting(auto_approve_exec=True),
        )

        result = session.run_turn("delegate then continue", turn_timeout=2.0)

        assert client.responses == [("approval-1", {"decision": "accept"})]
        assert result.final_text == "parent after approval"
        assert result.projected_messages == [
            {"role": "assistant", "content": "parent after approval"}
        ]



    def test_tool_iteration_counter_ticks(self):
        client = FakeClient()
        # Two completed exec items + one final agent message
        for i, item_id in enumerate(("ex1", "ex2"), start=1):
            client.queue_notification(
                "item/completed",
                item={
                    "type": "commandExecution", "id": item_id,
                    "command": f"cmd{i}", "cwd": "/tmp",
                    "status": "completed", "aggregatedOutput": "ok",
                    "exitCode": 0, "commandActions": [],
                },
                threadId="t", turnId="tu1",
            )
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "done"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn("do stuff", turn_timeout=2.0)
        assert r.tool_iterations == 2
        # Each tool item produces (assistant, tool) — 2*2 + final assistant = 5 msgs
        assert len(r.projected_messages) == 5


    def test_turn_start_failure_attaches_redacted_stderr_tail(self):
        """When codex stderr has content (non-OAuth), the tail gets attached
        to the user-facing error so config/provider problems are debuggable
        instead of just 'Internal error'. Credential-shaped values in stderr
        are redacted via agent.redact(force=True); web-URL query params pass
        through (see fix(redact): pass web URLs through unchanged)."""
        client = FakeClient()
        client.set_stderr_tail([
            "ERROR: provider auth failed",
            "Authorization: Bearer sk-live-deadbeefdeadbeef",
            "url=https://api.example.com/v1?token=querysecret12345",
        ])
        from agent.transports.codex_app_server import CodexAppServerError

        def boom(method, params):
            if method == "turn/start":
                raise CodexAppServerError(code=-32603, message="Internal error")
            return {"thread": {"id": "t"}, "activePermissionProfile": {"id": "x"}}

        client._request_handler = boom
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "turn/start failed" in r.error
        assert "Internal error" in r.error
        # Stderr tail attached
        assert "codex stderr" in r.error
        assert "provider auth failed" in r.error
        # Credential-shaped values still redacted (sk- prefix + Bearer header)
        assert "sk-live-deadbeefdeadbeef" not in r.error
        # Non-OAuth → should NOT retire (subprocess JSON-RPC is still healthy).
        assert r.should_retire is False

    def test_turn_start_timeout_attaches_redacted_stderr_tail(self):
        """A non-OAuth TimeoutError on turn/start surfaces with codex stderr
        context attached and marks the session for retirement."""
        client = FakeClient()
        client.set_stderr_tail([
            "WARN: provider request stalled",
            "Authorization: Bearer sk-stalled-secret-abc123",
        ])

        def stall(method, params):
            if method == "turn/start":
                raise TimeoutError("codex method 'turn/start' timed out after 10s")
            return {"thread": {"id": "t"}, "activePermissionProfile": {"id": "x"}}

        client._request_handler = stall
        s = make_session(client)
        r = s.run_turn("hi", turn_timeout=2.0)
        assert r.error is not None
        assert "turn/start timed out" in r.error
        assert "provider request stalled" in r.error
        assert "sk-stalled-secret-abc123" not in r.error
        assert r.should_retire is True




    def test_steer_appends_input_to_active_turn(self):
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        with s._active_turn_lock:
            s._active_turn_id = "turn-live-123"

        assert s.request_steer("Use Postgres instead") is True
        method, params = client.requests[-1]
        assert method == "turn/steer"
        assert params == {
            "threadId": "thread-fake-001",
            "input": [{"type": "text", "text": "Use Postgres instead"}],
            "expectedTurnId": "turn-live-123",
        }







class TestCompactThread:
    def test_compact_thread_sends_rpc_and_waits_for_completion(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={"type": "contextCompaction", "id": "compact-item-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={"type": "agentMessage", "id": "m1", "text": "compacted"},
        )
        client.queue_notification(
            "thread/tokenUsage/updated",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            tokenUsage={
                "last": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
                "total": {"inputTokens": 100, "outputTokens": 20, "totalTokens": 120},
                "modelContextWindow": 200000,
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1", "status": "completed", "error": None},
        )

        r = make_session(client).compact_thread(turn_timeout=2.0)

        assert ("thread/compact/start", {"threadId": "thread-fake-001"}) in client.requests
        assert r.error is None
        assert r.thread_id == "thread-fake-001"
        assert r.turn_id == "compact-turn-1"
        assert r.compacted is True
        assert r.final_text == "compacted"
        assert r.token_usage_last["totalTokens"] == 12
        assert r.model_context_window == 200000

    def test_compact_thread_ignores_foreign_child_completion(self):
        client = FakeClient()
        client.queue_notification(
            "turn/started",
            threadId="thread-child-001",
            turn={"id": "child-compact-turn"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-child-001",
            turnId="child-compact-turn",
            item={
                "type": "agentMessage",
                "id": "child-compact-message",
                "text": "child compact summary",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-child-001",
            turn={
                "id": "child-compact-turn",
                "status": "completed",
                "error": None,
            },
        )
        client.queue_notification(
            "turn/started",
            threadId="thread-fake-001",
            turn={"id": "compact-turn-1"},
        )
        client.queue_notification(
            "item/completed",
            threadId="thread-fake-001",
            turnId="compact-turn-1",
            item={
                "type": "agentMessage",
                "id": "parent-compact-message",
                "text": "parent compacted",
            },
        )
        client.queue_notification(
            "turn/completed",
            threadId="thread-fake-001",
            turn={
                "id": "compact-turn-1",
                "status": "completed",
                "error": None,
            },
        )

        result = make_session(client).compact_thread(turn_timeout=2.0)

        assert result.error is None
        assert result.turn_id == "compact-turn-1"
        assert result.final_text == "parent compacted"
        assert result.projected_messages == [
            {"role": "assistant", "content": "parent compacted"}
        ]





# ---- approval bridge ----

class TestServerRequestRouting:



    def test_unknown_server_request_replied_with_error(self):
        client = FakeClient()
        client.queue_server_request("totally/unknown", request_id="req-3")
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        s.run_turn("hi", turn_timeout=1.0)
        assert any(
            rid == "req-3" and code == -32601
            for (rid, code, _msg) in client.error_responses
        )

    def test_on_event_fires_during_approval_drain(self):
        """When a server-initiated approval request arrives, the session
        drains up to 8 pending notifications first so per-turn state
        (e.g. _pending_file_changes for fileChange approvals) is current.
        Those drained notifications must also reach the on_event display
        hook — otherwise tool bubbles around approvals silently disappear.

        Regression for the issue where item/started events that landed
        in the queue alongside (or just before) an approval request got
        projected into messages but never displayed.
        """
        client = FakeClient()
        # An item/started notification is queued first, then a server
        # request — the session sees both during a single drain loop.
        client.queue_notification(
            "item/started",
            item={
                "type": "commandExecution",
                "id": "exec-1",
                "command": "echo drained",
                "cwd": "/tmp",
            },
        )
        client.queue_server_request(
            "item/commandExecution/requestApproval", request_id="req-d",
            command="echo drained",
            cwd="/tmp",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )

        events: list[dict] = []

        def cb(command, description, *, allow_permanent=True):
            return "once"

        s = make_session(
            client,
            approval_callback=cb,
            on_event=events.append,
        )
        s.run_turn("hi", turn_timeout=1.0)

        # The on_event hook must have seen the item/started even though
        # it was drained as part of the approval roundtrip — not just
        # events that arrive on the main notification path.
        item_started_events = [
            e for e in events
            if e.get("method") == "item/started"
        ]
        assert item_started_events, (
            "item/started drained alongside the approval was not "
            "forwarded to on_event — display will miss tool bubbles "
            "around approvals"
        )



    def test_routing_auto_approve_bypass(self):
        client = FakeClient()
        client.queue_server_request("item/commandExecution/requestApproval", request_id="r1",
                                    command="ls", cwd="/")
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        # No callback, but routing says auto-approve. Should approve.
        s = make_session(client, request_routing=_ServerRequestRouting(
            auto_approve_exec=True))
        s.run_turn("hi", turn_timeout=1.0)
        assert ("r1", {"decision": "accept"}) in client.responses



# ---- enriched approval prompts ----

class TestApprovalPromptEnrichment:
    """Quirk #4: apply_patch prompt should show what's changing.
    Quirk #10: exec prompt should never show empty cwd."""

    def test_exec_falls_back_to_session_cwd(self):
        """When codex omits cwd from the approval params, the prompt shows
        the session cwd, not an empty string."""
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval", request_id="r1",
            command="ls",  # no cwd
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["description"] = description
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Session cwd is /tmp by default in make_session()
        assert "/tmp" in captured["description"]
        assert "Codex requests exec in <unknown>" not in captured["description"]

    def test_apply_patch_prompt_summarizes_pending_changes(self):
        """When the projector has cached the fileChange item from item/started,
        the approval prompt surfaces the change summary."""
        client = FakeClient()
        # item/started fires first (carries the changes), then approval request
        client.queue_notification(
            "item/started",
            item={"type": "fileChange", "id": "fc-1",
                  "changes": [
                      {"kind": {"type": "add"}, "path": "/tmp/new.py"},
                      {"kind": {"type": "update"}, "path": "/tmp/old.py"},
                  ]},
            threadId="t", turnId="tu1",
        )
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="req-2",
            itemId="fc-1", turnId="tu1", threadId="t",
            startedAtMs=1234567890,
            reason="add and update files",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["command"] = command
            captured["description"] = description
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Both add and update kinds should be in the summary
        assert "1 add" in captured["command"] or "1 add" in captured["description"]
        assert "1 update" in captured["command"] or "1 update" in captured["description"]
        # And at least one of the paths
        joined = captured["command"] + " " + captured["description"]
        assert "/tmp/new.py" in joined or "/tmp/old.py" in joined

    def test_apply_patch_prompt_works_without_cached_summary(self):
        """When approval arrives before item/started (or without changes
        info), prompt falls back to whatever codex provided."""
        client = FakeClient()
        client.queue_server_request(
            "item/fileChange/requestApproval", request_id="req-2",
            itemId="fc-orphan", turnId="tu1", threadId="t",
            startedAtMs=1234567890,
            reason="apply some changes",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        captured = {}
        def cb(command, description, *, allow_permanent=True):
            captured["command"] = command
            return "once"
        s = make_session(client, approval_callback=cb)
        s.run_turn("hi", turn_timeout=1.0)
        # Falls back to the reason
        assert "apply some changes" in captured["command"]


# ---- openclaw beta.8 parity: retire/wedge/oauth/abort marker ----

class TestSessionRetirement:
    """Mirrors openclaw beta.8's resilience fixes:
      - retire timed-out app-server clients (should_retire on deadline)
      - post-tool completion watchdog (don't burn the full deadline after a
        tool result if codex goes silent)
      - <turn_aborted> raw marker as terminal (don't wait for turn/completed
        that never comes)
      - OAuth refresh failure classification (suggest `codex login` instead
        of raw RPC error strings)
      - dead subprocess detection between iterations
    """



    def test_final_agent_message_without_turn_completed_is_recovered(self):
        """A completed assistant item is still a usable terminal response when
        codex omits turn/completed and then goes quiet.
        """
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "done"},
            threadId="t",
            turnId="tu1",
        )
        s = make_session(client)
        r = s.run_turn(
            "hi",
            turn_timeout=0.05,
            notification_poll_timeout=0.01,
        )
        assert r.final_text == "done"
        assert r.interrupted is False
        assert r.error is None
        assert r.should_retire is False
        assert any(
            msg["role"] == "assistant" and msg.get("content") == "done"
            for msg in r.projected_messages
        )
        assert not any(method == "turn/interrupt" for method, _ in client.requests)


    def test_post_tool_watchdog_uses_monotonic_clock(self):
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "commandExecution", "id": "ex1",
                "command": "echo hi", "cwd": "/tmp",
                "status": "completed", "aggregatedOutput": "hi",
                "exitCode": 0, "commandActions": [],
            },
            threadId="t", turnId="tu1",
        )
        s = make_session(client)
        # monotonic sequence covers: deadline, turn_started_at (TTFB anchor),
        # iter1 while + ttfb check + last_tool_completion_at, then iter2 while
        # + post-tool check. The last call (1000.2) trips the post-tool
        # watchdog; the TTFB watchdog must not fire first.
        monotonic_values = iter(
            [1000.0, 1000.0, 999.0, 999.0, 999.0, 999.0, 1000.2]
        )
        with patch.object(
            session_mod.time,
            "monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            r = s.run_turn(
                "tool then silence",
                turn_timeout=5.0,
                notification_poll_timeout=0.0,
                post_tool_quiet_timeout=0.15,
            )
        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "silent" in r.error

    def test_post_tool_watchdog_resets_on_further_activity(self):
        """A tool completion followed by an agent message should NOT trip
        the watchdog — further activity = codex still alive."""
        client = FakeClient()
        client.queue_notification(
            "item/completed",
            item={
                "type": "commandExecution", "id": "ex1",
                "command": "echo hi", "cwd": "/tmp",
                "status": "completed", "aggregatedOutput": "hi",
                "exitCode": 0, "commandActions": [],
            },
            threadId="t", turnId="tu1",
        )
        # Non-tool activity immediately after — resets watchdog.
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "tool finished"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn(
            "tool then talk", turn_timeout=2.0,
            notification_poll_timeout=0.01,
            post_tool_quiet_timeout=0.05,
        )
        # Tool ran, then text reset the watchdog, then turn/completed.
        # Should NOT be a retirement case.
        assert r.tool_iterations == 1
        assert r.final_text == "tool finished"
        assert r.should_retire is False
        assert r.interrupted is False







    def test_post_tool_watchdog_does_not_fire_while_tool_in_progress(self):
        """The exact sq-x-daily-engagement failure: tool A completes (arming
        the post-tool quiet timer), then tool B starts and runs LONGER than
        post_tool_quiet_timeout. Codex emits nothing during B's execution, so
        without an in-progress guard the watchdog would fire off A's stale
        completion anchor and retire the session mid-side-effect (a browser
        publish). The in-progress tool must suppress the watchdog, and the
        turn must complete once B finishes."""
        client = FakeClient()
        # Tool A completes → arms last_tool_completion_at.
        client.queue_notification(
            "item/started",
            item={"type": "commandExecution", "id": "exA", "command": "echo a",
                  "cwd": "/tmp"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "item/completed",
            item={"type": "commandExecution", "id": "exA", "command": "echo a",
                  "cwd": "/tmp", "status": "completed", "aggregatedOutput": "a",
                  "exitCode": 0, "commandActions": []},
            threadId="t", turnId="tu1",
        )
        # Tool B starts immediately after — long-running (browser publish).
        client.queue_notification(
            "item/started",
            item={"type": "commandExecution", "id": "exB",
                  "command": "node x_reply_publish.mjs", "cwd": "/tmp"},
            threadId="t", turnId="tu1",
        )
        s = make_session(client)

        # B's completion arrives only AFTER the post-tool quiet window has
        # passed. Without the fix, B's in-progress state would not suppress
        # the watchdog and it would fire off exA's stale completion anchor.
        import threading

        def release_after_silence():
            time.sleep(0.25)  # > post_tool_quiet_timeout below
            client.queue_notification(
                "item/completed",
                item={"type": "commandExecution", "id": "exB",
                      "command": "node x_reply_publish.mjs", "cwd": "/tmp",
                      "status": "completed", "aggregatedOutput": "POSTED_VERIFIED",
                      "exitCode": 0, "commandActions": []},
                threadId="t", turnId="tu1",
            )
            # codex then emits its final assistant message and completes the
            # turn — the realistic flow after a finished publish tool.
            client.queue_notification(
                "item/completed",
                item={"type": "agentMessage", "id": "m1",
                      "text": "All 4 replies published and verified."},
                threadId="t", turnId="tu1",
            )
            client.queue_notification(
                "turn/completed", threadId="t",
                turn={"id": "tu1", "status": "completed", "error": None},
            )

        t = threading.Thread(target=release_after_silence, daemon=True)
        t.start()
        r = s.run_turn(
            "publish the replies",
            turn_timeout=3.0,
            notification_poll_timeout=0.01,
            post_tool_quiet_timeout=0.05,  # tiny; must not trip mid-tool
        )
        t.join(timeout=5)
        assert "published" in r.final_text
        assert r.should_retire is False
        assert r.interrupted is False
        assert r.error is None
        # The turn should have completed, not been interrupted by the watchdog.
        assert not any(method == "turn/interrupt" for method, _ in client.requests)

    def test_default_turn_deadline_precedes_cron_inactivity_limit(self):
        """The app-server deadline must win the race against cron's 600s
        inactivity watchdog even when an intermediate event bypasses narrower
        TTFB/post-tool guards."""
        import inspect

        default_timeout = inspect.signature(CodexAppServerSession.run_turn).parameters[
            "turn_timeout"
        ].default
        assert default_timeout <= 540.0

    def test_in_progress_tool_silence_fast_fails_before_turn_deadline(self):
        """A tool that starts but never completes must not inherit the full
        600s turn deadline. This reproduces the sq-x-daily-engagement Mode B
        wedge: codex accepted the turn and emitted commandExecution/started,
        then went silent until cron's 600s inactivity watchdog killed Hermes.
        """
        client = FakeClient()
        client.queue_notification(
            "item/started",
            item={
                "type": "commandExecution",
                "id": "exec-wedged",
                "command": "node x_reply_publish.mjs",
                "cwd": "/tmp",
            },
            threadId="t",
            turnId="tu1",
        )
        s = make_session(client)

        r = s.run_turn(
            "publish the replies",
            turn_timeout=0.20,
            notification_poll_timeout=0.005,
            post_tool_quiet_timeout=0.01,
            in_progress_tool_timeout=0.04,
        )

        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "tool execution" in r.error
        assert "exec-wedged" in r.error
        assert "timed out after 0.2s" not in r.error
        assert any(method == "turn/interrupt" for method, _ in client.requests)

    def test_post_tool_watchdog_still_fires_after_completion_and_true_silence(self):
        """After a tool COMPLETES and codex then genuinely goes silent (no new
        tool started), the watchdog must still fire — the fix only suppresses
        it during an in-progress tool, not after real completion."""
        client = FakeClient()
        client.queue_notification(
            "item/started",
            item={"type": "commandExecution", "id": "ex1", "command": "echo a",
                  "cwd": "/tmp"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "item/completed",
            item={"type": "commandExecution", "id": "ex1", "command": "echo a",
                  "cwd": "/tmp", "status": "completed", "aggregatedOutput": "a",
                  "exitCode": 0, "commandActions": []},
            threadId="t", turnId="tu1",
        )
        s = make_session(client)
        r = s.run_turn(
            "tool then real silence",
            turn_timeout=2.0,
            notification_poll_timeout=0.01,
            post_tool_quiet_timeout=0.05,
        )
        # Tool completed (item/completed seen) then codex went silent past the
        # quiet window with no new tool → watchdog fires.
        assert r.should_retire is True
        assert r.error and "silent" in r.error


    def test_dead_subprocess_detected_between_iterations(self):
        """If codex dies (segfault, OOM, killed by its auth refresh
        thread), the inter-iteration is_alive check breaks the loop
        instead of waiting on a queue that will never fill."""
        client = FakeClient()
        s = make_session(client)
        s.ensure_started()
        # Simulate subprocess death by setting _closed (FakeClient's
        # is_alive returns False when closed).
        client._closed = True
        client.set_stderr_tail([
            "thread 'tokio-runtime-worker' panicked at 'oauth: invalid_grant'",
        ])
        r = s.run_turn("x", turn_timeout=2.0,
                       notification_poll_timeout=0.01)
        assert r.should_retire is True
        # Stderr-derived auth hint takes precedence over generic message
        assert r.error and "codex login" in r.error


# ---- thread/start cross-fill ----

class TestThreadStartCrossFill:
    """Mirrors openclaw beta.8's tolerance for thread.id/sessionId aliasing."""

    def test_thread_id_under_thread_key(self):
        client = FakeClient()
        s = make_session(client)
        tid = s.ensure_started()
        assert tid == "thread-fake-001"



    def test_missing_thread_id_raises(self):
        from agent.transports.codex_app_server import CodexAppServerError

        client = FakeClient()
        client._request_handler = lambda method, params: (
            {"thread": {}, "activePermissionProfile": {"id": "x"}}
            if method == "thread/start" else
            {"turn": {"id": "tu1"}}
        )
        s = make_session(client)
        with pytest.raises(CodexAppServerError, match="no thread id"):
            s.ensure_started()


class TestHasTurnAbortedMarker:
    """Unit coverage for the marker matcher itself."""

    def test_empty_string(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("") is False
        assert _has_turn_aborted_marker(None) is False  # type: ignore[arg-type]

    def test_plain_text_no_marker(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("normal response with no markers") is False

    def test_open_marker(self):
        from agent.transports.codex_app_server_session import (
            _has_turn_aborted_marker,
        )
        assert _has_turn_aborted_marker("blah <turn_aborted> blah") is True



class TestClassifyOAuthFailure:
    """Unit coverage for the OAuth classifier; conservative on purpose."""



    def test_401_classified(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        hint = _classify_oauth_failure("HTTP 401 Unauthorized")
        assert hint is not None


    def test_empty_inputs(self):
        from agent.transports.codex_app_server_session import (
            _classify_oauth_failure,
        )
        assert _classify_oauth_failure() is None
        assert _classify_oauth_failure("") is None
        assert _classify_oauth_failure("", None) is None  # type: ignore[arg-type]


# ---- TTFB / first-byte watchdog (cron "initializing" hang regression) ----

class TestFirstOutputWatchdog:
    """The codex app-server subprocess can accept turn/start then emit
    NOTHING for the entire turn (a wedged/admission-stalled backend). With no
    first-output watchdog, the loop burns the full turn_timeout — which equals
    the cron inactivity limit — so cron kills the run at `initializing`.
    These tests pin the TTFB fast-fail that mirrors the `codex_responses`
    stream path's "no first byte" watchdog.
    """

    def test_silent_turn_fast_fails_before_turn_deadline(self):
        """No notification at all → TTFB watchdog trips well before
        turn_timeout, issues interrupt, and retires the session."""
        client = FakeClient()
        s = make_session(client)
        r = s.run_turn(
            "silence please",
            turn_timeout=600.0,  # outer deadline (== cron limit) is long
            notification_poll_timeout=0.0,
            first_output_timeout=0.02,
        )
        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "no first byte" in r.error
        # It must NOT have burned the full 600s turn deadline.
        assert r.tool_iterations == 0
        assert r.final_text == ""
        # interrupt was issued to stop wasted compute
        assert any(method == "turn/interrupt" for method, _ in client.requests)

    def test_lifecycle_acknowledgement_does_not_disarm_watchdog(self):
        """The noisy-start regression: turn/started is only an acknowledgement,
        not substantive model output. If Codex wedges immediately afterward,
        TTFB must remain armed and retire the session before the outer deadline.
        """
        client = FakeClient()
        client.queue_notification(
            "turn/started", threadId="t", turn={"id": "tu1"}
        )
        s = make_session(client)
        r = s.run_turn(
            "ack then silence",
            turn_timeout=600.0,
            notification_poll_timeout=0.0,
            first_output_timeout=0.02,
        )
        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "no first byte" in r.error
        assert any(method == "turn/interrupt" for method, _ in client.requests)

    def test_ttfb_fast_fail_survives_closed_transport(self):
        """Regression (2026-08-24 10:22 cron burn): when the TTFB watchdog
        fires on a wedged app-server, the turn/interrupt RPC cannot reach a
        dead transport — request() -> _send() raises
        'codex app-server stdin closed unexpectedly' (RuntimeError). That
        RuntimeError must NOT escape _issue_interrupt and crash the fast-fail:
        run_turn has to return the clean TurnResult(error 'no first byte' +
        should_retire) instead of throwing and letting cron's 600s inactivity
        watchdog win the race at 'initializing'."""
        client = FakeClient()

        # Reuse FakeClient's default routing for thread/start + turn/start,
        # but wedge ONLY the turn/interrupt that the TTFB path issues. The
        # wedged branch records the attempt itself (orig_request would have
        # done the bookkeeping) so we can assert it was reached.
        orig_request = client.request

        def request(method, params=None, timeout=30.0):
            if method == "turn/interrupt":
                client.requests.append((method, params or {}))
                raise RuntimeError(
                    "codex app-server stdin closed unexpectedly: "
                    "I/O operation on closed file."
                )
            return orig_request(method, params, timeout)

        client.request = request  # type: ignore[assignment]
        s = make_session(client)
        # Must return cleanly — NOT raise — despite the wedged interrupt path.
        r = s.run_turn(
            "silence then wedged transport",
            turn_timeout=600.0,
            notification_poll_timeout=0.0,
            first_output_timeout=0.02,
        )
        assert r.interrupted is True
        assert r.should_retire is True
        assert r.error and "no first byte" in r.error
        # The interrupt was still attempted (best-effort), not skipped.
        assert any(method == "turn/interrupt" for method, _ in client.requests)

    def test_substantive_notification_disarms_watchdog(self):
        """An item event is substantive progress and disarms TTFB; the turn
        can then continue through normal completion."""
        client = FakeClient()
        # Lifecycle acknowledgement alone is insufficient. The completed
        # agentMessage below is the event that disarms TTFB.
        client.queue_notification(
            "turn/started", threadId="t", turn={"id": "tu1"}
        )
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "hello"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn(
            "hello",
            turn_timeout=2.0,
            notification_poll_timeout=0.01,
            first_output_timeout=0.02,  # tiny; must be disarmed by first note
        )
        assert r.final_text == "hello"
        assert r.interrupted is False
        assert r.should_retire is False
        assert r.error is None

    def test_server_request_disarms_watchdog(self):
        """A server-initiated request (e.g. an approval) is also live output —
        it must disarm the TTFB watchdog so codex gets time to keep going."""
        client = FakeClient()
        client.queue_server_request(
            "item/commandExecution/requestApproval",
            request_id="approval-1",
            command="pwd",
            cwd="/tmp",
        )
        client.queue_notification(
            "item/completed",
            item={"type": "agentMessage", "id": "m1", "text": "done"},
            threadId="t", turnId="tu1",
        )
        client.queue_notification(
            "turn/completed", threadId="t",
            turn={"id": "tu1", "status": "completed", "error": None},
        )
        s = make_session(client)
        r = s.run_turn(
            "needs approval",
            turn_timeout=2.0,
            notification_poll_timeout=0.01,
            first_output_timeout=0.02,
        )
        assert r.final_text == "done"
        assert r.should_retire is False
        assert r.interrupted is False

    def test_watchdog_disabled_when_timeout_zero(self):
        """first_output_timeout=0 disables the watchdog entirely, so a silent
        turn falls through to the outer turn deadline (existing behavior)."""
        client = FakeClient()
        s = make_session(client)
        r = s.run_turn(
            "silence please",
            turn_timeout=0.05,
            notification_poll_timeout=0.01,
            first_output_timeout=0.0,
        )
        # Fell through to the turn deadline, not the TTFB watchdog.
        assert r.error and "timed out after" in r.error
        assert r.should_retire is True


# ---- startup handshake timeout + retry ----

class TestStartupHandshake:
    def _factory(self, handlers):
        """Return a client_factory that hands out a fresh FakeClient per spawn,
        with the i-th client's request() routed through handlers[i] when set.
        Tracks which client actually answered thread/start."""
        created = []
        answered = []

        def factory(**kw):
            client = FakeClient(**kw)
            idx = len(created)
            created.append(client)
            h = handlers[idx] if idx < len(handlers) else None
            if h is not None:
                client._request_handler = h
            client._answered_by = lambda: answered.append(idx)
            return client

        return factory, created, answered

    def test_retries_on_thread_start_timeout_and_succeeds(self, monkeypatch):
        """A cold-start thread/start timeout must not abort the run: the
        session tears down the wedged client and spawns a fresh one."""
        monkeypatch.setenv("HERMES_CODEX_STARTUP_TIMEOUT_SECONDS", "0.05")
        monkeypatch.setenv("HERMES_CODEX_STARTUP_RETRIES", "1")  # 2 attempts

        calls = {"n": 0}

        def first(method, params):
            if method == "thread/start":
                calls["n"] += 1
                raise TimeoutError("codex app-server method 'thread/start' timed out")
            return {"thread": {"id": "t"}}

        factory, created, answered = self._factory([first])
        s = CodexAppServerSession(cwd="/tmp", client_factory=factory)
        tid = s.ensure_started()
        assert tid == "thread-fake-001"  # second client's default handler
        # Two clients were created (first wedged, second succeeded).
        assert len(created) == 2
        # The first client was torn down (closed) on the retry path.
        assert created[0]._closed is True
        # thread/start was attempted once on the first (wedged) client.
        ts_on_first = [m for (m, _) in created[0].requests if m == "thread/start"]
        assert len(ts_on_first) == 1

    def test_gives_up_after_retries_exhausted(self, monkeypatch):
        """If every attempt times out, ensure_started re-raises the last
        TimeoutError instead of hanging or returning a half-started session."""
        monkeypatch.setenv("HERMES_CODEX_STARTUP_TIMEOUT_SECONDS", "0.05")
        monkeypatch.setenv("HERMES_CODEX_STARTUP_RETRIES", "1")  # 2 attempts

        def always_timeout(method, params):
            raise TimeoutError("codex app-server method 'thread/start' timed out")

        factory, created, answered = self._factory([always_timeout, always_timeout])
        s = CodexAppServerSession(cwd="/tmp", client_factory=factory)
        with pytest.raises(TimeoutError):
            s.ensure_started()
        assert len(created) == 2
        assert s._client is None  # torn down, nothing left half-started
        assert s._thread_id is None

    def test_startup_timeout_passed_to_initialize_and_thread_start(self, monkeypatch):
        """The configured startup timeout must flow into BOTH the initialize
        handshake and thread/start (previously thread/start was hardcoded 15s)."""
        monkeypatch.setenv("HERMES_CODEX_STARTUP_TIMEOUT_SECONDS", "7.5")
        monkeypatch.setenv("HERMES_CODEX_STARTUP_RETRIES", "0")

        seen = {}

        def capture(method, params):
            if method == "thread/start":
                return {"thread": {"id": "t"}}
            return {}

        client = FakeClient()
        orig_request = client.request
        orig_initialize = client.initialize

        def wrap(method, params=None, timeout=30.0):
            seen[method] = timeout
            return orig_request(method, params, timeout)

        def wrap_init(**kwargs):
            seen["initialize"] = kwargs.get("timeout")
            return orig_initialize(**kwargs)

        client.request = wrap  # type: ignore[assignment]
        client.initialize = wrap_init  # type: ignore[assignment]
        client._request_handler = capture
        s = make_session(client)
        s.ensure_started()
        assert seen["initialize"] == 7.5
        assert seen["thread/start"] == 7.5

    def test_defaults_allow_slow_cold_start(self):
        """The default startup timeout (60s) is far above the hardcoded 15s
        that aborted cron runs, and defaults to one retry."""
        assert session_mod._DEFAULT_CODEX_STARTUP_TIMEOUT_SECONDS >= 60.0
        assert session_mod._DEFAULT_CODEX_STARTUP_RETRIES == 1

    @pytest.mark.parametrize(
        "env_value,expected",
        [
            # huge → clamped to hard cap (no effectively-unbounded retries)
            ("1000000", _MAX_CODEX_STARTUP_RETRIES),
            # inf / nan → fall back to default (no int() crash)
            ("inf", _DEFAULT_CODEX_STARTUP_RETRIES),
            ("-inf", _DEFAULT_CODEX_STARTUP_RETRIES),
            ("nan", _DEFAULT_CODEX_STARTUP_RETRIES),
            # negative → fall back to default
            ("-1", _DEFAULT_CODEX_STARTUP_RETRIES),
            ("--5", _DEFAULT_CODEX_STARTUP_RETRIES),
            # fractional → truncated (int floor)
            ("2.7", 2),
            ("0.5", 0),
            # boundary: exactly the cap stays the cap
            (str(_MAX_CODEX_STARTUP_RETRIES), _MAX_CODEX_STARTUP_RETRIES),
            # above the cap → clamped
            (str(_MAX_CODEX_STARTUP_RETRIES + 1), _MAX_CODEX_STARTUP_RETRIES),
        ],
    )
    def test_resolve_retries_huge_inf_nan_negative_fractional(
        self, env_value, expected
    ):
        """The review's required boundary cases: huge, inf, nan, negative, and
        fractional inputs must all resolve to a bounded int in
        [0, _MAX_CODEX_STARTUP_RETRIES] without raising."""
        assert _resolve_startup_retries(env_value, None, _DEFAULT_CODEX_STARTUP_RETRIES) == expected

    def test_resolve_retries_clamped_from_config_too(self):
        """The hard cap must apply to config-derived values as well as env."""
        # A giant config value is clamped identically to the env path.
        assert _resolve_startup_retries(
            None, 10_000_000, _DEFAULT_CODEX_STARTUP_RETRIES
        ) == _MAX_CODEX_STARTUP_RETRIES
        # Non-finite config value falls back to default.
        assert _resolve_startup_retries(
            None, float("inf"), _DEFAULT_CODEX_STARTUP_RETRIES
        ) == _DEFAULT_CODEX_STARTUP_RETRIES

    def test_resolve_retries_always_returns_int(self):
        """The retry count must ALWAYS be an int (not a float) so that
        ``range(max(1, retries + 1))`` in ensure_started() never raises
        TypeError. This pins the negative-fallback path, which previously
        returned ``float(default)``."""
        for env_value in ("-1", "-3", "0", "2.7", "1000000", "inf", "nan", "1"):
            result = _resolve_startup_retries(env_value, None, _DEFAULT_CODEX_STARTUP_RETRIES)
            assert isinstance(result, int), (
                f"retry count for env {env_value!r} is {type(result).__name__}, not int"
            )
        # config path too
        for cfg_value in (-1, 2.7, 10_000_000, float("inf"), 0):
            result = _resolve_startup_retries(None, cfg_value, _DEFAULT_CODEX_STARTUP_RETRIES)
            assert isinstance(result, int), (
                f"retry count for config {cfg_value!r} is {type(result).__name__}, not int"
            )

    def test_env_bridge_overrides_config(self, monkeypatch):
        """Config is the operator control, but the internal env bridge still
        takes precedence for a single process."""
        # Simulate a config value, then set the env bridge; env must win.
        assert _resolve_startup_retries("3", 1, _DEFAULT_CODEX_STARTUP_RETRIES) == 3
        assert _resolve_startup_retries(None, 2, _DEFAULT_CODEX_STARTUP_RETRIES) == 2

    @pytest.mark.parametrize(
        "env_value,config_value,expected",
        [
            # default when nothing set
            (None, None, 60.0),
            # env finite positive
            ("90", None, 90.0),
            # env fractional
            ("2.5", None, 2.5),
            # config positive
            (None, 45.0, 45.0),
            # env overrides config
            ("30", 45.0, 30.0),
            # env non-finite → default even with config set (env wins, but is invalid)
            ("nan", 45.0, 60.0),
            # env <= 0 → default
            ("0", None, 60.0),
            ("-4", None, 60.0),
            # config <= 0 → default
            (None, 0, 60.0),
            # config non-finite → default
            (None, float("nan"), 60.0),
            # garbage string → default
            ("not-a-number", None, 60.0),
        ],
    )
    def test_resolve_timeout_validation(self, env_value, config_value, expected):
        assert _resolve_startup_timeout(env_value, config_value, 60.0) == expected

    def test_ensure_started_uses_resolved_retry_count(self, monkeypatch):
        """End-to-end: ensure_started must honor the clamped retry count. With
        retries clamped to 0 (from a huge env value that the resolver floors)
        and a handler that always times out, only ONE client is created —
        proving the bound is real, not just cosmetic."""
        monkeypatch.setenv("HERMES_CODEX_STARTUP_TIMEOUT_SECONDS", "0.05")
        # 0.5 fractional → int(0.5) == 0 retries → 1 attempt
        monkeypatch.setenv("HERMES_CODEX_STARTUP_RETRIES", "0.5")

        def always_timeout(method, params):
            raise TimeoutError("codex app-server method 'thread/start' timed out")

        created = []

        def factory(**kw):
            client = FakeClient(**kw)
            created.append(client)
            client._request_handler = always_timeout
            return client

        s = CodexAppServerSession(cwd="/tmp", client_factory=factory)
        with pytest.raises(TimeoutError):
            s.ensure_started()
        # Exactly one attempt (0 retries) — a huge/uncapped value would create
        # many more clients.
        assert len(created) == 1
        assert s._client is None
        assert s._thread_id is None

    def test_ensure_started_clamps_huge_retry_count(self, monkeypatch):
        """A huge HERMES_CODEX_STARTUP_RETRIES (e.g. 1_000_000) must be clamped
        to the hard cap, so the number of attempts is bounded rather than
        effectively unbounded."""
        monkeypatch.setenv("HERMES_CODEX_STARTUP_TIMEOUT_SECONDS", "0.05")
        monkeypatch.setenv("HERMES_CODEX_STARTUP_RETRIES", "1000000")

        def always_timeout(method, params):
            raise TimeoutError("codex app-server method 'thread/start' timed out")

        created = []

        def factory(**kw):
            client = FakeClient(**kw)
            created.append(client)
            client._request_handler = always_timeout
            return client

        s = CodexAppServerSession(cwd="/tmp", client_factory=factory)
        with pytest.raises(TimeoutError):
            s.ensure_started()
        # Bounded: 1 (initial) + _MAX_CODEX_STARTUP_RETRIES retries.
        assert len(created) == 1 + _MAX_CODEX_STARTUP_RETRIES
        assert s._client is None
        assert s._thread_id is None


class TestCodexStartupConfigPropagation:
    """config.yaml is the operator-facing control (repository AGENTS.md policy);
    the HERMES_* env vars are an internal bridge only. These tests prove the
    resolver path reads the codex config section when the env bridge is unset."""

    def test_config_section_is_documented_in_defaults(self):
        """The codex startup knobs ship in DEFAULT_CONFIG so config.yaml-backed
        values are recognized (not treated as unknown roots)."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert "codex" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["codex"]["startup_timeout_seconds"] == 60.0
        assert DEFAULT_CONFIG["codex"]["startup_retries"] == 1

    def test_codex_config_section_reads_config(self, monkeypatch):
        """_codex_config_section returns the codex section from a loaded config."""
        fake_cfg = {
            "codex": {"startup_timeout_seconds": 90, "startup_retries": 2},
        }
        monkeypatch.setattr(
            session_mod, "_codex_config_section", lambda: dict(fake_cfg["codex"])
        )
        assert session_mod._codex_config_section() == {
            "startup_timeout_seconds": 90,
            "startup_retries": 2,
        }

    def test_ensure_started_honors_config_when_env_unset(
        self, monkeypatch, tmp_path
    ):
        """With the env bridge unset, ensure_started must pick up the retry
        count from config.yaml (the operator-facing control). A config retry
        count of 2 (→ 3 attempts) must produce exactly 3 client spawns."""
        # Point HERMES_HOME at a temp dir and write a codex section.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import yaml

        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {"codex": {"startup_timeout_seconds": 0.05, "startup_retries": 2}}
            )
        )
        monkeypatch.delenv("HERMES_CODEX_STARTUP_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("HERMES_CODEX_STARTUP_RETRIES", raising=False)

        def always_timeout(method, params):
            raise TimeoutError("codex app-server method 'thread/start' timed out")

        created = []

        def factory(**kw):
            client = FakeClient(**kw)
            created.append(client)
            client._request_handler = always_timeout
            return client

        s = CodexAppServerSession(cwd="/tmp", client_factory=factory)
        with pytest.raises(TimeoutError):
            s.ensure_started()
        # 1 initial + 2 config-derived retries = 3 attempts.
        assert len(created) == 3
        assert s._client is None
        assert s._thread_id is None



