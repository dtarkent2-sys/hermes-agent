"""Regression tests: prune disabled/absent MCP servers from the live registry.

Background
==========
``register_mcp_servers`` / ``discover_mcp_tools`` are add-only: they connect
servers enabled in config but never disconnect one whose ``enabled`` later
flips to ``false`` (or whose entry is removed). In a long-lived gateway
process, a config hot-reload that disables a server left its task registered —
it kept parking on reconnect and reloading its OAuth token on every discovery
pass / disk-watch. ``_prune_disabled_mcp_servers`` closes that gap and is wired
into ``discover_mcp_tools`` so the in-process registry converges to the
on-disk config on every discovery pass.
"""

from __future__ import annotations

from unittest.mock import patch


class _FakeTask:
    """Minimal stand-in for ``MCPServerTask`` covering the teardown surface
    that ``_shutdown_one_mcp_task`` touches when no live MCP loop exists."""

    def __init__(self, name: str):
        self.name = name
        self.shutdown_called = False
        self.session = None

        class _E:
            set_called = False

            def set(self):
                self.set_called = True
                return self

        self._shutdown_event = _E()
        self._reconnect_event = _E()
        self._registered_tool_names = [f"mcp__{name}__t1", f"mcp__{name}__t2"]

    def _deregister_tools(self):
        self.deregistered = True
        self._registered_tool_names = []


class _RecordingOAuth:
    """Stands in for ``tools.mcp_oauth_manager.get_manager``; records evicts."""

    def __init__(self):
        self.evicted = []

    def evict(self, name):
        self.evicted.append(name)


def _recording_oauth():
    return _RecordingOAuth()


def test_prune_removes_disabled_server(monkeypatch):
    import tools.mcp_tool as m

    live = _FakeTask("live")
    dead = _FakeTask("dead")
    monkeypatch.setitem(m._servers, "live", live)
    monkeypatch.setitem(m._servers, "dead", dead)
    mgr = _recording_oauth()

    with patch("tools.mcp_oauth_manager.get_manager", return_value=mgr):
        cfg = {
            "live": {"url": "http://live/mcp"},
            "dead": {"url": "http://dead/mcp", "enabled": False},
        }
        pruned = m._prune_disabled_mcp_servers(cfg)

    assert pruned == ["dead"], f"expected only 'dead' pruned, got {pruned}"
    assert "live" in m._servers and "dead" not in m._servers
    assert live.shutdown_called is False
    assert dead._shutdown_event.set_called is True
    assert dead._reconnect_event.set_called is True
    assert getattr(dead, "deregistered", False) is True
    assert mgr.evicted == ["dead"]


def test_prune_removes_absent_server(monkeypatch):
    """A registered server that no longer appears in config is pruned."""
    import tools.mcp_tool as m

    orphan = _FakeTask("orphan")
    kept = _FakeTask("kept")
    monkeypatch.setitem(m._servers, "orphan", orphan)
    monkeypatch.setitem(m._servers, "kept", kept)
    mgr = _recording_oauth()

    with patch("tools.mcp_oauth_manager.get_manager", return_value=mgr):
        pruned = m._prune_disabled_mcp_servers({"kept": {"url": "http://kept/mcp"}})

    assert pruned == ["orphan"]
    assert "kept" in m._servers and "orphan" not in m._servers
    assert mgr.evicted == ["orphan"]


def test_prune_empty_config_clears_all(monkeypatch):
    import tools.mcp_tool as m

    a = _FakeTask("a")
    b = _FakeTask("b")
    monkeypatch.setitem(m._servers, "a", a)
    monkeypatch.setitem(m._servers, "b", b)
    mgr = _recording_oauth()

    with patch("tools.mcp_oauth_manager.get_manager", return_value=mgr):
        pruned = m._prune_disabled_mcp_servers({})

    assert sorted(pruned) == ["a", "b"]
    assert m._servers == {}
    assert sorted(mgr.evicted) == ["a", "b"]


def test_prune_noop_when_registry_matches_config(monkeypatch):
    import tools.mcp_tool as m

    ok = _FakeTask("ok")
    monkeypatch.setitem(m._servers, "ok", ok)
    mgr = _recording_oauth()

    with patch("tools.mcp_oauth_manager.get_manager", return_value=mgr):
        pruned = m._prune_disabled_mcp_servers({"ok": {"url": "http://ok/mcp"}})

    assert pruned == []
    assert "ok" in m._servers
    assert ok.shutdown_called is False
    assert mgr.evicted == []


def test_prune_handles_missing_or_non_dict_entries(monkeypatch):
    """A non-dict config value for a registered server prunes it; an enabled
    absent server is ignored (no error, nothing pruned for it)."""
    import tools.mcp_tool as m

    weird = _FakeTask("weird")
    monkeypatch.setitem(m._servers, "weird", weird)
    mgr = _recording_oauth()

    with patch("tools.mcp_oauth_manager.get_manager", return_value=mgr):
        pruned = m._prune_disabled_mcp_servers({"weird": None, "ghost": {"enabled": True}})

    assert pruned == ["weird"]
    assert "weird" not in m._servers
    # 'ghost' was never registered, so nothing is pruned/evicted for it.
    assert mgr.evicted == ["weird"]


def test_discover_mcp_tools_prunes_disabled_before_connect(monkeypatch):
    """discover_mcp_tools must converge the registry to config: a server that
    is registered but disabled in config is evicted on a discovery pass, even
    when every remaining server is disabled (the no-SDK early-return path)."""
    import tools.mcp_tool as m

    dead = _FakeTask("dead")
    monkeypatch.setitem(m._servers, "dead", dead)
    mgr = _recording_oauth()

    with patch("tools.mcp_tool._load_mcp_config",
               return_value={"dead": {"enabled": False, "url": "http://x"}}), \
         patch("tools.mcp_tool._ensure_mcp_sdk", return_value=False), \
         patch("tools.mcp_oauth_manager.get_manager", return_value=mgr):
        result = m.discover_mcp_tools()

    # No-SDK early return, but the disabled registered task was pruned.
    assert result == []
    assert "dead" not in m._servers
    assert mgr.evicted == ["dead"]
