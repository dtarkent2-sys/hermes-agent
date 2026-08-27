"""Tests for webhook route-script execution (WebhookRouteProcessor.run_route_script).

Regression coverage for the WSL-stub bash-resolution defect: .sh/.bash
route scripts must resolve bash through the shared deterministic Git-Bash
ladder (tools.environments.local.resolve_script_bash — the same helper
cron's script runner uses), never a bare, PATH-dependent
shutil.which("bash"), which on Windows can land on the WSL launcher stub
(System32 bash.exe) that eats the backslashes from the native script path
and fails the route with exit 127 (surfacing as a logged "ignored
webhook").
"""

import json
import logging
import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gateway.platforms.webhook_filters import WebhookRouteProcessor  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so route scripts resolve under tmp scripts/."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _write_script(hermes_home: Path, name: str, body: str) -> Path:
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / name
    path.write_text(body, encoding="utf-8")
    return path


class TestRunRouteScriptPython:
    """Python route scripts keep their historical behaviour."""

    def test_python_script_transforms_payload(self, tmp_path):
        processor = WebhookRouteProcessor()
        _write_script(
            tmp_path,
            "route.py",
            (
                "import json, sys\n"
                "payload = json.load(sys.stdin)\n"
                "payload['transformed'] = True\n"
                "print(json.dumps(payload))\n"
            ),
        )

        cont, transformed = processor.run_route_script(
            "route.py", {"n": 1}
        )

        assert cont is True
        assert transformed == {"n": 1, "transformed": True}

    def test_python_script_nonzero_exit_ignores_webhook(self, tmp_path):
        processor = WebhookRouteProcessor()
        _write_script(tmp_path, "route.py", "import sys; sys.exit(3)\n")

        cont, transformed = processor.run_route_script("route.py", {"n": 1})

        assert (cont, transformed) == (False, None)


class TestResolveWebhookBash:
    """Bash resolution for .sh/.bash route scripts."""

    def test_module_uses_shared_script_bash_helper(self):
        """The webhook module must import the shared resolve_script_bash —
        not re-implement the ladder (drift guard against the WSL-stub
        regression)."""
        import gateway.platforms.webhook_filters as wf
        from tools.environments.local import resolve_script_bash

        assert wf._ROUTE_BASH_RESOLVER_AVAILABLE is True
        assert wf._resolve_route_bash is resolve_script_bash

    def test_delegates_to_shared_helper(self, monkeypatch):
        import gateway.platforms.webhook_filters as wf

        monkeypatch.setattr(wf, "_ROUTE_BASH_RESOLVER_AVAILABLE", True)
        monkeypatch.setattr(
            wf,
            "_resolve_route_bash",
            lambda: ("C:/Program Files/Git/bin/bash.exe", "shared-ladder"),
        )

        assert wf._resolve_webhook_bash() == (
            "C:/Program Files/Git/bin/bash.exe",
            "shared-ladder",
        )

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="win32 degraded branch: must refuse PATH-dependent bash resolution",
    )
    def test_win32_tools_missing_refuses_path_dependent_bash(self, monkeypatch):
        """win32 + unimportable tools.environments.local must return
        (None, reason) — never a bare shutil.which("bash") that could land
        on the WSL launcher stub (System32 bash.exe)."""
        import shutil as _shutil

        import gateway.platforms.webhook_filters as wf

        monkeypatch.setattr(wf, "_ROUTE_BASH_RESOLVER_AVAILABLE", False)
        monkeypatch.setattr(wf, "_resolve_route_bash", None)
        # Even with a bash "on PATH" (the WSL stub scenario), the degraded
        # win32 branch must refuse PATH-dependent resolution.
        monkeypatch.setattr(
            _shutil, "which", lambda name: "C:/Windows/System32/bash.exe" if name == "bash" else None
        )

        bash, source = wf._resolve_webhook_bash()

        assert bash is None
        assert "bash resolution unavailable" in source

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="POSIX degraded branch: keeps historical PATH->/bin/bash behaviour",
    )
    def test_posix_tools_missing_keeps_path_then_bin_bash(self, monkeypatch):
        import shutil as _shutil

        import gateway.platforms.webhook_filters as wf

        monkeypatch.setattr(wf, "_ROUTE_BASH_RESOLVER_AVAILABLE", False)
        monkeypatch.setattr(wf, "_resolve_route_bash", None)

        bash, source = wf._resolve_webhook_bash()

        if _shutil.which("bash"):
            assert source == "PATH"
            assert "bash" in os.path.basename(bash)
        elif os.path.isfile("/bin/bash"):
            assert (bash, source) == ("/bin/bash", "/bin/bash")
        else:
            assert bash is None


class TestRunRouteScriptBash:
    """End-to-end .sh execution through run_route_script."""

    def test_sh_script_runs_with_native_path_intact(self, tmp_path):
        """A real .sh route script must run rc=0 with the native Windows
        script path intact — the WSL stub would mangle the backslashes and
        fail the route.  The script echoes stdin back, so the payload (with
        a backslash-bearing path field) must round-trip unchanged."""
        from cron.scheduler import _resolve_cron_bash

        bash, _source = _resolve_cron_bash()
        if bash is None:
            pytest.skip("no bash available on this host")

        processor = WebhookRouteProcessor()
        _write_script(tmp_path, "route.sh", "#!/usr/bin/env bash\ncat\n")

        payload = {"file": "C:\\Users\\dtark\\report.txt", "n": 1}
        cont, transformed = processor.run_route_script("route.sh", payload)

        assert cont is True
        assert transformed == payload

    def test_bash_unavailable_ignores_webhook_with_reason(self, tmp_path, caplog, monkeypatch):
        """When bash cannot be resolved, the webhook is ignored cleanly and
        the warning names the resolution failure."""
        import gateway.platforms.webhook_filters as wf

        processor = WebhookRouteProcessor()
        _write_script(tmp_path, "route.sh", "#!/usr/bin/env bash\ncat\n")

        monkeypatch.setattr(
            wf,
            "_resolve_webhook_bash",
            lambda: (None, "forced-unavailable"),
        )

        with caplog.at_level(logging.WARNING, logger="gateway.platforms.webhook_filters"):
            cont, transformed = processor.run_route_script("route.sh", {"n": 1})

        assert (cont, transformed) == (False, None)
        assert any("bash not found" in rec.message for rec in caplog.records)
        assert any("forced-unavailable" in rec.message for rec in caplog.records)