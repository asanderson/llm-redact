"""Complexity-coverage tail: focused, real tests that EXECUTE branching
functions the rest of the suite never happened to reach.

The goal here is execution (the CC>1 coverage gate in scripts/complexity_gate.py),
not exhaustive behavior — but every test drives the function through a genuine
code path with realistic, secret-shaped fakes (corp.example, AKIA…EXAMPLE).
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import httpx
import pytest

from llm_redact.config import Config
from llm_redact.config_write import emit_config_toml


def _write_config(tmp_path: Path, config: Config | None = None) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(emit_config_toml(config or Config()), encoding="utf-8")
    return path


# --- cli.run_preview ---------------------------------------------------------


def test_run_preview_text_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.cli import run_preview

    cfg = _write_config(tmp_path)
    args = argparse.Namespace(config=cfg, text="Email jane@corp.example", json=False)
    assert run_preview(args) == 0
    assert "EMAIL" in capsys.readouterr().out

    args_json = argparse.Namespace(config=cfg, text="Email jane@corp.example", json=True)
    assert run_preview(args_json) == 0
    assert '"detections"' in capsys.readouterr().out


# --- license_cli.run_license_show / run_license_verify (Free / keyless) ------
# The valid-key path (audit-verify, license show/verify against a SIGNED key,
# users list/revoke, the Azure sink loop) needs the pro package and moved to the
# pro repo's coverage suite. The keyless Free paths stay here.


def test_license_show_and_verify_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.license_cli import run_license_show, run_license_verify

    monkeypatch.delenv("LLM_REDACT_LICENSE_KEY", raising=False)
    cfg = _write_config(tmp_path)

    show = argparse.Namespace(config=cfg, key=None, json=False)
    assert run_license_show(show) == 0
    assert "Free tier" in capsys.readouterr().out

    verify = argparse.Namespace(config=cfg, key=None, json=False)
    assert run_license_verify(verify) == 1


# --- plugin_cli.proxy_posture_hint / run_plugin ------------------------------


def test_proxy_posture_hint_returns_string(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_redact.plugin_cli as plugin_cli

    monkeypatch.delenv("LLM_REDACT_PROXY_URL", raising=False)
    # No network: the probe is a module-level default; force a clean "down" answer.
    monkeypatch.setattr(plugin_cli, "_default_probe", lambda url: False)
    hint = plugin_cli.proxy_posture_hint()
    assert isinstance(hint, str) and hint


def test_run_plugin_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.plugin_cli import run_plugin

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    args = argparse.Namespace(plugin_command="status")
    assert run_plugin(args) == 0
    assert "not installed" in capsys.readouterr().out


# --- providers.cohere.CohereAdapter._synthetic -------------------------------


def test_cohere_synthetic_covers_every_channel() -> None:
    import json

    from llm_redact.providers.cohere import CohereAdapter

    text = json.loads(CohereAdapter._synthetic(("text", 0), "hi").data)
    assert text["type"] == "content-delta"
    args = json.loads(CohereAdapter._synthetic(("args", 1), "{}").data)
    assert args["type"] == "tool-call-delta"
    plan = json.loads(CohereAdapter._synthetic(("tool_plan",), "plan").data)
    assert plan["type"] == "tool-plan-delta"
    v1 = json.loads(CohereAdapter._synthetic(("v1text",), "leftover").data)
    assert v1["event_type"] == "text-generation"


# --- providers.openai_responses.OpenAIResponsesAdapter.error_body ------------


def test_openai_responses_error_body_shapes() -> None:
    from llm_redact.providers.openai_responses import OpenAIResponsesAdapter

    adapter = OpenAIResponsesAdapter()
    big = adapter.error_body("too large", status=413)
    assert big["error"]["code"] == "request_too_large"
    other = adapter.error_body("nope", status=400)
    assert other["error"]["code"] is None


# --- eventstream.EventStreamMessage.exception_type ---------------------------


def test_eventstream_exception_type() -> None:
    from llm_redact.eventstream import EventStreamMessage, string_header

    with_exc = EventStreamMessage(headers=[string_header(":exception-type", "InternalFailure")])
    assert with_exc.exception_type == "InternalFailure"
    assert EventStreamMessage().exception_type is None


# --- bench.__main__.main -----------------------------------------------------


def test_bench_main_runs() -> None:
    from llm_redact.bench.__main__ import main as bench_main

    # No --check: just exercise the report path over a tiny corpus.
    assert bench_main(["--samples", "1"]) == 0


# --- init_cli._ask / _ask_yes_no ---------------------------------------------


def test_init_ask_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from llm_redact.init_cli import _ask, _ask_yes_no

    replies = iter(["", "custom", "y", "n", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(replies))

    assert _ask("Port", "8787") == "8787"  # empty -> default
    assert _ask("Port", "8787") == "custom"  # typed value wins
    assert _ask_yes_no("go?", default=False) is True  # "y"
    assert _ask_yes_no("go?", default=True) is False  # "n"
    assert _ask_yes_no("go?", default=True) is True  # empty -> default


# --- cloud_detect._http_probe (network probe, faked) -------------------------


def test_http_probe_faked(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_redact.cloud_detect as cloud_detect

    class _Resp:
        status_code = 200

    monkeypatch.setattr(cloud_detect.httpx, "request", lambda *a, **k: _Resp())
    assert cloud_detect._http_probe("GET", "http://meta.example", {}) is True

    def _boom(*a, **k):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(cloud_detect.httpx, "request", _boom)
    assert cloud_detect._http_probe("GET", "http://meta.example", {}) is False


# --- vault_cli._proxy_reachable (socket probe against a closed port) ---------


def test_proxy_reachable_false_when_nothing_listens(tmp_path: Path) -> None:
    from llm_redact.vault_cli import _proxy_reachable

    # Grab-then-release a port so the connect is guaranteed to be refused.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    closed_port = sock.getsockname()[1]
    sock.close()

    cfg = _write_config(tmp_path, Config(port=closed_port))
    args = argparse.Namespace(config=cfg)
    assert _proxy_reachable(args) is False
