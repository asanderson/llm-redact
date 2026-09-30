"""``llm-redact doctor``'s ``extraction`` rows (``extraction.extraction_checks``
through ``doctor_cli._check_extraction``): offline and value-free — what
keeps the section from starting, each service and where it runs, convert
mode, and what a clean scan lets through."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from llm_redact import extraction
from llm_redact.config import parse_extraction
from llm_redact.doctor_cli import run_doctor
from llm_redact.extraction import extraction_checks


def _rows(raw: dict[str, Any], environ: dict[str, str]) -> list[tuple[str, str]]:
    return extraction_checks(parse_extraction(raw), environ)


def test_disabled() -> None:
    [(level, message)] = _rows({}, {})
    assert level == "PASS" and "disabled" in message


def test_an_enabled_section() -> None:
    raw = {
        "enabled": True,
        # The local worker and both services, one after another (30 s each).
        "request_timeout_seconds": 90,
        "services": [
            {"kind": "tika", "url": "http://127.0.0.1:9998", "complete": True},
            {"kind": "docling", "url": "https://docling.corp.example", "trusted": True},
        ],
    }
    rows = _rows(raw, {})
    assert [level for level, _ in rows] == ["PASS", "WARN", "PASS"]
    assert "tika service at 127.0.0.1" in rows[0][1] and "counts as complete" in rows[0][1]
    assert "OFF this machine" in rows[1][1] and "never counts as complete" in rows[1][1]
    assert "client's own key only" in rows[2][1] and "extracted text only" in rows[2][1]
    proxy = _rows({"enabled": True, "proxy_credential": True}, {})
    assert "also under a credential the proxy holds" in proxy[-1][1]


def test_cloud_services_and_convert_are_warned_about() -> None:
    raw = {
        "enabled": True,
        "convert": ["pdf"],
        "request_timeout_seconds": 120,
        "services": [
            {"kind": "textract", "region": "eu-west-1", "trusted": True},
            {
                "kind": "azure_docintel",
                "url": "https://di.cognitiveservices.azure.com",
                "token_env": "DI_KEY",
                "trusted": True,
            },
        ],
    }
    environ = {"AWS_ACCESS_KEY_ID": "AKIDSECRETVALUE", "AWS_SECRET_ACCESS_KEY": "s3", "DI_KEY": "k"}
    rows = _rows(raw, environ)
    assert [level for level, _ in rows] == ["WARN", "WARN", "WARN", "PASS"]
    assert "textract service at textract.eu-west-1.amazonaws.com" in rows[0][1]
    assert "OFF this machine" in rows[1][1]
    assert "convert on for pdf" in rows[2][1] and "REDACTED EXTRACTED TEXT" in rows[2][1]
    assert "AKIDSECRETVALUE" not in json.dumps(rows)


def test_what_refuses_to_start_is_a_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {
        "enabled": True,
        "request_timeout_seconds": 120,
        "services": [
            {"kind": "tika", "url": "http://127.0.0.1:9998", "token_env": "TIKA_T"},
            {"kind": "textract", "region": "us-east-1", "trusted": True},
        ],
    }
    fails = [message for level, message in _rows(raw, {}) if level == "FAIL"]
    assert any("TIKA_T is not set" in m for m in fails)
    assert any("AWS_ACCESS_KEY_ID is not set" in m for m in fails)
    assert any("AWS_SECRET_ACCESS_KEY is not set" in m for m in fails)
    monkeypatch.setattr(extraction.importlib.util, "find_spec", lambda name: None)
    fails = [m for level, m in _rows({"enabled": True}, {}) if level == "FAIL"]
    assert fails and "llm-redact-proxy[extract]" in fails[0]


def test_doctor_prints_the_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "[extraction]\nenabled = true\nconvert = true\nrequest_timeout_seconds = 90\n"
        '[[extraction.services]]\nkind = "tika"\nurl = "http://127.0.0.1:9998"\n'
        'token_env = "LLM_REDACT_TEST_UNSET_TOKEN"\n'
    )
    code = run_doctor(argparse.Namespace(config=config, json=True, offline=True))
    checks = json.loads(capsys.readouterr().out)["checks"]
    rows = [row for row in checks if row["area"] == "extraction"]
    assert code == 1
    assert [row["level"] for row in rows] == ["FAIL", "PASS", "WARN", "PASS"]
    assert "LLM_REDACT_TEST_UNSET_TOKEN is not set" in rows[0]["message"]


class _Plugged:
    """A plugin's upload inspector (only what the doctor row reads)."""

    closed = False

    async def aclose(self) -> None:
        _Plugged.closed = True


@pytest.mark.parametrize("built", ["none", "raises", "plugged"])
def test_a_plugin_factory_the_proxy_would_refuse_is_a_fail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    built: str,
) -> None:
    # An older llm-redact-pro still replaces the factory and, reading the
    # section from its own config, builds none for the core's [extraction]:
    # the proxy refuses to start, so doctor must not pass it.
    from llm_redact import registry as registry_mod
    from llm_redact.config import ConfigError
    from llm_redact.registry import Registry

    def build(config: Any, tier: str) -> Any:
        if built == "raises":
            raise ConfigError("the plugin's own setting")
        return _Plugged() if built == "plugged" else None

    reg = Registry()
    reg.build_upload_inspector = build
    monkeypatch.setattr(registry_mod, "_registry", reg)
    config = tmp_path / "config.toml"
    config.write_text("[extraction]\nenabled = true\n")
    run_doctor(argparse.Namespace(config=config, json=True, offline=True))
    checks = json.loads(capsys.readouterr().out)["checks"]
    rows = [(row["level"], row["message"]) for row in checks if row["area"] == "extraction"]
    level, message = rows[0]
    if built == "plugged":
        assert level == "PASS" and "a plugin's upload inspector (_Plugged)" in message
        assert _Plugged.closed
    else:
        assert level == "FAIL" and "will refuse to start" in message
        assert ("the plugin's own setting" in message) is (built == "raises")
