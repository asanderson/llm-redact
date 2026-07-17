import argparse
from typing import Any

import pytest

from llm_redact.cli import main, run_status


def _args(port: int) -> argparse.Namespace:
    return argparse.Namespace(config=None, port=port, json=False)


def test_status_proxy_not_running(capsys: pytest.CaptureFixture[str]) -> None:
    # Nothing listens on this port: the command reports and exits nonzero.
    exit_code = run_status(_args(port=1))
    assert exit_code == 1
    assert "not reachable" in capsys.readouterr().out


def test_status_error_never_echoes_url_or_user_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Security review 3.1.1: a /u/<key> credential in LLM_REDACT_PROXY_URL must
    # never reach the terminal via an httpx exception (HTTPStatusError embeds
    # the full request URL). Only the netloc + a URL-free reason are printed.
    import httpx

    monkeypatch.setenv("LLM_REDACT_PROXY_URL", "http://127.0.0.1:8787/u/lrk_secretkey")

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("GET", url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError(
            f"404 Not Found for url '{url}'", request=request, response=response
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    args = argparse.Namespace(config=None, port=None, json=False, ca=None, cert=None, key=None)
    assert run_status(args) == 1
    out = capsys.readouterr().out
    assert "lrk_secretkey" not in out
    assert "/u/" not in out
    assert "127.0.0.1:8787" in out and "HTTP 404" in out


def test_posture_clean_when_nothing_opted_out(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.cli import _print_posture

    _print_posture({"warnings_total": {}, "detection": {}, "audit": {}})
    assert "all traffic redacted" in capsys.readouterr().out


def test_posture_surfaces_every_runtime_opt_out(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.cli import _print_posture

    _print_posture(
        {
            "warnings_total": {"PHONE": 3},
            "providers_detection_off": ["gemini"],
            "mcp_exempt_servers": 2,
            "detection": {"language_inactive_rules": ["french_nir"]},
            "compaction_forks": 1,
            "audit": {"s3": {"rows_dropped": 5}, "azure": {"rows_dropped": 0}},
            "providers_disabled": ["bedrock"],
        }
    )
    out = capsys.readouterr().out
    assert "PHONE×3" in out and "FORWARDED" in out
    assert "detection OFF for: gemini" in out
    assert "MCP exempt servers: 2" in out
    assert "french_nir" in out
    assert "compaction forks: 1" in out
    assert "audit.s3: 5 rows dropped" in out
    assert "audit.azure" not in out  # zero drops stay quiet
    assert "providers disabled (fail closed): bedrock" in out


def test_posture_surfaces_licensed_features_not_registered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from llm_redact.cli import _print_posture

    # Package present but no plugin registered → paid features silently OFF,
    # which the posture block must surface loudly.
    _print_posture(
        {"license": {"package_installed": True, "plugins": []}, "detection": {}, "audit": {}}
    )
    out = capsys.readouterr().out
    assert "no plugin registered" in out and "paid features OFF" in out

    # Installed AND registered → not an opt-out; stays quiet.
    _print_posture(
        {
            "license": {"package_installed": True, "plugins": ["llm_redact_pro"]},
            "detection": {},
            "audit": {},
        }
    )
    assert "all traffic redacted" in capsys.readouterr().out


def _full_status_payload(**license_extra: Any) -> dict[str, Any]:
    """A complete-enough /status body for run_status to render without KeyError."""
    return {
        "version": "4.1.0",
        "uptime_seconds": 1.0,
        "session": "default",
        "vault": {"backend": "memory", "entries": 0},
        "detections_total": {},
        "rehydrations_total": {},
        "audit": {"enabled": False, "rows": 0},
        "rehydration": {"fuzzy": True},
        "detection": {"ner_enabled": False},
        "providers": {},
        "license": {"tier": "free", "max_users": 1, "clouds": [], **license_extra},
    }


def _status_run(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    import httpx

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    args = argparse.Namespace(config=None, port=8787, json=False, ca=None, cert=None, key=None)
    assert run_status(args) == 0


def test_status_prints_licensed_features_not_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _status_run(monkeypatch, _full_status_payload(package_installed=False, plugins=[]))
    assert (
        "licensed-features package: not installed (FOSS core is complete)"
        in capsys.readouterr().out
    )


def test_status_prints_licensed_features_installed_active(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _status_run(
        monkeypatch, _full_status_payload(package_installed=True, plugins=["llm_redact_pro"])
    )
    assert "licensed-features package: installed (llm_redact_pro active)" in capsys.readouterr().out


def test_status_omits_licensed_features_line_on_old_proxy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A pre-4.1 proxy omits the field entirely — never invent a state for it.
    _status_run(monkeypatch, _full_status_payload())
    assert "licensed-features package" not in capsys.readouterr().out


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact import __version__

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_serve_port_override_reaches_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """--port must be baked into the config so /status reports the real port,
    and access logging stays off (access-log lines carry query strings,
    which can include provider API keys)."""
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured.update(kwargs)
        captured["config"] = app.state.proxy.config

    monkeypatch.setattr(uvicorn, "run", fake_run)
    main(["serve", "--port", "19999"])
    assert captured["port"] == 19999
    assert captured["config"].port == 19999  # /status reads this
    assert captured["access_log"] is False
