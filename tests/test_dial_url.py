"""Client-side probes dial the proxy they configure (P-19).

`status`, `doctor`, `run`, `plugin install`'s posture probe and `vault
rotate-key`'s liveness probe build a URL from the BIND address in the
config. A wildcard bind (`0.0.0.0`, `::`, empty) listens on every interface
but is dialed at loopback, and an IPv6 literal needs brackets in a URL:
`http://::1:8787` made httpx raise InvalidURL (not an HTTPError), so none of
them could reach an IPv6-bound proxy at all. No socket is opened here: every
probe is a monkeypatched `httpx.get`.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_redact.config import dial_host, dial_url

STATUS = "/__llm-redact/status"


@pytest.mark.parametrize(
    ("host", "dialed"),
    [
        ("0.0.0.0", "127.0.0.1"),
        ("", "127.0.0.1"),
        ("::", "[::1]"),
        ("0:0:0:0:0:0:0:0", "[::1]"),
        ("[::]", "[::1]"),
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "[::1]"),
        ("[::1]", "[::1]"),
        ("2001:db8::7", "[2001:db8::7]"),
        ("10.0.0.5", "10.0.0.5"),
        ("localhost", "localhost"),
        ("proxy.corp.example", "proxy.corp.example"),
    ],
)
def test_dial_host(host: str, dialed: str) -> None:
    assert dial_host(host) == dialed


def test_dial_url() -> None:
    assert dial_url("0.0.0.0", 8787) == "http://127.0.0.1:8787"
    assert dial_url("::", 8787, scheme="https") == "https://[::1]:8787"
    assert dial_url("::1", 9) == "http://[::1]:9"
    assert dial_url("proxy.corp.example", 443, scheme="https") == "https://proxy.corp.example:443"
    for host in ("0.0.0.0", "::", "", "::1", "2001:db8::7", "10.0.0.5", "localhost"):
        httpx.URL(dial_url(host, 8787) + STATUS)  # every one is a URL httpx accepts


def _config(tmp_path: Path, host: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(f'host = "{host}"\nport = 8787\n')
    return path


class _Recorder:
    """A stand-in for httpx.get: records every URL, answers 200 with a
    status body (or raises ``error``)."""

    def __init__(self, error: Exception | None = None) -> None:
        self.urls: list[str] = []
        self.error = error

    def __call__(self, url: str, **kwargs: Any) -> httpx.Response:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return httpx.Response(200, json=_status_payload(), request=httpx.Request("GET", url))


def _status_payload() -> dict[str, Any]:
    from llm_redact import __version__

    return {
        "version": __version__,
        "uptime_seconds": 1.0,
        "session": "default",
        "vault": {"backend": "memory", "entries": 0},
        "detections_total": {},
        "rehydrations_total": {},
        "audit": {"enabled": False, "rows": 0},
        "rehydration": {"fuzzy": True},
        "detection": {"ner_enabled": False},
        "providers": {},
    }


def _status_args(config: Path) -> argparse.Namespace:
    return argparse.Namespace(config=config, port=None, json=True, ca=None, cert=None, key=None)


@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("::1", "http://[::1]:8787"),
        ("::", "http://[::1]:8787"),
        ("0.0.0.0", "http://127.0.0.1:8787"),
        ("", "http://127.0.0.1:8787"),
        ("127.0.0.1", "http://127.0.0.1:8787"),
    ],
)
def test_status_dials_the_configured_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str, url: str
) -> None:
    from llm_redact.cli import run_status

    monkeypatch.delenv("LLM_REDACT_PROXY_URL", raising=False)
    monkeypatch.delenv("LLM_REDACT_HOST", raising=False)
    monkeypatch.delenv("LLM_REDACT_PORT", raising=False)
    get = _Recorder()
    monkeypatch.setattr(httpx, "get", get)
    assert run_status(_status_args(_config(tmp_path, host))) == 0
    assert get.urls == [url + STATUS]


def test_status_names_an_unreachable_ipv6_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.cli import run_status

    monkeypatch.delenv("LLM_REDACT_PROXY_URL", raising=False)
    monkeypatch.setenv("LLM_REDACT_HOST", "::")
    monkeypatch.delenv("LLM_REDACT_PORT", raising=False)
    monkeypatch.setattr(httpx, "get", _Recorder(httpx.ConnectError("refused")))
    assert run_status(_status_args(_config(tmp_path, "127.0.0.1"))) == 1
    out = capsys.readouterr().out
    assert "proxy not reachable at [::1]:8787 (ConnectError)" in out


def test_status_reports_a_host_no_url_can_carry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # httpx.InvalidURL is not an HTTPError: it is reported, never a traceback.
    from llm_redact.cli import run_status

    monkeypatch.delenv("LLM_REDACT_PROXY_URL", raising=False)
    monkeypatch.delenv("LLM_REDACT_HOST", raising=False)
    monkeypatch.setattr(httpx, "get", _Recorder(httpx.InvalidURL("bad host")))
    assert run_status(_status_args(_config(tmp_path, "proxy.corp.example"))) == 1
    assert "(InvalidURL)" in capsys.readouterr().out


def test_doctor_dials_and_names_the_configured_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact import doctor_cli
    from llm_redact.config import Config

    get = _Recorder()
    monkeypatch.setattr(httpx, "get", get)
    report = doctor_cli._Report(json_mode=True)
    doctor_cli._check_proxy(report, Config(host="::", port=8787))
    assert get.urls == ["http://[::1]:8787" + STATUS]
    assert report.rows[-1]["level"] == "PASS"
    assert report.rows[-1]["message"].endswith(" at [::1]:8787")


class _FakeSocket:
    """socket.socket for the port probe: records the family and the bind."""

    made: list[tuple[int, tuple[str, int]]] = []

    def __init__(self, family: int, kind: int) -> None:
        self.family = family

    def bind(self, address: tuple[str, int]) -> None:
        _FakeSocket.made.append((self.family, address))

    def close(self) -> None:
        pass


def test_doctor_probes_an_ipv6_port_with_an_ipv6_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_redact import doctor_cli
    from llm_redact.config import Config

    monkeypatch.setattr(httpx, "get", _Recorder(httpx.ConnectError("refused")))
    monkeypatch.setattr(doctor_cli.socket, "socket", _FakeSocket)
    _FakeSocket.made = []
    report = doctor_cli._Report(json_mode=True)
    doctor_cli._check_proxy(report, Config(host="::1", port=8787))
    assert _FakeSocket.made == [(socket.AF_INET6, ("::1", 8787))]
    messages = [row["message"] for row in report.rows]
    assert messages[0].startswith("not running at [::1]:8787 (")
    assert messages[1] == "port 8787 is free"
    _FakeSocket.made = []
    report = doctor_cli._Report(json_mode=True)
    doctor_cli._check_proxy(report, Config(host="0.0.0.0", port=8787))
    assert _FakeSocket.made == [(socket.AF_INET, ("0.0.0.0", 8787))]
    assert report.rows[0]["message"].startswith("not running at 127.0.0.1:8787 (")


def test_doctor_says_when_the_address_family_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An IPv6 bind on a host without IPv6: the port cannot be probed, and
    # doctor says so instead of claiming something else holds it.
    from llm_redact import doctor_cli
    from llm_redact.config import Config

    def no_ipv6(family: int, kind: int) -> Any:
        raise OSError(97, "Address family not supported by protocol")

    monkeypatch.setattr(httpx, "get", _Recorder(httpx.ConnectError("refused")))
    monkeypatch.setattr(doctor_cli.socket, "socket", no_ipv6)
    report = doctor_cli._Report(json_mode=True)
    doctor_cli._check_proxy(report, Config(host="::1", port=8787))
    assert report.rows[-1]["level"] == "WARN"
    assert "cannot probe port 8787" in report.rows[-1]["message"]


def test_run_exports_the_dialed_proxy_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    import llm_redact.run_cli as run_cli
    from llm_redact.run_cli import run_run

    monkeypatch.delenv("LLM_REDACT_PROXY_URL", raising=False)
    monkeypatch.delenv("LLM_REDACT_HOST", raising=False)
    monkeypatch.delenv("LLM_REDACT_PORT", raising=False)
    probed: list[str] = []
    monkeypatch.setattr(run_cli, "_proxy_running", lambda url: probed.append(url) or True)
    snippet = "import os; print('ANTH=' + os.environ['ANTHROPIC_BASE_URL'])"
    args = argparse.Namespace(
        config=_config(tmp_path, "::"),
        port=None,
        tools="claude",
        tool_command=["--", sys.executable, "-c", snippet],
        set_env=None,
        proxy_url=None,
    )
    assert run_run(args) == 0
    out, err = capfd.readouterr()
    assert probed == ["http://[::1]:8787"]
    assert "ANTH=http://[::1]:8787" in out
    assert "via http://[::1]:8787 (already running)" in err


def test_run_probe_survives_a_host_no_url_can_carry(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.run_cli import _proxy_running

    monkeypatch.setattr(httpx, "get", _Recorder(httpx.InvalidURL("bad host")))
    assert _proxy_running("http://bad host:8787") is False


def test_plugin_posture_probe_dials_the_configured_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.plugin_cli import _default_base_url

    monkeypatch.setenv("LLM_REDACT_CONFIG", str(_config(tmp_path, "127.0.0.1")))
    monkeypatch.setenv("LLM_REDACT_HOST", "::")
    monkeypatch.setenv("LLM_REDACT_PORT", "8788")
    assert _default_base_url() == "http://[::1]:8788"


def test_vault_liveness_probe_dials_the_configured_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.vault_cli import _proxy_reachable

    monkeypatch.delenv("LLM_REDACT_HOST", raising=False)
    monkeypatch.delenv("LLM_REDACT_PORT", raising=False)
    get = _Recorder()
    monkeypatch.setattr(httpx, "get", get)
    args = argparse.Namespace(config=_config(tmp_path, "::1"))
    assert _proxy_reachable(args) is True
    assert get.urls == ["http://[::1]:8787" + STATUS]
    # A host no URL can carry: no proxy can be reached there (the prompt
    # remains the backstop), never a traceback.
    monkeypatch.setattr(httpx, "get", _Recorder(httpx.InvalidURL("bad host")))
    assert _proxy_reachable(args) is False


def test_serve_banner_names_the_bind_and_the_dialed_status_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from llm_redact import serving
    from llm_redact.cli import main

    monkeypatch.setattr(serving, "run_server", lambda app, **kwargs: None)
    monkeypatch.setenv("LLM_REDACT_HOST", "::")
    monkeypatch.setenv("LLM_REDACT_INSECURE_BIND", "1")
    monkeypatch.setenv("LLM_REDACT_CONFIG", str(_config(tmp_path, "127.0.0.1")))
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        main(["serve", "--port", "19998"])
    assert (
        "serving on http://[::]:19998 — status http://[::1]:19998/__llm-redact/status"
        in caplog.text
    )
