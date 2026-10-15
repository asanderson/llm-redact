"""The llm NER backend's key for a server that requires one
([detection.ner.llm] api_key_env).

The configuration names an environment variable; its value is read at every
detector build and sent as ``Authorization: Bearer`` to the configured URL
only — never logged, echoed, in ``/status`` (only whether a key is
configured), in an error or in a repr.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.detection.llm_ner as llm_ner
from llm_redact.config import ConfigError
from llm_redact.config_write import emit_config_toml
from llm_redact.detection.base import DetectorUnavailable
from llm_redact.detection.engine import (
    DetectionConfig,
    build_allowlist,
    build_detectors,
    ner_backend_stats,
)
from llm_redact.redactor import Redactor
from llm_redact.vault import InMemoryVault
from ner_fakes import LLM_MODEL, FakeModelServer, install_llm, llm_ner_config
from test_llm_ner import LLM_TOML, NAME, PROXY, _app, _build, _chat, _doctor, _toml

KEY = "sk-llm-SERVERKEY-0123456789abcdef"
VAR = "LLM_SERVER_API_KEY"
OPENAI_TOML = (
    '[detection.ner]\nenabled = true\nbackend = "llm"\n\n[detection.ner.llm]\n'
    'api = "openai"\nmodel = "m"\nurl = "http://127.0.0.1:8000"\n'
)


def _keyed(
    monkeypatch: pytest.MonkeyPatch, server: FakeModelServer, **llm: Any
) -> tuple[Redactor, Any]:
    """A redactor over the rules plus the llm backend with ``api_key_env =
    VAR``, on ``server`` behind a guard answering 401 to a request that
    does not carry the key (and recording it); and the llm detector."""

    def guarded(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {KEY}":
            server.requests.append(request)
            return httpx.Response(401, json={"error": "unauthorized"})
        return server(request)

    monkeypatch.setattr(llm_ner, "make_transport", lambda: httpx.MockTransport(guarded))
    config = DetectionConfig(ner=llm_ner_config(api_key_env=VAR, **llm))
    detectors = build_detectors(config, startup=True)
    ((_n, detector, _s),) = ner_backend_stats(detectors)
    return Redactor(detectors, InMemoryVault(), build_allowlist(config)), detector


# --- configuration ----------------------------------------------------------------------


def test_api_key_env_parses_and_round_trips_as_a_name() -> None:
    config = _toml(OPENAI_TOML + f'api_key_env = "{VAR}"\n')
    assert config.detection.ner.llm.api_key_env == VAR
    emitted = emit_config_toml(config)
    assert f'api_key_env = "{VAR}"' in emitted
    assert _toml(emitted) == config
    assert "api_key_env" not in emit_config_toml(_toml(OPENAI_TOML))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (f'"{KEY}"', "must name an environment variable"),  # the key pasted in
        ('"MY VAR"', "must name an environment variable"),
        ('""', "must name an environment variable"),
        ("42", "must name an environment variable"),
        ('"LLM_REDACT_VAULT_KEY"', "must not name an LLM_REDACT_ variable"),
        ('"llm_redact_audit_hmac_key"', "must not name an LLM_REDACT_ variable"),
    ],
)
def test_a_bad_api_key_env_is_refused_without_echoing_it(value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message) as raised:
        _toml(OPENAI_TOML + f"api_key_env = {value}\n")
    assert KEY not in str(raised.value)


@pytest.mark.parametrize("value", [None, "", "   \n"], ids=["unset", "empty", "blank"])
def test_an_unset_or_empty_variable_stops_the_build(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    server = install_llm(monkeypatch, FakeModelServer())
    if value is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, value)
    with pytest.raises(ConfigError) as raised:
        build_detectors(DetectionConfig(ner=llm_ner_config(api_key_env=VAR)), startup=True)
    assert str(raised.value) == (
        f"[detection.ner.llm] api_key_env: the environment variable {VAR} is unset or empty"
    )
    assert server.requests == []  # refused before any request


def test_a_key_a_header_cannot_carry_is_refused_unshown(monkeypatch: pytest.MonkeyPatch) -> None:
    install_llm(monkeypatch, FakeModelServer())
    monkeypatch.setenv(VAR, "sk-two words")
    with pytest.raises(ConfigError, match="only visible ASCII") as raised:
        build_detectors(DetectionConfig(ner=llm_ner_config(api_key_env=VAR)))
    assert "two words" not in str(raised.value) and VAR in str(raised.value)


# --- the requests -----------------------------------------------------------------------


def test_every_request_to_an_openai_compatible_server_carries_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VAR, f"  {KEY}\n")  # edge blanks (a file's newline) trimmed
    server = FakeModelServer([(NAME, "person")], api="openai")
    _redactor, detector = _keyed(
        monkeypatch, server, api="openai", digest=None, url="http://127.0.0.1:8000"
    )
    assert [d.value for d in detector.detect(f"hi {NAME}")] == [NAME]
    assert [(r.url.path, r.headers["authorization"]) for r in server.requests] == [
        ("/v1/models", f"Bearer {KEY}"),
        ("/v1/chat/completions", f"Bearer {KEY}"),
    ]


def test_ollama_takes_the_key_too(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ollama behind an authenticating reverse proxy, or Ollama's own hosted
    # API, reads the same bearer header.
    monkeypatch.setenv(VAR, KEY)
    server = FakeModelServer([(NAME, "person")])
    redactor, detector = _keyed(monkeypatch, server)
    assert redactor.redact_text(f"hi {NAME}") == "hi «PERSON_001»"
    assert [(r.url.path, r.headers["authorization"]) for r in server.requests] == [
        ("/api/tags", f"Bearer {KEY}"),
        ("/api/chat", f"Bearer {KEY}"),
    ]
    assert detector.server_status()["api_key"] is True


def test_without_api_key_env_no_authorization_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VAR, KEY)  # set, but not named: never read
    server = FakeModelServer([(NAME, "person")])
    _redactor, detector = _build(monkeypatch, server)
    detector.detect(f"hi {NAME}")
    assert all("authorization" not in r.headers for r in server.requests)
    assert detector.server_status()["api_key"] is False


def test_a_wrong_key_is_a_failure_that_names_only_the_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(VAR, KEY)
    _redactor, detector = _keyed(monkeypatch, FakeModelServer([(NAME, "person")]))
    monkeypatch.setenv(VAR, "another-key")  # read at the build only: no effect
    assert [d.value for d in detector.detect(f"hi {NAME}")] == [NAME]
    detector._client._api_key = llm_ner.ApiKey("wrong-key")
    with pytest.raises(DetectorUnavailable) as raised:
        detector.detect(f"bye {NAME}")
    assert raised.value.reason == "HTTP 401"
    assert detector.breaker.status()["state"] == "closed"  # the server answers


def test_a_redirect_never_takes_the_key_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VAR, KEY)
    server = FakeModelServer([(NAME, "person")], fail="redirect")
    _redactor, detector = _build(monkeypatch, server, on_error="skip", api_key_env=VAR)
    assert detector.detect(f"hi {NAME}") == []
    assert [str(r.url) for r in server.requests] == [
        "http://127.0.0.1:11434/api/tags",
        "http://127.0.0.1:11434/api/chat",
    ]
    assert detector.stats.errors == 1


# --- never shown --------------------------------------------------------------------------


async def test_the_key_appears_in_no_log_repr_error_status_or_metric(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(VAR, KEY)
    server = FakeModelServer([(NAME, "person")], fail="status", fail_on="secret")
    with caplog.at_level(logging.DEBUG):
        app, _upstream = _app(monkeypatch, server, api_key_env=VAR)
        assert (await _chat(app, f"hi {NAME}")).status_code == 200
        refused = await _chat(app, f"secret {NAME}")
        assert refused.status_code == 503
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=PROXY
        ) as client:
            status = (await client.get("/__llm-redact/status")).text
            metrics = (await client.get("/__llm-redact/metrics")).text
    state = app.state.proxy
    ((_n, detector, _s),) = ner_backend_stats(state.detectors)
    entry = json.loads(status)["detection"]["ner"]["backends"]["llm"]
    assert entry["api_key"] is True
    client = detector._client
    shown = [
        caplog.text,
        refused.text,
        status,
        metrics,
        repr(detector),
        repr(client),
        repr(client._api_key),
        str(client._api_key),
        repr(vars(client)),
        repr(client._http.headers),
        repr(state.config),
        emit_config_toml(state.config),
        str(DetectorUnavailable("llm", "HTTP 500")),
        json.dumps(state.preview(f"secret {NAME}")),
    ]
    for text in shown:
        assert KEY not in text
    assert VAR not in status  # only whether a key is configured
    # The requests did carry it.
    assert {r.headers["authorization"] for r in server.requests} == {f"Bearer {KEY}"}


def test_the_real_transport_logs_no_key_at_debug(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The production transport (httpx over httpcore, whose DEBUG trace logs
    # each request), its network replaced by httpcore's in-memory backend.
    import httpcore

    sent: list[bytes] = []

    class Stream(httpcore.MockStream):
        def write(self, buffer: bytes, timeout: float | None = None) -> None:
            sent.append(bytes(buffer))

    class Backend(httpcore.MockBackend):
        def connect_tcp(self, *args: Any, **kwargs: Any) -> httpcore.MockStream:
            return Stream(list(self._buffer))

    answer = json.dumps({"models": []}).encode()
    reply = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
    reply += b"content-length: %d\r\n\r\n" % len(answer) + answer
    transport = httpx.HTTPTransport()
    transport._pool = httpcore.ConnectionPool(network_backend=Backend([reply]))
    monkeypatch.setattr(llm_ner, "make_transport", lambda: transport)
    client = llm_ner._Client(llm_ner_config().llm, ["person"], llm_ner.ApiKey(KEY))
    with caplog.at_level(logging.DEBUG):
        assert client.models(5) == {"models": []}
    client.close()
    assert any(record.name.startswith("httpcore") for record in caplog.records)
    assert KEY not in caplog.text
    # It did go out, on the wire.
    assert f"\r\nauthorization: Bearer {KEY}\r\n".encode() in b"".join(sent)


def test_a_build_failure_names_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(VAR, KEY)
    server = install_llm(monkeypatch, FakeModelServer(listed=False))
    with pytest.raises(ConfigError) as raised:
        build_detectors(DetectionConfig(ner=llm_ner_config(api_key_env=VAR)))
    assert KEY not in str(raised.value) and "does not serve this model" in str(raised.value)
    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"


async def test_a_reload_reads_the_variable_again_and_keeps_running_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    monkeypatch.setenv(VAR, KEY)
    server = FakeModelServer([(NAME, "person")])
    app, _upstream = _app(monkeypatch, server, api_key_env=VAR)
    state = app.state.proxy
    ((_n, before, _s),) = ner_backend_stats(state.detectors)
    monkeypatch.delenv(VAR)
    ner = dataclasses.replace(state.config.detection.ner, entities=("PERSON", "ADDRESS"))
    fresh = dataclasses.replace(
        state.config, detection=dataclasses.replace(state.config.detection, ner=ner)
    )
    with pytest.raises(ConfigError, match=f"{VAR} is unset or empty"):
        state.apply_config(fresh)
    ((_n, kept, _s),) = ner_backend_stats(state.detectors)
    assert kept is before  # the running detector, with the key it was built with
    assert (await _chat(app, f"hi {NAME}")).status_code == 200


# --- doctor -------------------------------------------------------------------------------


def test_doctor_checks_the_variable_offline_and_never_shows_the_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from llm_redact.doctor_cli import _check_llm, _Report

    server = install_llm(monkeypatch, FakeModelServer())
    toml = LLM_TOML + f'api_key_env = "{VAR}"\n'
    monkeypatch.delenv(VAR, raising=False)
    rows = _doctor(tmp_path, toml)
    assert rows["llm"][-1] == (
        "FAIL",
        f"[detection.ner.llm] api_key_env: the environment variable {VAR} is unset or empty"
        " — serve would refuse to start",
    )
    assert server.requests == []  # not asked without the key
    monkeypatch.setenv(VAR, KEY)
    report = _Report(json_mode=True)
    _check_llm(report, _toml(toml), offline=True)
    assert [(r["level"], r["message"]) for r in report.rows] == [
        ("PASS", "model server http://127.0.0.1:11434 is on loopback (ollama API)"),
        ("PASS", f"api key: {VAR} is set (sent as a bearer token to the model server only)"),
        ("PASS", "model server not asked (--offline)"),
    ]
    assert server.requests == []
    rows = _doctor(tmp_path, toml)
    assert rows["llm"][-1] == ("PASS", f"model server serves {LLM_MODEL!r} at the pinned digest")
    assert KEY not in json.dumps(rows)
    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"


def test_the_example_config_documents_both_keys() -> None:
    example = (Path(__file__).resolve().parent.parent / "config.example.toml").read_text()
    assert "# api_key_env = " in example and "# breaker_seconds = 30" in example
