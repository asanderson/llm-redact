"""The llm backend's destination is restart-only although ``[detection]``
reloads hot (config.RESTART_ONLY_PATHS: ``[detection.ner.llm]`` url,
allow_remote and api_key_env).

Whoever can change ``[detection]`` — a file edit and SIGHUP, the
llm-redact-pro config editor — must not point the backend at another
server, name another environment variable as the key it sends, or start
it: a running proxy never begins to send scanned text, or a credential, to a
destination it was not started with. A reload keeps the running values,
applies the rest of ``[detection]`` and names the kept keys (dotted) among
the restart-required ones; a reload that would start the backend, or that
the kept values make invalid, is refused with the running configuration
kept. The editor's dry run (validate_config) agrees with the apply.

The server is ``ner_fakes.FakeModelServer`` behind an
``httpx.MockTransport``: every request the backend makes is recorded, at
whatever URL it was sent to.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_redact import plugin_api
from llm_redact.config import (
    RESTART_ONLY_KEYS,
    RESTART_ONLY_PATHS,
    Config,
    ConfigError,
    llm_url_problem,
    load_config,
    pin_restart_only_paths,
    restart_only_name,
)
from llm_redact.detection.engine import DetectionConfig, LlmNerConfig, NerConfig, ner_backend_stats
from llm_redact.proxy import ProxyState
from ner_fakes import FakeModelServer, install_llm, llm_ner_config
from test_llm_ner import LLM_TOML, NAME, _app, _chat

NAMES = [restart_only_name(path) for path in RESTART_ONLY_PATHS]
REMOTE = "https://llm.elsewhere.example"
OTHER_VAR = "SOME_OTHER_SECRET"
OTHER_VALUE = "value-of-another-variable-0123456789"


def _with(config: Config, *, ner: dict[str, Any] | None = None, **llm: Any) -> Config:
    """``config`` with ``[detection.ner]`` fields ``ner`` and
    ``[detection.ner.llm]`` fields ``llm`` replaced."""
    current = config.detection.ner
    replaced = dataclasses.replace(
        current, llm=dataclasses.replace(current.llm, **llm), **ner or {}
    )
    return dataclasses.replace(
        config, detection=dataclasses.replace(config.detection, ner=replaced)
    )


def _sent(server: FakeModelServer) -> set[tuple[str, str | None]]:
    """Every (origin, authorization header) the backend sent to."""
    return {
        (
            f"{request.url.scheme}://{request.url.netloc.decode()}",
            request.headers.get("authorization"),
        )
        for request in server.requests
    }


def test_the_paths_name_the_llm_destination_and_no_top_level_key() -> None:
    assert NAMES == [
        "detection.ner.llm.url",
        "detection.ner.llm.allow_remote",
        "detection.ner.llm.api_key_env",
    ]
    # Inside a hot section: [detection] itself stays hot.
    assert "detection" not in RESTART_ONLY_KEYS
    for path in RESTART_ONLY_PATHS:
        assert hasattr(LlmNerConfig(), path[-1])


# --- a reload keeps the running destination ------------------------------------------------


async def test_a_reload_keeps_url_allow_remote_and_api_key_env_and_applies_the_rest(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(OTHER_VAR, OTHER_VALUE)
    server = FakeModelServer([(NAME, "person")])
    app, upstream = _app(monkeypatch, server)
    state = app.state.proxy
    ((_n, before, _s),) = ner_backend_stats(state.detectors)
    running = state.config.detection.ner.llm
    fresh = _with(
        state.config,
        ner={"entities": ("PERSON", "ADDRESS")},
        url=REMOTE,
        allow_remote=True,
        api_key_env=OTHER_VAR,
        timeout_ms=5000,
    )
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        assert state.apply_config(fresh) == NAMES
    llm = state.config.detection.ner.llm
    assert (llm.url, llm.allow_remote, llm.api_key_env) == (running.url, False, None)
    # The rest of [detection] applied: the backend was rebuilt with it.
    assert llm.timeout_ms == 5000
    assert state.config.detection.ner.entities == ("PERSON", "ADDRESS")
    ((_n, after, _s),) = ner_backend_stats(state.detectors)
    assert after is not before and after.server == "http://127.0.0.1:11434"
    assert (await _chat(app, f"hi {NAME}")).status_code == 200
    assert NAME not in upstream.bodies[-1].decode()
    # Every request — the rebuild's model list, the chat — went to the
    # server the proxy was started with, and carried no key.
    assert _sent(server) == {("http://127.0.0.1:11434", None)}
    assert all(OTHER_VALUE not in str(request.headers) for request in server.requests)
    messages = [record.getMessage() for record in caplog.records]
    for name in NAMES:
        assert f"config reload: {name} changes require restart; keeping current" in messages
    # Names only, never a value.
    assert REMOTE not in caplog.text and OTHER_VAR not in caplog.text


def test_a_sighup_reload_of_the_file_keeps_them_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    path = tmp_path / "config.toml"
    path.write_text(LLM_TOML)
    state = ProxyState(load_config(path), None, config_path=path)
    path.write_text(
        LLM_TOML
        + f'url = "{REMOTE}"\nallow_remote = true\napi_key_env = "{OTHER_VAR}"\n'
        + 'on_error = "skip"\n'
    )
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        state.reload()
    llm = state.config.detection.ner.llm
    assert (llm.url, llm.allow_remote, llm.api_key_env, llm.on_error) == (
        "http://127.0.0.1:11434",
        False,
        None,
        "skip",
    )
    assert "config reload failed" not in caplog.text
    assert "config reload: detection.ner.llm.url changes require restart" in caplog.text
    assert _sent(server) == {("http://127.0.0.1:11434", None)}


def test_only_the_changed_paths_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    install_llm(monkeypatch, FakeModelServer())
    state = ProxyState(Config(detection=DetectionConfig(ner=llm_ner_config())), None)
    assert state.apply_config(state.config) == []
    assert state.apply_config(_with(state.config, on_error="skip", breaker_failures=1)) == []
    assert state.apply_config(_with(state.config, api_key_env=OTHER_VAR)) == [
        "detection.ner.llm.api_key_env"
    ]
    # Beside a restart-only section, in the order apply_config reports them.
    fresh = dataclasses.replace(_with(state.config, url="http://localhost:11434"), port=9999)
    assert state.apply_config(fresh) == ["port", "detection.ner.llm.url"]


def test_the_dashboard_host_contract_names_the_dotted_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``plugin_api.DashboardHost`` is the plugin contract for apply_config:
    it must say that an entry is a section name OR a dotted path (a plugin
    reading each as a ``Config`` attribute would misreport the new ones),
    and every name apply_config returns is one of the two."""
    doc = " ".join((plugin_api.DashboardHost.__doc__ or "").split())
    for term in ("RESTART_ONLY_KEYS", "RESTART_ONLY_PATHS", "restart_only_name", NAMES[0]):
        assert term in doc, term
    install_llm(monkeypatch, FakeModelServer())
    state = ProxyState(Config(detection=DetectionConfig(ner=llm_ner_config())), None)
    fresh = _with(
        state.config, url="http://localhost:11434", allow_remote=True, api_key_env=OTHER_VAR
    )
    reported = state.apply_config(dataclasses.replace(fresh, host="127.0.0.2", port=9999))
    assert reported == ["host", "port", *NAMES]
    assert all(name in RESTART_ONLY_KEYS or name in NAMES for name in reported)
    assert [name for name in reported if not hasattr(Config, name)] == NAMES


def test_a_reload_that_removes_the_backend_keeps_its_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_llm(monkeypatch, FakeModelServer())
    state = ProxyState(Config(detection=DetectionConfig(ner=llm_ner_config())), None)
    off = _with(state.config, ner={"enabled": False}, url=REMOTE, allow_remote=True)
    assert state.apply_config(off) == NAMES[:2]
    assert not ner_backend_stats(state.detectors)
    assert state.config.detection.ner.llm.url == "http://127.0.0.1:11434"


# --- refused: a reload never starts the backend ----------------------------------------------


_NOT_RUNNING = {
    "the defaults (NER off, no table)": Config(),
    "llm the backend, NER off": Config(
        detection=DetectionConfig(ner=dataclasses.replace(llm_ner_config(), enabled=False))
    ),
    "its table set, another backend, NER off": Config(
        detection=DetectionConfig(
            ner=NerConfig(enabled=False, backends=("spacy",), llm=llm_ner_config().llm)
        )
    ),
}


@pytest.mark.parametrize("start", list(_NOT_RUNNING))
@pytest.mark.parametrize("destination", ["unchanged", "elsewhere"])
def test_a_reload_that_would_start_the_backend_is_refused(
    monkeypatch: pytest.MonkeyPatch, start: str, destination: str
) -> None:
    monkeypatch.setenv(OTHER_VAR, OTHER_VALUE)
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    state = ProxyState(_NOT_RUNNING[start], None)
    before = (state.config, state.detectors)
    running = state.config.detection.ner.llm
    llm = (
        {"url": running.url, "allow_remote": running.allow_remote}
        if destination == "unchanged"
        else {"url": REMOTE, "allow_remote": True, "api_key_env": OTHER_VAR}
    )
    fresh = dataclasses.replace(state.config, detection=DetectionConfig(ner=llm_ner_config(**llm)))
    for attempt in (state.validate_config, state.apply_config):
        with pytest.raises(ConfigError) as raised:
            attempt(fresh)
        message = str(raised.value)
        assert "a reload cannot start the llm backend" in message
        assert all(name in message for name in NAMES)
        assert REMOTE not in message and OTHER_VAR not in message
    assert (state.config, state.detectors) == before
    assert server.requests == []  # no server was asked anything


def test_a_backend_a_reload_stopped_is_not_started_again_by_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = install_llm(monkeypatch, FakeModelServer())
    path = tmp_path / "config.toml"
    path.write_text(LLM_TOML)
    state = ProxyState(load_config(path), None, config_path=path)
    path.write_text(LLM_TOML.replace("enabled = true", "enabled = false"))
    state.reload()
    assert not state.config.detection.ner.enabled
    asked = len(server.requests)
    path.write_text(LLM_TOML)
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        state.reload()
    assert not state.config.detection.ner.enabled  # the running configuration kept
    assert "config reload failed; keeping current config" in caplog.text
    assert "a reload cannot start the llm backend" in caplog.text
    assert len(server.requests) == asked


def test_a_reload_the_kept_url_makes_invalid_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # The running url, valid for the running api, is no base URL of the new
    # one: kept, it could not work — refused, never applied half way.
    server = install_llm(monkeypatch, FakeModelServer())
    dormant = dataclasses.replace(llm_ner_config(url="http://127.0.0.1:11434/v1"), enabled=False)
    state = ProxyState(Config(detection=DetectionConfig(ner=dormant)), None)
    before = state.config
    fresh = _with(state.config, api="openai", url="http://127.0.0.1:8000", digest=None)
    for attempt in (state.validate_config, state.apply_config):
        with pytest.raises(ConfigError, match="without /v1") as raised:
            attempt(fresh)
        assert "the running url is kept on a reload" in str(raised.value)
        assert all(name in str(raised.value) for name in NAMES)
    assert state.config is before and server.requests == []


# --- the editor's dry run agrees ---------------------------------------------------------------


def test_the_dry_run_builds_what_the_apply_would_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OTHER_VAR, OTHER_VALUE)
    server = install_llm(monkeypatch, FakeModelServer())
    state = ProxyState(Config(detection=DetectionConfig(ner=llm_ner_config())), None)
    asked = len(server.requests)
    # Only the kept keys changed: [detection] as applied is the running one,
    # so nothing is built (as apply_config keeps the live detectors).
    moved = _with(state.config, url=REMOTE, allow_remote=True, api_key_env=OTHER_VAR)
    state.validate_config(moved)
    assert len(server.requests) == asked
    # With another change too, the build asks the running server, keyless.
    state.validate_config(_with(moved, timeout_ms=5000))
    assert len(server.requests) == asked + 1
    assert _sent(server) == {("http://127.0.0.1:11434", None)}
    assert state.config.detection.ner.llm.timeout_ms == 30000  # a dry run changes nothing


# --- the pinning itself ------------------------------------------------------------------------


_URLS = (
    "http://127.0.0.1:11434",
    "http://localhost:8000",
    "https://llm.elsewhere.example",
    "http://127.0.0.1:11434/v1",
    "http://127.0.0.1:11434/api",
)


@st.composite
def _configs(draw: st.DrawFn) -> Config:
    llm = LlmNerConfig(
        url=draw(st.sampled_from(_URLS)),
        api=draw(st.sampled_from(("ollama", "openai"))),
        allow_remote=draw(st.booleans()),
        api_key_env=draw(st.sampled_from((None, "KEY_A", "KEY_B"))),
        timeout_ms=draw(st.sampled_from((1000, 30000))),
    )
    ner = NerConfig(
        enabled=draw(st.booleans()),
        backends=draw(st.sampled_from((("llm",), ("spacy",), ("spacy", "llm")))),
        llm=llm,
    )
    return Config(detection=DetectionConfig(ner=ner))


def _llm_runs(config: Config) -> bool:
    ner = config.detection.ner
    return ner.enabled and "llm" in ner.active_backends()


def _value(config: Config, path: tuple[str, ...]) -> Any:
    value: Any = config
    for name in path:
        value = getattr(value, name)
    return value


@given(_configs(), _configs())
def test_a_hot_apply_never_moves_or_starts_the_backend(fresh: Config, running: Config) -> None:
    try:
        pinned, changed = pin_restart_only_paths(fresh, running)
    except ConfigError as refused:
        message = str(refused)
        assert all(name in message for name in NAMES)
        assert "elsewhere" not in message and "KEY_" not in message
        # Refused only for a start, or for kept values that cannot work.
        kept = _with(
            fresh,
            **{path[-1]: _value(running, path) for path in RESTART_ONLY_PATHS},
        )
        llm = kept.detection.ner.llm
        starts = _llm_runs(fresh) and not _llm_runs(running)
        unusable = llm_url_problem(llm.url, allow_remote=llm.allow_remote) is not None
        not_a_base = llm.url.endswith({"ollama": "/api", "openai": "/v1"}[llm.api])
        assert starts or unusable or not_a_base
        return
    # Applied: the destination is the running one, everything else fresh's.
    for path in RESTART_ONLY_PATHS:
        assert _value(pinned, path) == _value(running, path)
    assert changed == [
        restart_only_name(path)
        for path in RESTART_ONLY_PATHS
        if _value(fresh, path) != _value(running, path)
    ]
    restored = _with(pinned, **{path[-1]: _value(fresh, path) for path in RESTART_ONLY_PATHS})
    assert restored == fresh
    assert not _llm_runs(pinned) or _llm_runs(running)
