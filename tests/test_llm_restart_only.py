"""The llm backend's destination is restart-only although ``[detection]``
reloads hot (config.RESTART_ONLY_PATHS: ``[detection.ner.llm]`` url,
allow_remote and api_key_env).

Whoever can change ``[detection]`` — a file edit and SIGHUP, the
llm-redact-pro config editor — must not point the backend at another
server or name another environment variable as the key it sends: a running
proxy never begins to send scanned text, or a credential, to a destination
it was not started with. A reload keeps the running values — the startup
ones: no apply moves them —, applies the rest of ``[detection]`` and names
the kept keys (dotted) among the restart-required ones. A reload STARTS
the backend (added to the backends, NER switched on, re-added after a
reload removed it) only to re-activate it: on a proxy started running it,
at those values. One that would start it on a proxy started without it
(at whatever values: a dormant table's, the defaults), or with any of them
changed, or that the kept values make invalid, is refused with the running
configuration kept. The editor's dry run (validate_config) and prepared
build (prepare_config) agree with the apply.

The server is ``ner_fakes.FakeModelServer`` behind an
``httpx.MockTransport``: every request the backend makes is recorded, at
whatever URL it was sent to.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from llm_redact import plugin_api
from llm_redact.config import (
    RESTART_ONLY_KEYS,
    RESTART_ONLY_PATHS,
    Config,
    ConfigError,
    llm_backend_runs,
    llm_url_problem,
    load_config,
    pin_restart_only_paths,
    restart_only_name,
)
from llm_redact.detection.engine import DetectionConfig, LlmNerConfig, NerConfig, ner_backend_stats
from llm_redact.proxy import ProxyState, create_app
from llm_redact.realtime import RealtimeRelay
from ner_fakes import LLM_MODEL, FakeModelServer, install_llm, llm_ner_config
from test_llm_ner import LLM_TOML, NAME, PROXY, _app, _chat, _Upstream

NAMES = [restart_only_name(path) for path in RESTART_ONLY_PATHS]
REMOTE = "https://llm.elsewhere.example"
OTHER_VAR = "SOME_OTHER_SECRET"
OTHER_VALUE = "value-of-another-variable-0123456789"
# The variable a startup configuration names as the model server's key.
KEY_VAR = "MODEL_SERVER_KEY"
KEY_VALUE = "key-of-the-model-server-0123456789"


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


async def _status(app: Any) -> Any:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PROXY) as c:
        return (await c.get("/__llm-redact/status")).json()


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


# --- a reload starts the backend again only where the proxy started it -----------------------


# The remote server a startup configuration names (never REMOTE, the
# server a reload names instead).
STARTED_REMOTE = "https://llm.started.example"


def _dormant(ner: NerConfig) -> Config:
    """A startup configuration holding ``ner``'s backend table, NER off."""
    return Config(detection=DetectionConfig(ner=dataclasses.replace(ner, enabled=False)))


# A proxy started WITHOUT the backend running: no reload starts it, at
# whatever values — the defaults, or a table its file held dormant.
_NOT_RUNNING = {
    "the defaults (NER off, no table)": Config(),
    "llm the backend, NER off": _dormant(llm_ner_config()),
    "its table set, another backend, NER off": Config(
        detection=DetectionConfig(
            ner=NerConfig(enabled=False, backends=("spacy",), llm=llm_ner_config().llm)
        )
    ),
    "its table naming a key, NER off": _dormant(llm_ner_config(api_key_env=KEY_VAR)),
    "its table on a remote server with a key, NER off": _dormant(
        llm_ner_config(url=STARTED_REMOTE, allow_remote=True, api_key_env=KEY_VAR)
    ),
}

# A proxy started running it, at these values.
_RUNNING = {
    "the default server": llm_ner_config(),
    "a key": llm_ner_config(api_key_env=KEY_VAR),
    "a remote server with a key": llm_ner_config(
        url=STARTED_REMOTE, allow_remote=True, api_key_env=KEY_VAR
    ),
}

# How a reload stopped it.
_STOPS = ("NER switched off", "its table and NER removed")

# One restart-only key changed, or all three: a start anywhere else.
_ELSEWHERE = ("url", "allow_remote", "api_key_env", "all three")


def _stopped(config: Config, how: str) -> Config:
    if how == "NER switched off":
        return _with(config, ner={"enabled": False})
    return dataclasses.replace(config, detection=DetectionConfig())


def _elsewhere(config: Config, moved: str) -> dict[str, Any]:
    """The ``[detection.ner.llm]`` fields a reload changes for ``moved``,
    each to a value other than ``config``'s."""
    _url, allow_remote, _key = _destination(config)
    every = {"url": REMOTE, "allow_remote": not allow_remote, "api_key_env": OTHER_VAR}
    return every if moved == "all three" else {moved: every[moved]}


def _destination(config: Config) -> tuple[str, bool, str | None]:
    llm = config.detection.ner.llm
    return llm.url, llm.allow_remote, llm.api_key_env


def _started(config: Config) -> Config:
    """``config`` with only the llm backend running, at ``config``'s own
    url, allow_remote and api_key_env."""
    url, allow_remote, api_key_env = _destination(config)
    ner = llm_ner_config(url=url, allow_remote=allow_remote, api_key_env=api_key_env)
    return dataclasses.replace(config, detection=DetectionConfig(ner=ner))


def _refusals(state: ProxyState, fresh: Config) -> list[str]:
    """The refusal of every way the proxy takes ``fresh``: the editor's dry
    run, its prepared build and the apply itself (SIGHUP's too) — which
    must all refuse it."""
    messages = []
    for attempt in (state.validate_config, state.prepare_config, state.apply_config):
        with pytest.raises(ConfigError) as raised:
            attempt(fresh)
        messages.append(str(raised.value))
    return messages


_VALUES = (REMOTE, STARTED_REMOTE, "elsewhere", "llm.started", OTHER_VAR, KEY_VAR)


@pytest.mark.parametrize("at", ["the startup values", *_ELSEWHERE])
@pytest.mark.parametrize("start", list(_NOT_RUNNING))
def test_a_reload_never_starts_the_backend_on_a_proxy_started_without_it(
    monkeypatch: pytest.MonkeyPatch, start: str, at: str
) -> None:
    """Not even at the url and key variable a dormant table names, nor at
    the defaults: a running proxy never begins to send scanned text, or a
    credential, to a model server it did not send to from its start."""
    monkeypatch.setenv(KEY_VAR, KEY_VALUE)
    monkeypatch.setenv(OTHER_VAR, OTHER_VALUE)
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    state = ProxyState(_NOT_RUNNING[start], None)
    assert not ner_backend_stats(state.detectors)
    before = (state.config, state.detectors)
    fresh = _started(state.config)
    if at != "the startup values":
        fresh = _with(fresh, **_elsewhere(state.config, at))
    for message in _refusals(state, fresh):
        assert message.startswith(
            "[detection.ner] a reload cannot start the llm backend: the proxy was not started"
            " running it ("
        )
        assert all(name in message for name in NAMES)
        # Names only, never a value.
        assert not any(value in message for value in _VALUES)
    assert (state.config, state.detectors) == before
    assert server.requests == []  # no server was asked anything


def test_a_sighup_never_switches_on_a_dormant_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The file the proxy started with held the backend's table — a remote
    server, a key variable — with NER off; a reload of it with NER on is
    refused and logged, the running configuration kept: nothing reaches
    the server, the key included."""
    monkeypatch.setenv(KEY_VAR, KEY_VALUE)
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    toml = LLM_TOML + (
        f'url = "{STARTED_REMOTE}"\nallow_remote = true\napi_key_env = "{KEY_VAR}"\n'
    )
    path = tmp_path / "config.toml"
    path.write_text(toml.replace("enabled = true", "enabled = false"))
    state = ProxyState(load_config(path), None, config_path=path)
    before = (state.config, state.detectors)
    path.write_text(toml)
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        state.reload()
    assert "config reload failed; keeping current config" in caplog.text
    assert "the proxy was not started running it" in caplog.text
    assert not any(value in caplog.text for value in _VALUES)
    assert (state.config, state.detectors) == before
    assert state.preview(f"hi {NAME}")["redacted"] == f"hi {NAME}"
    assert server.requests == []


@pytest.mark.parametrize("ran", [True, False], ids=["started running it", "started without it"])
async def test_the_sighup_reload_off_the_loop_follows_the_same_start_rule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    ran: bool,
) -> None:
    """What SIGHUP runs (``reload_off_loop``: the file read, its
    restart-only paths pinned and its ``[detection]`` built on a worker
    thread, then applied on the loop) starts the backend exactly when
    ``reload`` would: again, at the startup values, on a proxy started
    running it — the build off the loop knows the startup fact too —, and
    never on one started without it, where nothing reaches the server."""
    monkeypatch.setenv(KEY_VAR, KEY_VALUE)
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    toml = LLM_TOML + f'api_key_env = "{KEY_VAR}"\n'
    off = toml.replace("enabled = true", "enabled = false")
    path = tmp_path / "config.toml"
    path.write_text(toml if ran else off)
    state = ProxyState(load_config(path), None, config_path=path)
    if ran:
        path.write_text(off)
        assert await state.reload_off_loop() is True
        assert not llm_backend_runs(state.config.detection.ner)
    asked = len(server.requests)
    path.write_text(toml)
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        applied = await state.reload_off_loop()
    assert applied is ran
    assert llm_backend_runs(state.config.detection.ner) is ran
    redacted = state.preview(f"hi {NAME}")["redacted"]
    if ran:
        assert redacted == "hi «PERSON_001»"
        assert [request.url.path for request in server.requests[asked:]] == [
            "/api/tags",
            "/api/chat",
        ]
        assert _sent(server) == {("http://127.0.0.1:11434", f"Bearer {KEY_VALUE}")}
    else:
        assert redacted == f"hi {NAME}"
        assert "the proxy was not started running it" in caplog.text
        assert server.requests == []


@pytest.mark.parametrize("stop", _STOPS)
@pytest.mark.parametrize("start", list(_RUNNING))
def test_a_reload_starts_the_backend_again_at_the_values_the_proxy_started_with(
    monkeypatch: pytest.MonkeyPatch, start: str, stop: str
) -> None:
    monkeypatch.setenv(KEY_VAR, KEY_VALUE)
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    state = ProxyState(Config(detection=DetectionConfig(ner=_RUNNING[start])), None)
    startup = state.config
    ((_n, first, _s),) = ner_backend_stats(state.detectors)
    # The stop keeps the running values: a removed table's are reported.
    kept = [
        restart_only_name(path)
        for path in RESTART_ONLY_PATHS
        if stop != "NER switched off" and _value(startup, path) != _value(Config(), path)
    ]
    assert state.apply_config(_stopped(startup, stop)) == kept
    assert not ner_backend_stats(state.detectors)
    assert _destination(state.config) == _destination(startup)
    asked = len(server.requests)
    state.validate_config(startup)  # the dry run agrees, and changes nothing
    assert not ner_backend_stats(state.detectors) and not state.config.detection.ner.enabled
    # The editor's save: built off the loop, adopted by the apply.
    assert state.apply_config(startup, prepared=state.prepare_config(startup)) == []
    ((_n, again, _s),) = ner_backend_stats(state.detectors)
    assert again is not first and again.server == _destination(startup)[0]
    assert _destination(state.config) == _destination(startup)
    # The dry run and the prepared build each checked the server the proxy
    # was started with, with the key its startup named (read from the
    # environment at each build); the apply built nothing more.
    assert [request.url.path for request in server.requests[asked:]] == ["/api/tags"] * 2
    key = None if startup.detection.ner.llm.api_key_env is None else f"Bearer {KEY_VALUE}"
    assert _sent(server) == {(_destination(startup)[0], key)}
    assert state.preview(f"hi {NAME}")["redacted"] == "hi «PERSON_001»"
    assert _sent(server) == {(_destination(startup)[0], key)}


@pytest.mark.parametrize("moved", _ELSEWHERE)
@pytest.mark.parametrize("start", list(_RUNNING))
def test_a_reload_that_would_start_it_again_anywhere_else_is_refused(
    monkeypatch: pytest.MonkeyPatch, start: str, moved: str
) -> None:
    monkeypatch.setenv(KEY_VAR, KEY_VALUE)
    monkeypatch.setenv(OTHER_VAR, OTHER_VALUE)
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    state = ProxyState(Config(detection=DetectionConfig(ner=_RUNNING[start])), None)
    startup = state.config
    state.apply_config(_stopped(startup, "NER switched off"))
    before = (state.config, state.detectors)
    asked = len(server.requests)
    moves = _elsewhere(startup, moved)
    changed = [name for name in NAMES if name.rpartition(".")[2] in moves]
    for message in _refusals(state, _with(startup, **moves)):
        assert message.startswith(
            "[detection.ner] a reload cannot start the llm backend anywhere but where the proxy"
            f" was started to send it: {', '.join(changed)} changed ("
        )
        assert all(name in message for name in NAMES)
        # Names only, never a value.
        assert not any(value in message for value in _VALUES)
    assert (state.config, state.detectors) == before
    assert len(server.requests) == asked  # no server was asked anything


def test_an_unset_key_variable_refuses_the_start_and_keeps_the_running_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The build reads the variable the startup named, as at startup: unset
    # since, the start is refused before the server is asked.
    monkeypatch.setenv(KEY_VAR, KEY_VALUE)
    server = install_llm(monkeypatch, FakeModelServer())
    state = ProxyState(Config(detection=DetectionConfig(ner=_RUNNING["a key"])), None)
    startup = state.config
    state.apply_config(_stopped(startup, "NER switched off"))
    monkeypatch.delenv(KEY_VAR)
    before = (state.config, state.detectors)
    asked = len(server.requests)
    for attempt in (state.validate_config, state.prepare_config, state.apply_config):
        with pytest.raises(ConfigError, match=f"the environment variable {KEY_VAR} is unset"):
            attempt(startup)
    assert (state.config, state.detectors) == before
    assert len(server.requests) == asked


def test_a_backend_a_reload_stopped_is_started_again_by_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    path = tmp_path / "config.toml"
    path.write_text(LLM_TOML)
    state = ProxyState(load_config(path), None, config_path=path)
    ((_n, first, _s),) = ner_backend_stats(state.detectors)
    path.write_text(LLM_TOML.replace("enabled = true", "enabled = false"))
    state.reload()
    assert not ner_backend_stats(state.detectors)
    # Re-added at another url: refused, the running configuration kept.
    asked = len(server.requests)
    path.write_text(LLM_TOML + 'url = "http://localhost:11434"\n')
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        state.reload()
    assert "config reload failed; keeping current config" in caplog.text
    assert "a reload cannot start the llm backend" in caplog.text
    assert "detection.ner.llm.url changed" in caplog.text and "localhost" not in caplog.text
    assert not state.config.detection.ner.enabled and not ner_backend_stats(state.detectors)
    assert len(server.requests) == asked
    # Re-added as the proxy started: a new detector, on the same server.
    caplog.clear()
    path.write_text(LLM_TOML)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        state.reload()
    assert "config reload failed" not in caplog.text and "require restart" not in caplog.text
    ((_n, again, _s),) = ner_backend_stats(state.detectors)
    assert again is not first
    assert [request.url.path for request in server.requests[asked:]] == ["/api/tags"]
    assert state.preview(f"hi {NAME}")["redacted"] == "hi «PERSON_001»"
    assert _sent(server) == {("http://127.0.0.1:11434", None)}


async def test_a_backend_started_again_has_a_closed_breaker_and_new_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The breaker and the counters belong to the detector: one a reload
    # starts again is a new one, whatever its predecessor went through.
    server = FakeModelServer([(NAME, "person")], fail="timeout")
    app, _upstream = _app(monkeypatch, server, on_error="skip", breaker_failures=1)
    state = app.state.proxy
    await _chat(app, f"hi {NAME}")
    ((_n, before, stats),) = ner_backend_stats(state.detectors)
    assert before.breaker.status()["state"] == "open" and stats.errors == 1
    running = state.config
    state.apply_config(_with(running, ner={"enabled": False}))
    assert not ner_backend_stats(state.detectors)
    server.fail = None
    assert state.apply_config(running) == []
    ((_n, after, stats),) = ner_backend_stats(state.detectors)
    assert after is not before
    assert after.breaker.status()["state"] == "closed" and stats.errors == 0
    assert (await _chat(app, f"bye {NAME}")).status_code == 200
    assert server.chats == 2  # asked again: no breaker carried over


async def test_a_backend_a_reload_starts_again_redacts_reports_and_revokes_open_relays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end through the file: the proxy started running the backend
    — naming a key variable —, a SIGHUP reload of the file with NER off
    stopped it, and one with NER on again starts it again there. Requests
    are redacted with its findings, the key goes to the startup's server
    only, ``/status`` and ``doctor`` show it, and an open realtime relay
    admitted under the old detection policy is revoked (closed 1012: it
    reconnects under the new one)."""
    from llm_redact.doctor_cli import _check_llm, _Report

    monkeypatch.setenv(KEY_VAR, KEY_VALUE)
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    toml = LLM_TOML + f'api_key_env = "{KEY_VAR}"\nbreaker_failures = 1\n'
    path = tmp_path / "config.toml"
    path.write_text(toml)
    upstream = _Upstream()
    app = create_app(
        load_config(path), upstream_transport=httpx.MockTransport(upstream), config_path=path
    )
    state = app.state.proxy
    path.write_text(toml.replace("enabled = true", "enabled = false"))
    state.reload()
    relay = RealtimeRelay(
        "openai",
        state.config.providers["openai"],
        None,
        state.detectors,
        state.allowlist,
        state.modes,
    )
    state.realtime_relays.add(relay)
    assert (await _chat(app, f"hi {NAME}")).status_code == 200
    assert NAME in upstream.bodies[-1].decode()  # no detector of names now
    assert (await _status(app))["detection"]["ner"]["backends"] == {}
    asked = len(server.requests)  # the startup's model list only
    assert [request.url.path for request in server.requests] == ["/api/tags"]

    path.write_text(toml)
    state.reload()
    assert (relay.revoked, relay.close_code) == ("[detection] policy", 1012)
    assert (await _chat(app, f"hi {NAME}")).status_code == 200
    sent = upstream.bodies[-1].decode()
    assert NAME not in sent and "«PERSON_001»" in sent
    paths = [request.url.path for request in server.requests[asked:]]
    assert paths == ["/api/tags", "/api/chat"]
    assert _sent(server) == {("http://127.0.0.1:11434", f"Bearer {KEY_VALUE}")}
    status = await _status(app)
    assert status["detection"]["ner"]["enabled"] is True
    entry = status["detection"]["ner"]["backends"]["llm"]
    assert (entry["server"], entry["api"], entry["api_key"]) == (
        "http://127.0.0.1:11434",
        "ollama",
        True,
    )
    assert (entry["breaker"]["state"], entry["breaker"]["threshold"]) == ("closed", 1)
    assert entry["counters"]["scanned_whole"] == 1
    assert KEY_VALUE not in str(status) and KEY_VAR not in str(status)
    # doctor reads the same file: the server, the key's variable, the model.
    report = _Report(json_mode=True)
    _check_llm(report, load_config(path))
    assert [(row["level"], row["message"]) for row in report.rows] == [
        ("PASS", "model server http://127.0.0.1:11434 is on loopback (ollama API)"),
        (
            "PASS",
            f"api key: {KEY_VAR} is set (sent as a bearer token to the model server only)",
        ),
        ("PASS", f"model server serves {LLM_MODEL!r} at the pinned digest"),
    ]


# --- the running values are the startup values ------------------------------------------------


@pytest.mark.parametrize("ran", [True, False], ids=["started running it", "started without it"])
def test_no_apply_moves_the_running_values_off_the_startup_ones(
    monkeypatch: pytest.MonkeyPatch, ran: bool
) -> None:
    """The start rule compares a reload with the RUNNING configuration's
    values and with whether the backend ran at startup. The values are the
    ones the process started with because no apply moves them — every kind
    of apply, in turn, each after its dry run, which must answer alike —
    and the backend starts again only on a proxy started running it."""
    monkeypatch.setenv(KEY_VAR, KEY_VALUE)
    monkeypatch.setenv(OTHER_VAR, OTHER_VALUE)
    server = install_llm(monkeypatch, FakeModelServer([(NAME, "person")]))
    on = Config(detection=DetectionConfig(ner=llm_ner_config(api_key_env=KEY_VAR)))
    off = _with(on, ner={"enabled": False})
    state = ProxyState(on if ran else off, None)
    startup = _destination(state.config)
    url = {"url": "http://localhost:11434"}
    moved: dict[str, Any] = {"url": REMOTE, "allow_remote": True, "api_key_env": OTHER_VAR}
    refused = "refused"
    # A start at the startup values (alone, or with only kept keys moved)
    # is applied on a proxy started running the backend and refused on one
    # started without it.
    starts: tuple[list[str] | str, bool] = ([], True) if ran else (refused, False)
    keeps: tuple[list[str] | str, bool] = (NAMES, True) if ran else (refused, False)
    # (step, the reload, (what it reports, whether the backend then runs)).
    steps: list[tuple[str, Config, tuple[list[str] | str, bool]]] = [
        ("stopped, the url changed: kept", _with(off, **url), (NAMES[:1], False)),
        ("started at the changed url", _with(on, **url), (refused, False)),
        ("started with all three changed", _with(on, **moved), (refused, False)),
        ("started at the startup values", on, starts),
        ("all three changed: kept", _with(on, **moved), keeps),
        ("stopped, all three changed: kept", _with(off, **moved), (NAMES, False)),
        ("started again at the startup values", on, starts),
        ("its table and NER removed: its key kept", Config(), (NAMES[2:], False)),
        ("re-added, another variable", _with(on, api_key_env=OTHER_VAR), (refused, False)),
        ("re-added without the key", _with(on, api_key_env=None), (refused, False)),
        ("re-added at the startup values", on, starts),
    ]
    for step, fresh, (expected, runs) in steps:
        try:
            state.validate_config(fresh)
            dry: str | None = None
        except ConfigError as refusal:
            dry = str(refusal)
        try:
            outcome: list[str] | str = state.apply_config(fresh)
        except ConfigError as refusal:
            assert dry == str(refusal), step
            outcome = refused
        else:
            assert dry is None, step
        assert outcome == expected, step
        assert bool(ner_backend_stats(state.detectors)) is runs, step
        assert _destination(state.config) == startup, step
        assert state._llm_at_startup is ran, step
    # Every request went to the startup's server, with the startup's key —
    # and none at all from a proxy started without the backend.
    expected_sent = {("http://127.0.0.1:11434", f"Bearer {KEY_VALUE}")} if ran else set()
    assert _sent(server) == expected_sent


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


@given(_configs(), _configs(), st.booleans())
def test_a_hot_apply_never_moves_the_backend_or_starts_it_elsewhere(
    fresh: Config, running: Config, at_startup: bool
) -> None:
    moved = [path for path in RESTART_ONLY_PATHS if _value(fresh, path) != _value(running, path)]
    try:
        pinned, changed = pin_restart_only_paths(fresh, running, llm_at_startup=at_startup)
    except ConfigError as refused:
        message = str(refused)
        assert all(name in message for name in NAMES)
        assert "elsewhere" not in message and "KEY_" not in message
        # Refused only for a start on a process started without the
        # backend, or at other values, or for kept values that cannot work.
        kept = _with(
            fresh,
            **{path[-1]: _value(running, path) for path in RESTART_ONLY_PATHS},
        )
        llm = kept.detection.ner.llm
        starts = _llm_runs(fresh) and not _llm_runs(running)
        unusable = llm_url_problem(llm.url, allow_remote=llm.allow_remote) is not None
        not_a_base = llm.url.endswith({"ollama": "/api", "openai": "/v1"}[llm.api])
        assert (starts and (moved or not at_startup)) or unusable or not_a_base
        return
    # Applied: the destination is the running one, everything else fresh's.
    for path in RESTART_ONLY_PATHS:
        assert _value(pinned, path) == _value(running, path)
    assert changed == [restart_only_name(path) for path in moved]
    restored = _with(pinned, **{path[-1]: _value(fresh, path) for path in RESTART_ONLY_PATHS})
    assert restored == fresh
    # Started only on a process started running it, at the running values:
    # exactly the ones fresh named.
    if _llm_runs(pinned) and not _llm_runs(running):
        assert at_startup and changed == [] and pinned == fresh


def test_the_strict_answer_is_the_default() -> None:
    # A caller that does not say the backend ran at startup gets the strict
    # rule: no start by a reload at all, even at the running values.
    running = Config(
        detection=DetectionConfig(ner=dataclasses.replace(llm_ner_config(), enabled=False))
    )
    fresh = _with(running, ner={"enabled": True})
    with pytest.raises(ConfigError, match="the proxy was not started running it"):
        pin_restart_only_paths(fresh, running)
    assert pin_restart_only_paths(fresh, running, llm_at_startup=True) == (fresh, [])


@given(_configs(), st.lists(_configs(), max_size=8))
def test_no_sequence_of_hot_applies_moves_the_startup_values(
    startup: Config, reloads: list[Config]
) -> None:
    # apply_config's chain: the running configuration becomes the pinned
    # one, or stays when the reload is refused; whether the backend ran at
    # startup is recorded once, as ProxyState does.
    at_startup = llm_backend_runs(startup.detection.ner)
    running = startup
    for fresh in reloads:
        try:
            running, _changed = pin_restart_only_paths(fresh, running, llm_at_startup=at_startup)
        except ConfigError:
            continue
        for path in RESTART_ONLY_PATHS:
            assert _value(running, path) == _value(startup, path)
        # The backend runs only on a process started running it.
        assert at_startup or not _llm_runs(running)
