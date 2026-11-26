"""``llm-redact reload`` on Windows, which has no SIGHUP (winreload.py).

The proxy waits on a named auto-reset event, ``Local\\llm-redact-reload-<port>``
(the logon session's namespace), on a thread of its own; setting it asks the
reload scheduler for a reload exactly as SIGHUP does. The logic runs with a
fake event on every OS; the last test sets the real one on Windows."""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from llm_redact import winreload
from llm_redact.config import load_config
from llm_redact.proxy import create_app

EMAIL_ONLY = '[detection]\nenabled = ["email"]\n'
THREE_RULES = '[detection]\nenabled = ["email", "ipv4", "jwt"]\n'


class FakeEvents:
    """The named event and the listener's stop event, as a queue."""

    def __init__(self) -> None:
        self._asked: queue.Queue[bool | BaseException] = queue.Queue()
        self.closed = False

    def set(self) -> None:  # what `llm-redact reload` does
        self._asked.put(True)

    def fail(self, error: BaseException) -> None:
        self._asked.put(error)

    def wait(self) -> bool:
        answer = self._asked.get()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def stop(self) -> None:
        self._asked.put(False)

    def close(self) -> None:
        self.closed = True


def _until(condition: Any, seconds: float = 10) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_the_event_is_named_for_the_port_in_the_session_namespace() -> None:
    assert winreload.event_name(8787) == "Local\\llm-redact-reload-8787"


def test_each_set_asks_for_a_reload_until_stopped(caplog: pytest.LogCaptureFixture) -> None:
    events = FakeEvents()
    asked: list[str] = []
    calls = iter([None, RuntimeError("loop closed"), None])

    def on_signal() -> None:
        outcome = next(calls)
        asked.append("asked")
        if outcome is not None:
            raise outcome

    listener = winreload.ReloadListener(events, on_signal)
    listener.start()
    for _ in range(3):
        events.set()
    _until(lambda: len(asked) == 3)  # a failing call never ends the listener
    listener.stop()
    assert events.closed and not listener.running
    assert "llm-redact reload could not ask for a reload (RuntimeError)" in caplog.text


def test_a_failing_wait_ends_the_listener_with_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events = FakeEvents()
    listener = winreload.ReloadListener(events, lambda: None)
    listener.start()
    events.fail(OSError(6, "The handle is invalid"))
    _until(lambda: not listener.running)
    listener.stop()
    assert events.closed
    assert caplog.text.count("llm-redact reload stopped working (OSError)") == 1


def test_no_listener_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(winreload, "open_listening", lambda port: pytest.fail("opened"))
    assert winreload.start_listener(8787, lambda: None) is None


def test_an_event_name_another_program_holds_leaves_reload_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def taken(port: int) -> FakeEvents:
        raise winreload.EventTaken(winreload.event_name(port))

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(winreload, "open_listening", taken)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        assert winreload.start_listener(8787, lambda: None) is None
    assert "Local\\llm-redact-reload-8787" in caplog.text
    assert "llm-redact reload will not reach this proxy" in caplog.text


def _app(tmp_path: Path, body: str = EMAIL_ONLY) -> tuple[Any, Any, Path]:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    app = create_app(load_config(path), config_path=path)
    return app, app.state.proxy, path


async def test_the_lifespan_reloads_on_the_event_as_on_sighup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state, path = _app(tmp_path)
    opened: list[tuple[int, FakeEvents]] = []

    def open_listening(port: int) -> FakeEvents:
        events = FakeEvents()
        opened.append((port, events))
        return events

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", lambda *a: pytest.fail("SIGHUP on Windows"))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(winreload, "open_listening", open_listening)
    token = state.vault_manager.get(state.config.vault.session).placeholder_for(
        "EMAIL", "jane@corp.example"
    )
    async with app.router.lifespan_context(app):
        ((port, events),) = opened
        assert port == state.config.port
        path.write_text(THREE_RULES, encoding="utf-8")
        events.set()
        await _reloaded(state, 1)
        assert tuple(state.config.detection.enabled) == ("email", "ipv4", "jwt")
        assert state.reloads.status()["applied_total"] == 1
        # A bad file is logged and the running configuration stays.
        path.write_text("[detection]\nno_such_key = 1\n", encoding="utf-8")
        events.set()
        await _reloaded(state, 2)
        assert state.reloads.status()["failed_total"] == 1
        assert tuple(state.config.detection.enabled) == ("email", "ipv4", "jwt")
        vault = state.vault_manager.get(state.config.vault.session)
        assert vault.original_for(token) == "jane@corp.example"
    assert events.closed


async def _reloaded(state: Any, requested: int) -> None:
    deadline = time.monotonic() + 30
    while state.reloads.status()["requested_total"] < requested or state.reloads.running:
        assert time.monotonic() < deadline, "the reload was never asked for"
        await asyncio.sleep(0.01)
    await state.reloads.wait()


def _reload_args(tmp_path: Path, port: int | None) -> argparse.Namespace:
    config = tmp_path / "config.toml"
    config.write_text("port = 9123\n", encoding="utf-8")
    return argparse.Namespace(config=config, port=port)


def test_reload_off_windows_names_the_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert winreload.run_reload(_reload_args(tmp_path, None)) == 2
    out = capsys.readouterr().out
    assert "SIGHUP" in out and "kill -HUP" in out


@pytest.mark.parametrize(
    ("answer", "port", "code", "said"),
    [
        (True, None, 0, "asked the proxy on port 9123 to reload"),
        (True, 9200, 0, "asked the proxy on port 9200 to reload"),
        (False, None, 1, "no proxy started in this logon session listens on port 9123"),
        (OSError(5, "Access is denied"), None, 1, "cannot reach the proxy on port 9123"),
    ],
)
def test_reload_on_windows_sets_the_event(
    answer: bool | OSError,
    port: int | None,
    code: int,
    said: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    signalled: list[int] = []

    def signal(port: int) -> bool:
        signalled.append(port)
        if isinstance(answer, OSError):
            raise answer
        return answer

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(winreload, "signal", signal)
    assert winreload.run_reload(_reload_args(tmp_path, port)) == code
    assert signalled == [port or 9123]
    assert said in capsys.readouterr().out


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


@pytest.mark.skipif(sys.platform != "win32", reason="the named event is a Windows object")
def test_llm_redact_reload_reaches_a_running_windows_proxy(tmp_path: Path) -> None:
    from llm_redact import serving
    from llm_redact.cli import main

    port = _free_port()
    path = tmp_path / "config.toml"
    path.write_text(f"port = {port}\n{EMAIL_ONLY}", encoding="utf-8")
    app = create_app(load_config(path), config_path=path)
    state = app.state.proxy
    token = state.vault_manager.get(state.config.vault.session).placeholder_for(
        "EMAIL", "jane@corp.example"
    )
    import uvicorn

    server = serving.ProxyServer(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _until(lambda: server.started, 30)
        path.write_text(f"port = {port}\n{THREE_RULES}", encoding="utf-8")
        with pytest.raises(SystemExit) as exit_:
            main(["reload", "--config", str(path)])
        assert exit_.value.code == 0
        _until(lambda: state.reloads.status()["applied_total"] == 1, 30)
        assert tuple(state.config.detection.enabled) == ("email", "ipv4", "jwt")
        vault = state.vault_manager.get(state.config.vault.session)
        assert vault.original_for(token) == "jane@corp.example"
    finally:
        server.should_exit = True
        thread.join(30)
    # Once the proxy has stopped, nothing listens: exit 1.
    with pytest.raises(SystemExit) as exit_:
        main(["reload", "--port", str(port)])
    assert exit_.value.code == 1
