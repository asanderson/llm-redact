"""The access seam (plugin_api.AccessGate): the core holds no credential
logic, asks a registered gate to admit each request, and guards two leak
paths itself when no gate is present.

Keyless: a scripted fake gate on a bare Registry stands in for
llm-redact-pro's named-user gate.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import httpx
import pytest
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse, Response

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact.cli import build_parser, main
from llm_redact.completions import all_commands, bash_script, fish_script, zsh_script
from llm_redact.config import Config, ProviderConfig
from llm_redact.plugin_api import Admission, DashboardHost
from llm_redact.proxy import create_app
from llm_redact.registry import Registry

UPSTREAM = "https://upstream.test"
BODY = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hello"}],
}


class FakeGate:
    """Admits ``x-test-key: good`` (scrubbing it and a /u/<seg> prefix),
    refuses everything else when ``strict``."""

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self.closed = False
        self.handled: list[str] = []

    def admit(self, conn: HTTPConnection, surface: str) -> Admission:
        scope = conn.scope
        key = None
        kept = []
        for name, value in scope["headers"]:
            if name == b"x-test-key":
                key = value.decode()
            else:
                kept.append((name, value))
        scope["headers"] = kept
        if scope["path"].startswith("/u/"):
            segment, _, rest = scope["path"][3:].partition("/")
            key = key or segment
            scope["path"] = "/" + rest
            scope["raw_path"] = scope["path"].encode()
        for attr in ("_headers", "_url"):
            if hasattr(conn, attr):
                delattr(conn, attr)
        if key == "good":
            return Admission(subject=f"ada via {surface}")
        return Admission(refusal="a test key is required") if self.strict else Admission()

    def status(self) -> dict[str, Any]:
        return {"registry": True, "enforcement": self.strict, "verified": 2}

    async def handle(self, request: Request, host: DashboardHost) -> Response:
        self.handled.append(request.url.path)
        return JSONResponse({"from": "gate", "csrf_ok": bool(host.csrf_token)})

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> FakeGate:
    fake = FakeGate(strict=True)
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda config, license: fake
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return fake


def _app(received: list[httpx.Request], **provider: Any) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(
            200,
            json={"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            headers={"content-type": "application/json"},
        )

    config = Config(providers={"anthropic": ProviderConfig(UPSTREAM, **provider)})
    return create_app(config, upstream_transport=httpx.MockTransport(handler))


async def _call(
    app: Any, method: str, path: str, headers: dict[str, str] | None = None
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        if method == "POST":
            return await client.post(path, json=BODY, headers=headers)
        return await client.get(path, headers=headers)


# --- without a gate: the implicit single local user + two leak guards --------


async def test_own_headers_never_forwarded_without_a_gate() -> None:
    received: list[httpx.Request] = []
    app = _app(received)
    headers = {"x-llm-redact-user": "lrk_secret", "x-llm-redact-other": "v", "x-kept": "1"}
    assert (await _call(app, "POST", "/v1/messages", headers)).status_code == 200
    forwarded = {k.lower() for k in received[0].headers}
    assert "x-kept" in forwarded
    assert not any(name.startswith("x-llm-redact-") for name in forwarded)


async def test_unclaimed_identity_path_is_answered_locally(
    caplog: pytest.LogCaptureFixture,
) -> None:
    received: list[httpx.Request] = []
    app = _app(received)
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await _call(app, "POST", "/u/lrk_secret/v1/messages")
    assert response.status_code == 404
    assert received == []  # never forwarded
    assert "lrk_secret" not in response.text
    assert "lrk_secret" not in caplog.text
    rows = (await _call(app, "GET", "/__llm-redact/recent")).json()["entries"]
    assert rows == []  # never recorded (the path carries the key)


async def test_access_paths_without_a_gate() -> None:
    app = _app([])
    response = await _call(app, "GET", "/__llm-redact/users")
    assert response.status_code == 404
    assert "llm-redact-pro" in response.json()["error"]
    status = (await _call(app, "GET", "/__llm-redact/status")).json()
    assert status["users"] == {"registry": False, "enforcement": False}


# --- with a gate ----------------------------------------------------------------


async def test_gate_admits_scrubs_and_attributes(gate: FakeGate) -> None:
    received: list[httpx.Request] = []
    app = _app(received)
    ok = await _call(app, "POST", "/u/good/v1/messages")
    assert ok.status_code == 200
    ok = await _call(app, "POST", "/v1/messages", {"x-test-key": "good"})
    assert ok.status_code == 200
    for request in received:
        assert request.url.path == "/v1/messages"
        assert "x-test-key" not in request.headers
    rows = (await _call(app, "GET", "/__llm-redact/recent")).json()["entries"]
    assert [row["user"] for row in rows] == ["ada via http", "ada via http"]


async def test_gate_refusal_is_a_recorded_provider_shaped_403(gate: FakeGate) -> None:
    received: list[httpx.Request] = []
    app = _app(received)
    response = await _call(app, "POST", "/v1/messages")
    assert response.status_code == 403
    assert response.json()["type"] == "error"  # the Anthropic error envelope
    assert "a test key is required" in response.text
    assert received == []
    rows = (await _call(app, "GET", "/__llm-redact/recent")).json()["entries"]
    assert rows[0]["status"] == 403


async def test_disabled_provider_502_precedes_the_refusal(gate: FakeGate) -> None:
    response = await _call(_app([], enabled=False), "POST", "/v1/messages")
    assert response.status_code == 502


async def test_reserved_path_behind_a_stripped_prefix_is_not_forwarded(gate: FakeGate) -> None:
    received: list[httpx.Request] = []
    response = await _call(_app(received), "GET", "/u/good/__llm-redact/status")
    assert response.status_code == 404
    assert received == []


async def test_access_paths_dispatch_to_the_gate(gate: FakeGate) -> None:
    app = _app([])
    for path in ("/__llm-redact/users", "/__llm-redact/users/invite"):
        response = await _call(app, "GET", path)
        assert response.json() == {"from": "gate", "csrf_ok": True}
        assert response.headers["x-frame-options"] == "DENY"  # core-stamped
    assert gate.handled == ["/__llm-redact/users", "/__llm-redact/users/invite"]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://evil.example") as client:
        assert (await client.get("/__llm-redact/users")).status_code == 403  # host check
    assert len(gate.handled) == 2
    status = (await _call(app, "GET", "/__llm-redact/status")).json()
    assert status["users"] == {"registry": True, "enforcement": True, "verified": 2}


async def test_gate_closed_at_shutdown(gate: FakeGate) -> None:
    app = _app([])
    async with app.router.lifespan_context(app):
        assert not gate.closed
    assert gate.closed


# --- CLI seam ---------------------------------------------------------------------


class FakeCommand:
    name = "widgets"
    help = "manage widgets (a plugin command)"
    completion = (("list",), ("--json",))

    def __init__(self) -> None:
        self.ran: list[argparse.Namespace] = []

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("action", choices=["list"])
        parser.add_argument("--json", action="store_true")

    def run(self, args: argparse.Namespace) -> int:
        self.ran.append(args)
        return 7


class ShadowCommand(FakeCommand):
    name = "serve"  # collides with a core command: ignored


def test_plugin_cli_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    command = FakeCommand()
    reg = Registry()
    reg.cli_commands = [command, ShadowCommand()]
    monkeypatch.setattr(registry_mod, "_registry", reg)
    parser = build_parser()
    args = parser.parse_args(["widgets", "list", "--json"])
    assert args.json is True
    with pytest.raises(SystemExit) as exited:
        main(["widgets", "list"])
    assert exited.value.code == 7
    assert command.ran[0].action == "list"
    serve = parser.parse_args(["serve"])  # the core serve parser, not the shadow
    assert not hasattr(serve, "_plugin_command")
    assert all_commands()["widgets"] == (("list",), ("--json",))
    assert all_commands()["serve"] != ShadowCommand.completion
    for script in (bash_script(), zsh_script(), fish_script()):
        assert "widgets" in script
