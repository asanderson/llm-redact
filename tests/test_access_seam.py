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
from llm_redact.config import Config, ConfigError, ProviderConfig
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


# --- async admission, dashboard admission, gate paths, public origin -------------


class DashboardGate(FakeGate):
    """Opts into dashboard admission: admits a dashboard request carrying
    ``x-test-admin: yes`` and refuses the rest with a sign-in redirect."""

    guards_dashboard = True

    def __init__(self, *, origin: str | None = None, redirect: str | None = None) -> None:
        super().__init__(strict=False)
        self.origin = origin
        self.redirect = redirect or "/__llm-redact/auth/login?next=%2F__llm-redact%2F"
        self.surfaces: list[str] = []

    def public_origin(self) -> str | None:
        return self.origin

    async def admit(self, conn: HTTPConnection, surface: str) -> Admission:  # type: ignore[override]
        self.surfaces.append(surface)
        if surface == "dashboard":
            if conn.headers.get("x-test-admin") == "yes":
                return Admission(subject="root")
            return Admission(refusal="sign in first", redirect=self.redirect)
        return super().admit(conn, surface)


def _install(monkeypatch: pytest.MonkeyPatch, gate: Any) -> None:
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda config, license: gate
    monkeypatch.setattr(registry_mod, "_registry", reg)


async def test_async_admission_is_awaited(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = DashboardGate()
    _install(monkeypatch, gate)
    received: list[httpx.Request] = []
    response = await _call(_app(received), "POST", "/v1/messages", {"x-test-key": "good"})
    assert response.status_code == 200
    assert gate.surfaces == ["http"]
    rows = (await _call(_app([]), "GET", "/__llm-redact/recent", {"x-test-admin": "yes"})).json()
    assert rows["entries"] == []


async def test_dashboard_admission_is_opt_in(gate: FakeGate) -> None:
    # FakeGate has no guards_dashboard: reserved endpoints stay as before.
    response = await _call(_app([]), "GET", "/__llm-redact/status")
    assert response.status_code == 200


async def test_dashboard_admission_refuses_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = DashboardGate()
    _install(monkeypatch, gate)
    app = _app([])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        page = await client.get("/__llm-redact/status")
        assert page.status_code == 303
        assert page.headers["location"] == gate.redirect
        assert page.headers["x-frame-options"] == "DENY"  # still core-stamped
        post = await client.post("/__llm-redact/sessions/prune", json={})
        assert post.status_code == 403  # a non-GET is never redirected
        assert post.json() == {"error": "sign in first"}
        ok = await client.get("/__llm-redact/status", headers={"x-test-admin": "yes"})
        assert ok.status_code == 200
        for probe in ("/__llm-redact/healthz", "/__llm-redact/readyz", "/__llm-redact/metrics"):
            assert (await client.get(probe)).status_code == 200  # never admitted
    assert "dashboard" in gate.surfaces


@pytest.mark.parametrize(
    "redirect", ["https://evil.example/", "//evil.example/x", "/v1/messages", "/__llm-redact/\\\\x"]
)
async def test_only_same_proxy_redirects_are_honored(
    monkeypatch: pytest.MonkeyPatch, redirect: str
) -> None:
    _install(monkeypatch, DashboardGate(redirect=redirect))
    response = await _call(_app([]), "GET", "/__llm-redact/status")
    assert response.status_code == 403


async def test_gate_paths_skip_dashboard_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = DashboardGate()
    _install(monkeypatch, gate)
    app = _app([])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        for path in (
            "/__llm-redact/auth/login",
            "/__llm-redact/auth/callback",
            "/__llm-redact/auth/logout",
            "/__llm-redact/scim/v2/Users",
            "/__llm-redact/scim/v2",
        ):
            response = await client.get(path)
            assert response.json() == {"from": "gate", "csrf_ok": True}, path
        # SCIM clients are not browsers: a foreign Origin is not refused...
        scim = await client.get(
            "/__llm-redact/scim/v2/Users", headers={"origin": "https://idp.example"}
        )
        assert scim.status_code == 200
        # ...but the sign-in endpoints keep the Origin check.
        login = await client.get("/__llm-redact/auth/login", headers={"origin": "https://x.test"})
        assert login.status_code == 403
        # The admin endpoints are behind dashboard admission now.
        assert (await client.get("/__llm-redact/users")).status_code == 303
    assert gate.surfaces == ["dashboard"]  # only the /users request was admitted
    assert gate.handled[:5] == [
        "/__llm-redact/auth/login",
        "/__llm-redact/auth/callback",
        "/__llm-redact/auth/logout",
        "/__llm-redact/scim/v2/Users",
        "/__llm-redact/scim/v2",
    ]


async def test_gate_paths_without_a_gate() -> None:
    app = _app([])
    for path in ("/__llm-redact/auth/login", "/__llm-redact/scim/v2/Users"):
        response = await _call(app, "GET", path)
        assert response.status_code == 404
        assert "llm-redact-pro" in response.json()["error"]


AUTH_PREFIX_PATHS = (
    "/__llm-redact/auth/passkey",
    "/__llm-redact/auth/passkey/options",
    "/__llm-redact/auth/passkey/enroll/verify",
)


async def test_everything_under_the_auth_prefix_goes_to_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A sign-in method's own pages and JSON endpoints (passkeys) live below
    # the prefix: dispatched to the gate like the fixed three, never behind
    # dashboard admission (they are how a browser obtains it).
    gate = DashboardGate(origin="https://proxy.team.example")
    _install(monkeypatch, gate)
    transport = httpx.ASGITransport(app=_app([]))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://proxy.team.example"
    ) as client:
        for path in AUTH_PREFIX_PATHS:
            page = await client.get(path)
            assert page.json() == {"from": "gate", "csrf_ok": True}, path
            assert page.headers["content-security-policy"].startswith("default-src 'none'")
            assert page.headers["x-frame-options"] == "DENY"
            same = await client.post(
                path, json={}, headers={"origin": "https://proxy.team.example"}
            )
            assert same.json() == {"from": "gate", "csrf_ok": True}, path
            # POSTs keep the Origin check: a foreign page never reaches the gate.
            foreign = await client.post(path, json={}, headers={"origin": "https://evil.example"})
            assert foreign.status_code == 403
            assert foreign.json() == {"error": "origin not allowed"}
    assert gate.surfaces == []  # none of them went through dashboard admission
    assert gate.handled == [p for p in AUTH_PREFIX_PATHS for _ in range(2)]


async def test_the_auth_prefix_keeps_the_host_check(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = DashboardGate()
    _install(monkeypatch, gate)
    transport = httpx.ASGITransport(app=_app([]))
    async with httpx.AsyncClient(transport=transport, base_url="http://rebind.example") as client:
        response = await client.post("/__llm-redact/auth/passkey/verify", json={})
    assert response.status_code == 403
    assert response.json() == {"error": "host not allowed"}
    assert gate.handled == []


async def test_a_look_alike_of_the_auth_prefix_is_not_a_gate_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = DashboardGate()
    _install(monkeypatch, gate)
    transport = httpx.ASGITransport(app=_app([]))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        for path in ("/__llm-redact/authx", "/__llm-redact/auth"):
            # Behind dashboard admission like every other reserved path —
            # the bare prefix too: only paths BELOW it are the gate's, so a
            # bare POST can't reach a gate handler without admission...
            assert (await client.get(path)).status_code == 303, path
            assert (await client.post(path, json={})).status_code == 403, path
            admitted = await client.get(path, headers={"x-test-admin": "yes"})
            # ...and then an ordinary unknown reserved path, never the gate's.
            assert admitted.status_code == 404, path
    assert gate.handled == []


async def test_auth_prefix_paths_without_a_gate() -> None:
    received: list[httpx.Request] = []
    app = _app(received)
    for method in ("GET", "POST"):
        response = await _call(app, method, "/__llm-redact/auth/passkey/options")
        assert response.status_code == 404
        assert "llm-redact-pro" in response.json()["error"]
    assert received == []  # never forwarded


async def test_public_origin_widens_host_and_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, DashboardGate(origin="https://Proxy.Team.Example:443"))
    app = _app([])
    transport = httpx.ASGITransport(app=app)
    admin = {"x-test-admin": "yes"}
    async with httpx.AsyncClient(
        transport=transport, base_url="https://proxy.team.example"
    ) as client:
        assert (await client.get("/__llm-redact/recent", headers=admin)).status_code == 200
        response = await client.get(
            "/__llm-redact/users", headers={**admin, "origin": "https://proxy.team.example"}
        )
        assert response.json() == {"from": "gate", "csrf_ok": True}
        foreign = await client.get(
            "/__llm-redact/users", headers={**admin, "origin": "https://other.example"}
        )
        assert foreign.status_code == 403


async def test_public_origin_needs_dashboard_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = DashboardGate(origin="https://proxy.team.example")
    gate.guards_dashboard = False  # type: ignore[misc]
    _install(monkeypatch, gate)
    transport = httpx.ASGITransport(app=_app([]))
    async with httpx.AsyncClient(
        transport=transport, base_url="https://proxy.team.example"
    ) as client:
        assert (await client.get("/__llm-redact/recent")).status_code == 403  # host check


@pytest.mark.parametrize(
    "origin",
    ["ftp://proxy.example", "https://proxy.example/path", "https://u:p@proxy.example", "https://"],
)
def test_malformed_public_origin_fails_closed(monkeypatch: pytest.MonkeyPatch, origin: str) -> None:
    _install(monkeypatch, DashboardGate(origin=origin))
    with pytest.raises(ConfigError, match="public origin"):
        _app([])
