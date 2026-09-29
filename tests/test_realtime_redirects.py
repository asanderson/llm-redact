"""The realtime relay never follows an upstream handshake redirect.

``websockets.connect`` follows 3xx upgrade responses (up to ten, cross-origin
included) and resends its headers on every hop. Under ``auth = "identity"``
those headers carry the proxy's OWN cloud credential, so an upstream (or
anything answering as it) could send the token to another host or path — a
URL never authorized, not an exact identity path, possibly plain ``ws://``.
The HTTP side never follows redirects (httpx's default); the relay now
treats a redirect as a failed dial: accept-then-close 1011, counted in
``upstream_errors`` and recorded. Pass-through connections too: the
client's own key must not follow a redirect it never saw either.

Real sockets end to end (the test_realtime_relay harness).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

import pytest
import websockets
import websockets.asyncio.server

from test_realtime_identity import (
    AZURE_GA,
    VERTEX_V1,
    FakeAuth,
    _closed,
    _install,
    _recent,
    _status,
)
from test_realtime_relay import FakeUpstream, _config, _proxy

pytestmark = pytest.mark.asyncio


@contextlib.asynccontextmanager
async def _redirecting(location: str | None = None) -> AsyncIterator[tuple[int, list[str]]]:
    """An upstream answering an upgrade with ``302 Location`` (``location``,
    or a same-host ``/elsewhere`` path it would accept); yields its port and
    the paths it accepted."""
    accepted: list[str] = []

    def process_request(connection: Any, request: Any) -> Any:
        if request.path.startswith("/elsewhere"):
            accepted.append(request.path)
            return None  # the redirect target: a working upstream
        response = connection.respond(HTTPStatus.FOUND, "moved\n")
        response.headers["Location"] = location or "/elsewhere"
        return response

    async def handler(connection: Any) -> None:
        async for message in connection:
            await connection.send(message)

    server = await websockets.serve(handler, "127.0.0.1", 0, process_request=process_request)
    try:
        yield server.sockets[0].getsockname()[1], accepted
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize("cross_origin", [True, False], ids=["cross-origin", "same-host"])
@pytest.mark.parametrize(
    ("provider", "path", "identity"),
    [
        ("azure", AZURE_GA + "?model=gpt-realtime", True),
        ("vertex", VERTEX_V1, True),
        ("openai", "/v1/realtime?model=gpt-realtime", False),
    ],
    ids=["azure-identity", "vertex-identity", "passthrough"],
)
async def test_a_handshake_redirect_is_a_failed_dial(
    monkeypatch: pytest.MonkeyPatch, provider: str, path: str, identity: bool, cross_origin: bool
) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    kwargs = {"auth": "identity"} if identity else {}
    async with FakeUpstream() as foreign:
        location = f"ws://localhost:{foreign.port}/stolen" if cross_origin else None
        async with _redirecting(location) as (port, accepted):
            with _proxy(_config(port, provider, **kwargs)) as proxy_host:
                async with websockets.connect(
                    f"ws://{proxy_host}{path}",
                    additional_headers={"Authorization": "Bearer client-key"},
                ) as client:
                    # Bounded: a followed redirect would leave the relay
                    # connected to the redirect target, and nothing closes.
                    closed = await asyncio.wait_for(_closed(client), timeout=10)
                row = await _recent(proxy_host, lambda r: r["method"] == "WS")
                status = await _status(proxy_host)
    # Nothing reached the redirect target — another host or the same one:
    # not the proxy's credential, not the client's.
    assert foreign.paths == [] and foreign.headers == [] and accepted == []
    assert (closed.code, closed.reason) == (1011, "upstream websocket connect failed")
    assert (row["status"], row["provider"]) == (502, provider)
    assert status["upstream_errors_total"] == {provider: 1}
    assert len(auth.calls) == (1 if identity else 0)
