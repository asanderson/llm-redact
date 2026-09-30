"""A method override never turns a recognized route into another method upstream.

A matched route is redacted, restored, tracked and ownership-checked by the
request line's method. Google's front end honors ``X-HTTP-Method-Override``
(OData services ``X-HTTP-Method``, web frameworks a ``_method`` parameter):
a ``POST /v1beta/files`` (the metadata-only create) carrying ``GET`` came
back as the file LIST — restored whole in the caller's session (another
session's placeholder restored as the caller's own value) and every listed
file reported as created by the caller. So a matched route carrying any
override is refused (a recorded, provider-shaped 400) before the body is
read or any upstream contact, and the headers never leave on a matched route
(HTTP and realtime) as a second layer. Unmatched pass-through traffic (the
client's own key, nothing read) forwards them as sent.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from lent_routes import Owners, client, lent_app
from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import METHOD_OVERRIDE_HEADERS, _request_headers, create_app, method_override

GEMINI = "https://gemini.test"
KEY = {"x-goog-api-key": "AIza-client"}


class Google:
    """Honors X-HTTP-Method-Override like Google's front end."""

    def __init__(self) -> None:
        self.received: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.received.append(request)
        method = request.headers.get("x-http-method-override", request.method)
        if method == "GET" and request.url.path == "/v1beta/files":
            files = [{"name": "files/other1", "displayName": "notes of «EMAIL_001»"}]
            return httpx.Response(200, json={"files": files})
        return httpx.Response(200, json={"file": {"name": "files/new"}})


@pytest.mark.parametrize(
    ("headers", "query", "kind"),
    [
        ({"X-HTTP-Method-Override": "GET"}, "", "header"),
        ({"x-http-method": "GET"}, "", "header"),
        ({"X-Method-Override": "DELETE"}, "", "header"),
        ({}, "%24httpMethod=GET", "query parameter"),
        ({}, "_method=GET", "query parameter"),
        ({}, "alt=json&X-HTTP-Method-Override=GET", "query parameter"),
    ],
    ids=[
        "x-http-method-override",
        "x-http-method",
        "x-method-override",
        "$httpMethod",
        "_method",
        "query-header-name",
    ],
)
async def test_a_recognized_route_with_a_method_override_is_refused(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], query: str, kind: str
) -> None:
    upstream = Google()
    owners = Owners(listings=["/v1beta/files"])
    app, router = lent_app(monkeypatch, "gemini", GEMINI, upstream, owners=owners)
    path = "/v1beta/files" + (f"?{query}" if query else "")
    async with client(app) as http:
        reply = await http.post(
            path, content=b"{}", headers={**KEY, **headers, "content-type": "application/json"}
        )
        rows = (await http.get("/__llm-redact/recent")).json()["entries"]
    assert reply.status_code == 400
    message = reply.json()["error"]["message"]
    assert f"method override {kind}" in message and "GET" not in message
    assert upstream.received == [] and owners.objects == []
    assert all(not plan.begun for plan in router.plans)
    assert rows[0]["status"] == 400 and rows[0]["path"] == "/v1beta/files"


async def test_without_an_override_the_create_is_served(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = Google()
    owners = Owners(listings=["/v1beta/files"])
    app, _ = lent_app(monkeypatch, "gemini", GEMINI, upstream, owners=owners)
    async with client(app) as http:
        reply = await http.post(
            "/v1beta/files",
            content=b"{}",
            headers={**KEY, "content-type": "application/json", "x-override-note": "GET"},
        )
    assert reply.status_code == 200
    assert owners.objects == [("files/new", "user:n1:main")]


async def test_pass_through_forwards_an_override_as_sent() -> None:
    # An unmatched route with the client's own key: nothing is read, so the
    # pass-through contract holds — the request goes out as sent.
    received: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(200, json={})

    providers = {**Config().providers, "gemini": ProviderConfig(GEMINI)}
    app = create_app(Config(providers=providers), upstream_transport=httpx.MockTransport(upstream))
    async with client(app) as http:
        reply = await http.post(
            "/v1beta/tunedModels",
            json={},
            headers={**KEY, "X-HTTP-Method-Override": "GET"},
        )
    assert reply.status_code == 200
    (sent,) = received
    assert sent.headers["x-http-method-override"] == "GET" and sent.method == "POST"


def test_matched_headers_drop_every_override() -> None:
    from starlette.requests import Request

    raw = [(name.encode(), b"GET") for name in sorted(METHOD_OVERRIDE_HEADERS)]
    raw.append((b"x-kept", b"1"))
    request = Request({"type": "http", "headers": raw, "method": "POST", "path": "/"})
    matched = dict(_request_headers(request, matched=True))
    assert not METHOD_OVERRIDE_HEADERS & set(matched) and matched["x-kept"] == "1"
    unmatched = dict(_request_headers(request, matched=False))
    assert set(unmatched) >= METHOD_OVERRIDE_HEADERS


def test_the_realtime_relay_drops_every_override() -> None:
    from starlette.websockets import WebSocket

    from llm_redact.realtime import _filtered_headers

    raw = [(b"X-HTTP-Method-Override", b"DELETE"), (b"x-method-override", b"PUT")]
    raw.append((b"x-kept", b"1"))

    async def nothing() -> Any:  # pragma: no cover - never awaited
        return {}

    socket = WebSocket({"type": "websocket", "headers": raw, "path": "/"}, nothing, nothing)
    assert dict(_filtered_headers(socket)) == {"x-kept": "1"}


def test_method_override_reads_names_case_insensitively_and_decoded() -> None:
    assert method_override({"X-Http-Method": "GET"}, "") == "header"
    assert method_override({}, "a=1&%5Fmethod=GET") == "query parameter"
    assert method_override({}, "%24X-HTTP-METHOD-OVERRIDE=GET") == "query parameter"
    assert method_override({"x-goog-api-key": "k"}, "key=k&method=x&alt=sse") is None
    assert json.loads(json.dumps(sorted(METHOD_OVERRIDE_HEADERS))) == [
        "x-http-method",
        "x-http-method-override",
        "x-method-override",
    ]
