"""The realtime relay under the proxy's own cloud identity
(``[providers.azure|vertex] auth = "identity"``).

The core strips every client credential channel a WebSocket client can use
— upgrade headers, the query, and the offered subprotocols — hands the
registered ``plugin_api.UpstreamAuth`` the upgrade as the HTTP GET it is
(the upstream URL in its HTTP form, an empty body), and dials the upstream
with exactly the headers it returned. Only the exact documented realtime
paths are authorized; an authorizer failure is a recorded, counted
accept-then-close 1011 naming the credential SOURCE only. Real sockets end
to end (uvicorn on port 0 + a fake ``websockets`` upstream), driven by a
fake authorizer on a bare Registry — keyless and pro-free.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
import uvicorn
import websockets

import llm_redact.registry as registry_mod
from llm_redact.plugin_api import UpstreamAuthError
from llm_redact.proxy import _strip_credential_query, create_app
from llm_redact.realtime import (
    ALL_WS_ADAPTERS,
    AzureRealtimeWs,
    GeminiLiveWs,
    OpenAIRealtimeWs,
    VertexLiveWs,
    _close_reason,
    is_credential_subprotocol,
    ws_adapter_for,
)
from llm_redact.registry import Registry
from test_realtime_relay import FakeUpstream, _config, _proxy

pytestmark = pytest.mark.asyncio

EMAIL = "jane.doe@corp.example"
PROXY_TOKEN = "Bearer proxy-identity-token"
AZURE_PREVIEW = "/openai/realtime"
AZURE_GA = "/openai/v1/realtime"
VERTEX_V1 = "/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
VERTEX_V1BETA1 = "/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
GEMINI_LIVE = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"


class FakeAuth:
    """A scripted ``plugin_api.UpstreamAuth``: records every call and adds
    the proxy's bearer token, or raises."""

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str, list[tuple[str, str]], bytes]] = []

    async def authorize(
        self, method: str, url: str, headers: list[tuple[str, str]], body: bytes
    ) -> list[tuple[str, str]]:
        self.calls.append((method, url, list(headers), body))
        if self.error is not None:
            raise self.error
        return [*headers, ("authorization", PROXY_TOKEN)]

    def close(self) -> None:
        pass


def _install(monkeypatch: pytest.MonkeyPatch, auth: FakeAuth) -> None:
    reg = Registry()
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    monkeypatch.setattr(registry_mod, "_registry", reg)


async def _closed(client: Any) -> Any:
    with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
        await client.recv()
    assert closed.value.rcvd is not None
    return closed.value.rcvd


async def _recent(proxy_host: str, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=f"http://{proxy_host}") as http:
        for _ in range(100):
            for row in (await http.get("/__llm-redact/recent")).json()["entries"]:
                if predicate(row):
                    return dict(row)
            await asyncio.sleep(0.05)
    raise AssertionError("no matching /recent row")


async def _status(proxy_host: str) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=f"http://{proxy_host}") as http:
        return dict((await http.get("/__llm-redact/status")).json())


def _item_created(text: str) -> str:
    # Sent by the test client, echoed by the fake upstream: outbound it is
    # walked (redacted) like every client event, inbound it is a server
    # item echo, which the adapter rehydrates whole.
    return json.dumps(
        {
            "type": "conversation.item.created",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
    )


# --- routing ------------------------------------------------------------------------


ROUTES = {
    "/v1/realtime": "openai-realtime",
    AZURE_PREVIEW: "azure-realtime",
    AZURE_GA: "azure-realtime",
    "/openai/realtime/sub": "azure-realtime",
    "/openai/v1/realtime/sub": "azure-realtime",
    GEMINI_LIVE: "gemini-live",
    "/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent": (
        "gemini-live"
    ),
    VERTEX_V1: "vertex-live",
    VERTEX_V1BETA1: "vertex-live",
}
UNROUTED = (
    "/v1/other",
    "/openai/realtimex",
    "/openai/v2/realtime",
    "/openai/deployments/d/realtime",
    "/ws/google.cloud.aiplatform.v1beta.LlmBidiService/BidiGenerateContent",
    "/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent/x",
    "/ws/google.cloud.aiplatform.v1.LlmBidiService.BidiGenerateContent",
    "/model/anthropic.claude-3/invoke-with-bidirectional-stream",
)


@pytest.mark.parametrize("path", [*ROUTES, *UNROUTED])
async def test_every_path_is_claimed_by_at_most_one_adapter(path: str) -> None:
    adapters = [cls() for cls in ALL_WS_ADAPTERS]
    claims = [a.name for a in adapters if a.matches(path)]
    assert claims == ([ROUTES[path]] if path in ROUTES else [])
    found = ws_adapter_for(path, adapters)
    assert (found.name if found else None) == ROUTES.get(path)


async def test_vertex_live_inherits_gemini_live_and_targets_vertex() -> None:
    vertex = VertexLiveWs()
    assert isinstance(vertex, GeminiLiveWs)
    assert (vertex.name, vertex.provider) == ("vertex-live", "vertex")
    assert isinstance(AzureRealtimeWs(), OpenAIRealtimeWs)


async def test_only_the_exact_documented_paths_are_authorizable() -> None:
    authorizable = {
        (cls.name, path)
        for cls in ALL_WS_ADAPTERS
        for path in [*ROUTES, *UNROUTED]
        if cls().authorizable(path)
    }
    assert authorizable == {
        ("azure-realtime", AZURE_PREVIEW),
        ("azure-realtime", AZURE_GA),
        ("vertex-live", VERTEX_V1),
        ("vertex-live", VERTEX_V1BETA1),
    }


# --- credential channels --------------------------------------------------------------


@pytest.mark.parametrize(
    ("offer", "credential"),
    [
        ("realtime", False),
        ("openai-beta.realtime-v1", False),
        ("openai-insecure-api-key.sk-abc", True),
        ("OpenAI-Insecure-API-Key.sk-abc", True),
        ("x-api_key.k", True),
        ("apikey.k", True),
        ("authorization.bearer.t", True),
        ("bearer.t", True),
        ("ephemeral-token.t", True),
        ("client-secret.s", True),
        ("password.p", True),
    ],
)
async def test_credential_subprotocols(offer: str, credential: bool) -> None:
    assert is_credential_subprotocol(offer) is credential


async def test_authorization_query_parameters_are_credentials() -> None:
    assert (
        _strip_credential_query(
            "model=d&Authorization=Bearer%20t&x-ms-authorization=a&api-key=k&intent=x"
        )
        == "model=d&intent=x"
    )


async def test_close_reason_fits_a_close_frame() -> None:
    assert _close_reason("short") == "short"
    long = "é" * 100  # 200 bytes of UTF-8
    cut = _close_reason(long)
    assert len(cut.encode("utf-8")) <= 123
    assert cut == "é" * 61  # never splits a character


# --- end to end -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "query"),
    [
        (AZURE_PREVIEW, "api-version=2025-04-01-preview&deployment=gpt-realtime"),
        (AZURE_GA, "model=gpt-realtime"),
    ],
)
async def test_azure_realtime_authorized_with_the_proxy_identity(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, path: str, query: str
) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    caplog.set_level(logging.DEBUG, logger="llm_redact")
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, "azure", auth="identity")) as proxy_host:
            async with websockets.connect(
                f"ws://{proxy_host}{path}?{query}&api-key=client-query-key"
                "&Authorization=Bearer%20client-query-token",
                additional_headers={
                    "api-key": "client-header-key",
                    "Authorization": "Bearer client-entra-token",
                    "x-ms-client-request-id": "req-1",
                },
                subprotocols=[
                    websockets.Subprotocol("realtime"),
                    websockets.Subprotocol("openai-insecure-api-key.client-subprotocol-key"),
                    websockets.Subprotocol("openai-beta.realtime-v1"),
                ],
            ) as client:
                assert client.subprotocol == "realtime"
                await client.send(_item_created(f"mail {EMAIL} today"))
                echoed = json.loads(await client.recv())
                # Rehydration: the upstream echoed placeholders, the client
                # gets its original back.
                assert echoed["item"]["content"][0]["text"] == f"mail {EMAIL} today"

    # The authorizer saw the upgrade as an HTTP GET with an empty body, the
    # URL on the configured upstream minus query credentials, and no client
    # credential header.
    [(method, url, seen_headers, body)] = auth.calls
    assert (method, body) == ("GET", b"")
    assert url == f"http://127.0.0.1:{fake.port}{path}?{query}"
    seen = {name.lower(): value for name, value in seen_headers}
    assert "api-key" not in seen and "authorization" not in seen
    assert seen["x-ms-client-request-id"] == "req-1"  # the rest rides along

    # The upstream got exactly what the authorizer returned.
    assert fake.paths == [f"{path}?{query}"]
    upstream = fake.headers[0]
    assert upstream["authorization"] == PROXY_TOKEN
    assert "api-key" not in upstream
    assert upstream["sec-websocket-protocol"] == "realtime, openai-beta.realtime-v1"
    # Redaction still applies to frames.
    [frame] = fake.received
    assert isinstance(frame, str) and EMAIL not in frame and "«EMAIL_" in frame
    # No credential (or URL query) ever reaches a log line.
    for secret in ("client-", "proxy-identity-token", "api-version=", "model="):
        assert secret not in caplog.text


@pytest.mark.parametrize("path", [VERTEX_V1, VERTEX_V1BETA1])
async def test_vertex_live_authorized_with_the_proxy_identity(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, "vertex", auth="identity")) as proxy_host:
            async with websockets.connect(
                f"ws://{proxy_host}{path}?key=client-query-key",
                additional_headers={
                    "x-goog-api-key": "client-express-key",
                    "x-goog-user-project": "someone-elses-project",
                },
            ) as client:
                # Gemini Live JSON rides binary frames: binary in, binary out.
                message = {
                    "serverContent": {
                        "modelTurn": {"parts": [{"text": f"hi {EMAIL}"}]},
                        "turnComplete": True,
                    }
                }
                await client.send(json.dumps(message).encode())
                echoed = await client.recv()
                assert isinstance(echoed, bytes)
                assert json.loads(echoed) == message

    [(method, url, seen_headers, body)] = auth.calls
    assert (method, url, body) == ("GET", f"http://127.0.0.1:{fake.port}{path}", b"")
    assert not {name.lower() for name, _ in seen_headers} & {
        "x-goog-api-key",
        "x-goog-user-project",
    }
    assert fake.paths == [path]
    assert fake.headers[0]["authorization"] == PROXY_TOKEN
    assert "x-goog-api-key" not in fake.headers[0]
    [frame] = fake.received
    assert isinstance(frame, bytes) and EMAIL.encode() not in frame


@pytest.mark.parametrize(
    ("provider", "path"),
    [
        ("azure", "/openai/realtime/extra"),
        ("azure", "/openai/v1/realtime/extra"),
        ("vertex", VERTEX_V1 + "/extra"),
    ],
)
async def test_an_unrecognized_path_is_never_authorized(
    monkeypatch: pytest.MonkeyPatch, provider: str, path: str
) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, provider, auth="identity")) as proxy_host:
            async with websockets.connect(f"ws://{proxy_host}{path}") as client:
                closed = await _closed(client)
    assert closed.code == 1011
    # Azure subpaths route to the Azure adapter but are not authorizable;
    # a Vertex look-alike has no route at all. Either way: never dialled.
    assert ('auth = "identity"' in closed.reason) is (provider == "azure")
    assert auth.calls == [] and fake.paths == []


async def test_bedrock_has_no_realtime_route(monkeypatch: pytest.MonkeyPatch) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, "bedrock", auth="identity", region="us-east-1")) as host:
            for path in ("/model/m/invoke-with-bidirectional-stream", "/model/m/converse"):
                async with websockets.connect(f"ws://{host}{path}") as client:
                    closed = await _closed(client)
                assert (closed.code, closed.reason) == (1011, "no realtime route for this path")
    assert auth.calls == [] and fake.paths == []


@pytest.mark.parametrize(
    ("error", "shown", "hidden"),
    [
        (UpstreamAuthError("Microsoft Entra ID credentials"), "Microsoft Entra ID credentials", ""),
        (RuntimeError("token endpoint said secret-xyz"), "RuntimeError", "secret-xyz"),
    ],
)
async def test_authorizer_failure_refuses_counts_and_records(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    shown: str,
    hidden: str,
) -> None:
    auth = FakeAuth(error=error)
    _install(monkeypatch, auth)
    caplog.set_level(logging.DEBUG, logger="llm_redact")
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, "azure", auth="identity")) as proxy_host:
            async with websockets.connect(
                f"ws://{proxy_host}{AZURE_GA}?model=d",
                additional_headers={"api-key": "client-header-key"},
            ) as client:
                closed = await _closed(client)
            row = await _recent(proxy_host, lambda r: r["method"] == "WS")
            status = await _status(proxy_host)
    assert closed.code == 1011
    assert shown in closed.reason and "nothing was forwarded" in closed.reason
    assert fake.paths == []  # never dialled — no credential, no request
    assert (row["status"], row["provider"], row["path"]) == (502, "azure", AZURE_GA)
    assert status["upstream_errors_total"] == {"azure": 1}
    assert "client-header-key" not in caplog.text
    if hidden:
        assert hidden not in closed.reason and hidden not in caplog.text


async def test_upstream_connect_failure_is_counted_and_recorded() -> None:
    # A port nothing listens on: the dial fails after the audit START.
    async with FakeUpstream() as fake:
        dead_port = fake.port
    with _proxy(_config(dead_port)) as proxy_host:
        async with websockets.connect(f"ws://{proxy_host}/v1/realtime") as client:
            closed = await _closed(client)
        row = await _recent(proxy_host, lambda r: r["method"] == "WS")
        status = await _status(proxy_host)
    assert (closed.code, closed.reason) == (1011, "upstream websocket connect failed")
    assert (row["status"], row["provider"]) == (502, "openai")
    assert status["upstream_errors_total"] == {"openai": 1}


async def test_identity_provider_missing_its_authorizer_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Startup refuses an identity provider with no authorizer; a state that
    # nevertheless lacks one must still fail closed rather than forward the
    # client's credential.
    auth = FakeAuth()
    _install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        app = create_app(_config(fake.port, "azure", auth="identity"))
        app.state.proxy.upstream_auth = {}
        with _serve(app) as proxy_host:
            async with websockets.connect(f"ws://{proxy_host}{AZURE_GA}") as client:
                closed = await _closed(client)
    assert closed.code == 1011 and 'auth = "identity"' in closed.reason
    assert auth.calls == [] and fake.paths == []


@contextlib.contextmanager
def _serve(app: Any) -> Iterator[str]:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield f"127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def test_passthrough_keeps_every_subprotocol() -> None:
    # Without identity auth the client's own credential channels are the
    # contract: a browser key in a subprotocol still reaches its provider.
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, "azure")) as proxy_host:
            async with websockets.connect(
                f"ws://{proxy_host}{AZURE_GA}?model=d&api-key=k",
                subprotocols=[
                    websockets.Subprotocol("realtime"),
                    websockets.Subprotocol("openai-insecure-api-key.sk"),
                ],
            ) as client:
                assert client.subprotocol == "realtime"
    assert fake.paths == [f"{AZURE_GA}?model=d&api-key=k"]
    assert fake.headers[0]["sec-websocket-protocol"] == "realtime, openai-insecure-api-key.sk"
