"""Identity auth signs only bodies the proxy could redact.

Under ``[providers.NAME] auth = "identity"`` a non-empty request body on a
matched route is forwarded only when llm-redact actually walked it: a JSON
object, or canonical multipart on a route whose ``redact_multipart`` scans
it. Anything else (non-JSON bytes, invalid UTF-8, a top-level array or
scalar, whitespace, a content-encoded body, multipart elsewhere) is a
recorded, provider-shaped 400 before the authorizer or the upstream is
touched. An empty body still forwards, ``detection = false`` stays the
explicit unredacted opt-out, and passthrough-auth providers keep forwarding
non-JSON verbatim (never break the tool). The realtime relay applies the
same rule per frame: a non-JSON frame on an identity connection closes it
1008, never relayed.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import logging
from typing import Any

import httpx
import pytest
import websockets

from llm_redact.config import DetectionConfig, ProviderConfig
from llm_redact.providers.azure_openai import AzureOpenAIAdapter
from llm_redact.providers.bedrock import BedrockAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.vertex import VertexAdapter
from llm_redact.proxy import create_app
from llm_redact.realtime import GeminiLiveWs, OpenAIRealtimeWs, WsAdapter
from llm_redact.redactor import UnredactableRequest
from test_realtime_identity import AZURE_GA, VERTEX_V1, _closed, _item_created, _recent
from test_realtime_identity import FakeAuth as WsAuth
from test_realtime_identity import _install as ws_install
from test_realtime_relay import FakeUpstream, _proxy
from test_realtime_relay import _config as ws_config
from test_upstream_auth import (
    AZURE,
    AZURE_PATH,
    BEDROCK,
    BEDROCK_PATH,
    EMAIL,
    VERTEX,
    VERTEX_PATH,
    _client,
    _config,
    _identity,
    _install,
    _Upstream,
)

# (provider, base URL, a matched POST route) per identity-capable family.
FAMILIES = [
    ("bedrock", BEDROCK, BEDROCK_PATH),
    ("vertex", VERTEX, VERTEX_PATH),
    ("azure", AZURE, AZURE_PATH + "?api-version=1"),
]
# A matched body-less GET per family (REDACT_ONLY metadata routes).
EMPTY_GETS = [
    ("bedrock", BEDROCK, "/async-invoke"),
    ("vertex", VERTEX, "/v1/publishers/google/models/gemini-2.5-pro"),
    ("azure", AZURE, "/openai/models?api-version=1"),
]

_OBJECT = json.dumps({"messages": [{"role": "user", "content": f"mail {EMAIL}"}]}).encode()

# (body, extra headers): every one would be forwarded verbatim unscanned.
REFUSED: dict[str, tuple[bytes, dict[str, str]]] = {
    "text": (f"mail {EMAIL}".encode(), {"content-type": "text/plain"}),
    "json-content-type-non-json": (f"mail {EMAIL}".encode(), {"content-type": "application/json"}),
    "invalid-utf8": (b'{"t": "\xc3\x28 ' + EMAIL.encode() + b'"}', {}),
    "array": (json.dumps([{"content": EMAIL}]).encode(), {}),
    "string": (json.dumps(EMAIL).encode(), {}),
    "number": (b"4111111111111111", {}),
    "null": (b"null", {}),
    "whitespace": (b" \r\n\t", {}),
    "gzip": (gzip.compress(_OBJECT), {"content-encoding": "gzip"}),
    # Plain JSON claiming an encoding: the upstream would decode what the
    # proxy never saw.
    "encoded-claim": (_OBJECT, {"content-encoding": "deflate"}),
    "non-canonical-multipart": (
        b'--b\nContent-Disposition: form-data; name="prompt"\n\n' + EMAIL.encode() + b"\n--b--\n",
        {"content-type": "multipart/form-data; boundary=b"},
    ),
    # Canonical multipart, but on a JSON route: nothing would scan it.
    "multipart-off-route": (
        b'--b\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
        + EMAIL.encode()
        + b"\r\n--b--\r\n",
        {"content-type": "multipart/form-data; boundary=b"},
    ),
}


def _error_message(provider: str, payload: dict[str, Any]) -> str:
    if provider == "bedrock":
        return str(payload["message"])
    return str(payload["error"]["message"])


@pytest.mark.parametrize(("provider", "base", "path"), FAMILIES)
@pytest.mark.parametrize("kind", sorted(REFUSED))
async def test_unwalkable_body_refused_before_auth_and_upstream(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    base: str,
    path: str,
    kind: str,
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: _identity(base)}), upstream_transport=httpx.MockTransport(upstream)
    )
    body, headers = REFUSED[kind]
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(path, content=body, headers=headers)
    assert response.status_code == 400
    # Never sent, never signed: no credential fetch for a refused body.
    assert upstream.requests == [] and built[0].calls == []
    # Provider-shaped, naming the body's kind only.
    message = _error_message(provider, response.json())
    assert "proxy's own identity" in message
    assert EMAIL not in response.text and EMAIL not in caplog.text
    state = app.state.proxy
    assert state.recent[-1]["status"] == 400
    assert state.recent[-1]["provider"] == provider


@pytest.mark.parametrize(("provider", "base", "path"), FAMILIES)
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_OBJECT, id="object"),
        # json.loads reads a BOM / UTF-16 from bytes: the object IS walked.
        pytest.param(b"\xef\xbb\xbf" + _OBJECT, id="utf8-bom"),
        pytest.param(_OBJECT.decode().encode("utf-16"), id="utf16"),
    ],
)
async def test_json_object_still_redacted_and_signed(
    monkeypatch: pytest.MonkeyPatch, provider: str, base: str, path: str, body: bytes
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: _identity(base)}), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        response = await client.post(
            path,
            content=body,
            headers={"content-type": "application/json", "content-encoding": "identity"},
        )
    assert response.status_code == 200
    [sent] = upstream.requests
    assert EMAIL.encode() not in sent.content and "«EMAIL_".encode() in sent.content
    assert len(built[0].calls) == 1


@pytest.mark.parametrize(("provider", "base", "path"), EMPTY_GETS)
async def test_empty_body_still_forwarded(
    monkeypatch: pytest.MonkeyPatch, provider: str, base: str, path: str
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: _identity(base)}), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        response = await client.get(path)
    assert response.status_code == 200
    assert len(upstream.requests) == 1 and len(built[0].calls) == 1


async def test_body_less_post_still_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        response = await client.post("/openai/batches/batch_1/cancel?api-version=1")
    assert response.status_code == 200
    assert len(upstream.requests) == 1 and len(built[0].calls) == 1


async def test_multipart_files_upload_still_forwarded_under_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    line = json.dumps({"messages": [{"role": "user", "content": EMAIL}]}).encode()
    body = (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="in.jsonl"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n" + line + b"\r\n--b--\r\n"
    )
    async with _client(app) as client:
        response = await client.post(
            "/openai/files?api-version=1",
            content=body,
            headers={"content-type": "multipart/form-data; boundary=b"},
        )
    assert response.status_code == 200
    assert EMAIL.encode() not in upstream.requests[0].content


@pytest.mark.parametrize(("provider", "base", "path"), FAMILIES)
async def test_detection_off_stays_the_unredacted_opt_out(
    monkeypatch: pytest.MonkeyPatch, provider: str, base: str, path: str
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: _identity(base, detection=False)}),
        upstream_transport=httpx.MockTransport(upstream),
    )
    body = f"mail {EMAIL}".encode()
    async with _client(app) as client:
        response = await client.post(path, content=body, headers={"content-type": "text/plain"})
    assert response.status_code == 200
    assert upstream.requests[0].content == body and len(built[0].calls) == 1


@pytest.mark.parametrize(("provider", "base", "path"), FAMILIES)
@pytest.mark.parametrize("kind", ["text", "array", "gzip", "multipart-off-route"])
async def test_passthrough_auth_still_forwards_verbatim(
    monkeypatch: pytest.MonkeyPatch, provider: str, base: str, path: str, kind: str
) -> None:
    _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: ProviderConfig(base)}),
        upstream_transport=httpx.MockTransport(upstream),
    )
    body, headers = REFUSED[kind]
    async with _client(app) as client:
        response = await client.post(path, content=body, headers=headers)
    assert response.status_code == 200
    assert upstream.requests[0].content == body


def test_redacts_multipart_names_only_the_scanned_routes() -> None:
    assert not BedrockAdapter().redacts_multipart(BEDROCK_PATH)
    assert not VertexAdapter().redacts_multipart(VERTEX_PATH)
    azure = AzureOpenAIAdapter()
    assert azure.redacts_multipart("/openai/files")
    assert azure.redacts_multipart("/openai/v1/files")
    assert azure.redacts_multipart("/openai/deployments/d/images/edits")
    assert not azure.redacts_multipart(AZURE_PATH)
    assert OpenAIAdapter().redacts_multipart("/v1/files")
    assert OpenAIAdapter().redacts_multipart("/v1/videos")


# --- realtime -----------------------------------------------------------------------


def _ctx() -> Any:
    return None  # the non-JSON path never reads the context


@pytest.mark.parametrize("adapter", [WsAdapter(), OpenAIRealtimeWs(), GeminiLiveWs()])
@pytest.mark.parametrize("frame", ["not json", b"\x00\x01opaque", b"\xef\xbb\xbf{}"])
def test_ws_adapters_refuse_unparsed_frames_only_when_required(
    adapter: WsAdapter, frame: str | bytes
) -> None:
    assert adapter.redact_message(frame, _ctx()) == frame
    with pytest.raises(UnredactableRequest, match="not JSON"):
        adapter.redact_message(frame, _ctx(), require_json=True)


@pytest.mark.parametrize(
    ("provider", "path", "frame"),
    [
        ("azure", AZURE_GA, f"mail {EMAIL}"),
        ("azure", AZURE_GA, b"\x00\x01" + EMAIL.encode()),
        ("vertex", VERTEX_V1, EMAIL.encode()),
    ],
)
async def test_ws_identity_non_json_frame_closes_1008_never_relayed(
    monkeypatch: pytest.MonkeyPatch, provider: str, path: str, frame: str | bytes
) -> None:
    auth = WsAuth()
    ws_install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(ws_config(fake.port, provider, auth="identity")) as host:
            async with websockets.connect(f"ws://{host}{path}") as client:
                await client.send(frame)
                closed = await _closed(client)
            row = await _recent(host, lambda r: r["method"] == "WS")
    assert closed.code == 1008 and "not JSON" in closed.reason
    assert EMAIL not in closed.reason
    assert fake.received == []
    assert row["status"] == 400


async def test_ws_identity_json_frames_still_relayed(monkeypatch: pytest.MonkeyPatch) -> None:
    auth = WsAuth()
    ws_install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(ws_config(fake.port, "vertex", auth="identity")) as host:
            async with websockets.connect(f"ws://{host}{VERTEX_V1}") as client:
                # Gemini Live JSON in a BINARY frame stays allowed.
                message = {"clientContent": {"turns": [{"parts": [{"text": EMAIL}]}]}}
                await client.send(json.dumps(message).encode())
                echoed = await client.recv()
    assert isinstance(echoed, bytes)
    [sent] = fake.received
    assert isinstance(sent, bytes) and EMAIL.encode() not in sent


async def test_ws_identity_detection_off_relays_non_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = WsAuth()
    ws_install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(ws_config(fake.port, "azure", auth="identity", detection=False)) as host:
            async with websockets.connect(f"ws://{host}{AZURE_GA}") as client:
                await client.send("not-json ping")
                assert await client.recv() == "not-json ping"


async def test_ws_passthrough_azure_still_relays_non_json() -> None:
    async with FakeUpstream() as fake:
        with _proxy(ws_config(fake.port, "azure")) as host:
            async with websockets.connect(f"ws://{host}{AZURE_GA}") as client:
                await client.send(_item_created("hello"))
                await client.recv()
                await client.send(b"\x00\x01opaque")
                assert await client.recv() == b"\x00\x01opaque"
    assert fake.received[1:] == [b"\x00\x01opaque"]


async def test_ws_block_mode_closes_1008_ahead_of_the_upstream_close() -> None:
    # The policy close reaches the client before the upstream's 1000 is
    # mirrored (closing the upstream first used to let 1000 win the race).
    async with FakeUpstream() as fake:
        config = dataclasses.replace(
            ws_config(fake.port, "azure"),
            detection=DetectionConfig(modes=(("email", "block"),)),
        )
        with _proxy(config) as host:
            async with websockets.connect(f"ws://{host}{AZURE_GA}") as client:
                await client.send(_item_created(f"mail {EMAIL}"))
                closed = await _closed(client)
    assert closed.code == 1008 and "blocked" in closed.reason
    assert fake.received == []
