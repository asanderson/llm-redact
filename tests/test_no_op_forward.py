"""No-op short-circuit: when redaction/rehydration change nothing, the proxy
forwards the ORIGINAL bytes instead of a parse->dump round-trip."""

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig

captured: dict[str, bytes] = {}


def _fake_upstream() -> Starlette:
    async def messages(request: Request) -> Response:
        captured["request_bytes"] = await request.body()
        return Response(content=captured["response_bytes"], media_type="application/json")

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"])])


def _client(response_bytes: bytes) -> httpx.AsyncClient:
    from llm_redact.proxy import create_app

    captured.clear()
    captured["response_bytes"] = response_bytes
    config = Config(providers={**Config().providers, "anthropic": ProviderConfig("http://up")})
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


_JSON = {"content-type": "application/json"}


async def test_unredacted_request_forwards_original_bytes() -> None:
    # Distinctive whitespace + key order a re-serialize would normalize away.
    original = b'{"messages" :[{"role":"user","content":"hello there"}],   "model":"m"}'
    client = _client(b'{"content":[{"type":"text","text":"hi"}]}')
    response = await client.post("/v1/messages", content=original, headers=_JSON)
    assert response.status_code == 200
    # Nothing to redact -> the upstream got the EXACT original bytes.
    assert captured["request_bytes"] == original
    await client.aclose()


async def test_redacted_request_is_reserialized() -> None:
    original = b'{"model":"m","messages":[{"role":"user","content":"mail jane@corp.example"}]}'
    client = _client(b'{"content":[{"type":"text","text":"ok"}]}')
    response = await client.post("/v1/messages", content=original, headers=_JSON)
    assert response.status_code == 200
    body = captured["request_bytes"]
    assert b"jane@corp.example" not in body and b"EMAIL_001" in body  # redacted + re-serialized
    await client.aclose()


async def test_response_without_tokens_forwards_original_bytes() -> None:
    # Distinctive response formatting, no placeholders to restore.
    response_bytes = b'{"content" :  [{"type":"text","text":"plain reply"}]}'
    client = _client(response_bytes)
    request = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'
    response = await client.post("/v1/messages", content=request, headers=_JSON)
    assert response.status_code == 200
    assert response.content == response_bytes  # byte-identical pass-through
    await client.aclose()
