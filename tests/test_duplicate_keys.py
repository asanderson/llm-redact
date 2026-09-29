"""A repeated JSON key is never a redaction bypass.

``json.loads`` keeps the LAST occurrence of a repeated key, so the redactor
never walks the earlier ones — while an upstream parser may keep the FIRST.
Wherever the proxy would forward a request's ORIGINAL bytes because the walk
changed nothing (the HTTP no-op short-circuit, uploaded JSONL lines), a body
with a repeated key is re-serialized from the walked, parsed object instead;
the Bedrock CountTokens blob (inside the envelope the short-circuit cannot
see into) is refused. Realtime frames are always re-serialized. Pinned end to
end on key-authorized and identity-authorized providers.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
import websockets

from llm_redact.config import Config
from llm_redact.jsonwalk import loads_request
from llm_redact.proxy import create_app
from test_realtime_relay import _relay_setup
from test_upstream_auth import (
    AZURE,
    AZURE_PATH,
    BEDROCK,
    BEDROCK_PATH,
    EMAIL,
    _client,
    _config,
    _identity,
    _install,
    _Upstream,
)

CLEAN = [{"role": "user", "content": "hello there"}]
# The email rides the FIRST `messages`; the parse keeps the clean second one.
TOP_LEVEL_DUP = (
    '{"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "'
    + EMAIL
    + '"}], "messages": '
    + json.dumps(CLEAN)
    + "}"
).encode()
# A repeated key inside a nested object.
NESTED_DUP = (
    '{"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "'
    + EMAIL
    + '", "content": "hello there"}]}'
).encode()


# --- the parse ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value", "duplicated"),
    [
        ('{"a": 1, "b": 2}', {"a": 1, "b": 2}, False),
        ('[{"a": 1}, {"a": 2}]', [{"a": 1}, {"a": 2}], False),  # siblings, not repeats
        ('{"a": 1, "a": 2}', {"a": 2}, True),
        ('{"x": [{"a": {"b": 1, "b": 2}}]}', {"x": [{"a": {"b": 2}}]}, True),
        ('"text"', "text", False),
        (b'{"a": {"a": 1}}', {"a": {"a": 1}}, False),  # nesting reuses a name
    ],
)
def test_loads_request_flags_every_repeated_key(
    text: str | bytes, value: Any, duplicated: bool
) -> None:
    assert loads_request(text) == (value, duplicated)


def test_loads_request_raises_like_json_loads() -> None:
    with pytest.raises(ValueError):
        loads_request(b"not json")


# --- HTTP ------------------------------------------------------------------------------


@pytest.mark.parametrize("body", [TOP_LEVEL_DUP, NESTED_DUP], ids=["top-level", "nested"])
async def test_duplicate_key_body_reserialized_on_a_passthrough_provider(body: bytes) -> None:
    upstream = _Upstream(b'{"content": []}')
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(
            "/v1/messages", content=body, headers={"content-type": "application/json"}
        )
    assert response.status_code == 200
    [sent] = upstream.requests
    assert EMAIL.encode() not in sent.content
    # Exactly the walked (last-wins) object went upstream.
    assert json.loads(sent.content) == json.loads(body)


async def test_clean_body_without_duplicates_still_forwarded_byte_identical() -> None:
    body = b'{"model":"m",  "max_tokens":8,"messages":[{"role":"user","content":"hi"}]}'
    upstream = _Upstream(b'{"content": []}')
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        await client.post("/v1/messages", content=body)
    assert upstream.requests[0].content == body


@pytest.mark.parametrize(
    ("provider", "base", "path"),
    [("bedrock", BEDROCK, BEDROCK_PATH), ("azure", AZURE, AZURE_PATH + "?api-version=1")],
)
@pytest.mark.parametrize("body", [TOP_LEVEL_DUP, NESTED_DUP], ids=["top-level", "nested"])
async def test_duplicate_key_body_reserialized_under_identity(
    monkeypatch: pytest.MonkeyPatch, provider: str, base: str, path: str, body: bytes
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: _identity(base)}), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        response = await client.post(path, content=body)
    assert response.status_code == 200
    [sent] = upstream.requests
    assert EMAIL.encode() not in sent.content
    # The authorizer signed the re-serialized bytes, not the original.
    assert built[0].calls[0][3] == sent.content


# --- JSONL file lines -------------------------------------------------------------------


async def test_duplicate_key_jsonl_line_reserialized() -> None:
    clean_line = b'{"custom_id": "a",  "body": {"messages": []}}'
    dup_line = b'{"custom_id": "b", "body": {"q": "' + EMAIL.encode() + b'", "q": "hi"}}'
    body = (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="in.jsonl"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n"
        + clean_line
        + b"\n"
        + dup_line
        + b"\r\n--b--\r\n"
    )
    upstream = _Upstream(b'{"id": "file-1"}')
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            content=body,
            headers={
                "content-type": "multipart/form-data; boundary=b",
                "authorization": "Bearer sk-test",
            },
        )
    assert response.status_code == 200
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent
    assert clean_line in sent  # an unchanged, duplicate-free line stays byte-identical
    assert b'{"custom_id": "b", "body": {"q": "hi"}}' in sent


# --- Bedrock CountTokens blob -----------------------------------------------------------


@pytest.mark.parametrize("identity", [False, True])
async def test_duplicate_key_count_tokens_blob_refused(
    monkeypatch: pytest.MonkeyPatch, identity: bool
) -> None:
    _, built = _install(monkeypatch)
    inner = (
        b'{"messages": [{"role": "user", "content": "' + EMAIL.encode() + b'"}], "messages": []}'
    )
    envelope = {"input": {"invokeModel": {"body": base64.b64encode(inner).decode()}}}
    upstream = _Upstream(b'{"inputTokens": 3}')
    provider = _identity(BEDROCK) if identity else _config().providers["bedrock"]
    app = create_app(_config(bedrock=provider), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(
            "/model/anthropic.claude-v2/count-tokens",
            content=json.dumps(envelope).encode(),
            headers={"authorization": "Bearer bedrock-key"},
        )
    assert response.status_code == 400
    assert "repeats a JSON key" in response.json()["message"]
    assert upstream.requests == []
    assert all(auth.calls == [] for auth in built)


# --- realtime ---------------------------------------------------------------------------


async def test_duplicate_key_realtime_frame_never_relayed_raw() -> None:
    frame = (
        '{"type": "conversation.item.create", "item": {"type": "message", "role": "user",'
        ' "content": [{"type": "input_text", "text": "' + EMAIL + '", "text": "hello there"}]}}'
    )
    async with (
        _relay_setup() as (fake, proxy_host),
        websockets.connect(
            f"ws://{proxy_host}/v1/realtime?model=m",
            additional_headers={"Authorization": "Bearer sk-test"},
        ) as client,
    ):
        await client.send(frame)
        await client.recv()
    [sent] = fake.received
    assert isinstance(sent, str) and EMAIL not in sent
    assert json.loads(sent) == json.loads(frame)
