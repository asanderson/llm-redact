"""A JSON document nested too deep to walk: refused coming in, forwarded as it
came going out — never a bare 500.

The parser reads a document hundreds (3.11) to thousands (3.12+) of levels
deep, and every walk over it — redaction, MCP stripping, rehydration,
re-serialization — recursed once or twice per level. ``'{"messages":' +
'[' * 200000`` escaped the body parse as a RecursionError (a bare,
unrecorded 500), and a body a few hundred levels deep parsed fine and then
overflowed the walk. Every document the proxy reads now nests at most
``MAX_JSON_DEPTH`` levels (``jsonwalk.loads_bounded``/``loads_request``), so
no walk can overflow:

- a CLIENT document deeper than that is unreadable: a request body, an
  uploaded JSONL line or form field, a Bedrock count-tokens blob is a
  recorded 400 with nothing forwarded; a realtime client frame closes the
  connection 1008 (recorded 400); a reserved endpoint's POST is a 400;
- an UPSTREAM document deeper than that is forwarded exactly as it came —
  its placeholders left in place, never a wrong value, never a 500 or a
  cut stream: a buffered answer, an SSE event, an NDJSON line, a JSONL line
  of a file download or batch results, an event-stream payload, a realtime
  upstream frame.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
import websockets

from llm_redact.config import Config, ProviderConfig
from llm_redact.eventstream import EventStreamMessage, EventStreamParser, string_header
from llm_redact.eventstream import serialize as serialize_eventstream
from llm_redact.jsonwalk import MAX_JSON_DEPTH, JsonTooDeep, loads_bounded, loads_request
from llm_redact.proxy import CSRF_HEADER, create_app
from llm_redact.realtime import GeminiLiveWs, OpenAIRealtimeWs
from llm_redact.redactor import UnredactableRequest
from llm_redact.rehydrate import RehydratorPool
from local_refusals import refused_once
from test_realtime_relay import _relay_setup

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"  # the first email the static session sees
BEDROCK = "https://bedrock-runtime.us-east-1.amazonaws.com"
# Past the parser on every Python version (a RecursionError inside it).
PARSER_DEEP = 200_000
# Parsed on every version, but deeper than any walk over it could recurse.
WALK_DEEP = 600


def _nested(depth: int, leaf: Any = EMAIL) -> Any:
    """``leaf`` inside ``depth`` levels of alternating objects and arrays."""
    value = leaf
    for level in range(depth):
        value = [value] if level % 2 else {"k": value}
    return value


def _text(depth: int, leaf: str = f'"{EMAIL}"') -> str:
    """``_nested`` as JSON text, built without json.dumps (which recurses)."""
    return "[" * depth + leaf + "]" * depth


# --- the limit itself -----------------------------------------------------------------


def test_a_document_nesting_up_to_the_limit_is_read_and_one_level_more_is_not() -> None:
    at_limit = json.dumps(_nested(MAX_JSON_DEPTH))
    assert loads_bounded(at_limit) == _nested(MAX_JSON_DEPTH)
    assert loads_request(at_limit.encode()) == (_nested(MAX_JSON_DEPTH), False)
    over = json.dumps(_nested(MAX_JSON_DEPTH + 1))
    for document in (over, over.encode(), _text(WALK_DEEP), _text(PARSER_DEEP)):
        with pytest.raises(JsonTooDeep) as raised:
            loads_bounded(document)
        assert str(raised.value) == f"the JSON document nests deeper than {MAX_JSON_DEPTH} levels"
        with pytest.raises(JsonTooDeep):
            loads_request(document)
    # A repeated key does not hide the depth (the second, plain parse).
    repeated = '{"a": 1, "a": ' + over + "}"
    with pytest.raises(JsonTooDeep):
        loads_request(repeated)
    # A ValueError: every caller that treats an unparseable document as such
    # treats this one the same way.
    assert issubclass(JsonTooDeep, ValueError)
    # Scalars and shallow documents, and every level counted: arrays and
    # objects alike, the widest level too.
    assert loads_bounded("1") == 1 and loads_bounded('"x"') == "x"
    assert loads_bounded("[[], {}, [[]]]") == [[], {}, [[]]]
    wide = {"a": [_nested(MAX_JSON_DEPTH - 2), 1], "b": {"c": 2}}
    assert loads_bounded(json.dumps(wide)) == wide
    wide["b"] = _nested(MAX_JSON_DEPTH)
    with pytest.raises(JsonTooDeep):
        loads_bounded(json.dumps(wide))


# --- client documents: refused, recorded, never forwarded ----------------------------


def _config() -> Config:
    providers = dict(Config().providers)
    providers["bedrock"] = ProviderConfig(BEDROCK)
    return Config(providers=providers)


class _Upstream:
    def __init__(self, content_type: str = "application/json", body: bytes = b"{}") -> None:
        self.requests: list[httpx.Request] = []
        self.content_type = content_type
        self.body = body

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, content=self.body, headers={"content-type": self.content_type})


async def _send(
    app: Any, method: str, path: str, content: bytes, headers: dict[str, str]
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.request(method, path, content=content, headers=headers)


ANTHROPIC = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
OPENAI = {"authorization": "Bearer sk-proj-FAKE", "content-type": "application/json"}


def _messages_body(nested: str) -> bytes:
    return (
        '{"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "mail '
        + EMAIL
        + '"}], "metadata": '
        + nested
        + "}"
    ).encode()


@pytest.mark.parametrize(
    "content",
    [
        ('{"messages":' + "[" * PARSER_DEEP + "]" * PARSER_DEEP + "}").encode(),
        _messages_body(_text(PARSER_DEEP)),
        _messages_body(_text(WALK_DEEP)),
        _messages_body(json.dumps(_nested(MAX_JSON_DEPTH))),  # the body adds a level
    ],
    ids=["reported", "parser-deep", "walk-deep", "one-over"],
)
async def test_a_request_body_nested_too_deep_is_a_recorded_400(content: bytes) -> None:
    upstream = _Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    response = await _send(app, "POST", "/v1/messages", content, ANTHROPIC)
    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]
    assert f"nests JSON deeper than {MAX_JSON_DEPTH} levels" in message
    assert EMAIL not in response.text
    assert upstream.requests == []
    (row,) = app.state.proxy.recent
    assert row["status"] == 400 and row["provider"] == "anthropic"


async def test_a_body_at_the_limit_is_redacted_and_forwarded() -> None:
    upstream = _Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    at_limit = json.dumps(_nested(MAX_JSON_DEPTH - 1))  # the body's own object is a level
    response = await _send(app, "POST", "/v1/messages", _messages_body(at_limit), ANTHROPIC)
    assert response.status_code == 200, response.text
    (sent,) = upstream.requests
    assert EMAIL not in sent.content.decode() and TOKEN in sent.content.decode()


def _multipart(content: bytes) -> tuple[bytes, dict[str, str]]:
    body = (
        b"--XyZ\r\n"
        b'Content-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
        b"--XyZ\r\n"
        b'Content-Disposition: form-data; name="file"; filename="in.jsonl"\r\n'
        b"Content-Type: application/jsonl\r\n\r\n" + content + b"\r\n--XyZ--\r\n"
    )
    return body, {**OPENAI, "content-type": "multipart/form-data; boundary=XyZ"}


async def test_an_uploaded_line_nested_too_deep_refuses_the_upload() -> None:
    upstream = _Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    line = json.dumps({"custom_id": "a", "body": {"input": EMAIL}}).encode()
    for deep in (_text(PARSER_DEEP), _text(WALK_DEEP)):
        body, headers = _multipart(line + b"\n" + ('{"x": ' + deep + "}").encode() + b"\n")
        response = await _send(app, "POST", "/v1/files", body, headers)
        assert response.status_code == 400, response.text
        assert EMAIL not in response.text
    assert upstream.requests == []


async def test_a_count_tokens_blob_nested_too_deep_is_refused() -> None:
    upstream = _Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    inner = ('{"messages": [{"role": "user", "content": "' + EMAIL + '"}], "x": ').encode()
    for deep in (_text(PARSER_DEEP), _text(WALK_DEEP)):
        blob = base64.b64encode(inner + deep.encode() + b"}").decode()
        body = json.dumps({"input": {"invokeModel": {"body": blob}}}).encode()
        response = await _send(
            app,
            "POST",
            "/model/m/count-tokens",
            body,
            {"authorization": "Bearer ABSK", "content-type": "application/json"},
        )
        assert response.status_code == 400, response.text
    assert upstream.requests == []


async def test_a_reserved_post_nested_too_deep_is_a_400(tmp_path: Any) -> None:
    from llm_redact.config import VaultConfig

    config = Config(vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")))
    app = create_app(config)
    try:
        for deep in (_text(PARSER_DEEP, "1"), _text(WALK_DEEP, "1")):
            response = await _send(
                app,
                "POST",
                "/__llm-redact/sessions/prune",
                ('{"older_than_days": 30, "x": ' + deep + "}").encode(),
                {CSRF_HEADER: app.state.proxy.csrf_token, "content-type": "application/json"},
            )
            assert response.status_code == 400, response.text
            assert "invalid JSON" in response.json()["error"]
    finally:
        app.state.proxy.vault_manager.close()


# --- upstream documents: forwarded as they came ----------------------------------------


async def _issued(app: Any) -> None:
    """Make «EMAIL_001» the static session's token for EMAIL."""
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]})
    response = await _send(app, "POST", "/v1/chat/completions", body.encode(), OPENAI)
    assert response.status_code == 200


@pytest.mark.parametrize("depth", [PARSER_DEEP, WALK_DEEP])
async def test_a_buffered_answer_nested_too_deep_is_delivered_as_it_came(depth: int) -> None:
    answer = ('{"id": "x", "choices": [{"message": {"content": "' + TOKEN + '"}}], "x": ').encode()
    answer += _text(depth).encode() + b"}"
    upstream = _Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    await _issued(app)
    upstream.body = answer
    request = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    response = await _send(app, "POST", "/v1/chat/completions", request.encode(), OPENAI)
    assert response.status_code == 200, response.text[:300]
    assert response.content == answer  # its placeholder left in place, never guessed


def _sse(*payloads: bytes) -> bytes:
    return b"".join(b"data: " + payload + b"\n\n" for payload in payloads)


def _chunk(text: str) -> bytes:
    return json.dumps({"choices": [{"index": 0, "delta": {"content": text}}]}).encode()


def _deep_chunk(depth: int) -> bytes:
    return b'{"choices": [{"index": 0, "delta": {"x": ' + _text(depth).encode() + b"}}]}"


def _gemini_chunk(text: str, depth: int | None = None) -> bytes:
    part = '{"text": "' + text + '"'
    if depth is not None:
        part += ', "x": ' + _text(depth)
    return ('{"candidates": [{"index": 0, "content": {"parts": [' + part + "}]}}]}").encode()


def _frame(event_type: str, payload: bytes) -> bytes:
    return serialize_eventstream(
        EventStreamMessage(
            headers=[
                string_header(":message-type", "event"),
                string_header(":event-type", event_type),
                string_header(":content-type", "application/json"),
            ],
            payload=payload,
        )
    )


def _deep_payloads(depth: int) -> dict[str, tuple[str, str, dict[str, str], str, bytes, bytes]]:
    """name -> (method, path, headers, content type, upstream bytes, the deep
    unit the client must receive byte-identically)."""
    deep_chunk = _deep_chunk(depth)
    deep_gemini = _gemini_chunk("x", depth)
    deep_line = b'{"model": "m", "message": {"role": "assistant", "content": "x", "x": '
    deep_line += _text(depth).encode() + b"}}"
    deep_result = b'{"custom_id": "b", "result": ' + _text(depth).encode() + b"}"
    deep_converse = b'{"contentBlockIndex": 0, "delta": {"text": "x", "x": '
    deep_converse += _text(depth).encode() + b"}}"
    ollama_done = b'{"model": "m", "message": {"role": "assistant", "content": ""}, "done": true}'
    succeeded = json.dumps(
        {"custom_id": "a", "result": {"type": "succeeded", "message": {"content": TOKEN}}}
    ).encode()
    return {
        "openai-sse": (
            "POST",
            "/v1/chat/completions",
            OPENAI,
            "text/event-stream",
            _sse(_chunk(f"hi {TOKEN}"), deep_chunk, _chunk(f"bye {TOKEN}")) + b"data: [DONE]\n\n",
            deep_chunk,
        ),
        "gemini-sse": (
            "POST",
            "/v1beta/models/m:streamGenerateContent?alt=sse",
            {"x-goog-api-key": "AIzaFAKE", "content-type": "application/json"},
            "text/event-stream",
            _sse(_gemini_chunk(f"hi {TOKEN}"), deep_gemini, _gemini_chunk(f"bye {TOKEN}")),
            deep_gemini,
        ),
        "ollama-ndjson": (
            "POST",
            "/api/chat",
            {"content-type": "application/json"},
            "application/x-ndjson",
            b'{"model": "m", "message": {"role": "assistant", "content": "hi '
            + TOKEN.encode()
            + b'"}}\n'
            + deep_line
            + b"\n"
            + ollama_done
            + b"\n",
            deep_line,
        ),
        "anthropic-batch-results": (
            "GET",
            "/v1/messages/batches/msgbatch_1/results",
            ANTHROPIC,
            "application/x-jsonl",
            succeeded + b"\n" + deep_result + b"\n",
            deep_result,
        ),
        "openai-file-content": (
            "GET",
            "/v1/files/file-1/content",
            OPENAI,
            "application/octet-stream",
            succeeded + b"\n" + deep_result + b"\n",
            deep_result,
        ),
        "bedrock-eventstream": (
            "POST",
            "/model/m/converse-stream",
            {"authorization": "Bearer ABSK", "content-type": "application/json"},
            "application/vnd.amazon.eventstream",
            _frame(
                "contentBlockDelta",
                json.dumps({"contentBlockIndex": 0, "delta": {"text": f"hi {TOKEN}"}}).encode(),
            )
            + _frame("contentBlockDelta", deep_converse)
            + _frame("contentBlockStop", b'{"contentBlockIndex": 0}'),
            deep_converse,
        ),
    }


@pytest.mark.parametrize("depth", [PARSER_DEEP, WALK_DEEP])
@pytest.mark.parametrize("name", sorted(_deep_payloads(1)))
async def test_a_streamed_unit_nested_too_deep_is_forwarded_as_it_came(
    name: str, depth: int
) -> None:
    method, path, headers, content_type, stream, deep = _deep_payloads(depth)[name]
    upstream = _Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    await _issued(app)
    upstream.content_type, upstream.body = content_type, stream
    request = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    response = await _send(
        app, method, path, request.encode() if method == "POST" else b"", headers
    )
    assert response.status_code == 200
    content = response.content
    if content_type == "application/vnd.amazon.eventstream":
        payloads = [message.payload for message in EventStreamParser().feed(content)]
        assert deep in payloads
        assert f"hi {EMAIL}".encode() in b"".join(payloads)
    else:
        assert deep in content  # the deep unit byte-identical, the stream whole
        assert EMAIL.encode() in content  # the other units restored


# --- realtime frames -----------------------------------------------------------------


@pytest.mark.parametrize("adapter", [OpenAIRealtimeWs(), GeminiLiveWs()], ids=["openai", "gemini"])
def test_a_realtime_frame_nested_too_deep(adapter: Any) -> None:
    deep = '{"type": "x", "x": ' + _text(WALK_DEEP) + "}"
    with pytest.raises(UnredactableRequest, match=f"deeper than {MAX_JSON_DEPTH} levels"):
        adapter.redact_message(deep, None)
    for frame in (deep, deep.encode(), _text(PARSER_DEEP)):
        assert adapter.rehydrate_message(frame, RehydratorPool(None)) == [frame]


async def test_the_relay_refuses_a_client_frame_nested_too_deep() -> None:
    async with (
        _relay_setup() as (fake, proxy_host),
        websockets.connect(f"ws://{proxy_host}/v1/realtime") as client,
    ):
        await client.send('{"type": "x", "x": ' + _text(PARSER_DEEP) + "}")
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
            await client.recv()
    assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
    assert f"deeper than {MAX_JSON_DEPTH} levels" in closed.value.rcvd.reason
    assert fake.received == []


async def test_the_relay_forwards_an_upstream_frame_nested_too_deep_as_it_came() -> None:
    deep = '{"type": "x", "x": ' + _text(PARSER_DEEP) + "}"
    async with _relay_setup() as (fake, proxy_host):
        fake.greeting = deep
        async with websockets.connect(f"ws://{proxy_host}/v1/realtime") as client:
            assert await client.recv() == deep
            await client.send('{"type": "session.update", "session": {}}')
            assert json.loads(await client.recv()) == {"type": "session.update", "session": {}}


# --- the stored-object check's reading of an upload --------------------------------------


def test_the_ownership_check_cannot_read_an_upload_nested_too_deep() -> None:
    from llm_redact.upload_view import TOO_DEEP, UploadView, read_upload

    deep = ('{"custom_id": "a", "x": ' + _text(WALK_DEEP) + "}").encode()
    field = (
        b'--XyZ\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n'
        + deep
        + b"\r\n--XyZ--\r\n"
    )
    for body in (_multipart(deep)[0], field):
        assert read_upload(body, b"XyZ", max_json_bytes=10**7, max_lines=10_000) == UploadView(
            [], problem=TOO_DEEP
        )
    assert f"a multipart part nests JSON deeper than {MAX_JSON_DEPTH} levels" == TOO_DEEP


async def test_under_identity_with_detection_off_an_upload_nested_too_deep_is_never_signed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing is redacted there, so the stored-object check is the only
    reader of an upload's lines: one it cannot walk (a batch line citing a
    stored object below a deep branch) is refused, never signed."""
    from license_fixtures import resolved
    from llm_redact.registry import Registry
    from llm_redact.upload_view import TOO_DEEP
    from test_object_access_seams import AZURE, FakeAuth, LineRouter, Upstream, _app, _client

    auth = FakeAuth()
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("pro")
    reg.build_access_gate = lambda cfg, lic: None
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    router = LineRouter()
    upstream = Upstream()
    app = _app(
        monkeypatch,
        router,
        upstream,
        registry=reg,
        providers={"azure": ProviderConfig(AZURE, auth="identity", detection=False)},
    )
    line = '{"custom_id": "a", "body": {"previous_response_id": "resp_x"}, "x": '
    body, headers = _multipart((line + _text(WALK_DEEP) + "}").encode())
    async with _client(app) as client:
        response = await client.post("/openai/v1/files", content=body, headers=headers)
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert TOO_DEEP in message and "proxy's own identity" in message
    assert upstream.requests == [] and auth.calls == 0 and router.checks == []
    refused_once(app.state.proxy, "unchecked_body", "azure")


# --- detector validators reading client JSON ------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


PAYLOAD = _b64url(b'{"sub": "x"}')


@pytest.mark.parametrize("depth", [PARSER_DEEP, WALK_DEEP])
def test_a_jwt_segment_nested_too_deep_to_parse_is_judged_by_its_opening(depth: int) -> None:
    import re

    from llm_redact.detection.regex_rules import _jwt_header_ok
    from llm_redact.detection.validators import VALIDATORS

    deep_object = _b64url(b'{"alg": ' + b"[" * depth + b"]" * depth + b"}")
    spaced_object = _b64url(b' \n{"alg": ' + b"[" * depth + b"]" * depth + b"}")
    deep_array = _b64url(b" " + b"[" * depth + b"]" * depth)
    for header, is_jwt in ((deep_object, True), (spaced_object, True), (deep_array, False)):
        match = re.fullmatch(r".+", f"{header}.{PAYLOAD}.signature")
        assert match is not None
        assert _jwt_header_ok(match) is is_jwt
        assert VALIDATORS["jwt"](match) is is_jwt


def test_a_jwt_segment_that_is_not_base64_is_no_jwt() -> None:
    import re

    from llm_redact.detection.validators import VALIDATORS

    # Five characters: one more than a multiple of four, which no base64
    # encoding ends with.
    match = re.fullmatch(r".+", f"abcde.{PAYLOAD}.signature")
    assert match is not None
    assert VALIDATORS["jwt"](match) is False


async def test_a_jwt_whose_header_nests_too_deep_is_redacted_not_a_500() -> None:
    header = _b64url(b'{"alg": ' + b"[" * PARSER_DEEP + b"]" * PARSER_DEEP + b"}")
    token = f"{header}.{PAYLOAD}.signature"
    upstream = _Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    body = {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": token}]}
    response = await _send(app, "POST", "/v1/messages", json.dumps(body).encode(), ANTHROPIC)
    assert response.status_code == 200, response.text[:200]
    (sent,) = upstream.requests
    assert token not in sent.content.decode() and "«JWT_001»" in sent.content.decode()
