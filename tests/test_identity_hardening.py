"""Regression suite for the identity-auth / request-target review findings.

Each block names the finding it pins (reviewer probes reused as tests):
dot segments, credential query/header variants, the realtime relay's base
path, the Azure batch LIST, Bedrock CountTokens' base64 invoke body, the
realtime subprotocol allowlist, and the smaller hardening items.
"""

import base64
import json
import logging
import socket
import threading
import time
from typing import Any
from urllib.parse import unquote

import httpx
import pytest
import uvicorn
import websockets

from llm_redact.config import Config, ConfigError, ProviderConfig, parse_config
from llm_redact.log import setup_logging
from llm_redact.providers.bedrock import _APPLY_GUARDRAIL, _ASYNC_INVOKE_ITEM, _COUNT_TOKENS, _ROUTE
from llm_redact.proxy import (
    _same_upstream,
    _strip_credential_query,
    create_app,
    has_dot_segment,
    strip_client_credentials,
)
from llm_redact.realtime import identity_subprotocols
from test_realtime_identity import AZURE_GA, VERTEX_V1
from test_realtime_identity import FakeAuth as WsAuth
from test_realtime_identity import _install as ws_install
from test_realtime_relay import FakeUpstream, _proxy
from test_realtime_relay import _config as ws_config
from test_upstream_auth import (
    AZURE,
    BEDROCK,
    VERTEX,
    _client,
    _config,
    _identity,
    _install,
    _Upstream,
)

EMAIL = "jane.doe@corp.example"


async def _raw(app: Any, method: str, raw_path: str, body: bytes = b"") -> int:
    """What uvicorn/h11 hands the app for a raw request line: the path is
    percent-decoded, the raw path untouched (no normalization)."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": unquote(raw_path),
        "raw_path": raw_path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1:8787"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8787),
    }
    sent: list[dict[str, Any]] = []
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive() -> dict[str, Any]:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async with app.router.lifespan_context(app):
        await app(scope, receive, send)
    return int(next(m for m in sent if m["type"] == "http.response.start")["status"])


# --- dot segments ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "base", "method", "raw"),
    [
        ("bedrock", BEDROCK, "GET", "/async-invoke/../../some/unrecognized/op"),
        ("bedrock", BEDROCK, "GET", "/async-invoke/%2e%2e/%2E%2E/some/op"),
        ("bedrock", BEDROCK, "POST", "/model/x/../../any/invoke"),
        ("bedrock", BEDROCK, "POST", "/model/..%2F..%2Fany/invoke"),
        ("vertex", VERTEX, "DELETE", "/v1/projects/p/locations/l/cachedContents/.."),
        ("vertex", VERTEX, "GET", "/v1/projects/p/locations/l/cachedContents/."),
        ("azure", AZURE, "DELETE", "/openai/v1/conversations/../items/x"),
        ("azure", AZURE, "DELETE", "/openai/v1/conversations/..%5Citems"),
        ("azure", AZURE, "GET", "/openai/files/.%2E"),
    ],
)
async def test_dot_segments_refused_before_any_upstream_contact(
    monkeypatch: pytest.MonkeyPatch, provider: str, base: str, method: str, raw: str
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: _identity(base)}), upstream_transport=httpx.MockTransport(upstream)
    )
    assert await _raw(app, method, raw) == 400
    assert [call for auth in built for call in auth.calls] == []
    assert upstream.requests == []


async def test_dot_segments_refused_on_passthrough_providers_too() -> None:
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": EMAIL}]}).encode()
    assert await _raw(app, "POST", "/v1/chat/../chat/completions", body) == 400
    assert await _raw(app, "GET", "/v1/%2e/models") == 400
    assert upstream.requests == []


def test_dot_segments_refused_over_a_real_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    # h11 hands the raw target through unnormalized; the refusal holds there.
    _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(bedrock=_identity(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        assert time.time() < deadline
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        with socket.create_connection(("127.0.0.1", port)) as conn:
            conn.sendall(
                b"GET /async-invoke/../../any/path/on/the/host HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\nConnection: close\r\n\r\n"
            )
            data = b""
            while chunk := conn.recv(4096):
                data += chunk
    finally:
        server.should_exit = True
        thread.join(10)
    assert data.split(b"\r\n")[0] == b"HTTP/1.1 400 Bad Request"
    assert upstream.requests == []


@pytest.mark.parametrize(
    ("path", "dotted"),
    [
        ("/a/../b", True),
        ("/a/./b", True),
        ("/a/..", True),
        ("/..", True),
        ("/a/%2E%2e/b", True),
        ("/a/.%2e", True),
        ("/a\\..\\b", True),
        ("/a/%5C../b", True),
        ("/a/.../b", False),
        ("/a/.b/c", False),
        ("/a/b..", False),
        ("/v1/models/gemini-1.5-pro:generateContent", False),
        ("/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A1%3Aprofile%2Fus.claude.v1%3A0/invoke", False),
    ],
)
async def test_has_dot_segment(path: str, dotted: bool) -> None:
    assert has_dot_segment({"raw_path": path.encode(), "path": unquote(path)}) is dotted
    assert has_dot_segment({"path": unquote(path)}) is dotted  # servers without raw_path


async def test_bedrock_matchers_reject_dot_segments_but_keep_arns() -> None:
    arn = "arn:aws:bedrock:us-east-1:1:inference-profile/us.anthropic.claude-v1:0"
    assert _ROUTE.match(f"/model/{arn}/invoke")
    assert _COUNT_TOKENS.match(f"/model/{arn}/count-tokens")
    assert _APPLY_GUARDRAIL.match("/guardrail/arn:aws:bedrock:g/x/version/1/apply")
    assert _ASYNC_INVOKE_ITEM.match("/async-invoke/arn:aws:bedrock:us-east-1:1:async-invoke/j")
    for bad in (
        "/model/../invoke",
        "/model/x/../invoke",
        "/model/./invoke",
        "/model//invoke",
    ):
        assert not _ROUTE.match(bad), bad
    assert not _ASYNC_INVOKE_ITEM.match("/async-invoke/../x")
    assert not _APPLY_GUARDRAIL.match("/guardrail/g/version/../apply")


async def test_same_upstream_exact_path() -> None:
    base = "https://gw.example/api"
    assert _same_upstream(f"{base}/v1/x?y=1", base, exact_path="/api/v1/x")
    assert not _same_upstream(f"{base}/v1/x", base, exact_path="/api/v1/y")
    # httpx re-encodes what it would send differently: refused under identity.
    assert not _same_upstream(f"{base}/v1/a{{b}}", base, exact_path="/api/v1/a{b}")


async def test_identity_refuses_a_path_httpx_would_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(bedrock=_identity(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = json.dumps({"messages": [{"role": "user", "content": [{"text": "hi"}]}]}).encode()
    assert await _raw(app, "POST", "/model/a{b}/converse", body) == 400
    assert built[0].calls == [] and upstream.requests == []


# --- credential query / header variants --------------------------------------------


@pytest.mark.parametrize(
    "param",
    [
        "$key=k",
        "%24key=k",
        "KEY=k",
        "$userProject=victim",
        "%24userProject=victim",
        "userproject=victim",
        "quotaUser=q",
        "$quotaUser=q",
        "subscription-key=apim",
        "Subscription-Key=apim",
        "password=pw",
        "passwd=pw",
        "oauth_token=t",
    ],
)
async def test_credential_query_variants_stripped(param: str) -> None:
    assert _strip_credential_query(f"api-version=1&{param}&model=m") == "api-version=1&model=m"


async def test_credential_header_variants_stripped() -> None:
    _, headers = strip_client_credentials(
        f"{VERTEX}/v1/x",
        [
            ("password", "pw"),
            ("Passwd", "pw"),
            ("x-goog-quota-user", "q"),
            ("x-request-id", "keep"),
        ],
    )
    assert headers == [("x-request-id", "keep")]


async def test_http_identity_strips_the_query_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(vertex=_identity(VERTEX)), upstream_transport=httpx.MockTransport(upstream)
    )
    path = "/v1/projects/p/locations/l/publishers/google/models/g:generateContent"
    query = "$key=K&%24userProject=victim&quotaUser=q&alt=sse"
    async with _client(app) as client:
        response = await client.post(
            f"{path}?{query}",
            json={"contents": [{"parts": [{"text": "hi"}]}]},
            headers={"password": "pw", "x-goog-quota-user": "q"},
        )
    assert response.status_code == 200
    sent = upstream.requests[0]
    assert sent.url.query == b"alt=sse"
    assert built[0].calls[0][1].endswith("?alt=sse")
    assert "password" not in sent.headers and "x-goog-quota-user" not in sent.headers


async def test_ws_identity_strips_query_headers_and_foreign_subprotocols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = WsAuth()
    ws_install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(ws_config(fake.port, "vertex", auth="identity")) as host:
            async with websockets.connect(
                f"ws://{host}{VERTEX_V1}?$key=CLIENT_API_KEY&%24userProject=victim&quotaUser=q",
                subprotocols=[
                    websockets.Subprotocol("realtime"),
                    websockets.Subprotocol("eyJhbGciOiJIUzI1NiJ9.CLIENTJWT.sig"),
                ],
                additional_headers={"password": "CLIENT_PW", "x-goog-quota-user": "q"},
            ) as client:
                await client.send(json.dumps({"realtimeInput": {"text": "hi"}}).encode())
                await client.recv()
    assert fake.paths == [VERTEX_V1]
    assert auth.calls[0][1] == f"http://127.0.0.1:{fake.port}{VERTEX_V1}"
    upstream = fake.headers[0]
    assert upstream["sec-websocket-protocol"] == "realtime"
    assert "password" not in upstream and "x-goog-quota-user" not in upstream


async def test_ws_identity_strips_apim_subscription_key(monkeypatch: pytest.MonkeyPatch) -> None:
    auth = WsAuth()
    ws_install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(ws_config(fake.port, "azure", auth="identity")) as host:
            async with websockets.connect(
                f"ws://{host}{AZURE_GA}?model=d&subscription-key=CLIENT_APIM_KEY"
            ) as client:
                await client.send(json.dumps({"type": "session.update", "session": {}}))
                await client.recv()
    assert fake.paths == [f"{AZURE_GA}?model=d"]


# --- the realtime relay keeps the upstream base path -------------------------------


@pytest.mark.parametrize("identity", [True, False])
async def test_ws_relay_keeps_the_base_path(
    monkeypatch: pytest.MonkeyPatch, identity: bool
) -> None:
    auth = WsAuth()
    ws_install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        base = f"http://127.0.0.1:{fake.port}/my-apim-api/"
        provider = ProviderConfig(base, auth="identity" if identity else "passthrough")
        with _proxy(Config(providers={**Config().providers, "azure": provider})) as host:
            async with websockets.connect(f"ws://{host}{AZURE_GA}?model=d") as client:
                await client.send(json.dumps({"type": "session.update", "session": {}}))
                await client.recv()
    assert fake.paths == [f"/my-apim-api{AZURE_GA}?model=d"]
    if identity:
        assert auth.calls[0][1] == f"http://127.0.0.1:{fake.port}/my-apim-api{AZURE_GA}?model=d"
        assert fake.headers[0]["authorization"] == "Bearer proxy-identity-token"
    else:
        assert auth.calls == []


async def test_same_upstream_requires_the_base_path() -> None:
    assert _same_upstream("http://gw/api/openai/v1/realtime", "http://gw/api")
    assert _same_upstream("http://gw/api", "http://gw/api/")
    assert not _same_upstream("http://gw/other/openai/v1/realtime", "http://gw/api")
    assert not _same_upstream("http://gw/apix/openai", "http://gw/api")


# --- subprotocol allowlist ----------------------------------------------------------


@pytest.mark.parametrize(
    ("offered", "kept"),
    [
        (["realtime", "openai-beta.realtime-v1"], ["realtime", "openai-beta.realtime-v1"]),
        (["realtime", "openai-insecure-api-key.sk-1", "openai-beta.realtime-v1"], None),
        (["realtime", "eyJhbGciOi.CLIENTJWT.sig"], ["realtime"]),
        (["openai-organization.org-1", "openai-project.proj-1", "realtime"], ["realtime"]),
        (["bearer", "realtime", "realtime"], ["realtime"]),  # the value after a marker
        (["Authorization", "openai-beta.x", "realtime"], ["realtime"]),
        (["openai-insecure-api-key.", "realtime"], []),  # a bare marker spelled with a dot
        (["Realtime"], []),  # exact match only
        ([], []),
    ],
)
async def test_identity_subprotocol_allowlist(offered: list[str], kept: list[str] | None) -> None:
    expected = kept if kept is not None else ["realtime", "openai-beta.realtime-v1"]
    assert identity_subprotocols(offered) == expected


# --- Azure batch list is never rehydrated ------------------------------------------


async def test_azure_batch_list_passes_through_but_a_batch_is_restored() -> None:
    upstream = _Upstream()
    app = create_app(
        _config(azure=ProviderConfig(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        created = await client.post(
            "/openai/batches?api-version=2024-10-21",
            json={"input_file_id": "file-1", "metadata": {"owner": EMAIL}},
        )
        assert created.status_code == 200
        token = json.loads(upstream.requests[0].content)["metadata"]["owner"]
        assert token.startswith("«EMAIL_")
        batch = json.dumps({"id": "b1", "metadata": {"owner": token}}).encode()
        upstream.content = json.dumps({"object": "list", "data": [json.loads(batch)]}).encode()
        listing = await client.get("/openai/batches?api-version=2024-10-21")
        upstream.content = batch
        single = await client.get("/openai/batches/b1?api-version=2024-10-21")
    assert listing.json()["data"][0]["metadata"]["owner"] == token  # not restored
    assert single.json()["metadata"]["owner"] == EMAIL  # the batch's own read: restored


# --- Bedrock CountTokens: the base64 invoke body is redacted -----------------------


def _invoke_body(text: str) -> dict[str, Any]:
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": text}],
    }


def _count_tokens(inner: dict[str, Any] | str) -> dict[str, Any]:
    raw = inner if isinstance(inner, str) else base64.b64encode(json.dumps(inner).encode()).decode()
    return {"input": {"invokeModel": {"body": raw}}}


COUNT_PATH = "/model/anthropic.claude-3-5-sonnet/count-tokens"


@pytest.mark.parametrize("identity", [True, False])
async def test_count_tokens_invoke_body_is_redacted(
    monkeypatch: pytest.MonkeyPatch, identity: bool
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b'{"inputTokens": 12}')
    provider = _identity(BEDROCK) if identity else ProviderConfig(BEDROCK)
    app = create_app(_config(bedrock=provider), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(COUNT_PATH, json=_count_tokens(_invoke_body(f"email {EMAIL}")))
    assert response.status_code == 200 and response.json() == {"inputTokens": 12}
    sent = json.loads(upstream.requests[0].content)
    decoded = json.loads(base64.b64decode(sent["input"]["invokeModel"]["body"]))
    text = decoded["messages"][0]["content"]
    assert EMAIL not in text and text.startswith("email «EMAIL_")
    assert decoded["anthropic_version"] == "bedrock-2023-05-31"  # the rest untouched
    if identity:
        assert built[0].calls[0][3] == upstream.requests[0].content  # signed what was sent


async def test_count_tokens_without_a_secret_is_forwarded_byte_identical() -> None:
    upstream = _Upstream(b'{"inputTokens": 3}')
    app = create_app(
        _config(bedrock=ProviderConfig(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = json.dumps(_count_tokens(_invoke_body("hello"))).encode()
    async with _client(app) as client:
        response = await client.post(
            COUNT_PATH, content=body, headers={"content-type": "application/json"}
        )
    assert response.status_code == 200
    assert upstream.requests[0].content == body


@pytest.mark.parametrize(
    "blob",
    [
        "not base64!!",
        base64.b64encode(b"\xff\xfe not utf-8").decode(),
        base64.b64encode(b"plain text, not json").decode(),
    ],
)
async def test_undecodable_count_tokens_body_fails_closed(
    monkeypatch: pytest.MonkeyPatch, blob: str
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(bedrock=_identity(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        response = await client.post(COUNT_PATH, json=_count_tokens(blob))
    assert response.status_code == 400
    assert "invokeModel.body" in response.json()["message"]  # Bedrock's error shape
    assert blob not in response.text
    assert upstream.requests == [] and built[0].calls == []
    assert app.state.proxy.recent[-1]["status"] == 400


async def test_count_tokens_block_mode_inside_the_blob() -> None:
    upstream = _Upstream(b"{}")
    config = parse_config(
        {
            "providers": {"bedrock": {"upstream_base_url": BEDROCK}},
            "detection": {"modes": {"email": "block"}},
        },
        "<t>",
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(COUNT_PATH, json=_count_tokens(_invoke_body(EMAIL)))
    assert response.status_code == 400 and "blocked" in response.text
    assert upstream.requests == []


async def test_count_tokens_converse_form_unchanged() -> None:
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(bedrock=ProviderConfig(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = {"input": {"converse": {"messages": [{"role": "user", "content": [{"text": EMAIL}]}]}}}
    async with _client(app) as client:
        assert (await client.post(COUNT_PATH, json=body)).status_code == 200
    assert EMAIL.encode() not in upstream.requests[0].content


# --- the smaller hardening items ----------------------------------------------------


async def test_websockets_logger_pinned_at_warning() -> None:
    logger = logging.getLogger("websockets")
    previous = logger.level
    try:
        logger.setLevel(logging.DEBUG)
        setup_logging("text")
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(previous)


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://bedrock-runtime.us-east-1.amazonaws.com", True),
        ("http://127.0.0.1:9000", True),
        ("http://localhost:9000/base", True),
        ("http://[::1]:9000", True),
        ("http://bedrock-runtime.us-east-1.amazonaws.com", False),
        ("http://10.0.0.5:8080", False),
        ("ws://gw.example", False),
    ],
)
async def test_identity_requires_https_off_loopback(url: str, ok: bool) -> None:
    raw = {"providers": {"bedrock": {"upstream_base_url": url, "auth": "identity"}}}
    if ok:
        assert parse_config(raw, "<t>").providers["bedrock"].upstream_base_url == url.rstrip("/")
    else:
        with pytest.raises(ConfigError, match="must be https"):
            parse_config(raw, "<t>")
    # Passthrough keeps accepting any URL (the client's own credential).
    raw["providers"]["bedrock"]["auth"] = "passthrough"
    parse_config(raw, "<t>")


async def test_identity_https_rule_also_holds_for_configs_built_in_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _install(monkeypatch)
    with pytest.raises(ConfigError, match="must be https"):
        create_app(_config(azure=_identity("http://res.openai.azure.com")))
    assert built == []


async def test_ws_identity_refusal_of_an_unauthorizable_path_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = WsAuth()
    ws_install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(ws_config(fake.port, "azure", auth="identity")) as host:
            async with websockets.connect(f"ws://{host}/openai/realtime/extra") as client:
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await client.recv()
            assert closed.value.rcvd is not None and closed.value.rcvd.code == 1011
            async with httpx.AsyncClient(base_url=f"http://{host}") as http:
                rows = (await http.get("/__llm-redact/recent")).json()["entries"]
    assert fake.paths == [] and auth.calls == []
    assert any(r["method"] == "WS" and r["status"] == 403 for r in rows), rows


def _multipart(boundary: str, body: str) -> bytes:
    return body.replace("\n", "\r\n").replace("BOUNDARY", boundary).encode()


async def test_non_canonical_multipart_refused_under_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    # LF-only line endings: outside the codec's canonical CRLF grammar.
    body = b'--b\nContent-Disposition: form-data; name="prompt"\n\n' + EMAIL.encode() + b"\n--b--\n"
    async with _client(app) as client:
        response = await client.post(
            "/openai/deployments/img/images/edits?api-version=1",
            content=body,
            headers={"content-type": "multipart/form-data; boundary=b"},
        )
    assert response.status_code == 400
    assert upstream.requests == [] and built[0].calls == []


async def test_canonical_multipart_still_redacted_under_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = (
        b"--b\r\n"
        b'Content-Disposition: form-data; name="prompt"\r\n\r\n'
        + f"edit for {EMAIL}".encode()
        + b"\r\n--b--\r\n"
    )
    async with _client(app) as client:
        response = await client.post(
            "/openai/deployments/img/images/edits?api-version=1",
            content=body,
            headers={"content-type": "multipart/form-data; boundary=b"},
        )
    assert response.status_code == 200
    assert EMAIL.encode() not in upstream.requests[0].content


@pytest.mark.parametrize(
    ("path", "headers", "host"),
    [
        ("/v1/models", {"x-goog-api-key": "g"}, "generativelanguage.googleapis.com"),
        ("/v1/models/gemini-2.5-pro?key=g", {}, "generativelanguage.googleapis.com"),
        ("/v1/models?%24key=g", {}, "generativelanguage.googleapis.com"),
        ("/v1/models", {"authorization": "Bearer sk-1"}, "api.openai.com"),
        ("/v1/models?keyring=x", {}, "api.openai.com"),
    ],
)
async def test_google_authenticated_v1_passthrough_goes_to_gemini(
    path: str, headers: dict[str, str], host: str
) -> None:
    upstream = _Upstream(b'{"models": []}')
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        assert (await client.get(path, headers=headers)).status_code == 200
    assert upstream.requests[0].url.host == host
