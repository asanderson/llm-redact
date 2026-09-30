"""A recognized route forwards only a request body llm-redact scanned.

The rule the proxy's own identity already followed now holds on EVERY
matched route where redaction applies — the client's own key included —
and on every matched route whose request spends a credential the PROXY
holds (its cloud identity, or a routed plan's operator key:
``RoutePlan.proxy_credential``, a plan that does not say counting as the
proxy's). A non-empty body the proxy cannot parse or scan — any Content-
Encoding but identity, bytes after the JSON value, invalid UTF-8 (a
Latin-1 body), a top-level array or scalar, whitespace, multipart outside
the canonical form or on a route that does not scan it, a second
Content-Type — was forwarded VERBATIM: lenient upstreams (Go's JSON
decoder behind Ollama, Express body-parser, Jackson) decode the first JSON
value, substitute bad bytes or inflate gzip, and ran the unredacted
content. It is now a recorded, provider-shaped 400 (415, with
``Accept-Encoding: identity``, for a content coding) before any upstream
contact, naming the body's kind only. ``[providers.NAME] detection =
false`` with the client's own key, and unmatched pass-through, keep
forwarding such bodies verbatim.
"""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any

import httpx
import pytest

from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from llm_redact.config import AuditConfig, Config, ProviderConfig
from llm_redact.proxy import create_app
from test_routing_seam import FakeAudit

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"
OPENAI_OFF = ProviderConfig("https://api.openai.com", detection=False)

_OBJECT = json.dumps({"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]})
_MULTIPART = {"content-type": "multipart/form-data; boundary=b"}

# kind -> (body, extra headers, status): each was forwarded verbatim, unscanned.
REFUSED: dict[str, tuple[bytes, Any, int]] = {
    "text": (f"mail {EMAIL}".encode(), {"content-type": "text/plain"}, 400),
    "json-content-type-non-json": (f"mail {EMAIL}".encode(), {}, 400),
    "invalid-utf8": (b'{"model": "m", "t": "\xc3\x28 ' + EMAIL.encode() + b'"}', {}, 400),
    # Windows PowerShell 5.1: a string -Body without a charset goes out
    # ISO-8859-1, so "Müller" is the single byte 0xFC.
    "latin1": (_OBJECT.replace("mail", "Müller").encode("latin-1"), {}, 400),
    "trailing-bytes": (_OBJECT.encode() + b" xx", {}, 400),
    "trailing-nul": (_OBJECT.encode() + b"\x00", {}, 400),
    "second-json-value": (_OBJECT.encode() + b"\n" + _OBJECT.encode(), {}, 400),
    "array": (json.dumps([{"content": EMAIL}]).encode(), {}, 400),
    "string": (json.dumps(EMAIL).encode(), {}, 400),
    "number": (b"4111111111111111", {}, 400),
    "null": (b"null", {}, 400),
    "whitespace": (b" \r\n\t", {}, 400),
    "gzip": (gzip.compress(_OBJECT.encode()), {"content-encoding": "gzip"}, 415),
    # Plain JSON claiming a coding: the upstream would decode what the proxy
    # never saw.
    "deflate-claim": (_OBJECT.encode(), {"content-encoding": "deflate"}, 415),
    "br-claim": (_OBJECT.encode(), {"content-encoding": "br"}, 415),
    "coding-list": (_OBJECT.encode(), {"content-encoding": "identity, gzip"}, 415),
    "second-coding-header": (
        _OBJECT.encode(),
        [("content-encoding", "identity"), ("content-encoding", "zstd")],
        415,
    ),
    "second-content-type": (
        _OBJECT.encode(),
        [("content-type", "application/json"), ("content-type", "multipart/form-data; boundary=x")],
        400,
    ),
    "non-canonical-multipart": (
        b'--b\nContent-Disposition: form-data; name="prompt"\n\n' + EMAIL.encode() + b"\n--b--\n",
        _MULTIPART,
        400,
    ),
    # Canonical multipart, but on a JSON route: nothing scans it.
    "multipart-off-route": (
        b'--b\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
        + EMAIL.encode()
        + b"\r\n--b--\r\n",
        _MULTIPART,
        400,
    ),
}

# Matched chat routes under the client's own key, one per protocol family.
ROUTES = {
    "openai": "/v1/chat/completions",
    "anthropic": "/v1/messages",
    "ollama": "/api/chat",
    "gemini": "/v1beta/models/m:generateContent",
    "cohere": "/v2/chat",
    "custom": "/custom/lm/v1/chat/completions",
}


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True})


def _config(**overrides: Any) -> Config:
    providers = {
        **Config().providers,
        "custom:lm": ProviderConfig("http://127.0.0.1:1234/v1"),
        **overrides.pop("providers", {}),
    }
    return Config(providers=providers, **overrides)


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _message(payload: dict[str, Any]) -> str:
    error = payload.get("error")
    if isinstance(error, dict):
        return str(error["message"])
    return str(error if error is not None else payload["message"])


def _assert_refused(
    response: httpx.Response, status: int, clause: str, app: Any, caplog: Any
) -> None:
    assert response.status_code == status
    message = _message(response.json())
    assert message.startswith("llm-redact: ") and clause in message
    assert "the request was not forwarded" in message
    if status == 415:
        assert "content-encoded" in message and "send it uncompressed" in message
        assert response.headers["accept-encoding"] == "identity"
    assert EMAIL not in response.text and EMAIL not in caplog.text
    (row,) = app.state.proxy.recent
    assert row["status"] == status and row["detections"] == {}


# --- the client's own key, detection on --------------------------------------------------


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("kind", sorted(REFUSED))
async def test_key_auth_refuses_what_it_cannot_scan(
    route: str, kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    body, headers, status = REFUSED[kind]
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(ROUTES[route], content=body, headers=headers)
    _assert_refused(response, status, "forwards only bodies it has redacted", app, caplog)
    assert upstream.requests == []


@pytest.mark.parametrize("kind", sorted(REFUSED))
async def test_detection_off_with_the_clients_own_key_still_forwards_verbatim(kind: str) -> None:
    # The explicit opt-out: no scanning, no body rule — honest in /status.
    upstream = Upstream()
    app = create_app(
        _config(providers={"openai": OPENAI_OFF}), upstream_transport=httpx.MockTransport(upstream)
    )
    body, headers, _ = REFUSED[kind]
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", content=body, headers=headers)
    assert response.status_code == 200
    [sent] = upstream.requests
    assert sent.content == body


@pytest.mark.parametrize("kind", ["gzip", "trailing-bytes", "invalid-utf8", "array"])
async def test_unmatched_pass_through_still_forwards_verbatim(kind: str) -> None:
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    body, headers, _ = REFUSED[kind]
    async with _client(app) as client:
        response = await client.post("/v1/assistants", content=body, headers=headers)
    assert response.status_code == 200
    [sent] = upstream.requests
    assert sent.content == body


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(_OBJECT.encode(), id="object"),
        # json.loads reads a BOM / UTF-16 from bytes: the object IS walked.
        pytest.param(b"\xef\xbb\xbf" + _OBJECT.encode(), id="utf8-bom"),
        pytest.param(_OBJECT.encode("utf-16"), id="utf16"),
        pytest.param(b" \n" + _OBJECT.encode() + b"\n ", id="surrounding-whitespace"),
    ],
)
async def test_a_json_object_is_still_redacted_and_sent(body: bytes) -> None:
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    headers = [("content-type", "application/json"), ("content-encoding", " Identity ,identity")]
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", content=body, headers=headers)
    assert response.status_code == 200
    [sent] = upstream.requests
    assert EMAIL.encode() not in sent.content and TOKEN.encode() in sent.content


async def test_an_empty_body_is_still_forwarded() -> None:
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(
            "/v1/batches/batch_1/cancel", headers={"content-encoding": "gzip"}
        )
    assert response.status_code == 200 and len(upstream.requests) == 1


def _form(*parts: bytes, preamble: bytes = b"") -> bytes:
    body = preamble
    for part in parts:
        body += b"--b\r\n" + part + b"\r\n"
    return body + b"--b--\r\n"


def _file(content: bytes, *, name: str = "in.jsonl", headers: bytes = b"") -> bytes:
    return (
        b'Content-Disposition: form-data; name="file"; filename="'
        + name.encode()
        + b'"\r\nContent-Type: application/octet-stream\r\n'
        + headers
        + b"\r\n"
        + content
    )


_LINE = json.dumps({"messages": [{"role": "user", "content": EMAIL}]}).encode()
_PURPOSE = b'Content-Disposition: form-data; name="purpose"\r\n\r\nbatch'


@pytest.mark.parametrize("path", ["/v1/files", "/custom/lm/v1/files", "/openai/files"])
async def test_a_jsonl_upload_is_still_redacted_and_sent(path: str) -> None:
    upstream = Upstream()
    azure = ProviderConfig("https://res.openai.azure.com")
    app = create_app(
        _config(providers={"azure": azure}), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        response = await client.post(
            path, content=_form(_PURPOSE, _file(_LINE)), headers=_MULTIPART
        )
    assert response.status_code == 200
    [sent] = upstream.requests
    assert EMAIL.encode() not in sent.content and TOKEN.encode() in sent.content


async def test_an_image_edit_keeps_its_media_parts() -> None:
    # The image part is media (the documented non-goal); the prompt is scanned.
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    image = b'Content-Disposition: form-data; name="image"; filename="a.png"\r\n\r\n\x89PNG\x00\xff'
    prompt = b'Content-Disposition: form-data; name="prompt"\r\n\r\nto ' + EMAIL.encode()
    async with _client(app) as client:
        response = await client.post(
            "/v1/images/edits", content=_form(image, prompt), headers=_MULTIPART
        )
    assert response.status_code == 200
    [sent] = upstream.requests
    assert b"\x89PNG\x00\xff" in sent.content and EMAIL.encode() not in sent.content


# Inside an upload the same rule holds part by part.
_PDF_UPLOAD = _form(_PURPOSE, _file(b"%PDF-1.7\n\xe2\xe3\xcf\xd3 " + EMAIL.encode(), name="a.pdf"))
_TEXT_UPLOAD = _form(_PURPOSE, _file(b"notes: " + EMAIL.encode(), name="notes.txt"))

UNSCANNED_UPLOADS: dict[str, tuple[bytes, str]] = {
    "field-not-utf8": (
        _form(b'Content-Disposition: form-data; name="user"\r\n\r\n\xfc ' + EMAIL.encode()),
        "a multipart form field is not UTF-8 text llm-redact can redact",
    ),
    "preamble": (
        _form(_PURPOSE, _file(_LINE), preamble=EMAIL.encode() + b"\r\n"),
        "the multipart body carries a preamble or epilogue llm-redact does not redact",
    ),
    "transfer-encoded": (
        _form(
            _PURPOSE,
            _file(b"eyJtZXNzYWdlcyI6W119", headers=b"Content-Transfer-Encoding: base64\r\n"),
        ),
        "a multipart part declares a Content-Transfer-Encoding llm-redact does not decode",
    ),
}


# A multipart content type with more than one reading: a reader taking the
# LAST boundary (or splitting quoted parameters naively) parses the body
# with "c" — here the form field "user" holds a whole "c"-delimited body
# whose file part is a batch input file.
_INNER = b"\r\n--c\r\n" + _file(_LINE) + b"\r\n--c\r\n" + _PURPOSE + b"\r\n--c--"
_TWO_BOUNDARIES = _form(_PURPOSE, b'Content-Disposition: form-data; name="user"\r\n\r\nx' + _INNER)
TWO_READINGS = [
    pytest.param("multipart/form-data; boundary=b; boundary=c", id="repeated-boundary"),
    pytest.param('multipart/form-data; x="; boundary=c"; boundary=b', id="naive-split"),
]


@pytest.mark.parametrize("content_type", TWO_READINGS)
async def test_key_auth_refuses_a_multipart_content_type_with_two_readings(
    content_type: str, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=_TWO_BOUNDARIES, headers={"content-type": content_type}
        )
    _assert_refused(response, 400, "forwards only bodies it has redacted", app, caplog)
    assert upstream.requests == []


@pytest.mark.parametrize("kind", sorted(UNSCANNED_UPLOADS))
async def test_key_auth_refuses_an_upload_part_it_cannot_scan(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    body, why = UNSCANNED_UPLOADS[kind]
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post("/v1/files", content=body, headers=_MULTIPART)
    _assert_refused(response, 400, "forwards only bodies it has redacted", app, caplog)
    assert why in _message(response.json())
    assert upstream.requests == []


async def test_key_auth_redacts_a_text_file_and_forwards_a_binary_one() -> None:
    # Decided by content: text is redacted; a binary file cannot be, and
    # with the client's own key it goes out unscanned (binary_uploads =
    # "forward", the default — counted in /status).
    upstream = Upstream()
    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        text = await client.post("/v1/files", content=_TEXT_UPLOAD, headers=_MULTIPART)
        pdf = await client.post("/v1/files", content=_PDF_UPLOAD, headers=_MULTIPART)
    assert text.status_code == 200 and pdf.status_code == 200
    assert EMAIL.encode() not in upstream.requests[0].content
    assert TOKEN.encode() in upstream.requests[0].content
    assert upstream.requests[1].content == _PDF_UPLOAD
    assert app.state.proxy.unscanned_uploads == {"openai": 1}


async def test_detection_off_forwards_an_unscanned_upload_verbatim() -> None:
    upstream = Upstream()
    app = create_app(
        _config(providers={"openai": OPENAI_OFF}), upstream_transport=httpx.MockTransport(upstream)
    )
    body = _PDF_UPLOAD
    async with _client(app) as client:
        response = await client.post("/v1/files", content=body, headers=_MULTIPART)
    assert response.status_code == 200
    assert upstream.requests[0].content == body


# --- a routed plan spending the proxy's credential ------------------------------------------


def _routed(
    monkeypatch: pytest.MonkeyPatch, *, proxy_credential: bool | None, **config: Any
) -> tuple[Any, FakeRouter, Upstream, FakeAudit]:
    kwargs = {} if proxy_credential is None else {"proxy_credential": proxy_credential}
    router = FakeRouter(
        {"r": [Hop("a", "http://a.example/v1/chat/completions"), Stop()]},
        plan_kwargs={"r": kwargs},
    )
    audit = FakeAudit()
    install(monkeypatch, router, audit=audit)
    upstream = Upstream()
    app = create_app(
        routed_config(audit=AuditConfig(enabled=True, required=True), **config),
        upstream_transport=httpx.MockTransport(upstream),
    )
    return app, router, upstream, audit


@pytest.mark.parametrize("detection", [True, False], ids=["detection-on", "detection-off"])
@pytest.mark.parametrize("proxy_credential", [None, True], ids=["plan-silent", "operator-key"])
@pytest.mark.parametrize("kind", sorted(REFUSED))
async def test_a_routed_plan_spending_the_proxys_credential_never_sends_it(
    kind: str,
    proxy_credential: bool | None,
    detection: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    providers = {**Config().providers, **({} if detection else {"openai": OPENAI_OFF})}
    app, router, upstream, audit = _routed(
        monkeypatch, proxy_credential=proxy_credential, providers=providers
    )
    body, headers, status = REFUSED[kind]
    request_headers = httpx.Headers(headers)
    request_headers[ROUTE_HEADER] = "r"
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", content=body, headers=request_headers)
    # detection = false turns redaction off, never this rule: the proxy's
    # credential carries only a body the proxy read.
    _assert_refused(response, status, "the proxy's own provider credential", app, caplog)
    assert upstream.requests == []
    (plan,) = router.plans
    assert plan.begun == [] and audit.begun == []  # no hop, no audit START row


@pytest.mark.parametrize("kind", ["gzip", "trailing-bytes", "array"])
async def test_a_plan_on_the_clients_own_key_falls_to_the_key_auth_rule(
    kind: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    app, router, upstream, audit = _routed(monkeypatch, proxy_credential=False)
    body, headers, status = REFUSED[kind]
    request_headers = httpx.Headers(headers)
    request_headers[ROUTE_HEADER] = "r"
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", content=body, headers=request_headers)
    _assert_refused(response, status, "forwards only bodies it has redacted", app, caplog)
    assert upstream.requests == [] and router.plans[0].begun == []


async def test_a_plan_on_the_clients_own_key_with_detection_off_forwards_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers = {**Config().providers, "openai": OPENAI_OFF}
    app, router, upstream, _ = _routed(monkeypatch, proxy_credential=False, providers=providers)
    body, headers, _ = REFUSED["trailing-bytes"]
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions", content=body, headers={**headers, ROUTE_HEADER: "r"}
        )
    assert response.status_code == 200
    assert router.plans[0].begun[0][0] == body  # the plan carries the bytes as sent


async def test_a_routed_json_object_is_still_redacted_and_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, router, upstream, audit = _routed(monkeypatch, proxy_credential=True)
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions", content=_OBJECT.encode(), headers={ROUTE_HEADER: "r"}
        )
    assert response.status_code == 200
    [sent] = upstream.requests
    assert EMAIL.encode() not in sent.content and len(audit.begun) == 1


@pytest.mark.parametrize("proxy_credential", [None, True], ids=["plan-silent", "operator-key"])
async def test_a_routed_binary_upload_is_never_sent_with_the_proxys_credential(
    proxy_credential: bool | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app, router, upstream, audit = _routed(monkeypatch, proxy_credential=proxy_credential)
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=_PDF_UPLOAD, headers={**_MULTIPART, ROUTE_HEADER: "r"}
        )
    _assert_refused(response, 400, "the proxy's own provider credential", app, caplog)
    assert "an uploaded file is binary" in _message(response.json())
    assert upstream.requests == [] and router.plans[0].begun == [] and audit.begun == []
    assert app.state.proxy.unscanned_uploads == {}


async def test_a_routed_upload_on_the_proxys_credential_still_sends_a_redacted_text_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, router, upstream, audit = _routed(monkeypatch, proxy_credential=True)
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=_TEXT_UPLOAD, headers={**_MULTIPART, ROUTE_HEADER: "r"}
        )
    assert response.status_code == 200
    sent = router.plans[0].begun[0][0]
    assert EMAIL.encode() not in sent and TOKEN.encode() in sent


async def test_a_routed_binary_upload_on_the_clients_own_key_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, router, upstream, audit = _routed(monkeypatch, proxy_credential=False)
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=_PDF_UPLOAD, headers={**_MULTIPART, ROUTE_HEADER: "r"}
        )
    assert response.status_code == 200
    assert router.plans[0].begun[0][0] == _PDF_UPLOAD
    assert app.state.proxy.unscanned_uploads == {"openai": 1}


# --- the proxy's own cloud identity -----------------------------------------------------


@pytest.mark.parametrize("detection", [True, False], ids=["detection-on", "detection-off"])
@pytest.mark.parametrize("kind", sorted(REFUSED))
async def test_identity_refuses_the_same_catalogue(
    kind: str, detection: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from test_upstream_auth import AZURE, AZURE_PATH, _identity, _install

    _, built = _install(monkeypatch)
    upstream = Upstream()
    app = create_app(
        _config(providers={"azure": _identity(AZURE, detection=detection)}),
        upstream_transport=httpx.MockTransport(upstream),
    )
    body, headers, status = REFUSED[kind]
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(AZURE_PATH + "?api-version=1", content=body, headers=headers)
    _assert_refused(response, status, "the proxy's own identity", app, caplog)
    assert upstream.requests == [] and built[0].calls == []  # never signed
