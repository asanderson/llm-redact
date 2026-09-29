"""Identity auth signs a multipart upload only when every piece was scanned.

On the multipart routes llm-redact redacts (Files uploads, image edits),
a key-authorized upload forwards whatever the adapter does not rewrite
byte-identically: non-JSON JSONL lines, text or binary files, structural
form fields (``purpose``, ``model``, ``size`` …), a form field that is not
UTF-8, and the bytes outside every part (every other plain form field is
scanned as text, like its JSON twin). Under ``auth = "identity"`` each of
those is either scanned (every plain form field, as UTF-8 text) or the
WHOLE request is refused with a recorded 400 naming its kind — never signed
unscanned. Image/video parts on the media routes stay the documented media
non-goal (like base64 media in a JSON body).
"""

from __future__ import annotations

import logging

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.providers.base import ProviderAdapter
from llm_redact.providers.bedrock import BedrockAdapter
from llm_redact.proxy import create_app
from llm_redact.redactor import UnredactableRequest
from test_upstream_auth import (
    AZURE,
    EMAIL,
    _client,
    _config,
    _identity,
    _install,
    _Upstream,
)

AZURE_FILES = "/openai/files?api-version=1"
AZURE_EDITS = "/openai/deployments/img/images/edits?api-version=1"
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe" + b"\x00" * 16
CLEAN_LINE = b'{"custom_id": "a", "body": {"messages": [{"role": "user", "content": "hi"}]}}'


def _field(name: str, value: bytes) -> tuple[bytes, bytes]:
    return (f'Content-Disposition: form-data; name="{name}"'.encode(), value)


def _file(name: str, filename: str, value: bytes, content_type: str) -> tuple[bytes, bytes]:
    headers = (
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}"
    ).encode()
    return (headers, value)


def _form(*parts: tuple[bytes, bytes], preamble: bytes = b"", epilogue: bytes = b"\r\n") -> bytes:
    out = preamble
    for headers, value in parts:
        out += b"--b\r\n" + headers + b"\r\n\r\n" + value + b"\r\n"
    return out + b"--b--" + epilogue


def _jsonl(*lines: bytes) -> tuple[bytes, bytes]:
    return _file("file", "in.jsonl", b"\n".join(lines), "application/octet-stream")


PURPOSE = _field("purpose", b"batch")

# (route, body): every one holds a piece the adapter would forward unscanned.
REFUSED: dict[str, tuple[str, bytes]] = {
    "non-json-line": (
        AZURE_FILES,
        _form(PURPOSE, _jsonl(CLEAN_LINE, b"notes for " + EMAIL.encode(), b"")),
    ),
    "non-object-line": (AZURE_FILES, _form(PURPOSE, _jsonl(b'["' + EMAIL.encode() + b'"]'))),
    "text-file": (
        AZURE_FILES,
        _form(
            _field("purpose", b"assistants"),
            _file("file", "notes.txt", b"call " + EMAIL.encode(), "text/plain"),
        ),
    ),
    "binary-file": (
        AZURE_FILES,
        _form(
            _field("purpose", b"assistants"),
            _file("file", "doc.pdf", b"%PDF-1.7\n\xe2\xe3\xcf\xd3\n", "application/pdf"),
        ),
    ),
    "non-utf8-field": (AZURE_FILES, _form(_field("purpose", b"\xff\xfe"), _jsonl(CLEAN_LINE))),
    "preamble": (
        AZURE_FILES,
        _form(PURPOSE, _jsonl(CLEAN_LINE), preamble=EMAIL.encode() + b"\r\n"),
    ),
    "epilogue": (
        AZURE_FILES,
        _form(PURPOSE, _jsonl(CLEAN_LINE), epilogue=b"\r\n" + EMAIL.encode()),
    ),
    "non-utf8-prompt": (AZURE_EDITS, _form(_field("prompt", b"\xff" + EMAIL.encode()))),
    "non-utf8-media-field": (
        AZURE_EDITS,
        _form(_field("prompt", b"brighter"), _field("user", b"\xff" + EMAIL.encode())),
    ),
}
KINDS = {
    "non-json-line": "JSONL line",
    "non-object-line": "JSONL line",
    "text-file": "JSONL line",
    "binary-file": "JSONL line",
    "non-utf8-field": "form field",
    "preamble": "preamble or epilogue",
    "epilogue": "preamble or epilogue",
    "non-utf8-prompt": "form field",
    "non-utf8-media-field": "form field",
}


def _headers() -> dict[str, str]:
    return {"content-type": "multipart/form-data; boundary=b"}


@pytest.mark.parametrize("case", sorted(REFUSED))
async def test_unscanned_multipart_piece_refused_under_identity(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, case: str
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    path, body = REFUSED[case]
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(path, content=body, headers=_headers())
    assert response.status_code == 400
    assert upstream.requests == [] and built[0].calls == []
    message = response.json()["error"]["message"]
    assert KINDS[case] in message and "proxy's own identity" in message
    assert EMAIL not in response.text and EMAIL not in caplog.text
    assert app.state.proxy.recent[-1]["status"] == 400


@pytest.mark.parametrize("case", sorted(REFUSED))
async def test_passthrough_still_forwards_the_same_upload_verbatim(case: str) -> None:
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    path, body = REFUSED[case]
    openai_path = "/v1/images/edits" if "edits" in path else "/v1/files"
    async with _client(app) as client:
        response = await client.post(
            openai_path, content=body, headers={**_headers(), "authorization": "Bearer sk-t"}
        )
    assert response.status_code == 200
    assert upstream.requests[0].content == body


async def test_form_fields_scanned_as_text_under_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = _form(
        _field("prompt", b"make it brighter"),
        _field("user", EMAIL.encode()),
        _file("image", "in.png", PNG, "image/png"),
    )
    async with _client(app) as client:
        response = await client.post(AZURE_EDITS, content=body, headers=_headers())
    assert response.status_code == 200
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent and "«EMAIL_".encode() in sent
    # The image itself is media (the documented non-goal): byte-identical.
    assert PNG in sent
    assert built[0].calls[0][3] == sent


async def test_plain_form_fields_scanned_on_passthrough_structural_ones_as_sent() -> None:
    # Key auth scans plain form fields like the JSON strings they mirror (a
    # chat body's `user` is redacted by the walk; so is the form's), and
    # forwards the structural ones — enums, sizes, the model — as sent.
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    structural = [
        _field("model", b"gpt-image-1"),
        _field("size", b"1024x1024"),
        _field("n", b"1"),
        _field("response_format", b"b64_json"),
        _field("quality", b"high"),
    ]
    body = _form(
        _field("prompt", b"brighter"),
        _field("user", EMAIL.encode()),
        *structural,
        _file("image", "in.png", PNG, "image/png"),
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/images/edits", content=body, headers={**_headers(), "authorization": "Bearer t"}
        )
    assert response.status_code == 200
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent and "«EMAIL_001»".encode() in sent
    for headers, value in structural:
        assert headers + b"\r\n\r\n" + value + b"\r\n" in sent
    assert PNG in sent  # the image itself: media, byte-identical


async def test_structural_form_field_forwarded_as_sent_under_key_auth_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A structural field's value is protocol (the `model` a JSON walk skips
    # too): key auth sends it as is — the proxy's own identity signs even
    # it only once scanned.
    body = _form(_field("purpose", EMAIL.encode()), _jsonl(CLEAN_LINE))
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        await client.post(
            "/v1/files", content=body, headers={**_headers(), "authorization": "Bearer t"}
        )
    assert upstream.requests[0].content == body
    _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        await client.post(AZURE_FILES, content=body, headers=_headers())
    assert EMAIL.encode() not in upstream.requests[0].content


async def test_unknown_and_nameless_form_fields_are_scanned_on_passthrough() -> None:
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    nameless = (b"Content-Type: text/plain", b"reach " + EMAIL.encode())
    body = _form(
        _field("purpose", b"batch"),
        _field("note", b"from " + EMAIL.encode()),
        nameless,
        _jsonl(CLEAN_LINE),
    )
    async with _client(app) as client:
        await client.post(
            "/v1/files", content=body, headers={**_headers(), "authorization": "Bearer t"}
        )
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent and sent.count("«EMAIL_001»".encode()) == 2
    assert b'name="purpose"\r\n\r\nbatch\r\n' in sent


async def test_form_user_and_json_user_share_one_placeholder() -> None:
    # The multipart field and its JSON twin now redact alike, into the same
    # vault identity.
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    auth = {"authorization": "Bearer t"}
    async with _client(app) as client:
        await client.post(
            "/v1/chat/completions",
            json={"model": "m", "user": EMAIL, "messages": [{"role": "user", "content": "hi"}]},
            headers=auth,
        )
        await client.post(
            "/v1/images/edits",
            content=_form(_field("prompt", b"x"), _field("user", EMAIL.encode())),
            headers={**_headers(), **auth},
        )
    assert "«EMAIL_001»".encode() in upstream.requests[0].content
    assert "«EMAIL_001»".encode() in upstream.requests[1].content
    assert all(EMAIL.encode() not in request.content for request in upstream.requests)


async def test_scanned_jsonl_with_blank_and_crlf_lines_still_signed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    line = b'{"custom_id": "b", "body": {"q": "' + EMAIL.encode() + b'"}}'
    body = _form(PURPOSE, _jsonl(CLEAN_LINE + b"\r", b"", line + b"\r", b"  ", b""))
    async with _client(app) as client:
        response = await client.post(AZURE_FILES, content=body, headers=_headers())
    assert response.status_code == 200
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent and CLEAN_LINE + b"\r" in sent
    assert len(built[0].calls) == 1


def test_base_multipart_hook_scans_nothing() -> None:
    adapter: ProviderAdapter = BedrockAdapter()
    body = _form(PURPOSE)
    assert adapter.redact_multipart("/p", body, b"b", None, inject_note=False) is None  # type: ignore[arg-type]
    with pytest.raises(UnredactableRequest, match="not one llm-redact redacts"):
        adapter.redact_multipart(
            "/p",
            body,
            b"b",
            None,  # type: ignore[arg-type]
            inject_note=False,
            require_scanned=True,
        )


async def test_azure_key_auth_scans_the_user_field_too() -> None:
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=ProviderConfig(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = _form(
        _field("prompt", b"brighter"),
        _field("user", EMAIL.encode()),
        _field("size", b"1024x1024"),
        _file("image", "in.png", PNG, "image/png"),
    )
    async with _client(app) as client:
        response = await client.post(
            AZURE_EDITS, content=body, headers={**_headers(), "api-key": "azure-key"}
        )
    assert response.status_code == 200
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent and b"1024x1024" in sent and PNG in sent
