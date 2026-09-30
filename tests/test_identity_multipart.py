"""A multipart upload is forwarded only when every piece was scanned.

On the multipart routes llm-redact redacts (Files uploads, image edits),
every piece of an upload is scanned — a file part by its CONTENT (JSONL
line by line, any other text file as one text), a form field as UTF-8 text
— or the WHOLE request is refused with a recorded 400 naming its kind:
under ``auth = "identity"`` (never signed unscanned) and, since the
scanned-body rule (tests/test_scanned_body.py), under the client's own key
wherever redaction applies too. A BINARY file part is the one piece that
cannot be scanned: under the proxy's identity it refuses the upload; with
the client's own key it is forwarded unscanned ([detection]
binary_uploads = "forward", the default — counted) or refused ("refuse").
Every plain form field is scanned as UTF-8 text, structural ones
(``purpose``, ``model``, ``size`` …) included. ``detection = false`` with
the client's own key stays the explicit opt-out: the upload is forwarded
as sent. Image/video parts on the media routes stay the documented media
non-goal (like base64 media in a JSON body).
"""

from __future__ import annotations

import base64
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

# Bare-LF pseudo-headers inside a part without a CRLF CRLF separator.
_QP_PART = (
    b'Content-Disposition: form-data; name="file"; filename="notes.txt"\n'
    b"Content-Type: text/plain\nContent-Transfer-Encoding: quoted-printable\n\n"
    b"contact " + EMAIL.replace("@", "=40").encode()
)
_BASE64_PART = (
    b'Content-Disposition: form-data; name="file"; filename="doc.pdf"\n'
    b"Content-Type: application/pdf\nContent-Transfer-Encoding: base64\n\n"
    + base64.b64encode(b"%PDF-1.7\n\xe2\xe3\xcf\xd3\n(" + EMAIL.encode() + b")\n%%EOF\n")
)
_ASSISTANTS = b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nassistants\r\n'
HEADERLESS_QP = _ASSISTANTS + b"--b\r\n" + _QP_PART + b"\r\n--b--\r\n"
HEADERLESS_BASE64 = _ASSISTANTS + b"--b\r\n" + _BASE64_PART + b"\r\n--b--\r\n"

# (route, body): every one holds a piece the adapter would forward unscanned.
REFUSED: dict[str, tuple[str, bytes]] = {
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
    "too-deep-jsonl-line": (
        AZURE_FILES,
        _form(PURPOSE, _jsonl(CLEAN_LINE, b'{"a": ' * 200 + b"1" + b"}" * 200)),
    ),
    # A reader accepting a bare LF as a line break ends this part's header
    # block after its Content-Type: the rest is a JSONL line there, never
    # scanned here (it is a header value to the canonical grammar).
    "bare-lf-in-a-header": (
        AZURE_FILES,
        _form(
            PURPOSE,
            (
                b'Content-Disposition: form-data; name="file"; filename="in.jsonl"\r\n'
                b"Content-Type: application/jsonl\n\n"
                b'{"custom_id": "a", "body": {"input": "' + EMAIL.encode() + b'"}}',
                b"",
            ),
        ),
    ),
    # A part with NO header/body separator was once read as a plain field
    # with no header at all; a reader accepting a bare LF as a line break
    # finds a file part there, decoding quoted-printable (or base64: a
    # binary file read as text) content nobody scanned.
    "headerless-quoted-printable": (AZURE_FILES, HEADERLESS_QP),
    "headerless-base64-pdf": (AZURE_FILES, HEADERLESS_BASE64),
    "headerless-media-route": (
        AZURE_EDITS,
        _form(_field("prompt", b"brighter")).replace(
            b"--b--", b"--b\r\n" + _QP_PART + b"\r\n--b--"
        ),
    ),
}
KINDS = {
    "headerless-quoted-printable": "part header",
    "headerless-base64-pdf": "part header",
    "headerless-media-route": "part header",
    "non-utf8-field": "form field",
    "preamble": "preamble or epilogue",
    "epilogue": "preamble or epilogue",
    "non-utf8-prompt": "form field",
    "non-utf8-media-field": "form field",
    "too-deep-jsonl-line": "JSONL line",
    "bare-lf-in-a-header": "part header",
}

# Text files that are not JSONL: redacted as one text and sent, under every
# credential (the proxy read them).
TEXT_FILES: dict[str, bytes] = {
    "non-json-line": _form(PURPOSE, _jsonl(CLEAN_LINE, b"notes for " + EMAIL.encode(), b"")),
    "non-object-line": _form(PURPOSE, _jsonl(b'["' + EMAIL.encode() + b'"]')),
    "text-file": _form(
        _field("purpose", b"assistants"),
        _file("file", "notes.txt", b"call " + EMAIL.encode(), "text/plain"),
    ),
    "utf16-text-file": _form(
        _field("purpose", b"assistants"),
        _file("file", "notes.txt", b"\xff\xfe" + f"call {EMAIL}".encode("utf-16-le"), "text/plain"),
    ),
}

PDF = b"%PDF-1.7\n\xe2\xe3\xcf\xd3\n(" + EMAIL.encode() + b")\n%%EOF\n"
BINARY_FILE = _form(
    _field("purpose", b"assistants"),
    _file("file", f"{EMAIL} doc.pdf", PDF, "application/pdf"),
)


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
async def test_key_auth_refuses_the_same_upload(case: str) -> None:
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    path, body = REFUSED[case]
    openai_path = "/v1/images/edits" if "edits" in path else "/v1/files"
    async with _client(app) as client:
        response = await client.post(
            openai_path, content=body, headers={**_headers(), "authorization": "Bearer sk-t"}
        )
    assert response.status_code == 400 and upstream.requests == []
    message = response.json()["error"]["message"]
    assert KINDS[case] in message and "forwards only bodies it has redacted" in message


@pytest.mark.parametrize("case", sorted(REFUSED))
async def test_detection_off_still_forwards_the_same_upload_verbatim(case: str) -> None:
    upstream = _Upstream(b"{}")
    off = ProviderConfig("https://api.openai.com", detection=False)
    app = create_app(_config(openai=off), upstream_transport=httpx.MockTransport(upstream))
    path, body = REFUSED[case]
    openai_path = "/v1/images/edits" if "edits" in path else "/v1/files"
    async with _client(app) as client:
        response = await client.post(
            openai_path, content=body, headers={**_headers(), "authorization": "Bearer sk-t"}
        )
    assert response.status_code == 200
    assert upstream.requests[0].content == body


@pytest.mark.parametrize(
    "part",
    [
        pytest.param(b"", id="empty-part"),
        # An EMPTY header block (the part opens with CRLF): every reader
        # finds the content where the proxy scans it.
        pytest.param(b"\r\ncall " + EMAIL.encode(), id="empty-header-block"),
    ],
)
async def test_a_part_every_reader_finds_empty_headers_in_is_scanned_and_signed(
    monkeypatch: pytest.MonkeyPatch, part: bytes
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = _ASSISTANTS + b"--b\r\n" + part + b"\r\n--b--\r\n"
    async with _client(app) as client:
        response = await client.post(AZURE_FILES, content=body, headers=_headers())
    assert response.status_code == 200, response.text
    (sent,) = upstream.requests
    assert EMAIL.encode() not in sent.content and built[0].calls[0][3] == sent.content


@pytest.mark.parametrize("case", sorted(TEXT_FILES))
async def test_a_text_file_is_redacted_and_signed_under_identity(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        response = await client.post(AZURE_FILES, content=TEXT_FILES[case], headers=_headers())
    assert response.status_code == 200
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent and EMAIL.encode("utf-16-le") not in sent
    assert "EMAIL_001" in sent.decode("utf-8", "replace").replace("\x00", "")
    assert built[0].calls[0][3] == sent  # exactly what was signed is sent
    assert app.state.proxy.unscanned_uploads == {}


@pytest.mark.parametrize("case", sorted(TEXT_FILES))
async def test_a_text_file_is_redacted_under_key_auth(case: str) -> None:
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            content=TEXT_FILES[case],
            headers={**_headers(), "authorization": "Bearer sk-t"},
        )
    assert response.status_code == 200
    sent = upstream.requests[0].content
    assert EMAIL.encode() not in sent and EMAIL.encode("utf-16-le") not in sent


async def test_a_binary_file_is_never_signed_under_identity(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # binary_uploads = "forward" is the default, yet the proxy's own
    # identity vouches only for what it read.
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(AZURE_FILES, content=BINARY_FILE, headers=_headers())
    assert response.status_code == 400
    assert upstream.requests == [] and built[0].calls == []
    message = response.json()["error"]["message"]
    assert "an uploaded file is binary" in message and "proxy's own identity" in message
    assert EMAIL not in response.text and EMAIL not in caplog.text
    assert app.state.proxy.recent[-1]["status"] == 400
    assert app.state.proxy.unscanned_uploads == {}


async def test_a_binary_file_is_forwarded_unscanned_with_the_clients_own_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream = _Upstream(rb'{"id": "file_1", "filename": "\u00abEMAIL_001\u00bb doc.pdf"}')
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=BINARY_FILE, headers={**_headers(), "authorization": "Bearer t"}
        )
        status = (await client.get("/__llm-redact/status")).json()
        metrics = (await client.get("/__llm-redact/metrics")).text
    assert response.status_code == 200
    # The echoed (redacted) file name is restored in the answer.
    assert response.json()["filename"] == f"{EMAIL} doc.pdf"
    sent = upstream.requests[0].content
    # The file's bytes go out exactly as sent — unscanned (the value inside
    # the PDF leaves the machine: the documented, counted trade) — while
    # its file NAME is redacted.
    assert PDF in sent
    assert sent.replace(PDF, b"").find(EMAIL.encode()) == -1
    assert 'filename="«EMAIL_001» doc.pdf"'.encode() in sent
    assert status["unscanned_uploads_total"] == {"openai": 1}
    assert 'llm_redact_unscanned_uploads_total{provider="openai"} 1' in metrics
    assert "forwarded 1 binary upload file part(s) unscanned" in caplog.text
    assert "/v1/files" in caplog.text
    assert EMAIL not in caplog.text and "doc.pdf" not in caplog.text


async def test_a_binary_file_is_refused_with_binary_uploads_refuse() -> None:
    from llm_redact.detection.engine import DetectionConfig

    upstream = _Upstream(b"{}")
    app = create_app(
        Config(detection=DetectionConfig(binary_uploads="refuse")),
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=BINARY_FILE, headers={**_headers(), "authorization": "Bearer t"}
        )
        # A text file is still redacted and sent.
        text = await client.post(
            "/v1/files",
            content=TEXT_FILES["text-file"],
            headers={**_headers(), "authorization": "Bearer t"},
        )
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "an uploaded file is binary" in message
    assert "forwards only bodies it has redacted" in message
    assert text.status_code == 200 and len(upstream.requests) == 1
    assert app.state.proxy.unscanned_uploads == {}


async def test_detection_off_forwards_a_binary_file_uncounted() -> None:
    # The whole provider is the opt-out (its own /status line): the upload
    # is never read, so nothing is counted as a binary part.
    upstream = _Upstream(b"{}")
    off = ProviderConfig("https://api.openai.com", detection=False)
    app = create_app(_config(openai=off), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=BINARY_FILE, headers={**_headers(), "authorization": "Bearer t"}
        )
    assert response.status_code == 200 and upstream.requests[0].content == BINARY_FILE
    assert app.state.proxy.unscanned_uploads == {}


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


async def test_a_structural_form_field_is_scanned_under_every_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A structural field's value is protocol (the `model` a JSON walk skips
    # too), yet it is scanned before it leaves: the scanned-body rule holds
    # under the client's own key as under the proxy's own identity.
    body = _form(_field("purpose", EMAIL.encode()), _jsonl(CLEAN_LINE))
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        await client.post(
            "/v1/files", content=body, headers={**_headers(), "authorization": "Bearer t"}
        )
    assert EMAIL.encode() not in upstream.requests[0].content
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
