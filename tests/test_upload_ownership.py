"""What an upload cites is checked like a JSON body, whatever the route does
with its bytes.

The session router's stored-object check (``object_access_refusal``,
llm-redact-pro named users) reads the parsed request body. A
multipart/form-data upload is not JSON, yet it cites stored objects the
provider acts on: the lines of an uploaded batch input file are requests
the provider runs later, with the credential the upload is sent with, and a
form field can name a file. ``upload_view.read_upload`` reads an upload for
the check alone — its lines and form fields — so:

- a route with ``[providers.NAME] detection = false`` (nothing redacted) is
  checked exactly like a redacted one, before anything is sent, and a line
  repeating a key is sent re-serialized: exactly what was checked;
- a routed PASS-THROUGH upload (no adapter matches it) sent with the proxy's
  own credential is checked too, its bytes forwarded verbatim; one the check
  cannot read (outside the canonical grammar, a transfer encoding, a field
  that is not UTF-8 text, a line repeating a key, more JSON than
  ``max_body_bytes``) is refused 400/413, like the JSON case;
- under the proxy's credential a matched route's body the check cannot read
  (content-encoded, two Content-Types) is refused too.

Keyless: scripted fakes on a bare Registry stand in for llm-redact-pro.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import create_app
from llm_redact.upload_view import (
    AMBIGUOUS,
    CHARSET,
    NOT_TEXT,
    OUTSIDE_GRAMMAR,
    REPEATED_KEY,
    TRANSFER_ENCODED,
    UploadView,
    read_upload,
)
from test_object_access_seams import (
    AZURE,
    REFUSAL,
    UPSTREAM,
    FakeAuth,
    LineRouter,
    Upstream,
    _app,
    _batch_line,
    _client,
    _registry,
)

BOUNDARY = b"XyZ"
EMAIL = "ada@corp.example"


def _part(head: bytes, content: bytes) -> bytes:
    return b"--" + BOUNDARY + b"\r\n" + head + b"\r\n\r\n" + content + b"\r\n"


def _field(name: str, value: bytes, extra: bytes = b"") -> bytes:
    return _part(b'Content-Disposition: form-data; name="' + name.encode() + b'"' + extra, value)


def _file(content: bytes, extra: bytes = b"", filename: bytes = b"in.jsonl") -> bytes:
    head = b'Content-Disposition: form-data; name="file"; filename="' + filename + b'"'
    return _part(head + b"\r\nContent-Type: application/jsonl" + extra, content)


def _form(*parts: bytes) -> bytes:
    return b"".join(parts) + b"--" + BOUNDARY + b"--\r\n"


def _lines(*objects: Any) -> bytes:
    return b"\n".join(json.dumps(obj).encode() for obj in objects) + b"\n"


HEADERS = {"content-type": "multipart/form-data; boundary=XyZ"}


# --- the reader ---------------------------------------------------------------------


def test_an_upload_cites_its_json_lines_and_its_form_fields() -> None:
    body = _form(
        _field("purpose", b"batch"),
        _field("file_ids[]", b"file-1"),
        _field("tool_resources[code_interpreter][file_ids][0]", b"file-2"),
        _field("metadata", b' {"source": {"type": "file_id", "id": "file-3"}} '),
        _field("tags", b"[not json"),
        _field("odd[name", b"v"),
        _file(_lines(_batch_line("file-4"), [1, 2], "x") + b"not json\n\xff\xfe\n"),
        _file(b"\x89PNG\r\n\x1a\n\x00binary", filename=b"image.png"),
        _part(b"Content-Type: text/plain", b"a part without a name"),
    )
    view = read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=10_000)
    assert view == UploadView(
        [
            {"purpose": "batch"},
            {"file_ids": ["file-1"]},
            {"tool_resources": {"code_interpreter": {"file_ids": ["file-2"]}}},
            {"metadata": {"source": {"type": "file_id", "id": "file-3"}}},
            {"tags": "[not json"},
            {"odd[name": "v"},
            _batch_line("file-4"),
        ]
    )


def test_a_part_with_no_header_block_is_no_field() -> None:
    body = b"--XyZ\r\nno header block at all\r\n--XyZ--"
    assert read_upload(body, BOUNDARY, max_json_bytes=100, max_lines=10_000) == UploadView([])


def test_a_line_repeating_a_key_is_read_as_the_provider_reads_it() -> None:
    first = _lines({"custom_id": "1", "body": {"file_id": "file-own"}})
    repeated = b'{"body": {"file_id": "file-a"}, "body": {"file_id": "file-own"}}\n'
    body = _form(_field("purpose", b"batch"), _file(first + repeated))
    view = read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=10_000)
    assert view.cited[-1] == {"body": {"file_id": "file-own"}}  # the LAST occurrence
    assert view.normalized is not None and b"file-a" not in view.normalized
    # Only the repeating line changed; every other byte is the upload's own.
    assert view.normalized == body.replace(
        repeated, json.dumps({"body": {"file_id": "file-own"}}).encode() + b"\n"
    )


def test_a_rewritten_file_before_other_parts_is_still_normalized() -> None:
    # Whatever parts follow the file whose line was rewritten, the body sent
    # is the rewritten one.
    repeated = b'{"body": {"file_id": "file-a"}, "body": {"file_id": "file-own"}}\n'
    body = _form(_file(repeated), _field("purpose", b"batch"), _file(b"plain text\n"))
    view = read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=10_000)
    assert view.normalized == body.replace(
        repeated, json.dumps({"body": {"file_id": "file-own"}}).encode() + b"\n"
    )


def test_a_rewritten_line_keeps_a_lone_surrogate_as_the_same_json_value() -> None:
    # A \ud800-style escape can carry a lone surrogate, which has no UTF-8
    # form: the rewritten line escapes it again (jsonwalk.json_bytes — only
    # the surrogate, every other character as the upload had it; the same
    # JSON value) instead of failing the upload.
    repeated = b'{"body": {"file_id": "file-a"}, "body": {"input": "\\ud800 \xc2\xab"}}\n'
    body = _form(_file(repeated))
    view = read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=10_000)
    assert view.cited == [{"body": {"input": "\ud800 «"}}]
    assert view.normalized == body.replace(repeated, b'{"body": {"input": "\\ud800 \xc2\xab"}}\n')


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        pytest.param(b"--XyZ\nlf only\n--XyZ--", OUTSIDE_GRAMMAR, id="outside-the-grammar"),
        pytest.param(
            _form(_part(b'Content-Disposition: form-data; name="a"; name="b"', b"x")),
            AMBIGUOUS,
            id="ambiguous-disposition",
        ),
        pytest.param(
            _form(_file(b"e30=", b"\r\nContent-Transfer-Encoding: base64")),
            TRANSFER_ENCODED,
            id="transfer-encoded-file",
        ),
        pytest.param(
            _form(_field("file_id", b"file-a", b"\r\nContent-Transfer-Encoding: quoted-printable")),
            TRANSFER_ENCODED,
            id="transfer-encoded-field",
        ),
        pytest.param(
            _form(
                _field("file_id", b"f\x00i\x00", b"\r\nContent-Type: text/plain; charset=utf-16")
            ),
            CHARSET,
            id="field-charset",
        ),
        pytest.param(
            _form(_field("_charset_", b"iso-8859-1"), _field("file_id", b"file-a")),
            CHARSET,
            id="default-charset-field",
        ),
        pytest.param(_form(_field("file_id", b"\xff\xfe")), NOT_TEXT, id="not-utf8"),
        pytest.param(
            _form(_field("metadata", b'{"file_id": "file-a", "file_id": "file-b"}')),
            REPEATED_KEY,
            id="json-field-repeating-a-key",
        ),
    ],
)
def test_what_the_check_cannot_read_is_named(body: bytes, problem: str) -> None:
    assert read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=10_000) == UploadView(
        [], problem=problem
    )


def test_plain_encodings_and_charsets_are_read() -> None:
    body = _form(
        _field("_charset_", b"UTF-8"),
        _field("file_id", b"file-a", b"\r\nContent-Type: text/plain; charset=US-ASCII"),
        _file(_lines({"a": 1}), b"\r\nContent-Transfer-Encoding: 8bit"),
    )
    view = read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=10_000)
    assert view == UploadView([{"_charset_": "UTF-8"}, {"file_id": "file-a"}, {"a": 1}])


def test_more_json_than_the_check_reads_is_oversized() -> None:
    line = _lines({"custom_id": "x" * 40})
    body = _form(_file(line * 3))
    assert read_upload(body, BOUNDARY, max_json_bytes=len(line) * 3, max_lines=10_000).cited
    assert read_upload(
        body, BOUNDARY, max_json_bytes=len(line) * 3 - 4, max_lines=10_000
    ) == UploadView([], oversized=True)
    fields = _form(_field("prompt", b"p" * 64))
    assert read_upload(fields, BOUNDARY, max_json_bytes=63, max_lines=10_000) == UploadView(
        [], oversized=True
    )
    # Media files carry no JSON: never counted, whatever their size.
    media = _form(_file(b"\x00" * 10_000, filename=b"a.mp3"))
    assert read_upload(media, BOUNDARY, max_json_bytes=16, max_lines=10_000) == UploadView([])


def test_the_budget_may_be_spent_to_the_last_byte() -> None:
    # Exactly the budget is read; one byte more is oversized — for a file's
    # lines (the newline is not counted) and for a form field alike.
    line = json.dumps({"custom_id": "x" * 40}).encode()
    body = _form(_file(line + b"\n" + line + b"\n"))
    assert (
        read_upload(body, BOUNDARY, max_json_bytes=2 * len(line), max_lines=10_000).cited
        == [{"custom_id": "x" * 40}] * 2
    )
    assert read_upload(body, BOUNDARY, max_json_bytes=2 * len(line) - 1, max_lines=10_000).oversized
    fields = _form(_field("prompt", b"p" * 64))
    assert read_upload(fields, BOUNDARY, max_json_bytes=64, max_lines=10_000) == UploadView(
        [{"prompt": "p" * 64}]
    )


def test_every_object_line_is_read_whatever_precedes_it() -> None:
    # Blank lines, lines that are not JSON and JSON lines that are not
    # objects are skipped — never the end of the reading.
    content = b"\n   \nnot json\n[1, 2]\n" + json.dumps(_batch_line("file-9")).encode() + b"\n"
    view = read_upload(_form(_file(content)), BOUNDARY, max_json_bytes=10_000, max_lines=10_000)
    assert view == UploadView([_batch_line("file-9")])


def test_a_file_named_only_by_its_extended_filename_is_read_as_a_file() -> None:
    body = _form(
        _part(
            b"Content-Disposition: form-data; name=\"file\"; filename*=UTF-8''in.jsonl",
            _lines(_batch_line("file-7")),
        )
    )
    view = read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=10_000)
    assert view == UploadView([_batch_line("file-7")])


def test_a_json_array_form_field_is_read_as_json() -> None:
    body = _form(_field("file_ids", b' ["file-1", "file-2"] '))
    view = read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=10_000)
    assert view == UploadView([{"file_ids": ["file-1", "file-2"]}])


def test_a_rewritten_line_keeps_its_non_ascii_text_as_utf8() -> None:
    repeated = '{"body": {"input": "a"}, "body": {"input": "«é»"}}\n'.encode()
    body = _form(_file(repeated))
    view = read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=10_000)
    assert view.cited == [{"body": {"input": "«é»"}}]
    assert view.normalized == body.replace(repeated, '{"body": {"input": "«é»"}}\n'.encode())


# --- detection = false: checked like a redacted route ---------------------------------

OFF = {"openai": ProviderConfig(UPSTREAM, detection=False)}


@pytest.mark.parametrize("detection", [True, False], ids=["redacted", "detection-off"])
@pytest.mark.parametrize(
    "citing",
    [
        pytest.param(
            _form(_field("purpose", b"batch"), _file(_lines(_batch_line("file-a")))), id="line"
        ),
        pytest.param(_form(_field("purpose", b"batch"), _field("file_id", b"file-a")), id="field"),
        pytest.param(
            _form(_field("purpose", b"batch"), _field("file_ids[]", b"file-a")), id="list-field"
        ),
    ],
)
async def test_an_upload_citing_a_refused_object_is_never_sent(
    detection: bool, citing: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = LineRouter()
    upstream = Upstream()
    providers = None if detection else OFF
    app = _app(monkeypatch, router, upstream, providers=providers)
    async with _client(app) as client:
        response = await client.post("/v1/files", content=citing, headers=HEADERS)
    assert response.status_code == 403 and REFUSAL in response.text
    assert upstream.requests == []
    ((adapter, method, path, body, identity),) = router.checks
    assert (adapter, method, path, identity) == ("openai", "POST", "/v1/files", False)
    assert "file-a" in json.dumps(body)


async def test_an_allowed_upload_with_detection_off_is_forwarded_unredacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream, providers=OFF)
    upload = _form(_field("purpose", b"batch"), _file(_lines({"body": {"input": EMAIL}})))
    async with _client(app) as client:
        response = await client.post("/v1/files", content=upload, headers=HEADERS)
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert sent.content == upload  # byte for byte: nothing redacted
    assert router.checks[0][3] == [{"purpose": "batch"}, {"body": {"input": EMAIL}}]


async def test_detection_off_sends_a_repeated_key_line_as_it_was_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream, providers=OFF)
    repeated = b'{"custom_id": "1", "body": {"input": [{"file_id": "file-a"}]}, "body": {}}\n'
    upload = _form(_file(repeated))
    async with _client(app) as client:
        response = await client.post("/v1/files", content=upload, headers=HEADERS)
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert b"file-a" not in sent.content  # the occurrence the check never saw
    assert router.checks[0][3] == [{"custom_id": "1", "body": {}}]


PDF = b"%PDF-1.7\n(" + EMAIL.encode() + b")\n%%EOF\n"


async def test_with_the_clients_own_key_a_binary_file_passes_the_check_unread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Content-decided uploads: the check reads the form fields; a binary
    # file cites nothing and never makes the request fail — it is forwarded
    # byte for byte (binary_uploads = "forward"), counted.
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream)
    upload = _form(_field("purpose", b"user_data"), _file(PDF, filename=b"a.pdf"))
    async with _client(app) as client:
        response = await client.post("/v1/files", content=upload, headers=HEADERS)
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert sent.content == upload
    assert router.checks[0][3] == [{"purpose": "user_data"}]
    assert app.state.proxy.unscanned_uploads == {"openai": 1}


async def test_a_text_file_sends_its_repeated_key_line_as_it_was_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A JSON line inside a text file is read by the check (the LAST
    # occurrence of a repeated key); the file is then redacted as text, so
    # what goes out is the line the check read — never the occurrence it
    # did not.
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream)
    repeated = b'{"body": {"input": [{"file_id": "file-a"}]}, "body": {"q": "' + EMAIL.encode()
    upload = _form(_file(b"notes\n" + repeated + b'"}}\n', filename=b"notes.txt"))
    async with _client(app) as client:
        response = await client.post("/v1/files", content=upload, headers=HEADERS)
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert b"file-a" not in sent.content and EMAIL.encode() not in sent.content
    assert "«EMAIL_001»".encode() in sent.content
    assert router.checks[0][3] == [{"body": {"q": EMAIL}}]


async def test_under_identity_a_binary_file_is_refused_before_it_is_signed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_redact.registry import Registry

    auth = FakeAuth()
    reg = Registry()
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    router = LineRouter()
    upstream = Upstream()
    app = _app(
        monkeypatch,
        router,
        upstream,
        registry=reg,
        providers={"azure": ProviderConfig(AZURE, auth="identity")},
    )
    upload = _form(_field("purpose", b"user_data"), _file(PDF, filename=b"a.pdf"))
    async with _client(app) as client:
        response = await client.post("/openai/files?api-version=1", content=upload, headers=HEADERS)
    assert response.status_code == 400
    assert "an uploaded file is binary" in response.json()["error"]["message"]
    assert upstream.requests == [] and auth.calls == 0
    assert router.checks[0][3] == [{"purpose": "user_data"}]  # checked, then refused


@pytest.mark.parametrize("detection", [True, False], ids=["redacted", "detection-off"])
async def test_an_uploaded_line_holding_a_lone_surrogate_is_sent_as_the_same_value(
    detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A line the proxy rewrites (a redacted value, a repeated key) whose
    # JSON escapes carry a lone surrogate is sent re-escaped — never a 500.
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream, providers=None if detection else OFF)
    line = b'{"body": {"input": "\\ud800 ' + EMAIL.encode() + b'"}, "body": {"input": "\\ud800 '
    line += EMAIL.encode() + b'"}}\n'
    async with _client(app) as client:
        response = await client.post("/v1/files", content=_form(_file(line)), headers=HEADERS)
    assert response.status_code == 200
    (sent,) = upstream.requests
    (value,) = [json.loads(row) for row in sent.content.split(b"\r\n") if row.startswith(b"{")]
    assert value["body"]["input"].startswith("\ud800 ")
    assert (EMAIL in value["body"]["input"]) is not detection  # redacted unless detection is off


async def test_under_identity_with_detection_off_an_unreadable_part_is_never_signed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from license_fixtures import resolved
    from llm_redact.registry import Registry

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
    hidden = _form(_file(b"eyJmaWxlX2lkIjogImZpbGUtYSJ9", b"\r\nContent-Transfer-Encoding: base64"))
    async with _client(app) as client:
        response = await client.post("/openai/v1/files", content=hidden, headers=HEADERS)
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert TRANSFER_ENCODED in message and "proxy's own identity" in message
    assert upstream.requests == [] and auth.calls == 0 and router.checks == []


# --- routed pass-through uploads under the proxy's credential -------------------------


def _routed(
    monkeypatch: pytest.MonkeyPatch,
    router: Any,
    path: str,
    *,
    proxy_credential: bool | None = None,
    **config: Any,
) -> tuple[Any, FakeRouter, Upstream]:
    kwargs = {} if proxy_credential is None else {"proxy_credential": proxy_credential}
    fake = FakeRouter(
        {"r": [Hop("x", f"http://x.example{path}"), Stop()]}, plan_kwargs={"r": kwargs}
    )
    reg, _ = install(monkeypatch, fake)
    _registry(monkeypatch, router, reg)
    upstream = Upstream({"id": "part_1"})
    app = create_app(routed_config(**config), upstream_transport=httpx.MockTransport(upstream))
    return app, fake, upstream


PARTS = "/v1/uploads/upload_1/parts"  # pass-through: no adapter matches it


@pytest.mark.parametrize(
    ("upload", "config"),
    [
        pytest.param(
            _form(_file(_lines(_batch_line("file-a")), filename=b"part.jsonl")), {}, id="line"
        ),
        pytest.param(_form(_field("file_ids[]", b"file-a")), {}, id="field"),
        pytest.param(
            _form(_field("note", EMAIL.encode()), _file(_lines({"q": EMAIL}), filename=b"p.jsonl")),
            {},
            id="allowed",
        ),
        pytest.param(b"--XyZ\nlf only\n--XyZ--", {}, id="lf-only"),
        pytest.param(
            _form(_file(b"eyJ9", b"\r\nContent-Transfer-Encoding: base64")),
            {},
            id="transfer-encoded",
        ),
        pytest.param(_form(_field("file_id", b"\xff")), {}, id="not-text"),
        pytest.param(
            _form(_file(b'{"file_id": "file-a", "file_id": "file-own"}\n')),
            {},
            id="repeated-key-line",
        ),
        pytest.param(_form(_file(_lines({"q": "x" * 80}))), {"max_body_bytes": 64}, id="oversized"),
        pytest.param(
            _form(*[_field("purpose", b"batch")] * 3), {"max_body_strings": 2}, id="over-part-cap"
        ),
    ],
)
async def test_a_routed_pass_through_upload_under_the_proxys_credential_is_refused_unread(
    upload: bytes, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrecognized route (the Uploads API's parts) is never sent with a
    credential the proxy holds (C-R2-04): refused before the upload is
    read — whatever it holds, readable or not, allowed or not — or the
    ownership check asked; nothing upstream."""
    import llm_redact.proxy as proxy_module

    def no_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("read an upload on a refused route")

    monkeypatch.setattr(proxy_module, "read_upload", no_read)
    router = LineRouter()
    app, fake, upstream = _routed(monkeypatch, router, PARTS, **config)
    async with _client(app) as client:
        response = await client.post(PARTS, content=upload, headers={**HEADERS, ROUTE_HEADER: "r"})
    assert response.status_code == 403
    assert "credential the proxy holds" in response.json()["error"]
    assert router.checks == [] and fake.plans[0].begun == [] and upstream.requests == []
    (row,) = app.state.proxy.recent
    assert row["status"] == 403 and row["provider"] == "openai"


async def test_with_the_clients_own_credential_a_pass_through_upload_is_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = LineRouter()
    app, fake, _ = _routed(monkeypatch, router, PARTS, proxy_credential=False)
    upload = b"--XyZ\nlf only, unread\n--XyZ--"
    async with _client(app) as client:
        response = await client.post(PARTS, content=upload, headers={**HEADERS, ROUTE_HEADER: "r"})
    assert response.status_code == 200 and fake.plans[0].begun[0][0] == upload
    assert router.checks == [(None, "POST", PARTS, None, False)]


# --- a matched route's body under the proxy's credential -------------------------------


@pytest.mark.parametrize(
    ("headers", "why", "status"),
    [
        pytest.param(
            [("content-type", "application/json"), ("content-encoding", "gzip")],
            "content-encoded",
            415,
            id="content-encoded",
        ),
        pytest.param(
            [
                ("content-type", "application/json"),
                ("content-type", "multipart/form-data; boundary=b"),
            ],
            "more than one Content-Type",
            400,
            id="two-content-types",
        ),
    ],
)
async def test_a_matched_body_the_check_cannot_read_is_never_sent_with_the_proxys_credential(
    headers: list[tuple[str, str]], why: str, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = LineRouter()
    app, fake, upstream = _routed(
        monkeypatch,
        router,
        "/v1/chat/completions",
        providers={**Config().providers, "openai": ProviderConfig(UPSTREAM)},
    )
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=body,
            headers=httpx.Headers([*headers, (ROUTE_HEADER, "r")]),
        )
    assert response.status_code == status
    assert why in response.json()["error"]["message"]  # provider-shaped: a matched route
    assert router.checks == [] and fake.plans[0].begun == [] and upstream.requests == []
    # With the client's own credential and detection = false (the explicit
    # opt-out) the same body is not refused: nothing reads it at all.
    app, fake, _ = _routed(
        monkeypatch,
        LineRouter(),
        "/v1/chat/completions",
        proxy_credential=False,
        providers={**Config().providers, "openai": ProviderConfig(UPSTREAM, detection=False)},
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=body,
            headers=httpx.Headers([*headers, (ROUTE_HEADER, "r")]),
        )
    assert response.status_code == 200


# --- max_body_strings: parts are counted before the check reads an upload -------------


async def test_an_upload_over_the_part_cap_is_never_read(monkeypatch: pytest.MonkeyPatch) -> None:
    # A body of many empty parts costs the event loop per part: the check
    # never parses one over max_body_strings. Where the scanned-body rule
    # holds — redaction applies, or the proxy's credential is spent — the
    # parts are counted before the rule parses anything and the upload is
    # refused 413 before the check is asked — nothing upstream; with the
    # client's own key and detection = false it goes out as sent, unread.
    import llm_redact.proxy as proxy_module

    def no_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("read an upload over the part cap")

    monkeypatch.setattr(proxy_module, "read_upload", no_read)
    upload = _form(*[_field("purpose", b"batch")] * 3)

    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream, max_body_strings=2)
    async with _client(app) as client:
        response = await client.post("/v1/files", content=upload, headers=HEADERS)
    assert response.status_code == 413 and "max_body_strings (2)" in response.text
    # The check saw no upload content (never read), and nothing went upstream.
    assert all(body is None for _, _, _, body, _ in router.checks)
    assert upstream.requests == []

    router = LineRouter()
    app, fake, upstream = _routed(
        monkeypatch,
        router,
        "/v1/files",
        max_body_strings=2,
        providers={**Config().providers, "openai": ProviderConfig(UPSTREAM)},
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=upload, headers={**HEADERS, ROUTE_HEADER: "r"}
        )
    assert response.status_code == 413
    assert "max_body_strings (2)" in response.json()["error"]["message"]
    assert router.checks == [] and fake.plans[0].begun == [] and upstream.requests == []

    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream, max_body_strings=2, providers=OFF)
    async with _client(app) as client:
        response = await client.post("/v1/files", content=upload, headers=HEADERS)
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert sent.content == upload  # forwarded as sent, never read
    assert router.checks == [("openai", "POST", "/v1/files", None, False)]


# --- the reading never changes what a file part is ---------------------------------------

# A UTF-16BE text whose SECOND line's bytes spell an ASCII JSON object that
# repeats a key: every byte pair is a valid UTF-16 character (``{"``, ``a"``,
# ``:1`` ... ``}\n``), so the file is UTF-16 text; spliced UTF-8 bytes of a
# different length would misalign the rest of it.
_UTF16_TEXT = (
    b"\xfe\xff"
    + "hi\n".encode("utf-16-be")
    + b'{"a":1,"a":2}\n'
    + f"mail {EMAIL}\n".encode("utf-16-be")
)


@pytest.mark.parametrize(
    ("content", "cited"),
    [
        # UTF-16LE behind its mark: a bytes parse guesses UTF-16 from the
        # mark; the line is no UTF-8 text, so it is neither read nor rewritten.
        (b"\xff\xfe" + f'{{"a": 1, "a": 2}}\r\nmail {EMAIL}\r\n'.encode("utf-16-le"), []),
        (b"\xff\xfe\x00\x00" + '{"a": 1, "a": 2}\n'.encode("utf-32-le"), []),
        # A line that IS UTF-8 JSON in a UTF-16 file: cited, never rewritten.
        (_UTF16_TEXT, [{"a": 2}]),
        # A binary file: cited, never rewritten (its bytes must not move).
        (b"%PDF-1.7\n" + b'{"a": 1, "a": 2}\n%%EOF\n', [{"a": 2}]),
        (b'\x00{"a": 1}\n{"a": 1, "a": 2}\n', [{"a": 2}]),
    ],
    ids=["utf-16-le", "utf-32-le", "utf-16-be-json-line", "pdf", "nul"],
)
def test_only_a_utf8_files_repeated_key_line_is_rewritten(content: bytes, cited: Any) -> None:
    from llm_redact.upload_content import classify_file

    assert classify_file(content).codec != "utf-8"
    view = read_upload(
        _form(_file(content, filename=b"n.txt")), BOUNDARY, max_json_bytes=10_000, max_lines=10_000
    )
    assert view == UploadView(cited)


@pytest.mark.parametrize(
    ("content", "cited", "rewritten"),
    [
        # JSONL, the first line behind a UTF-8 byte-order mark (read as the
        # JSONL redaction reads it; the rewrite drops the mark, as its does).
        (
            b'\xef\xbb\xbf{"a": 1, "a": 2}\n{"b": 3}\n',
            [{"a": 2}, {"b": 3}],
            b'{"a": 2}\n{"b": 3}\n',
        ),
        # UTF-8 text: the line the check read is what goes out.
        (b'notes\n{"a": 1, "a": 2}\n', [{"a": 2}], b'notes\n{"a": 2}\n'),
    ],
    ids=["jsonl-bom", "utf-8-text"],
)
def test_a_utf8_files_repeated_key_line_is_rewritten(
    content: bytes, cited: Any, rewritten: bytes
) -> None:
    view = read_upload(
        _form(_file(content, filename=b"n.txt")), BOUNDARY, max_json_bytes=10_000, max_lines=10_000
    )
    assert view == UploadView(cited, normalized=_form(_file(rewritten, filename=b"n.txt")))


@pytest.mark.parametrize(
    ("content", "encoded_email"),
    [
        (
            b"\xff\xfe" + f'{{"a": 1, "a": 2}}\r\nmail {EMAIL}\r\n'.encode("utf-16-le"),
            EMAIL.encode("utf-16-le"),
        ),
        (_UTF16_TEXT, EMAIL.encode("utf-16-be")),
    ],
    ids=["utf-16-le", "utf-16-be-json-line"],
)
async def test_a_utf16_text_file_is_redacted_whatever_its_lines_parse_as(
    monkeypatch: pytest.MonkeyPatch, content: bytes, encoded_email: bytes
) -> None:
    # The client's own key, binary_uploads = "forward" (the default): the
    # check's re-reading once rewrote the first line as UTF-8, the file then
    # read as "binary" and went out unscanned.
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream)
    upload = _form(_field("purpose", b"user_data"), _file(content, filename=b"n.txt"))
    headers = {**HEADERS, "authorization": "Bearer sk-own"}
    async with _client(app) as client:
        response = await client.post("/v1/files", content=upload, headers=headers)
        status = (await client.get("/__llm-redact/status")).json()
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert encoded_email not in sent.content
    codec = "utf-16-le" if encoded_email == EMAIL.encode("utf-16-le") else "utf-16-be"
    assert "«EMAIL_001»".encode(codec) in sent.content
    assert status["unscanned_uploads_total"] == {}


async def test_a_reading_that_changes_a_parts_kind_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The proxy's belt: whatever the reader returns, a re-serialized upload
    # whose file part redaction would read as something else never goes out
    # (here a text file turned binary) — with the client's own key too.
    import llm_redact.proxy as proxy_module

    text = _form(_file(f"mail {EMAIL}\n".encode(), filename=b"n.txt"))
    binary = _form(_file(b"\x00" + EMAIL.encode(), filename=b"n.txt"))
    monkeypatch.setattr(
        proxy_module,
        "read_upload",
        lambda body, boundary, **kw: UploadView([], normalized=binary),
    )
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream)
    headers = {**HEADERS, "authorization": "Bearer sk-own"}
    async with _client(app) as client:
        response = await client.post("/v1/files", content=text, headers=headers)
        recent = (await client.get("/__llm-redact/recent")).json()["entries"]
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "changed what a file part is" in message and "credential" not in message
    assert EMAIL not in response.text
    assert upstream.requests == [] and router.checks == []
    assert recent[0]["status"] == 400


@pytest.mark.parametrize(
    "normalized",
    [
        _form(_file(b"mail\n", filename=b"n.txt"), _field("purpose", b"batch")),
        b"not multipart at all",
    ],
    ids=["another-part", "outside-grammar"],
)
async def test_a_reading_that_changes_the_parts_is_refused(
    monkeypatch: pytest.MonkeyPatch, normalized: bytes
) -> None:
    # A re-serialized upload with another part count, or none the grammar
    # reads, counts as a changed reading: refused, whatever the credential.
    import llm_redact.proxy as proxy_module

    monkeypatch.setattr(
        proxy_module,
        "read_upload",
        lambda body, boundary, **kw: UploadView([], normalized=normalized),
    )
    upstream = Upstream()
    app = _app(monkeypatch, LineRouter(), upstream)
    text = _form(_file(b"mail\n", filename=b"n.txt"))
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=text, headers={**HEADERS, "authorization": "Bearer sk-own"}
        )
    assert response.status_code == 400
    assert "changed what a file part is" in response.json()["error"]["message"]
    assert upstream.requests == []


async def test_a_reading_that_keeps_every_parts_kind_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The belt compares kinds part by part, in order: an unchanged reading
    # (a JSONL line re-serialized, still JSONL) goes out as the check read it.
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream)
    upload = _form(_field("purpose", b"batch"), _file(b'{"a": 1, "a": 2}\n'))
    async with _client(app) as client:
        response = await client.post("/v1/files", content=upload, headers=HEADERS)
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert b'{"a": 2}' in sent.content and b'"a": 1' not in sent.content


# --- the check parses only lines that can be objects, each counted first ------------------


def test_only_lines_that_can_be_objects_are_parsed_and_counted() -> None:
    # Blank lines, prose, arrays and strings are never parsed nor counted; a
    # line whose first byte past whitespace and one byte-order mark is "{"
    # is counted before it is parsed, whether it parses or not.
    line = json.dumps(_batch_line("file-9")).encode()
    content = (
        b'\n  \nprose { brace\n[1]\n"s"\n\t\xef\xbb\xbf ' + line + b"\n{not json\n" + line + b"\n"
    )
    body = _form(_file(content))
    view = read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=3)
    assert view == UploadView([_batch_line("file-9")] * 2)
    assert read_upload(body, BOUNDARY, max_json_bytes=10_000, max_lines=2) == UploadView(
        [], too_many_lines=True
    )
    # One count across the upload's file parts.
    two = _form(_file(_lines({"a": 1})), _file(_lines({"b": 2}), filename=b"b.jsonl"))
    assert read_upload(two, BOUNDARY, max_json_bytes=10_000, max_lines=2).cited == [
        {"a": 1},
        {"b": 2},
    ]
    assert read_upload(two, BOUNDARY, max_json_bytes=10_000, max_lines=1).too_many_lines


def test_a_repeated_key_line_is_rewritten_in_place_among_other_lines() -> None:
    # The rewrite splices exactly the repeated-key lines (their own
    # surrounding whitespace included), every other byte kept.
    content = b'prose\r\n {"a": 1, "a": 2}\r\n{"b": 1}\n\n{"c": 1, "c": 3}'
    view = read_upload(_form(_file(content)), BOUNDARY, max_json_bytes=10_000, max_lines=10)
    assert view.cited == [{"a": 2}, {"b": 1}, {"c": 3}]
    assert view.normalized == _form(_file(b'prose\r\n{"a": 2}\n{"b": 1}\n\n{"c": 3}'))


@pytest.mark.parametrize(
    "content",
    [
        b"\n" * (10 * 2**20),  # blank lines: never parsed
        b"a\n" * (5 * 2**20),  # prose
        b" " * (10 * 2**20),  # one line of whitespace: no backtracking
        b"{\n" * (5 * 2**20),  # candidates: refused at the line budget
    ],
    ids=["blank", "prose", "whitespace", "braces"],
)
def test_a_huge_line_count_is_read_in_bounded_time(content: bytes) -> None:
    # 10 MiB of blank lines once took about a minute (a JSON parse per line).
    started = time.perf_counter()
    view = read_upload(
        _form(_file(content)), BOUNDARY, max_json_bytes=10 * 2**20, max_lines=100_000
    )
    assert time.perf_counter() - started < 5
    assert view.cited == []
    assert view.too_many_lines is content.startswith(b"{")


async def test_an_upload_with_more_json_lines_than_the_budget_is_refused_unread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Under the proxy's credential an upload whose file holds more lines to
    # parse than max_body_strings is refused 413 before the check is asked
    # and before anything is sent.
    upload = _form(_file(_lines(*[{"q": index} for index in range(5)])))
    router = LineRouter()
    app, fake, upstream = _routed(
        monkeypatch,
        router,
        "/v1/files",
        max_body_strings=4,
        providers={**Config().providers, "openai": ProviderConfig(UPSTREAM, detection=False)},
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/files", content=upload, headers={**HEADERS, ROUTE_HEADER: "r"}
        )
    assert response.status_code == 413
    assert "more JSON lines than llm-redact max_body_strings (4)" in response.text
    assert router.checks == [] and fake.plans[0].begun == [] and upstream.requests == []


async def test_a_huge_blank_upload_is_checked_in_bounded_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End to end with the stored-object check on: 10 MiB of newlines once
    # blocked the event loop for about a minute.
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream, providers=OFF)
    upload = _form(_file(b"\n" * (10 * 2**20 - 200)))
    started = time.perf_counter()
    async with _client(app) as client:
        response = await client.post("/v1/files", content=upload, headers=HEADERS)
    assert time.perf_counter() - started < 10
    assert response.status_code == 200
    assert router.checks == [("openai", "POST", "/v1/files", [], False)]
