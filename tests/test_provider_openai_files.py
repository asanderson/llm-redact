"""OpenAI Files + Batches: multipart upload redaction, output rehydration.

The upload's JSONL file part is redacted line by line (batch lines get the
system note inside body; fine-tune lines directly); part filenames are
redacted, and restored in the file objects the provider echoes; batch
create, retrieve, cancel and list redact and restore the caller's
``metadata``. Through the proxy every piece of an upload must be scanned
(the scanned-body rule): plain form fields are scanned as text, a file
that is not JSONL is redacted as one text when it is text, and a binary
file refuses the upload unless the caller forwards binary parts
(``forward_binary``: the client's own key). The lenient reading
(``require_scanned=False``) keeps the structural form fields and binary
parts byte-identical. A downloaded file is restored by the same reading.
"""

import json
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_redact.config import Config, ProviderConfig, parse_config
from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
from llm_redact.multipart import parse, parse_boundary
from llm_redact.providers.base import SYSTEM_NOTE, RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor, UnredactableRequest
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"
BOUNDARY = b"testboundary123"


def _redactor(vault: InMemoryVault) -> Redactor:
    return Redactor(
        detectors=build_detectors(DetectionConfig()),
        vault=vault,
        allowlist=Allowlist(exact=frozenset(), patterns=()),
    )


def _upload_body(*lines: str) -> bytes:
    jsonl = "".join(f"{line}\n" for line in lines).encode()
    return (
        b"--testboundary123\r\n"
        b'Content-Disposition: form-data; name="purpose"\r\n'
        b"\r\n"
        b"batch\r\n"
        b"--testboundary123\r\n"
        b'Content-Disposition: form-data; name="file"; filename="input.jsonl"\r\n'
        b"Content-Type: application/jsonl\r\n"
        b"\r\n" + jsonl + b"\r\n"
        b"--testboundary123--\r\n"
    )


def test_files_routing() -> None:
    adapter = OpenAIAdapter()
    assert adapter.matches("POST", "/v1/files") is RouteKind.CHAT
    assert adapter.matches("GET", "/v1/files/file_abc/content") is RouteKind.CHAT
    # File objects echo the (redacted) upload filename: restored.
    assert adapter.matches("GET", "/v1/files") is RouteKind.CHAT
    assert adapter.matches("GET", "/v1/files/file_abc") is RouteKind.CHAT
    # A file's delete carries its id only: recognized, redact-only (a
    # body-less no-op), so a proxy-held credential may reach it.
    assert adapter.matches("DELETE", "/v1/files/file_abc") is RouteKind.REDACT_ONLY
    assert adapter.matches("DELETE", "/v1/files") is RouteKind.NONE
    assert not adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1/files/file_abc")
    # Batches: create, retrieve, cancel and the list carry (or echo) the
    # caller's `metadata`; unknown batch sub-routes stay pass-through.
    assert adapter.matches("POST", "/v1/batches") is RouteKind.CHAT
    assert adapter.matches("GET", "/v1/batches/batch_1") is RouteKind.CHAT
    assert adapter.matches("POST", "/v1/batches/batch_1/cancel") is RouteKind.CHAT
    assert adapter.matches("GET", "/v1/batches") is RouteKind.CHAT
    assert adapter.matches("POST", "/v1/batches/batch_1") is RouteKind.NONE
    assert adapter.matches("DELETE", "/v1/batches/batch_1") is RouteKind.NONE
    assert adapter.matches("GET", "/v1/batches/batch_1/cancel") is RouteKind.NONE
    for path in ("/v1/batches", "/v1/batches/batch_1", "/v1/batches/batch_1/cancel"):
        assert not adapter.wants_system_note(RouteKind.CHAT, path)


def test_upload_batch_lines_redacted_and_noted() -> None:
    adapter = OpenAIAdapter()
    vault = InMemoryVault()
    batch_line = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": f"to {EMAIL}"}]},
        }
    )
    clean_line = json.dumps({"custom_id": "req-2", "body": {"messages": []}})
    body = _upload_body(batch_line, "", clean_line)

    out = adapter.redact_multipart("/v1/files", body, BOUNDARY, _redactor(vault), inject_note=True)
    assert out is not None
    parsed = parse(out, BOUNDARY)
    assert parsed is not None
    assert parsed.parts[0].content == b"batch"  # form field untouched

    lines = parsed.parts[1].content.split(b"\n")
    rewritten = json.loads(lines[0])
    assert EMAIL not in lines[0].decode()
    assert "«EMAIL_001»" in rewritten["body"]["messages"][-1]["content"]
    assert rewritten["body"]["messages"][0] == {"role": "system", "content": SYSTEM_NOTE}
    assert rewritten["custom_id"] == "req-1"
    assert lines[1] == b""  # a blank line byte-identical
    assert json.loads(lines[2]) == json.loads(clean_line)  # clean line unchanged bytes
    assert lines[2] == clean_line.encode()


def test_upload_fine_tune_lines() -> None:
    adapter = OpenAIAdapter()
    vault = InMemoryVault()
    ft_line = json.dumps({"messages": [{"role": "user", "content": f"contact {EMAIL}"}]})
    out = adapter.redact_multipart(
        "/v1/files", _upload_body(ft_line), BOUNDARY, _redactor(vault), inject_note=True
    )
    assert out is not None
    parsed = parse(out, BOUNDARY)
    assert parsed is not None
    rewritten = json.loads(parsed.parts[1].content.split(b"\n")[0])
    assert "«EMAIL_001»" in rewritten["messages"][-1]["content"]
    assert rewritten["messages"][0]["content"] == SYSTEM_NOTE


def test_upload_without_secrets_forwards_verbatim() -> None:
    adapter = OpenAIAdapter()
    body = _upload_body(json.dumps({"custom_id": "req-1", "body": {"messages": []}}))
    out = adapter.redact_multipart(
        "/v1/files", body, BOUNDARY, _redactor(InMemoryVault()), inject_note=True
    )
    assert out is None  # nothing changed: proxy forwards the ORIGINAL bytes


def test_upload_binary_file_untouched() -> None:
    adapter = OpenAIAdapter()
    binary = (
        b"--testboundary123\r\n"
        b'Content-Disposition: form-data; name="file"; filename="doc.pdf"\r\n'
        b"\r\n"
        b"%PDF-1.7 \x00\x01\x02 not jsonl\r\n"
        b"--testboundary123--\r\n"
    )
    assert (
        adapter.redact_multipart(
            "/v1/files", binary, BOUNDARY, _redactor(InMemoryVault()), inject_note=True
        )
        is None
    )


def test_output_file_rehydrated() -> None:
    adapter = OpenAIAdapter()
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    rehydrator = Rehydrator(vault)
    output_line = json.dumps(
        {
            "id": "batch_req_1",
            "custom_id": "req-1",
            "response": {
                "status_code": 200,
                "body": {"choices": [{"message": {"content": f"sent to {token}"}}]},
            },
        }
    ).encode()
    raw = output_line + b"\n" + json.dumps({"id": "batch_req_2", "note": token}).encode() + b"\n"
    out = adapter.rehydrate_raw_body("/v1/files/file_abc/content", raw, rehydrator)
    assert out is not None
    restored = json.loads(out.split(b"\n")[0])
    assert restored["response"]["body"]["choices"][0]["message"]["content"] == f"sent to {EMAIL}"
    assert json.loads(out.split(b"\n")[1]) == {"id": "batch_req_2", "note": EMAIL}
    # A file mixing JSON lines with other lines is TEXT, restored as one
    # text: a token the provider wrote escaped stays a placeholder (never a
    # wrong value); a raw one is restored.
    mixed = output_line + b"\nnot json " + token.encode() + b"\n"
    out = adapter.rehydrate_raw_body("/v1/files/file_abc/content", mixed, rehydrator)
    assert out == output_line + b"\nnot json " + EMAIL.encode() + b"\n"
    # Non-file-content paths and token-free bodies stay untouched.
    assert adapter.rehydrate_raw_body("/v1/other", raw, rehydrator) is None
    assert (
        adapter.rehydrate_raw_body("/v1/files/file_abc/content", b"plain text\n", rehydrator)
        is None
    )


def _file_upload(content: bytes, *, filename: str = "notes.txt", headers: bytes = b"") -> bytes:
    return (
        b"--testboundary123\r\n"
        b'Content-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
        b"--testboundary123\r\n"
        b'Content-Disposition: form-data; name="file"; filename="'
        + filename.encode()
        + b'"\r\n'
        + headers
        + b"\r\n"
        + content
        + b"\r\n--testboundary123--\r\n"
    )


def _file_content(out: bytes | None) -> bytes:
    assert out is not None
    parsed = parse(out, BOUNDARY)
    assert parsed is not None
    return parsed.parts[1].content


def _scanned(body: bytes, **kwargs: Any) -> bytes | None:
    return OpenAIAdapter().redact_multipart(
        "/v1/files",
        body,
        BOUNDARY,
        _redactor(InMemoryVault()),
        inject_note=True,
        require_scanned=True,
        **kwargs,
    )


def test_a_text_file_is_redacted_as_one_text() -> None:
    # Mixed lines (a JSON object among prose) make it text: redacted as the
    # text it is, every byte but the value kept — no note, no re-serializing.
    content = (
        b'contact: {"email": "' + EMAIL.encode() + b'"}\r\n'
        b"name,email\r\njane," + EMAIL.encode() + b"\r\n"
    )
    out = _file_content(_scanned(_file_upload(content)))
    assert out == content.replace(EMAIL.encode(), "«EMAIL_001»".encode())
    assert SYSTEM_NOTE.encode() not in out


def test_a_text_file_with_a_bom_keeps_its_encoding() -> None:
    for bom, codec in ((b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe", "utf-16-le")):
        text = f"mail {EMAIL}\n"
        out = _file_content(_scanned(_file_upload(bom + text.encode(codec))))
        assert out == bom + "mail «EMAIL_001»\n".encode(codec)


def test_a_utf16_text_file_may_declare_its_own_charset_only() -> None:
    body = _file_upload(
        b"\xff\xfe" + f"mail {EMAIL}".encode("utf-16-le"),
        headers=b"Content-Type: text/plain; charset=UTF-16\r\n",
    )
    assert b"EMAIL_001" in _file_content(_scanned(body)).replace(b"\x00", b"")
    # A charset naming another encoding than the content's is refused: the
    # upstream would decode what the proxy never read.
    for charset in (b"utf-8", b"utf-32", b"latin-1"):
        mismatched = _file_upload(
            b"\xff\xfe" + b"h\x00i\x00",
            headers=b"Content-Type: text/plain; charset=" + charset + b"\r\n",
        )
        with pytest.raises(UnredactableRequest, match="charset"):
            _scanned(mismatched)
    # A UTF-8 text file keeps today's rule: utf-8 / us-ascii only.
    assert (
        _scanned(_file_upload(b"hi", headers=b"Content-Type: text/plain; charset=us-ascii\r\n"))
        is None
    )
    with pytest.raises(UnredactableRequest, match="charset"):
        _scanned(_file_upload(b"hi", headers=b"Content-Type: text/plain; charset=utf-16\r\n"))


PDF = b"%PDF-1.7\n1 0 obj (" + EMAIL.encode() + b") endobj\n%%EOF\n"


def test_a_binary_file_is_refused_unless_forwarded() -> None:
    body = _file_upload(PDF, filename=f"{EMAIL} report.pdf")
    with pytest.raises(UnredactableRequest, match="an uploaded file is binary"):
        _scanned(body)
    counted: list[int] = []
    out = _scanned(body, forward_binary=counted.append)
    # Byte-identical content; the file NAME is still redacted.
    assert _file_content(out) == PDF
    assert out is not None and EMAIL.encode() not in out.replace(PDF, b"")
    assert "«EMAIL_001» report.pdf".encode() in out
    assert counted == [1]


def test_forwarded_binary_parts_are_counted_once_and_only_when_sent() -> None:
    two = (
        _file_upload(PDF, filename="a.pdf")[: -len(b"--testboundary123--\r\n")]
        + b"--testboundary123\r\n"
        b'Content-Disposition: form-data; name="file"; filename="b.png"\r\n\r\n'
        b"\x89PNG\r\n\x1a\n\x00\r\n--testboundary123--\r\n"
    )
    counted: list[int] = []
    assert _scanned(two, forward_binary=counted.append) is None  # nothing changed
    assert counted == [2]
    # A later piece refusing the upload: nothing was sent, nothing counted.
    refused = two.replace(b'name="purpose"\r\n\r\nuser_data', b'name="user"\r\n\r\n\xff')
    counted.clear()
    with pytest.raises(UnredactableRequest, match="form field"):
        _scanned(refused, forward_binary=counted.append)
    assert counted == []
    # Text only: the callable is not called.
    assert _scanned(_file_upload(b"no secrets"), forward_binary=counted.append) is None
    assert counted == []


def test_a_forwarded_binary_part_still_refuses_a_transfer_encoding() -> None:
    body = _file_upload(PDF, headers=b"Content-Transfer-Encoding: base64\r\n")
    with pytest.raises(UnredactableRequest, match="Content-Transfer-Encoding"):
        _scanned(body, forward_binary=lambda count: None)
    # A binary part's charset is not read (its content never is).
    declared = _file_upload(PDF, headers=b"Content-Type: application/pdf; charset=latin-1\r\n")
    assert _scanned(declared, forward_binary=lambda count: None) is None


def test_a_jsonl_line_nesting_too_deep_still_refuses_the_upload() -> None:
    deep = b'{"a": ' * 200 + b"1" + b"}" * 200
    body = _file_upload(b'{"custom_id": "x"}\n' + deep + b"\n", filename="in.jsonl")
    with pytest.raises(UnredactableRequest, match="JSONL line"):
        _scanned(body, forward_binary=lambda count: None)


def test_a_downloaded_text_file_is_restored() -> None:
    adapter = OpenAIAdapter()
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    rehydrator = Rehydrator(vault)
    path = "/v1/files/file_abc/content"
    # A CSV (fine-tune results, a text file uploaded redacted): as text.
    csv = f"name,email\r\njane,{token}\r\n".encode()
    assert adapter.rehydrate_raw_body(path, csv, rehydrator) == csv.replace(
        token.encode(), EMAIL.encode()
    )
    # UTF-16 with its mark: restored and re-encoded the same way.
    utf16 = b"\xff\xfe" + f"to {token}".encode("utf-16-le")
    assert adapter.rehydrate_raw_body(path, utf16, rehydrator) == b"\xff\xfe" + (
        f"to {EMAIL}".encode("utf-16-le")
    )
    # A text file is restored as ONE text, the way it was redacted: a JSON
    # line among its text lines gets the value exactly as it was sent, never
    # JSON-escaped (the upload redacted the raw characters).
    quoted = vault.placeholder_for("DENY", 'say "hi"')
    mixed = f'prose {token}\r\n  {{"q": "{quoted}"}}\r\nend'.encode()
    restored = adapter.rehydrate_raw_body(path, mixed, rehydrator)
    assert restored == f'prose {EMAIL}\r\n  {{"q": "say "hi""}}\r\nend'.encode()
    # A JSONL file is restored line by line as JSON: the value escaped where
    # it lands, a line's own CR kept.
    jsonl = f'{{"p": "{token}"}}\r\n  {{"q": "{quoted}"}}\r\n'.encode()
    restored = adapter.rehydrate_raw_body(path, jsonl, rehydrator)
    assert restored == f'{{"p": "{EMAIL}"}}\r\n  {{"q": "say \\"hi\\""}}\r\n'.encode()
    # A binary file is never touched, whatever it carries.
    pdf = b"%PDF-1.7 " + token.encode()
    assert adapter.rehydrate_raw_body(path, pdf, rehydrator) is None
    assert adapter.rehydrate_raw_body(path, b"\x89PNG " + token.encode(), rehydrator) is None
    # Nothing to restore: untouched (no re-encoding of an unchanged file).
    assert adapter.rehydrate_raw_body(path, b"\xef\xbb\xbfplain", rehydrator) is None
    assert adapter.rehydrate_raw_body(path, "«NOPE_001» only".encode(), rehydrator) is None
    # A line nesting JSON too deep stays as it came; a scalar JSON line with
    # a token is restored as JSON.
    deep = '{"a": ' * 200 + f'"{token}"' + "}" * 200
    assert adapter.rehydrate_raw_body(path, deep.encode(), rehydrator) is None
    assert adapter.rehydrate_raw_body(path, f'"{token}"'.encode(), rehydrator) == (
        f'"{EMAIL}"'.encode()
    )


# --- integration: real app, fake files/batches upstream ---------------------

received: dict[str, Any] = {}


def _fake_upstream() -> Starlette:
    async def upload(request: Request) -> Response:
        received["upload_raw"] = await request.body()
        received["upload_content_type"] = request.headers.get("content-type", "")
        return JSONResponse({"id": "file_abc", "object": "file", "purpose": "batch"})

    async def create_batch(request: Request) -> Response:
        received["batch_create"] = await request.json()
        return JSONResponse({"id": "batch_1", "status": "in_progress"})

    async def content(request: Request) -> Response:
        # Echo every placeholder seen in the upload as a batch output file.
        boundary = parse_boundary(received["upload_content_type"])
        assert boundary is not None
        parsed = parse(received["upload_raw"], boundary)
        assert parsed is not None
        import re as _re

        # The injected system note itself contains example tokens
        # («TYPE_NNN», «EMAIL_000», a number the vault never issues) — echo
        # only vault-issued EMAIL tokens from the actual message content.
        tokens = _re.findall("«EMAIL_(?!000»)[0-9]+»", parsed.parts[1].content.decode())
        lines = b"".join(
            json.dumps(
                {
                    "custom_id": f"req-{i}",
                    "response": {"status_code": 200, "body": {"content": f"echo {token}"}},
                }
            ).encode()
            + b"\n"
            for i, token in enumerate(dict.fromkeys(tokens))
        )
        return Response(content=lines, media_type="application/octet-stream")

    return Starlette(
        routes=[
            Route("/v1/files", upload, methods=["POST"]),
            Route("/v1/batches", create_batch, methods=["POST"]),
            Route("/v1/files/file_abc/content", content, methods=["GET"]),
        ]
    )


@pytest.fixture
def client() -> httpx.AsyncClient:
    received.clear()
    config = Config(providers={"openai": ProviderConfig(upstream_base_url="http://upstream")})
    app = create_app(config, upstream_transport=httpx.ASGITransport(app=_fake_upstream()))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")


@pytest.mark.anyio
async def test_files_batch_round_trip(client: httpx.AsyncClient) -> None:
    line = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": f"to {EMAIL}"}]},
        }
    )
    uploaded = await client.post(
        "/v1/files",
        data={"purpose": "batch"},
        files={"file": ("input.jsonl", f"{line}\n".encode(), "application/jsonl")},
    )
    assert uploaded.status_code == 200

    # The upstream never saw the real value — and its multipart framing
    # still parses with a standard parser (starlette would have failed the
    # request otherwise; assert on the recorded raw body too).
    assert EMAIL.encode() not in received["upload_raw"]
    assert "«EMAIL_001»".encode() in received["upload_raw"]

    # Batch creation with nothing to redact: forwarded byte-identical.
    batch = await client.post("/v1/batches", json={"input_file_id": "file_abc"})
    assert batch.json()["id"] == "batch_1"
    assert received["batch_create"] == {"input_file_id": "file_abc"}

    # Output download restores the original value.
    output = await client.get("/v1/files/file_abc/content")
    assert output.status_code == 200
    first = json.loads(output.content.splitlines()[0])
    assert first["response"]["body"]["content"] == f"echo {EMAIL}"


def test_pass_through_provider_inference_covers_uploads(tmp_path):
    # /v1/uploads (the multipart Uploads API) has NO adapter — its content
    # is a documented non-goal — but pass-through must still reach the
    # OPENAI upstream. It was missing from the inference prefixes, so
    # Uploads traffic fell through to the anthropic default and hit the
    # wrong provider entirely.
    from llm_redact.config import load_config
    from llm_redact.proxy import ProxyState

    config_path = tmp_path / "config.toml"
    config_path.write_text("")
    state = ProxyState(load_config(config_path), None, config_path=config_path)
    for path in ("/v1/uploads", "/v1/uploads/upload_abc/parts", "/v1/uploads/upload_abc/complete"):
        assert state.provider_for(None, path) == "openai"
    # The siblings and the default stay as they were.
    assert state.provider_for(None, "/v1/batches") == "openai"
    assert state.provider_for(None, "/v1/messages/something-unknown") == "anthropic"
    # Anthropic's beta Files API (same paths, anthropic-version header)
    # still wins over the OpenAI inference.
    assert state.provider_for(None, "/v1/files", {"anthropic-version": "2023-06-01"}) == "anthropic"


@pytest.mark.parametrize(
    "original",
    [
        # A quoted CSV cell: not JSON as sent (``\j``), a JSON string once
        # redacted; restored as the text it was, never JSON-escaped.
        b'account\n"CORP\\jdoe"\nnote: CORP\\jdoe\n',
        # An object line in a text file, likewise.
        b'prose\n{"path": "CORP\\jdoe"}\n',
    ],
    ids=["quoted-cell", "object-line"],
)
async def test_a_text_file_round_trips_byte_for_byte(original: bytes) -> None:
    # Upload through the proxy, the provider stores what it received, the
    # download through the proxy returns exactly the uploaded bytes.
    stored: dict[str, bytes] = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            boundary = parse_boundary(request.headers["content-type"])
            assert boundary is not None
            parsed = parse(request.content, boundary)
            assert parsed is not None
            stored["file"] = parsed.parts[1].content
            return httpx.Response(200, json={"id": "file-1", "object": "file"})
        return httpx.Response(
            200, content=stored["file"], headers={"content-type": "application/octet-stream"}
        )

    config = parse_config({"detection": {"deny": ["CORP\\jdoe"]}}, "t")
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    headers = {"authorization": "Bearer sk-own"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        upload = await client.post(
            "/v1/files",
            content=_file_upload(original, filename="accounts.csv"),
            headers={
                **headers,
                "content-type": f"multipart/form-data; boundary={BOUNDARY.decode()}",
            },
        )
        download = await client.get("/v1/files/file-1/content", headers=headers)
    assert upload.status_code == 200 and download.status_code == 200
    assert b"CORP" not in stored["file"] and "«DENY_001»".encode() in stored["file"]
    assert download.content == original
