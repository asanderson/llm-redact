"""Convert mode (``[extraction] convert``) and the file-name extension
(``UploadPart.extension``), through the real app and the real extractors.

A binary upload whose COMPLETE reading holds values to redact is replaced by
its redacted display text — a ``text/plain`` part named ``*.txt`` — on a
route whose provider takes a text file for the upload's purpose, remembered
so its download is restored like a text upload's. Everywhere else (another
route or purpose, an incomplete reading, a block-mode value, a spreadsheet,
a credential the proxy holds without ``proxy_credential``) the upload is
refused as without convert mode. A file whose name's extension names
another format than its bytes is never vouched for."""

from __future__ import annotations

from collections import Counter
from typing import Any

import httpx
import pytest

from document_fixtures import docx, odt, pdf, xlsx
from fake_router import install, routed_config
from lent_routes import LentRouter
from llm_redact import extract_worker, multipart
from llm_redact.cli import _print_posture
from llm_redact.config import parse_config, parse_extraction
from llm_redact.multipart import MultipartPart
from llm_redact.plugin_api import Inspection
from llm_redact.providers import openai
from llm_redact.providers.gemini import GeminiAdapter
from llm_redact.providers.openai import text_file_name, with_content_type
from llm_redact.proxy import create_app
from llm_redact.redactor import BlockedRequest
from llm_redact.upload_inspection import file_extension, judge

EMAIL = "jane.doe@corp.example"
FORM = {"authorization": "Bearer sk-client", "content-type": "multipart/form-data; boundary=b"}
PDF = "application/pdf"


def _upload(
    content: bytes,
    filename: str = "report.pdf",
    content_type: str = PDF,
    *,
    purpose: str | None = "user_data",
    extra: bytes = b"",
) -> bytes:
    field = (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\n'
        + purpose.encode()
        + b"\r\n"
        if purpose is not None
        else b""
    )
    return (
        field
        + b'--b\r\nContent-Disposition: form-data; name="file"; filename="'
        + filename.encode()
        + b'"\r\nContent-Type: '
        + content_type.encode()
        + b"\r\n\r\n"
        + content
        + b"\r\n"
        + extra
        + b"--b--\r\n"
    )


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.stored = b""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET" and request.url.path.endswith("/content"):
            return httpx.Response(200, content=self.stored, headers={"content-type": "text/plain"})
        self.stored = _file_part(request)[1]
        return httpx.Response(200, json={"id": "file-1", "object": "file"})


def _file_part(request: httpx.Request) -> tuple[bytes, bytes]:
    """The (headers, content) of the upload's file part as sent."""
    parsed = multipart.parse(request.content, b"b")
    assert parsed is not None
    for part in parsed.parts:
        if part.filename is not None:
            assert part.headers is not None
            return part.headers, part.content
    raise AssertionError("no file part")


def _app(upstream: Upstream, **extraction: Any) -> Any:
    raw: dict[str, Any] = {"extraction": {"enabled": True, "convert": True, **extraction}}
    detection = extraction.pop("detection", None)
    if detection is not None:
        raw = {"extraction": raw["extraction"], "detection": detection}
        raw["extraction"].pop("detection", None)
    return create_app(parse_config(raw, "t"), upstream_transport=httpx.MockTransport(upstream))


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


@pytest.fixture(autouse=True)
def _fresh_text_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai, "RAW_TEXT_FILES", openai._RawTextFiles())


async def test_a_document_holding_a_value_is_sent_as_its_redacted_text() -> None:
    upstream = Upstream()
    app = _app(upstream)
    document = pdf([f"payroll contact {EMAIL}", "second page"])
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(document), headers=FORM)
        download = await client.get("/v1/files/file-1/content", headers=FORM)
        status = (await client.get("/__llm-redact/status")).json()
    assert reply.status_code == 200
    headers, sent = _file_part(upstream.requests[0])
    assert b'filename="report.txt"' in headers
    assert b"Content-Type: text/plain; charset=utf-8" in headers
    assert PDF.encode() not in headers
    text = sent.decode("utf-8")
    assert EMAIL not in text and "«EMAIL_001»" in text
    assert "payroll contact" in text and "second page" in text
    # The download is restored like a text upload's.
    assert download.status_code == 200 and EMAIL in download.text
    assert status["inspected_uploads_total"] == {"openai": {"converted": 1}}
    assert status["unscanned_uploads_total"] == {}
    assert status["upload_inspector"]["inspector"]["convert"] == [
        "pdf",
        "office",
        "markup",
        "rtf",
        "image",
        "other",
    ]


async def test_a_clean_document_is_still_sent_as_the_original() -> None:
    upstream = Upstream()
    clean = pdf(["quarterly figures only"])
    async with _client(_app(upstream)) as client:
        reply = await client.post("/v1/files", content=_upload(clean), headers=FORM)
    assert reply.status_code == 200 and _file_part(upstream.requests[0])[1] == clean


@pytest.mark.parametrize(
    ("path", "headers", "body"),
    [
        (
            "/v1/containers/cntr_1/files",
            FORM,
            _upload(docx([f"mail {EMAIL}"]), "a.docx", "application/octet-stream", purpose=None),
        ),
        (
            "/v1/files",
            {**FORM, "x-api-key": "k", "anthropic-version": "2023-06-01"},
            _upload(odt([f"mail {EMAIL}"]), "notes.odt", "application/octet-stream", purpose=None),
        ),
        ("/v1/files", FORM, _upload(pdf([f"mail {EMAIL}"]), purpose="assistants")),
    ],
    ids=["container", "anthropic", "assistants"],
)
async def test_routes_that_take_a_text_file(
    path: str, headers: dict[str, str], body: bytes
) -> None:
    upstream = Upstream()
    async with _client(_app(upstream)) as client:
        reply = await client.post(path, content=body, headers=headers)
    assert reply.status_code == 200
    part_headers, sent = _file_part(upstream.requests[0])
    assert b".txt" in part_headers and EMAIL.encode() not in sent and b"EMAIL_001" in sent


@pytest.mark.parametrize(
    ("purpose", "document", "convert"),
    [
        ("vision", pdf([f"mail {EMAIL}"]), True),
        ("batch", pdf([f"mail {EMAIL}"]), True),
        ("fine-tune", pdf([f"mail {EMAIL}"]), True),
        ("user_data", pdf([f"mail {EMAIL}"]), ["office"]),  # not a class it converts
        ("user_data", xlsx([f"mail {EMAIL}"]), True),  # a spreadsheet: no display reading
        ("user_data", pdf([f"mail {EMAIL}"], image_page=True), True),  # incomplete
    ],
    ids=["vision", "batch", "fine-tune", "class", "spreadsheet", "incomplete"],
)
async def test_everything_else_is_refused_as_without_convert(
    purpose: str, document: bytes, convert: Any
) -> None:
    upstream = Upstream()
    app = _app(upstream, convert=convert)
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files",
            content=_upload(document, "report.bin", "application/octet-stream", purpose=purpose),
            headers=FORM,
        )
    assert reply.status_code == 400 and upstream.requests == []
    assert ("openai", "converted") not in app.state.proxy.inspected_uploads


async def test_a_block_mode_value_still_refuses() -> None:
    upstream = Upstream()
    app = _app(upstream, detection={"modes": {"email": "block"}})
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(pdf([EMAIL])), headers=FORM)
    assert reply.status_code == 400 and upstream.requests == []
    assert app.state.proxy.inspected_uploads == {("openai", "blocked"): 1}


async def test_a_converted_part_of_a_refused_upload_counts_as_converted_refused() -> None:
    upstream = Upstream()
    app = _app(upstream, detection={"binary_uploads": "refuse"})
    image = (
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="scan.png"\r\n'
        b"Content-Type: image/png\r\n\r\n\x89PNG\r\n\x1a\n" + bytes(32) + b"\r\n"
    )
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files", content=_upload(pdf([EMAIL]), extra=image), headers=FORM
        )
    assert reply.status_code == 400 and upstream.requests == []
    assert app.state.proxy.inspected_uploads == Counter(
        {("openai", "converted_refused"): 1, ("openai", "incomplete"): 1}
    )


@pytest.mark.parametrize("allowed", [False, True], ids=["refused", "allowed"])
async def test_under_a_proxy_credential_only_with_proxy_credential(
    monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    upstream = Upstream()
    install(monkeypatch, LentRouter("https://api.openai.com"))
    section = parse_extraction({"enabled": True, "convert": True, "proxy_credential": allowed})
    app = create_app(
        routed_config(extraction=section), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(pdf([EMAIL])), headers=FORM)
    if allowed:
        assert reply.status_code == 200
        assert EMAIL.encode() not in _file_part(upstream.requests[0])[1]
    else:
        assert reply.status_code == 400 and upstream.requests == []


def test_the_gemini_upload_is_never_converted() -> None:
    assert GeminiAdapter().converts_upload("/upload/v1beta/files", None) is False  # type: ignore[arg-type]


async def test_status_prints_a_posture_line_for_conversions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _print_posture({"inspected_uploads_total": {"openai": {"converted": 2}}})
    out = capsys.readouterr().out
    assert "openai×2 file part(s) CONVERTED" in out


# --- the file-name extension -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("disposition", "extension"),
    [
        (b'form-data; name="file"; filename="Report.PDF"', "pdf"),
        (b'form-data; name="file"; filename="C:\\\\dir\\\\a.b.docx"', "docx"),
        (b'form-data; name="file"; filename="dir/archive"', None),
        (b'form-data; name="file"; filename=".bashrc"', None),
        (b'form-data; name="file"; filename="x.tar-gz"', None),
        (b'form-data; name="file"; filename="x.abcdefghijk"', None),
        (b'form-data; name="file"; filename="a.pdf"; filename*=UTF-8\'\'a.pdf', "pdf"),
        (b'form-data; name="file"; filename="a.pdf"; filename*=UTF-8\'\'a.html', ""),
        (b'form-data; name="file"; filename="a.pdf"; filename*=UTF-8\'\'a', ""),
        (b'form-data; name="file"', None),
        (b'form-data; name="file"; filename="a.pdf"; filename="b.pdf"', None),
    ],
)
def test_file_extension(disposition: bytes, extension: str | None) -> None:
    part = MultipartPart(b"Content-Disposition: " + disposition, b"%PDF-")
    assert file_extension(part) == extension


@pytest.mark.parametrize(
    ("filename", "content_type", "complete"),
    [
        ("report.pdf", PDF, True),
        ("report.bin", PDF, True),
        ("report", PDF, True),
        ("report.docx", PDF, False),  # a docx reader may open it
        ("report.txt", "application/octet-stream", False),
        ("report.weird", PDF, False),
    ],
)
async def test_a_file_named_as_another_format_is_not_vouched_for(
    filename: str, content_type: str, complete: bool
) -> None:
    upstream = Upstream()
    app = _app(upstream, convert=False, detection={"binary_uploads": "refuse"})
    clean = pdf(["quarterly figures only"])
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files", content=_upload(clean, filename, content_type), headers=FORM
        )
    assert reply.status_code == (200 if complete else 400)
    outcome = "clean" if complete else "incomplete"
    assert app.state.proxy.inspected_uploads == {("openai", outcome): 1}


# --- units -------------------------------------------------------------------------------------


def test_text_file_name() -> None:
    assert text_file_name("report.pdf") == "report.txt"
    assert text_file_name("«EMAIL_001».docx") == "«EMAIL_001».txt"
    assert text_file_name("README") == "README.txt"
    assert text_file_name(".profile") == ".profile.txt"
    assert text_file_name("dir.d/file") == "dir.d/file.txt"


def test_with_content_type() -> None:
    line = b"Content-Type: text/plain; charset=utf-8"
    block = b'Content-Disposition: form-data; name="file"\r\ncontent-type: application/pdf'
    assert with_content_type(block, line) == block.split(b"\r\n")[0] + b"\r\n" + line
    assert with_content_type(b"", line) == line
    sized = block + b"\r\nContent-Length: 99"
    assert with_content_type(sized, line) == block.split(b"\r\n")[0] + b"\r\n" + line


class _Scan:
    """A Redactor stand-in for ``judge``: finds EMAIL in any text holding
    the address, blocks on BLOCK."""

    def scan_text(self, text: str) -> Counter[str]:
        if "BLOCK" in text:
            raise BlockedRequest("EMAIL")
        return Counter({"EMAIL": 1}) if EMAIL in text else Counter()


def test_a_convert_text_past_the_text_budget_is_not_converted() -> None:
    inspection = Inspection(EMAIL, True, "x", convert_text=EMAIL * 10)
    verdict = judge(
        {0: inspection},
        _Scan(),  # type: ignore[arg-type]
        identity=False,
        text_budget=len(EMAIL) * 5,
        convertible=True,
    )
    assert verdict.outcomes == Counter({"detected": 1}) and not verdict.converted


@pytest.mark.parametrize(
    ("inspection", "identity", "convertible", "outcome"),
    [
        (Inspection(EMAIL, True, "x", convert_text=EMAIL), False, True, "converted"),
        (Inspection(EMAIL, True, "x", convert_text=EMAIL), False, False, "detected"),
        (Inspection(EMAIL, True, "x"), False, True, "detected"),
        (Inspection(EMAIL, False, "x", convert_text=EMAIL), False, True, "detected"),
        (Inspection(EMAIL, True, "x", convert_text=EMAIL), True, True, "detected"),
        (
            Inspection(EMAIL, True, "x", proxy_credential=True, convert_text=EMAIL),
            True,
            True,
            "converted",
        ),
        (Inspection("BLOCK", True, "x", convert_text="BLOCK"), False, True, "blocked"),
        (Inspection("clean", True, "x", convert_text="clean"), False, True, "clean"),
    ],
)
def test_judge_converts_only_what_it_may(
    inspection: Inspection, identity: bool, convertible: bool, outcome: str
) -> None:
    verdict = judge(
        {0: inspection},
        _Scan(),  # type: ignore[arg-type]
        identity=identity,
        text_budget=1000,
        convertible=convertible,
    )
    assert verdict.outcomes == Counter({outcome: 1})
    assert dict(verdict.converted) == ({0: EMAIL} if outcome == "converted" else {})
    assert not verdict.detected if outcome != "detected" else verdict.detected


# --- the worker's display reading --------------------------------------------------------------


def _extract(data: bytes) -> dict[str, Any]:
    return extract_worker.extract(
        data, formats=set(extract_worker.FORMATS), max_chars=100_000, max_inflated=1 << 24
    )


def test_display_readings() -> None:
    reading = _extract(pdf(["first page", "second page"]))
    assert reading["complete"] is True
    assert reading["display"].split("\n\n") == ["first page", "second page"]
    document = _extract(docx(["Dear ", "Jane"], ["second"]))
    assert document["display"] == "Dear Jane\nsecond"  # no properties, no relationships
    markup = _extract(
        b'<html><head><meta charset="utf-8"><title>T</title><style>p{}</style>'
        b"<script>var s='x';</script></head>"
        b"<body><p>Hello <b>you</b></p>\n\n  <p>there</p></body></html>"
    )
    assert markup["display"] == "THello you\nthere"
    rtf = _extract(b"{\\rtf1\\ansi hello {\\v hidden}world}")
    assert rtf["display"] == "hello world"
    assert _extract(xlsx(["cell"]))["display"] is None
    assert _extract(pdf([None], image_page=True))["display"] is None


def test_the_display_reading_is_bounded() -> None:
    reading = extract_worker.Reading(10)
    reading.show("12345")
    reading.show("678901")
    assert reading.display() is None and reading.shown == []
    reading.show("more")
    assert reading.display() is None


# --- token floors of the text sent in the file's place ------------------------------------------


def test_the_floors_cover_the_convert_text() -> None:
    # The display reading joins what the full reading keeps apart: the
    # placeholder it carries must still bound the request's new numbers.
    inspection = Inspection(f"«EMAIL_x001» {EMAIL}", True, "x", convert_text=f"«EMAIL_001» {EMAIL}")
    verdict = judge(
        {0: inspection},
        _Scan(),  # type: ignore[arg-type]
        identity=False,
        text_budget=1000,
        convertible=True,
    )
    assert verdict.outcomes == Counter({"converted": 1}) and verdict.floors == {"EMAIL": 1}


async def test_a_converted_file_never_carries_one_number_for_two_values() -> None:
    # Windows-1252 markup (a binary upload) whose shown text holds
    # «EMAIL_001» only once a script is left out: the value it holds must
    # not be numbered 001 too, or the download (and any echo) would
    # restore the address in place of the token the file carried.
    upstream = Upstream()
    app = _app(upstream)
    page = (
        '<html><head><meta charset="windows-1252"></head><body><p>caf\xe9 '
        f"\xabEMAIL_<script>x</script>001\xbb {EMAIL}</p></body></html>"
    ).encode("cp1252")
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files", content=_upload(page, "page.html", "text/html"), headers=FORM
        )
        download = await client.get("/v1/files/file-1/content", headers=FORM)
    assert reply.status_code == 200
    headers, sent = _file_part(upstream.requests[0])
    assert b"text/plain" in headers and EMAIL.encode() not in sent
    assert "«EMAIL_001»".encode() in sent and "«EMAIL_002»".encode() in sent
    assert download.text == f"café «EMAIL_001» {EMAIL}"


@pytest.mark.parametrize("between", [b"<script>x</script>", b"<style>x</style>", b"<!-- c -->"])
async def test_a_value_a_browser_shows_joined_is_found(between: bytes) -> None:
    # A browser shows "SSN 123-45-6789" (no script, style or comment in the
    # line); the full reading alone once read "123-45-x6789", scanned clean
    # and sent the file under binary_uploads = "refuse".
    upstream = Upstream()
    app = _app(upstream, convert=False, detection={"binary_uploads": "refuse"})
    page = (
        b'<html><head><meta charset="windows-1252"></head><body><p>caf\xe9 SSN 123-45-'
        + between
        + b"6789</p></body></html>"
    )
    async with _client(app) as client:
        reply = await client.post(
            "/v1/files", content=_upload(page, "page.html", "text/html"), headers=FORM
        )
    assert reply.status_code == 400 and "SSN" in reply.text and upstream.requests == []
    assert app.state.proxy.inspected_uploads == {("openai", "detected"): 1}
