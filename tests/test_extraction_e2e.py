"""Binary uploads read by the real extractors, through the real app.

With ``[extraction] enabled``, a PDF or Office file uploaded to a Files API
is read by a real worker process before redaction: a value in its text
refuses the upload (the core cannot redact inside the file) — or, in
convert mode, the file is replaced by its redacted text — a complete clean
reading sends it byte-identical (with the client's own key, and under a
credential the proxy holds only with ``proxy_credential = true``), and a
reading that is not complete (an image-only page, a file named or declared
as another format) keeps the core's unscanned-binary rules.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from document_fixtures import docx, pdf, pdf_objects, pdf_stream, workbook
from fake_router import install, routed_config
from lent_routes import LentRouter
from llm_redact.config import parse_config, parse_extraction
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
FORM = {"authorization": "Bearer sk-client", "content-type": "multipart/form-data; boundary=b"}


_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "rtf": "application/rtf",
}


def _upload(content: bytes, filename: str = "report.pdf", content_type: str | None = None) -> bytes:
    declared = content_type or _TYPES[filename.rpartition(".")[2]]
    return (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nuser_data\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="'
        + filename.encode()
        + b'"\r\nContent-Type: '
        + declared.encode()
        + b"\r\n\r\n"
        + content
        + b"\r\n--b--\r\n"
    )


class Files:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"id": "file-1", "object": "file"})


def _raw(**extraction: Any) -> dict[str, Any]:
    return {"extraction": {"enabled": True, **extraction}}


def _app(raw: dict[str, Any], files: Files) -> Any:
    return create_app(
        parse_config(raw, "config.toml"), upstream_transport=httpx.MockTransport(files)
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _file_sent(request: httpx.Request) -> bytes:
    return request.content.split(b"\r\n\r\n")[-1].rsplit(b"\r\n--b--", 1)[0]


async def test_a_document_holding_a_value_is_refused_and_a_clean_one_sent() -> None:
    files = Files()
    app = _app(_raw(), files)
    dirty_pdf = pdf([f"payroll contact {EMAIL}"])
    clean_pdf = pdf(["quarterly figures only"])
    dirty_docx = docx(["mail jane.", "doe@corp.example"])
    async with _client(app) as client:
        refused = await client.post("/v1/files", content=_upload(dirty_pdf), headers=FORM)
        split = await client.post("/v1/files", content=_upload(dirty_docx, "a.docx"), headers=FORM)
        sent = await client.post("/v1/files", content=_upload(clean_pdf), headers=FORM)
        status = (await client.get("/__llm-redact/status")).json()
    for reply in (refused, split):
        assert reply.status_code == 400
        assert "EMAIL" in reply.text and EMAIL not in reply.text
    assert sent.status_code == 200
    (request,) = files.requests
    assert _file_sent(request) == clean_pdf
    assert status["inspected_uploads_total"] == {"openai": {"clean": 1, "detected": 2}}
    assert status["unscanned_uploads_total"] == {}
    inspector = status["upload_inspector"]["inspector"]
    assert inspector["formats"] == ["pdf", "ooxml", "odf", "html", "rtf"]
    assert inspector["readings_total"] == {
        "local:ooxml": {"complete": 1},
        "local:pdf": {"complete": 2},
    }


@pytest.mark.parametrize("mode", ["forward", "refuse"])
async def test_an_image_only_page_keeps_the_unscanned_rules(mode: str) -> None:
    files = Files()
    raw = {**_raw(), "detection": {"binary_uploads": mode}}
    app = _app(raw, files)
    scan = pdf(["cover page"], image_page=True)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(scan), headers=FORM)
    state = app.state.proxy
    assert state.inspected_uploads == {("openai", "incomplete"): 1}
    if mode == "forward":
        assert reply.status_code == 200 and _file_sent(files.requests[0]) == scan
        assert state.unscanned_uploads == {"openai": 1}
    else:
        assert reply.status_code == 400 and files.requests == []


# A page saying "Hello" whose file-attachment annotation carries a file with
# an e-mail address the page text never shows.
ATTACHED = pdf_objects(
    [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 7 0 R >> >> /Annots [5 0 R] >>",
        pdf_stream(b"", b"BT /F1 12 Tf 72 712 Td (Hello) Tj ET"),
        b"<< /Type /Annot /Subtype /FileAttachment /Rect [0 0 10 10]"
        b" /FS << /Type /Filespec /F (a.txt) /EF << /F 6 0 R >> >> >>",
        pdf_stream(b"/Type /EmbeddedFile", f"contact {EMAIL}".encode()),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
)


# A page whose words are drawn as outlines: it paints, yet has no text layer.
OUTLINES = pdf(["x"], raw_content=b"0 0 1 rg 72 700 m 80 720 l 88 700 l f")


@pytest.mark.parametrize("mode", ["forward", "refuse"])
@pytest.mark.parametrize("document", [ATTACHED, OUTLINES], ids=["attached", "outlines"])
async def test_content_the_reader_cannot_vouch_for_keeps_the_unscanned_rules(
    mode: str, document: bytes
) -> None:
    files = Files()
    app = _app({**_raw(), "detection": {"binary_uploads": mode}}, files)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(document), headers=FORM)
    state = app.state.proxy
    assert state.inspected_uploads == {("openai", "incomplete"): 1}
    if mode == "forward":
        # Sent as without [extraction]: unscanned, and counted so.
        assert reply.status_code == 200 and state.unscanned_uploads == {"openai": 1}
    else:
        assert reply.status_code == 400 and files.requests == []


async def test_a_number_typed_with_non_breaking_hyphens_is_found() -> None:
    files = Files()
    app = _app(_raw(), files)
    document = (
        b'<w:document xmlns:w="urn:w"><w:body><w:p><w:r><w:t>SSN 078</w:t><w:noBreakHyphen/>'
        b"<w:t>05</w:t><w:noBreakHyphen/><w:t>1120</w:t></w:r></w:p></w:body></w:document>"
    )
    upload = _upload(docx(["x"], extra={"word/document.xml": document}), "a.docx")
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=upload, headers=FORM)
    assert reply.status_code == 400 and "SSN" in reply.text and files.requests == []


@pytest.mark.parametrize(
    "between",
    [
        b'<w:del w:id="1" w:author="R"><w:r><w:delText>x</w:delText></w:r></w:del>',
        b'<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:t>QUOTE x</w:t></w:r>'
        b'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        b'<w:r><w:fldChar w:fldCharType="end"/></w:r>',
    ],
    ids=["tracked-deletion", "field-instruction"],
)
async def test_a_value_split_by_a_tracked_deletion_is_found(between: bytes) -> None:
    # Word shows "SSN 123-45-6789" (neither the deletion nor a field's
    # instruction is in the line); the joined reading alone once read
    # "123-45-x6789", scanned clean and sent the file under
    # binary_uploads = "refuse".
    files = Files()
    app = _app({**_raw(), "detection": {"binary_uploads": "refuse"}}, files)
    document = (
        b'<w:document xmlns:w="urn:w"><w:body><w:p><w:r><w:t>SSN 123-45-</w:t></w:r>'
        + between
        + b"<w:r><w:t>6789</w:t></w:r></w:p></w:body></w:document>"
    )
    upload = _upload(docx(["x"], extra={"word/document.xml": document}), "a.docx")
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=upload, headers=FORM)
    assert reply.status_code == 400 and "SSN" in reply.text and files.requests == []
    assert app.state.proxy.inspected_uploads == {("openai", "detected"): 1}


@pytest.mark.parametrize(
    "book",
    [
        # The cell stores 123456789 (bare digits never match); Excel's Special
        # > Social Security Number format shows it as 123-45-6789.
        workbook([("123456789", "000\\-00\\-0000")], ["000\\-00\\-0000"]),
        # A date (6789-01-01 12:03:45) whose format shows 123-45-6789.
        workbook([("1785673.5026041667", "hhm-ss-yyyy")], ["hhm-ss-yyyy"]),
        workbook([("1785673", '"123-45-"yyyy')], ['"123-45-"yyyy']),
    ],
    ids=["special-format", "date-codes", "date-and-literal"],
)
async def test_a_number_shown_through_its_format_is_found(book: bytes) -> None:
    files = Files()
    app = _app({**_raw(), "detection": {"binary_uploads": "refuse"}}, files)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(book, "staff.xlsx"), headers=FORM)
    assert reply.status_code == 400 and "SSN" in reply.text and files.requests == []


@pytest.mark.parametrize(
    ("content", "filename", "declared", "status"),
    [
        # A clean PDF with a Word file appended, uploaded as that Word file:
        # the provider opens the docx (python-docx, Tika) and reads the SSN.
        (
            pdf(["quarterly figures only"]) + docx(["SSN 123-45-6789"]),
            "report.docx",
            None,
            400,
        ),
        # A PDF appended past the first 1024 bytes of a (Windows-1252, so
        # binary) web page, uploaded as statement.pdf: pypdf, poppler and
        # pdf.js find it by its trailer and read the SSN.
        (
            b'<html><head><meta charset="windows-1252"></head><body>Caf\xe9'
            + b" " * 1500
            + b"</body></html>\n"
            + pdf(["SSN 123-45-6789"], compress=True),
            "statement.pdf",
            "application/octet-stream",
            400,
        ),
        # A clean PDF declared as a Word document.
        (pdf(["quarterly figures only"]), "report.docx", None, 400),
        (pdf(["quarterly figures only"]), "report.pdf", None, 200),
        (pdf(["quarterly figures only"]), "report.pdf", "application/octet-stream", 200),
    ],
    ids=["polyglot", "pdf-after-a-page", "declared-docx", "declared-pdf", "generic"],
)
async def test_a_file_read_as_another_format_than_declared_keeps_the_unscanned_rules(
    content: bytes, filename: str, declared: str | None, status: int
) -> None:
    files = Files()
    app = _app({**_raw(), "detection": {"binary_uploads": "refuse"}}, files)
    upload = _upload(content, filename, declared)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=upload, headers=FORM)
    assert reply.status_code == status
    outcome = "clean" if status == 200 else "incomplete"
    assert app.state.proxy.inspected_uploads == {("openai", outcome): 1}


@pytest.mark.parametrize("declared", [None, "application/octet-stream", "application/pdf"])
async def test_a_service_reading_does_not_clear_a_polyglot(declared: str | None) -> None:
    # A clean PDF with a Word file appended, uploaded as report.docx without
    # a Word type: Tika, configured complete for PDFs, reads the PDF by its
    # leading bytes; the provider may open the Word file and its SSN.
    files = Files()
    raw = {**_raw(), "detection": {"binary_uploads": "refuse"}}
    raw["extraction"]["services"] = [
        {"kind": "tika", "url": "http://127.0.0.1:9998", "complete": True, "formats": ["pdf"]}
    ]
    app = _app(raw, files)
    asked: list[httpx.Request] = []

    def tika(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        return httpx.Response(200, json=[{"X-TIKA:content": "quarterly figures only"}])

    app.state.proxy.upload_inspector._transport = httpx.MockTransport(tika)
    polyglot = pdf(["quarterly figures only"]) + docx(["SSN 123-45-6789"])
    upload = _upload(polyglot, "report.docx", declared or "application/octet-stream")
    if declared is None:
        upload = upload.replace(b"Content-Type: application/octet-stream\r\n", b"")
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=upload, headers=FORM)
    assert reply.status_code == 400 and files.requests == [] and len(asked) == 1
    assert app.state.proxy.inspected_uploads == {("openai", "incomplete"): 1}


async def test_a_page_with_a_decoy_charset_is_read_as_declared() -> None:
    # Shift_JIS (so not UTF-8 text: a binary part), a koi8-r mention in a
    # comment ahead of the real meta tag: once decoded as koi8-r, no digit
    # of the full-width SSN was read and the page went out as clean.
    files = Files()
    app = _app({**_raw(), "detection": {"binary_uploads": "refuse"}}, files)
    page = (
        '<html><head><!-- saved as charset=koi8-r --><meta charset="shift_jis"></head>'
        "<body>社員 SSN １２３-４５-６７８９</body></html>"
    ).encode("shift_jis")
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(page, "p.html"), headers=FORM)
    assert reply.status_code == 400 and "SSN" in reply.text and files.requests == []


async def test_a_value_written_as_unicode_escapes_in_rtf_is_found() -> None:
    # Every RTF reader skips the one fallback character after each \uN; the
    # decoder once kept them ("1?2?3?-…") and the file went out as clean.
    files = Files()
    app = _app({**_raw(), "detection": {"binary_uploads": "refuse"}}, files)
    rtf = (
        b"{\\rtf1\\ansi\\ansicpg1252 Caf\xe9 SSN \\u49?\\u50?\\u51?-\\u52?\\u53?-"
        b"\\u54?\\u55?\\u56?\\u57?\\par}"
    )
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(rtf, "a.rtf"), headers=FORM)
    assert reply.status_code == 400 and "SSN" in reply.text and files.requests == []


# An Acrobat form field formatted as a Social Security Number: its value
# holds bare digits, its appearance — what a viewer, and a provider's page
# image, shows — the formatted number.
FORMATTED_FIELD = pdf_objects(
    [
        b"<< /Type /Catalog /Pages 2 0 R /AcroForm << /Fields [5 0 R] >> >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 7 0 R >> >> /Annots [5 0 R] >>",
        pdf_stream(b"", b"BT /F1 12 Tf 72 712 Td (Social Security Number:) Tj ET"),
        b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (ssn) /V (123456789)"
        b" /Rect [72 600 300 620] /AP << /N 6 0 R >> >>",
        pdf_stream(
            b"/Type /XObject /Subtype /Form /BBox [0 0 228 20]"
            b" /Resources << /Font << /F1 7 0 R >> >>",
            b"/Tx BMC BT /F1 12 Tf 2 5 Td (123-45-6789) Tj ET EMC",
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
)


# The page's own text ends "SSN 123-45-"; a field drawn right after it on
# the line shows "6789" — or glyphs "xxxx" stand for it (/ActualText).
_HALF_PAGE = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
    b" /Resources << /Font << /F1 7 0 R >> >> /Annots [5 0 R] >>",
]
JOINED_FIELD = pdf_objects(
    [
        *_HALF_PAGE,
        pdf_stream(b"", b"BT /F1 12 Tf 72 712 Td (SSN 123-45-) Tj ET"),
        b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (last4) /V (6789) /F 4"
        b" /Rect [150 708 200 724] /AP << /N 6 0 R >> >>",
        pdf_stream(
            b"/Type /XObject /Subtype /Form /BBox [0 0 50 16]"
            b" /Resources << /Font << /F1 7 0 R >> >>",
            b"/Tx BMC BT /F1 12 Tf 0 4 Td (6789) Tj ET EMC",
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
)
STOOD_FOR = pdf_objects(
    [
        *_HALF_PAGE,
        pdf_stream(
            b"",
            b"BT /F1 12 Tf 72 712 Td (SSN 123-45-) Tj /Span << /ActualText (6789) >> BDC"
            b" (xxxx) Tj EMC ET",
        ),
        b"<< /Type /Annot /Subtype /Text /Rect [0 0 1 1] >>",
        b"<< >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
)


@pytest.mark.parametrize(
    "document",
    [FORMATTED_FIELD, JOINED_FIELD, STOOD_FOR],
    ids=["field-appearance", "field-on-the-line", "actual-text"],
)
async def test_a_value_shown_only_by_a_field_appearance_is_found(document: bytes) -> None:
    files = Files()
    app = _app({**_raw(), "detection": {"binary_uploads": "refuse"}}, files)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(document), headers=FORM)
    assert reply.status_code == 400 and "SSN" in reply.text and files.requests == []


async def test_a_hostile_file_is_killed_and_counts_as_unread() -> None:
    files = Files()
    app = _app({**_raw(timeout_seconds=0.5), "detection": {"binary_uploads": "refuse"}}, files)
    content = b"BT /F1 12 Tf 72 712 Td " + b"(a) Tj " * 2_000_000 + b"ET"
    hostile = pdf(["x"], raw_content=content, compress=True)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(hostile), headers=FORM)
    assert reply.status_code == 400 and files.requests == []
    inspector = app.state.proxy.upload_inspector
    assert inspector.killed == 1 and inspector.running == 0


def _routed(
    monkeypatch: pytest.MonkeyPatch, files: Files, *, proxy_credential: bool, **extraction: Any
) -> Any:
    """The app behind a (fake) router lending the operator's key: a
    credential the proxy holds."""
    install(monkeypatch, LentRouter("https://api.openai.com"))
    section = parse_extraction(
        {"enabled": True, "proxy_credential": proxy_credential, **extraction}
    )
    return create_app(
        routed_config(extraction=section), upstream_transport=httpx.MockTransport(files)
    )


@pytest.mark.parametrize("allowed", [False, True], ids=["refused", "allowed"])
async def test_an_operator_key_sends_a_clean_file_only_when_configured(
    monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    files = Files()
    app = _routed(monkeypatch, files, proxy_credential=allowed)
    clean_pdf = pdf(["quarterly figures only"])
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(clean_pdf), headers=FORM)
        dirty = await client.post("/v1/files", content=_upload(pdf([EMAIL])), headers=FORM)
    assert dirty.status_code == 400
    if allowed:
        assert reply.status_code == 200
        (request,) = files.requests
        assert _file_sent(request) == clean_pdf
    else:
        assert reply.status_code == 400 and files.requests == []


def test_the_section_needs_no_plugin_and_no_license() -> None:
    config = parse_config(_raw(), "config.toml")
    assert config.extraction.enabled is True and "extraction" not in config.extensions
    assert registry_mod.Registry().build_upload_inspector(config, "free") is not None


@pytest.mark.parametrize("allowed", [False, True], ids=["refused", "allowed"])
async def test_a_service_sees_no_file_the_operator_key_would_refuse_anyway(
    monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    # Under a credential the proxy holds without proxy_credential, the
    # upload is refused whatever the reading says: no service is sent it.
    files = Files()
    app = _routed(
        monkeypatch,
        files,
        proxy_credential=allowed,
        services=[{"kind": "tika", "url": "http://127.0.0.1:9998", "complete": True}],
    )
    asked: list[httpx.Request] = []

    def tika(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        return httpx.Response(200, json=[{"X-TIKA:content": "quarterly figures only"}])

    app.state.proxy.upload_inspector._transport = httpx.MockTransport(tika)
    scan = pdf(["cover page"], image_page=True)  # the local reading is incomplete
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(scan), headers=FORM)
    if allowed:
        assert reply.status_code == 200 and len(asked) == 1
        assert _file_sent(files.requests[0]) == scan
    else:
        assert reply.status_code == 400 and asked == [] and files.requests == []


def _hidden(catalog: bytes = b"", page: bytes = b"", *extra: bytes) -> bytes:
    """A one-page PDF showing only "Hello world", with ``catalog`` and
    ``page`` entries (and ``extra`` objects from 6) no viewer shows."""
    return pdf_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R" + catalog + b" >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
            b" /Resources << /Font << /F1 5 0 R >> >>" + page + b" >>",
            pdf_stream(b"", b"BT /F1 12 Tf 72 712 Td (Hello world) Tj ET"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            *extra,
        ]
    )


_MAIL = EMAIL.encode()


@pytest.mark.parametrize(
    "document",
    [
        _hidden(b" /OpenAction << /S /JavaScript /JS (var m='%s';) >>" % _MAIL),
        _hidden(page=b" /AA << /O << /S /JavaScript /JS (app.alert('%s')) >> >>" % _MAIL),
        _hidden(b" /OpenAction << /S /URI /URI (mailto:%s) >>" % _MAIL),
        _hidden(b" /PieceInfo << /App << /Private (%s) >> >>" % _MAIL),
        _hidden(
            b"",
            b" /Metadata 6 0 R",
            pdf_stream(b"/Type /Metadata /Subtype /XML", b"<x>%s</x>" % _MAIL),
        ),
    ],
    ids=["openaction-js", "page-aa-js", "openaction-uri", "private-data", "page-metadata"],
)
async def test_a_value_no_viewer_shows_still_refuses_the_upload(document: bytes) -> None:
    # binary_uploads = "refuse": a PDF whose only value sits in a script, an
    # action, private data or a page's metadata once read clean and went
    # out byte-identical; the value is in the file the provider receives.
    files = Files()
    app = _app({**_raw(), "detection": {"binary_uploads": "refuse"}}, files)
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(document), headers=FORM)
    assert reply.status_code == 400 and "EMAIL" in reply.text and EMAIL not in reply.text
    assert files.requests == []
    assert app.state.proxy.inspected_uploads == {("openai", "detected"): 1}


async def test_a_stream_no_reader_reads_keeps_the_upload_unscanned() -> None:
    files = Files()
    app = _app({**_raw(), "detection": {"binary_uploads": "refuse"}}, files)
    document = _hidden(b"", b"", pdf_stream(b"", b"note " + _MAIL))
    async with _client(app) as client:
        reply = await client.post("/v1/files", content=_upload(document), headers=FORM)
    assert reply.status_code == 400 and files.requests == []
    assert app.state.proxy.inspected_uploads == {("openai", "incomplete"): 1}
