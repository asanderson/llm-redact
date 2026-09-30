"""Small real documents, generated in the test run (never committed): PDFs
assembled longhand (objects, a cross-reference table with real offsets),
OOXML/ODF packages and zip bombs built with the stdlib ``zipfile``."""

from __future__ import annotations

import io
import zipfile
import zlib


def _pdf_string(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return b"(" + escaped.encode("latin-1") + b")"


def pdf(
    pages: list[str | None],
    *,
    info: dict[str, str] | None = None,
    link: str | None = None,
    image_page: bool = False,
    compress: bool = False,
    raw_content: bytes | None = None,
) -> bytes:
    """A PDF with one page per entry: its text drawn in Helvetica (None: an
    empty page). ``image_page`` adds a page that only draws an image;
    ``link`` a URI link annotation on the first page; ``info`` the document
    information dictionary; ``raw_content`` replaces the first page's
    content stream."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    pages_id = len(objects) + 1 + 0  # placeholder, fixed below
    kids: list[tuple[bytes, bytes]] = []
    for index, text in enumerate(pages):
        if raw_content is not None and index == 0:
            stream = raw_content
        elif text is None:
            stream = b""
        else:
            stream = b"BT /F1 12 Tf 72 712 Td " + _pdf_string(text) + b" Tj ET"
        kids.append((stream, b""))
    image = None
    if image_page:
        image = add(
            b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray"
            b" /BitsPerComponent 8 /Length 1 >>\nstream\n\x80\nendstream"
        )
        kids.append((b"q 100 0 0 100 0 0 cm /Im1 Do Q", b" /XObject << /Im1 %d 0 R >>" % image))
    annot = None
    if link is not None:
        annot = add(
            b"<< /Type /Annot /Subtype /Link /Rect [0 0 10 10] /A << /S /URI /URI "
            + _pdf_string(link)
            + b" >> >>"
        )
    pages_id = len(objects) + 1 + 2 * len(kids)
    for index, (stream, extra_resources) in enumerate(kids):
        if compress:
            data = zlib.compress(stream, 9)
            content = add(
                b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data)
                + data
                + b"\nendstream"
            )
        else:
            content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        annots = b" /Annots [%d 0 R]" % annot if (annot is not None and index == 0) else b""
        page_ids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Contents %d 0 R"
                b" /Resources << /Font << /F1 %d 0 R >>%s >>%s >>"
                % (pages_id, content, font, extra_resources, annots)
            )
        )
    assert (
        add(
            b"<< /Type /Pages /Kids ["
            + b" ".join(b"%d 0 R" % p for p in page_ids)
            + b"] /Count %d >>" % len(page_ids)
        )
        == pages_id
    )
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)
    info_id = None
    if info:
        info_id = add(
            b"<< "
            + b" ".join(b"/" + k.encode() + b" " + _pdf_string(v) for k, v in info.items())
            + b" >>"
        )
    out = io.BytesIO()
    out.write(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    trailer = b"<< /Size %d /Root %d 0 R" % (len(objects) + 1, catalog)
    if info_id is not None:
        trailer += b" /Info %d 0 R" % info_id
    out.write(b"trailer\n" + trailer + b" >>\nstartxref\n%d\n%%%%EOF\n" % xref)
    return out.getvalue()


def pdf_stream(entries: bytes, data: bytes) -> bytes:
    """The body of a PDF stream object: its dictionary ``entries`` (plus
    the length) and ``data``, unfiltered."""
    return b"<< /Length %d %s >>\nstream\n" % (len(data), entries) + data + b"\nendstream"


def pdf_objects(objects: list[bytes], *, packed: frozenset[int] = frozenset()) -> bytes:
    """A PDF assembled longhand from raw object bodies numbered from 1
    (object 1 is the catalog). The objects numbered in ``packed`` go into
    an object stream behind a cross-reference stream (PDF 1.5) — where most
    producers put their dictionaries today; the rest are written plainly."""
    out = io.BytesIO()
    out.write(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number, body in enumerate(objects, start=1):
        if number not in packed:
            offsets[number] = out.tell()
            out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    if not packed:
        xref = out.tell()
        out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
        for number in range(1, len(objects) + 1):
            out.write(b"%010d 00000 n \n" % offsets[number])
        out.write(
            b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref)
        )
        return out.getvalue()
    stream_number = len(objects) + 1
    header, bodies = b"", b""
    for number in sorted(packed):
        header += b"%d %d " % (number, len(bodies))
        bodies += objects[number - 1] + b"\n"
    offsets[stream_number] = out.tell()
    out.write(
        b"%d 0 obj\n" % stream_number
        + pdf_stream(b"/Type /ObjStm /N %d /First %d" % (len(packed), len(header)), header + bodies)
        + b"\nendobj\n"
    )
    xref_number = stream_number + 1
    offsets[xref_number] = out.tell()
    rows = b"\x00" + (0).to_bytes(4, "big") + b"\xff\xff"
    index = {number: position for position, number in enumerate(sorted(packed))}
    for number in range(1, xref_number + 1):
        if number in packed:
            rows += b"\x02" + stream_number.to_bytes(4, "big") + index[number].to_bytes(2, "big")
        else:
            rows += b"\x01" + offsets[number].to_bytes(4, "big") + b"\x00\x00"
    out.write(
        b"%d 0 obj\n" % xref_number
        + pdf_stream(b"/Type /XRef /Size %d /W [1 4 2] /Root 1 0 R" % (xref_number + 1), rows)
        + b"\nendobj\nstartxref\n%d\n%%%%EOF\n" % offsets[xref_number]
    )
    return out.getvalue()


def package(entries: dict[str, bytes], *, stored: tuple[str, ...] = ()) -> bytes:
    """A zip package, entries deflated in order (``stored`` ones not)."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, data in entries.items():
            method = zipfile.ZIP_STORED if name in stored else zipfile.ZIP_DEFLATED
            archive.writestr(zipfile.ZipInfo(name), data, compress_type=method)
    return out.getvalue()


_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_CT = (
    b'<?xml version="1.0" encoding="UTF-8"?><Types'
    b' xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>'
)


def docx(*runs_per_paragraph: list[str], extra: dict[str, bytes] | None = None) -> bytes:
    """A .docx whose paragraphs hold these runs (a value may be split
    across runs, as Word does)."""
    paragraphs = "".join(
        "<w:p>" + "".join(f"<w:r><w:t>{run}</w:t></w:r><w:proofErr/>" for run in runs) + "</w:p>"
        for runs in runs_per_paragraph
    )
    document = f'<?xml version="1.0"?><w:document {_W}><w:body>{paragraphs}</w:body></w:document>'
    return package(
        {
            "[Content_Types].xml": _CT,
            "_rels/.rels": b'<?xml version="1.0"?><Relationships/>',
            "word/document.xml": document.encode(),
            "docProps/core.xml": b'<?xml version="1.0"?><cp:coreProperties'
            b' xmlns:cp="urn:cp" xmlns:dc="urn:dc"><dc:creator>Ada</dc:creator>'
            b"</cp:coreProperties>",
            **(extra or {}),
        }
    )


def xlsx(shared: list[str], sheet_name: str = "Sheet1") -> bytes:
    strings = "".join(f"<si><t>{s}</t></si>" for s in shared)
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    return package(
        {
            "[Content_Types].xml": _CT,
            "xl/workbook.xml": f'<workbook {ns}><sheets><sheet name="{sheet_name}"'
            ' sheetId="1"/></sheets></workbook>'.encode(),
            "xl/sharedStrings.xml": f"<sst {ns}>{strings}</sst>".encode(),
            "xl/worksheets/sheet1.xml": f'<worksheet {ns}><sheetData><row r="1"><c r="A1"'
            ' t="s"><v>0</v></c><c r="B1"><f>SUM(1,2)</f><v>3</v></c></row></sheetData>'
            "</worksheet>".encode(),
        }
    )


def workbook(
    cells: list[tuple[str, str | None]],
    formats: list[str],
    *,
    styles_part: str = "xl/styles.xml",
    extra: dict[str, bytes] | None = None,
    styles_extra: str = "",
    kinds: dict[int, str] | None = None,
    workbook_part: str = "xl/workbook.xml",
    workbook_pr: str = "",
) -> bytes:
    """A .xlsx whose first row holds ``cells`` — (stored value, the custom
    number format code it is shown with, or None) — numeric cells (or of
    the ``t`` type ``kinds`` gives by position) styled through ``cellXfs``
    with the codes in ``formats`` (ids from 164); ``workbook_pr`` the
    workbook's properties element (its date system), in ``workbook_part``,
    written after the sheet."""
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    codes = "".join(
        f'<numFmt numFmtId="{164 + n}" formatCode="{_xml_attribute(code)}"/>'
        for n, code in enumerate(formats)
    )
    xfs = '<xf numFmtId="0"/>' + "".join(f'<xf numFmtId="{164 + n}"/>' for n in range(len(formats)))
    row = "".join(
        f'<c r="{chr(65 + n)}1"'
        + (f' s="{formats.index(code) + 1}"' if code is not None else "")
        + (f' t="{(kinds or {})[n]}"' if n in (kinds or {}) else "")
        + f"><v>{value}</v></c>"
        for n, (value, code) in enumerate(cells)
    )
    return package(
        {
            "[Content_Types].xml": _CT,
            "xl/worksheets/sheet1.xml": f'<worksheet {ns}><sheetData><row r="1">{row}</row>'
            "</sheetData></worksheet>".encode(),
            workbook_part: f'<workbook {ns}>{workbook_pr}<sheets><sheet name="S" sheetId="1"/>'
            "</sheets></workbook>".encode(),
            styles_part: f"<styleSheet {ns}><numFmts>{codes}</numFmts><cellXfs>{xfs}</cellXfs>"
            f"{styles_extra}</styleSheet>".encode(),
            **(extra or {}),
        }
    )


def _xml_attribute(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def pptx(texts: list[str], *, thumbnail: bool = True) -> bytes:
    ns = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
    body = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in texts)
    entries = {
        "[Content_Types].xml": _CT,
        "ppt/presentation.xml": b"<p:presentation xmlns:p='urn:p'/>",
        "ppt/slides/slide1.xml": f"<p:sld xmlns:p='urn:p' {ns}><a:txBody>{body}</a:txBody>"
        "</p:sld>".encode(),
    }
    if thumbnail:
        entries["docProps/thumbnail.jpeg"] = b"\xff\xd8\xff\xe0 not a real jpeg"
    return package(entries)


def odt(paragraphs: list[str], *, picture: bool = False) -> bytes:
    ns = 'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
    body = "".join(f"<text:p>{p}</text:p>" for p in paragraphs)
    entries = {
        "mimetype": b"application/vnd.oasis.opendocument.text",
        "content.xml": f"<office:document-content xmlns:office='urn:o' {ns}><office:body>"
        f"{body}</office:body></office:document-content>".encode(),
        "meta.xml": b"<office:document-meta xmlns:office='urn:o'/>",
        "META-INF/manifest.xml": b"<manifest:manifest xmlns:manifest='urn:m'/>",
        "Thumbnails/thumbnail.png": b"\x89PNG\r\n\x1a\n not a real png",
    }
    if picture:
        entries["Pictures/1.png"] = b"\x89PNG\r\n\x1a\n not a real png"
    return package(entries, stored=("mimetype",))


def zip_bomb(size: int = 64 << 20) -> bytes:
    """A document package whose one XML part inflates to ``size`` bytes."""
    return package({"[Content_Types].xml": _CT, "word/document.xml": b"<" + b"a" * size})
