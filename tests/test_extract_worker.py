"""The document extraction worker's readers, in process, over real small
files generated here (tests/document_fixtures.py): what text each format
yields and when a reading is COMPLETE — the claim that lets the core send a
clean file. The worker's process isolation and limits are exercised for
real in tests/test_extraction.py."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from typing import Any

import pytest

from document_fixtures import (
    docx,
    odt,
    package,
    pdf,
    pdf_objects,
    pdf_stream,
    pptx,
    workbook,
    xlsx,
    zip_bomb,
)
from llm_redact import extract_worker
from llm_redact.extract_worker import (
    FORMATS,
    Reading,
    apply_limits,
    carries_another,
    declared_charset,
    decode_markup,
    detect,
    extract,
    is_markup,
    iso_serial,
    keeps_ascii,
    main,
    number_format,
    render_number,
    xml_text,
)

EMAIL = "jane.doe@corp.example"
ALL = set(FORMATS)


def _read(data: bytes, formats: set[str] = ALL, **limits: int) -> dict[str, Any]:
    return extract(
        data,
        formats=formats,
        max_chars=limits.get("max_chars", 1_000_000),
        max_inflated=limits.get("max_inflated", 64 << 20),
    )


# --- PDF ------------------------------------------------------------------------------


def test_a_text_pdf_is_read_completely() -> None:
    result = _read(
        pdf(
            [f"contact {EMAIL}", None, "page three"],
            info={"Author": "Ada Lovelace", "Title": "Payroll"},
            link="https://files.example/share?token=abc123",
        )
    )
    assert result["format"] == "pdf" and result["complete"] is True
    assert result["pages"] == 3  # what a cloud OCR reading must cover
    for expected in (EMAIL, "page three", "Ada Lovelace", "Payroll", "token=abc123"):
        assert expected in result["text"]


def test_a_page_drawing_an_image_is_incomplete() -> None:
    result = _read(pdf(["text page"], image_page=True))
    assert result["complete"] is False and "text page" in result["text"]
    assert result["pages"] == 2


def test_a_malformed_pdf_is_incomplete_with_what_was_read() -> None:
    result = _read(b"%PDF-1.7\nthis is not a pdf at all\n%%EOF\n")
    assert result["complete"] is False and result["reason"] == "error"
    assert result["pages"] is None  # never opened: its pages unknown


SECRET = b"contact alice.secret@example.com"
HELLO = pdf_stream(b"", b"BT /F1 12 Tf 72 712 Td (Hello) Tj ET")
HELVETICA = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"


def _page(extra: bytes = b"", *, resources: bytes = b"") -> bytes:
    """Object 3: the one page of ``_document``, drawing ``Hello`` (object 4)
    in Helvetica (object 5)."""
    return (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >>" + resources + b" >>" + extra + b" >>"
    )


def _document(
    *more: bytes,
    page: bytes | None = None,
    catalog: bytes = b"",
    packed: frozenset[int] = frozenset(),
    content: bytes | None = None,
) -> bytes:
    """A one-page PDF (catalog 1, pages 2, page 3, content 4 — ``Hello``
    unless ``content`` — and font 5) with ``more`` objects numbered from 6."""
    return pdf_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R" + catalog + b" >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            page if page is not None else _page(),
            HELLO if content is None else pdf_stream(b"", content),
            HELVETICA,
            *more,
        ],
        packed=packed,
    )


def test_the_plain_document_is_complete() -> None:
    result = _read(_document())
    assert result["complete"] is True and result["text"] == "Hello"
    assert _read(_document(packed=frozenset({1, 2, 3, 5})))["complete"] is True


_ATTACHMENT = b"<< /Type /Filespec /F (a.txt) /EF << /F 7 0 R >> >>"
_EMBEDDED = pdf_stream(b"/Type /EmbeddedFile", SECRET)


@pytest.mark.parametrize(
    "document",
    [
        # A file attached to a page (the annotation's own file specification).
        _document(
            b"<< /Type /Annot /Subtype /FileAttachment /Rect [0 0 10 10] /FS "
            + _ATTACHMENT
            + b" >>",
            _EMBEDDED,
            page=_page(b" /Annots [6 0 R]"),
        ),
        # The same, every dictionary packed into an object stream.
        _document(
            b"<< /Type /Annot /Subtype /FileAttachment /Rect [0 0 10 10] /FS "
            + _ATTACHMENT
            + b" >>",
            _EMBEDDED,
            page=_page(b" /Annots [6 0 R]"),
            packed=frozenset({1, 2, 3, 5, 6}),
        ),
        # A PDF 2.0 associated file of the page.
        _document(_ATTACHMENT, _EMBEDDED, page=_page(b" /AF [6 0 R]")),
        # A file specification reached from nowhere the reader walks.
        _document(_ATTACHMENT, _EMBEDDED, catalog=b" /Extra 6 0 R"),
        # An embedded file stream alone.
        _document(_EMBEDDED, catalog=b" /Extra 6 0 R"),
        # A portfolio.
        _document(catalog=b" /Collection << /Type /Collection >>"),
    ],
    ids=["annotation", "object-stream", "associated", "unwalked", "stream", "portfolio"],
)
def test_a_pdf_carrying_a_file_is_incomplete(document: bytes) -> None:
    result = _read(document)
    assert result["complete"] is False and "Hello" in result["text"]


@pytest.mark.parametrize("subtype", [b"RichMedia", b"Movie", b"Sound", b"3D", b"Screen"])
def test_a_media_annotation_is_incomplete(subtype: bytes) -> None:
    annotation = b"<< /Type /Annot /Subtype /" + subtype + b" /Rect [0 0 10 10] >>"
    result = _read(_document(annotation, page=_page(b" /Annots [6 0 R]")))
    assert result["complete"] is False


def test_the_object_walk_is_bounded_in_depth() -> None:
    deep = b"[" * 40 + b"1" + b"]" * 40
    assert _read(_document(catalog=b" /Extra " + deep))["complete"] is False
    shallow = b"[" * 8 + b"1" + b"]" * 8
    assert _read(_document(catalog=b" /Extra " + shallow))["complete"] is True


IMAGE = pdf_stream(
    b"/Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray"
    b" /BitsPerComponent 8",
    b"\x80",
)
INLINE_IMAGE = b"q 10 0 0 10 0 0 cm BI /W 1 /H 1 /CS /G /BPC 8 ID \x80 EI Q"
TO_UNICODE = pdf_stream(
    b"",
    b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /T def"
    b" 1 begincodespacerange <00> <FF> endcodespacerange 1 beginbfchar <01> <0041>"
    b" endbfchar endcmap CMapName currentdict /CMap defineresource pop end end",
)


def _font_document(font: bytes, *more: bytes, text: bytes = b"(Hello)") -> bytes:
    """``_document`` with its font (object 5) replaced and its page showing
    ``text``."""
    return pdf_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            _page(),
            pdf_stream(b"", b"BT /F1 12 Tf 72 712 Td " + text + b" Tj ET"),
            font,
            *more,
        ]
    )


def _type3(extra: bytes = b"") -> bytes:
    return (
        b"<< /Type /Font /Subtype /Type3 /FontBBox [0 0 10 10] /FontMatrix [0.1 0 0 0.1 0 0]"
        b" /CharProcs << /g1 6 0 R >> /Encoding << /Type /Encoding /Differences [1 /g1] >>"
        b" /FirstChar 1 /LastChar 1 /Widths [10]" + extra + b" >>"
    )


@pytest.mark.parametrize(
    "document",
    [
        # Glyphs as outlines: the page paints, yet reads as no text.
        _document(content=b"0 0 1 rg 72 700 m 80 720 l 88 700 l f"),
        # An image painted through a tiling pattern (never a page image).
        pdf_objects(
            [
                b"<< /Type /Catalog /Pages 2 0 R >>",
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
                b" /Resources << /Pattern << /P1 5 0 R >> >> >>",
                pdf_stream(b"", b"/Pattern cs /P1 scn 0 0 612 792 re f"),
                pdf_stream(
                    b"/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1"
                    b" /BBox [0 0 612 792] /XStep 612 /YStep 792"
                    b" /Resources << /XObject << /Im1 6 0 R >> >>",
                    b"q 612 0 0 792 0 0 cm /Im1 Do Q",
                ),
                IMAGE,
            ]
        ),
        # An inline image in a tiling pattern, and in a form.
        _document(
            pdf_stream(b"/Type /Pattern /PatternType 1 /BBox [0 0 1 1]", INLINE_IMAGE),
            catalog=b" /Extra 6 0 R",
        ),
        _document(
            pdf_stream(b"/Type /XObject /Subtype /Form /BBox [0 0 1 1]", INLINE_IMAGE),
            catalog=b" /Extra 6 0 R",
        ),
        # A stamp whose appearance draws an image.
        _document(
            b"<< /Type /Annot /Subtype /Stamp /Rect [0 0 100 100] /AP << /N 7 0 R >> >>",
            pdf_stream(
                b"/Type /XObject /Subtype /Form /BBox [0 0 100 100]"
                b" /Resources << /XObject << /Im1 8 0 R >> >>",
                b"q 100 0 0 100 0 0 cm /Im1 Do Q",
            ),
            IMAGE,
            page=_page(b" /Annots [6 0 R]"),
        ),
        # An appearance showing text it gives no font for: read as nothing.
        _document(
            b"<< /Type /Annot /Subtype /Widget /Rect [0 0 1 1] /AP << /N 7 0 R >> >>",
            pdf_stream(b"/Subtype /Form /BBox [0 0 1 1]", b"BT (value) Tj ET"),
            page=_page(b" /Annots [6 0 R]"),
        ),
        # A glyph procedure showing text of its own.
        _font_document(
            _type3(b" /ToUnicode 7 0 R"),
            pdf_stream(b"", b"10 0 d0 BT /F1 1 Tf (x) Tj ET"),
            TO_UNICODE,
            text=b"(\x01)",
        ),
        # A PostScript XObject.
        _document(pdf_stream(b"/Type /XObject /Subtype /PS", b"showpage"), catalog=b" /E 6 0 R"),
        # Fonts pypdf cannot map to the text their glyphs show.
        _font_document(_type3(), pdf_stream(b"", b"10 0 d0 0 0 10 10 re f"), text=b"(\x01)"),
        _font_document(
            _type3(b" /ToUnicode 7 0 R"),
            pdf_stream(b"", b"10 0 d0 " + INLINE_IMAGE),
            TO_UNICODE,
            text=b"(\x01)",
        ),
        _font_document(
            b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H"
            b" /DescendantFonts [6 0 R] >>",
            b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /X"
            b" /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> >>",
            text=b"<0041>",
        ),
        _font_document(b"<< /Type /Font /Subtype /TrueType /BaseFont /Arial >>"),
        _font_document(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding /MacExpertEncoding >>"
        ),
        _font_document(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding << /BaseEncoding /Custom >> >>"
        ),
        _font_document(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /X"
            b" /Encoding << /Differences [72 /H /g7] >> >>"
        ),
        # Found through the page's resources: no /Type /Font to go by.
        _font_document(b"<< /Subtype /TrueType /BaseFont /Arial >>"),
    ],
    ids=[
        "outlines",
        "pattern-image",
        "pattern-inline-image",
        "form-inline-image",
        "stamp-image",
        "unreadable-appearance",
        "type3-glyph-text",
        "postscript",
        "type3",
        "type3-glyph-image",
        "identity-cids",
        "no-encoding",
        "encoding-name",
        "base-encoding",
        "glyph-name",
        "untyped-font",
    ],
)
def test_content_the_text_layer_does_not_show_is_incomplete(document: bytes) -> None:
    assert _read(document)["complete"] is False


@pytest.mark.parametrize(
    "document",
    [
        # Painting nothing reads as nothing.
        _document(content=b"q 1 0 0 1 0 0 cm Q"),
        # An appearance drawing no text or image.
        _document(
            b"<< /Type /Annot /Subtype /Highlight /Rect [0 0 1 1] /AP << /N 7 0 R >> >>",
            pdf_stream(b"/Subtype /Form /BBox [0 0 1 1]", b"1 1 0 rg 0 0 1 1 re f"),
            page=_page(b" /Annots [6 0 R]"),
        ),
        # Fonts pypdf maps: a ToUnicode map, standard encodings and glyph names.
        _font_document(
            _type3(b" /ToUnicode 7 0 R"),
            pdf_stream(b"", b"10 0 d0 0 0 10 10 re f"),
            TO_UNICODE,
            text=b"(\x01)",
        ),
        _font_document(
            b"<< /Type /Font /Subtype /TrueType /BaseFont /Arial /Encoding /WinAnsiEncoding >>"
        ),
        _font_document(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /X /Encoding"
            b" << /BaseEncoding /MacRomanEncoding /Differences [72 /H /eacute] >> >>"
        ),
        # A resource dictionary naming a mapped font directly.
        _font_document(HELVETICA.replace(b"/Type /Font ", b"")),
    ],
    ids=[
        "paints-nothing",
        "highlight",
        "type3-mapped",
        "win-ansi",
        "differences",
        "untyped-mapped-font",
    ],
)
def test_content_the_text_layer_shows_is_complete(document: bytes) -> None:
    result = _read(document)
    assert result["complete"] is True, result


def _form(content: bytes, entries: bytes = b"") -> bytes:
    """A form XObject drawing ``content`` in Helvetica (object 5)."""
    return pdf_stream(
        b"/Type /XObject /Subtype /Form /BBox [0 0 300 100] /Resources << /Font << /F1 5 0 R >> >>"
        + entries,
        content,
    )


_SSN = b"BT /F1 12 Tf 5 50 Td (SSN 123-45-6789) Tj ET"


@pytest.mark.parametrize(
    "document",
    [
        # A form field whose value is harmless but whose appearance, what
        # every viewer and page rasterizer draws, shows the SSN.
        _document(
            b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (name) /V (harmless)"
            b" /Rect [72 600 300 620] /AP << /N 7 0 R >> >>",
            _form(_SSN),
            page=_page(b" /Annots [6 0 R]"),
            catalog=b" /AcroForm << /Fields [6 0 R] >>",
        ),
        # Acrobat's Special > Social Security Number format: /V holds bare
        # digits (never matched), the appearance the formatted number.
        _document(
            b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (ssn) /V (123456789)"
            b" /Rect [72 600 300 620] /AP << /N 7 0 R >>"
            b" /AA << /F << /S /JavaScript /JS (AFSpecial_Format\\(3\\);) >> >> >>",
            _form(b"/Tx BMC BT /F1 12 Tf 2 5 Td (123-45-6789) Tj ET EMC"),
            page=_page(b" /Annots [6 0 R]"),
            catalog=b" /AcroForm << /Fields [6 0 R] >>",
        ),
        # A link, and free text (its down state), drawing words.
        _document(
            b"<< /Type /Annot /Subtype /Link /Rect [0 0 300 100] /AP << /N 7 0 R >>"
            b" /A << /S /URI /URI (https://example.org/) >> >>",
            _form(_SSN),
            page=_page(b" /Annots [6 0 R]"),
        ),
        _document(
            b"<< /Type /Annot /Subtype /FreeText /Rect [0 0 300 100]"
            b" /AP << /D << /On 7 0 R >> >> >>",
            _form(_SSN),
            page=_page(b" /Annots [6 0 R]"),
        ),
        # A tiling pattern whose cell draws the words, filling a rectangle.
        _document(
            pdf_stream(
                b"/Type /Pattern /PatternType 1 /PaintType 1 /TilingType 1 /BBox [0 0 300 20]"
                b" /XStep 612 /YStep 792 /Resources << /Font << /F1 5 0 R >> >>",
                _SSN,
            ),
            page=_page(resources=b" /Pattern << /P1 6 0 R >>"),
            content=b"BT /F1 12 Tf 72 712 Td (Cover letter) Tj ET /Pattern cs /P1 scn"
            b" 72 600 300 20 re f",
        ),
        # A form drawn only as a soft mask.
        _document(_form(_SSN), catalog=b" /Extra << /Type /Mask /S /Luminosity /G 6 0 R >>"),
        # Glyphs standing for other text: copy and paste read the ActualText.
        _document(
            content=b"BT /F1 12 Tf 72 712 Td /Span << /ActualText (SSN 123-45-6789) >> BDC"
            b" (see attached) Tj EMC ET"
        ),
        # The same through a named property list and the structure tree.
        _document(
            page=_page(resources=b" /Properties << /MC0 << /ActualText (SSN 123-45-6789) >> >>"),
            content=b"BT /F1 12 Tf 72 712 Td /Span /MC0 BDC (see attached) Tj EMC ET",
        ),
        _document(catalog=b" /StructTreeRoot << /K << /S /Span /Alt (SSN 123-45-6789) >> >>"),
        # A field value, and an annotation's rich contents, as text streams.
        _document(
            b"<< /FT /Tx /T (ssn) /V 7 0 R >>",
            pdf_stream(b"", b"SSN 123-45-6789"),
            catalog=b" /AcroForm << /Fields [6 0 R] >>",
        ),
        _document(
            b"<< /Type /Annot /Subtype /Text /Rect [0 0 1 1] /RC 7 0 R >>",
            pdf_stream(b"", b"\xfe\xff" + "SSN 123-45-6789".encode("utf-16-be")),
            page=_page(b" /Annots [6 0 R]"),
        ),
    ],
    ids=[
        "widget-appearance",
        "acrobat-ssn-format",
        "link-appearance",
        "freetext-appearance",
        "tiling-pattern",
        "soft-mask",
        "actual-text",
        "named-actual-text",
        "structure-alt",
        "field-value-stream",
        "rich-contents-stream",
    ],
)
def test_text_drawn_or_stood_for_outside_the_page_content_is_read(document: bytes) -> None:
    result = _read(document)
    assert result["complete"] is True, result
    assert "123-45-6789" in result["text"]


_HALF = b"BT /F1 12 Tf 72 712 Td (SSN 123-45-) Tj ET"
_FOUR = pdf_stream(
    b"/Type /XObject /Subtype /Form /BBox [0 0 50 16] /Resources << /Font << /F1 5 0 R >> >>",
    b"BT /F1 12 Tf 0 4 Td (6789) Tj ET",
)


def _widget(
    rect: bytes = b"150 708 200 724", entries: bytes = b" /AP << /N 7 0 R >>", flags: int = 4
) -> bytes:
    return (
        b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (last4) /V (6789) /F %d /Rect [%s]%s >>"
        % (
            flags,
            rect,
            entries,
        )
    )


def _stands_for(marked: bytes, **more: Any) -> bytes:
    """A page showing ``SSN 123-45-`` and then ``marked`` (glyphs inside
    marked content whose ActualText is the rest)."""
    return _document(content=b"BT /F1 12 Tf 72 712 Td (SSN 123-45-) Tj " + marked + b" ET", **more)


@pytest.mark.parametrize(
    "document",
    [
        # A form field drawn right after the page's own text, on its line.
        _document(_widget(), _FOUR, page=_page(b" /Annots [6 0 R]"), content=_HALF),
        # The same through the appearance state the annotation shows.
        _document(
            _widget(entries=b" /AP << /N << /On 7 0 R /Off 8 0 R >> >> /AS /On"),
            _FOUR,
            pdf_stream(b"/Type /XObject /Subtype /Form /BBox [0 0 50 16]", b""),
            page=_page(b" /Annots [6 0 R]"),
            content=_HALF,
        ),
        # A form XObject the page draws on the same line.
        _document(
            _FOUR,
            page=_page(resources=b" /XObject << /Fm1 6 0 R >>"),
            content=_HALF + b" q 1 0 0 1 150 708 cm /Fm1 Do Q",
        ),
        # Glyphs standing for the rest (/ActualText), inline, in a named
        # property list (shown once over two strings), over TJ and the
        # quote operators, for a form's own marked content, and through a
        # structure element standing for its MCID.
        _stands_for(b"/Span << /ActualText (6789) >> BDC (xxxx) Tj EMC"),
        _stands_for(
            b"/Span /MC0 BDC (xx) Tj (xx) Tj EMC",
            page=_page(resources=b" /Properties << /MC0 << /ActualText (6789) >> >>"),
        ),
        _stands_for(b"/Span << /ActualText (6789) >> BDC [(x) -20 (x)] TJ (x) ' 1 2 (x) \" EMC"),
        _document(
            pdf_stream(
                b"/Type /XObject /Subtype /Form /BBox [0 0 50 16] /Resources << /Font"
                b" << /F1 5 0 R >> /Properties << /P << /ActualText (6789) >> >> >>",
                b"BT /F1 12 Tf 0 4 Td /Span /P BDC (xxxx) Tj EMC ET",
            ),
            page=_page(resources=b" /XObject << /Fm1 6 0 R >>"),
            content=_HALF + b" q 1 0 0 1 150 708 cm /Fm1 Do Q",
        ),
        _stands_for(
            b"/Span << /MCID 0 >> BDC (xxxx) Tj EMC",
            catalog=b" /StructTreeRoot << /K [<< /S /P /Pg 3 0 R /K [<< /S /Span"
            b" /ActualText (6789) /K 0 >>] >> << /Type /MCR /Pg 3 0 R /MCID 1 >>] >>",
        ),
    ],
    ids=[
        "widget-on-the-line",
        "appearance-state",
        "form-on-the-line",
        "actual-text",
        "named-actual-text",
        "actual-text-operators",
        "form-actual-text",
        "structure-actual-text",
    ],
)
def test_a_value_is_read_as_the_page_shows_it(document: bytes) -> None:
    # Two drawings meeting on a line, or glyphs and the text they stand
    # for, were read apart ("SSN 123-45-" and "6789"): clean and complete.
    result = _read(document)
    assert result["complete"] is True, result
    assert "SSN 123-45-6789" in result["text"]


@pytest.mark.parametrize(
    "widget",
    [
        _widget(rect=b"150 300 200 316"),
        _widget(flags=2),
        _widget(rect=b"150 708 150 724"),
    ],
    ids=["another-line", "hidden", "degenerate"],
)
def test_drawings_on_other_lines_or_not_shown_are_not_joined(widget: bytes) -> None:
    result = _read(_document(widget, _FOUR, page=_page(b" /Annots [6 0 R]"), content=_HALF))
    assert result["complete"] is True and "6789" in result["text"]
    assert "SSN 123-45-6789" not in result["text"]


def test_malformed_annotations_and_resources_are_no_failure() -> None:
    # A null annotation, an appearance dictionary that is a number, a form
    # name the resources do not hold: nothing to place, nothing failed.
    result = _read(
        _document(
            b"<< /Type /Annot /Subtype /Widget /Rect [0 0 9 9] /AP 5 /F (x) >>",
            page=_page(b" /Annots [null 6 0 R]", resources=b" /XObject << /Im 7 >>"),
            content=_HALF + b" /Missing Do /Span /Nowhere BDC (x) Tj EMC",
        )
    )
    assert result["reason"] == "ok" and "SSN 123-45-" in result["text"]


def test_forms_drawing_forms_are_followed_a_bounded_depth() -> None:
    # A form that draws itself: pypdf skips the cycle, the row reading
    # stops at its depth and form budget.
    looping = pdf_stream(
        b"/Type /XObject /Subtype /Form /BBox [0 0 50 16] /Resources << /Font << /F1 5 0 R >>"
        b" /XObject << /Fm1 6 0 R >> >>",
        b"BT /F1 12 Tf 0 4 Td (6789) Tj ET /Fm1 Do",
    )
    result = _read(
        _document(
            looping,
            page=_page(resources=b" /XObject << /Fm1 6 0 R >>"),
            content=_HALF + b" q 1 0 0 1 150 708 cm /Fm1 Do Q",
        )
    )
    assert result["complete"] is True and "SSN 123-45-6789" in result["text"]


def test_javascript_that_may_redraw_a_field_is_incomplete() -> None:
    field = (
        b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (ssn) /V (123456789) /Rect [0 0 1 1]"
        b" /AA << /F << /S /JavaScript /JS (AFSpecial_Format\\(3\\);) >> >>%s >>"
    )
    appearance = _form(b"BT /F1 12 Tf 2 5 Td (123-45-6789) Tj ET")
    drawn = field % b" /AP << /N 7 0 R >>"
    # The viewer regenerates the appearance (its format script runs), or
    # draws a widget that has none: what it shows is not read.
    for document in (
        _document(field % b"", page=_page(b" /Annots [6 0 R]")),
        _document(
            drawn,
            appearance,
            page=_page(b" /Annots [6 0 R]"),
            catalog=b" /AcroForm << /Fields [6 0 R] /NeedAppearances true >>",
        ),
    ):
        assert _read(document)["complete"] is False
    # A widget without an appearance and no script shows its value (read).
    plain = b"<< /Type /Annot /Subtype /Widget /FT /Tx /T (ssn) /V (123456789) /Rect [0 0 1 1] >>"
    assert _read(_document(plain, page=_page(b" /Annots [6 0 R]")))["complete"] is True


def test_pdf_strings_follow_dictionaries_and_arrays_but_not_back_links() -> None:
    from pypdf.generic import (
        ArrayObject,
        ByteStringObject,
        DictionaryObject,
        NameObject,
        TextStringObject,
    )

    deep: Any = TextStringObject("too deep")
    for _ in range(8):
        deep = ArrayObject([deep])
    annotation = DictionaryObject(
        {
            NameObject("/Contents"): TextStringObject("note"),
            NameObject("/T"): ByteStringObject(b"author"),
            NameObject("/Parent"): TextStringObject("never followed"),
            NameObject("/Kids"): ArrayObject([TextStringObject("kid"), deep]),
        }
    )
    reading = Reading(1000)
    extract_worker._pdf_strings(annotation, reading)
    assert reading.pieces == ["note", "author", "kid"]


def test_outline_titles_nest() -> None:
    class Item:
        def __init__(self, title: str) -> None:
            self.title = title

    reading = Reading(1000)
    nested: Any = [Item("deep")]
    for _ in range(20):
        nested = [nested]
    extract_worker._outline_titles([Item("one"), [Item("two"), nested]], reading)
    assert reading.pieces == ["one", "two"]


# --- document packages --------------------------------------------------------------------


def test_a_value_split_across_runs_is_read_whole() -> None:
    result = _read(docx(["jane.", "doe@corp.example"], ["next"]))
    assert result["format"] == "ooxml" and result["complete"] is True
    assert EMAIL in result["text"]  # the joined reading
    assert "Ada" in result["text"]  # document properties too


_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_MC = 'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'
_REVISION = 'w:id="1" w:author="Reviewer" w:date="2026-01-01T00:00:00Z"'


def _word(body: str) -> bytes:
    """A .docx whose document body is ``body`` (WordprocessingML)."""
    document = f'<?xml version="1.0"?><w:document {_W} {_MC}><w:body>{body}</w:body></w:document>'
    return docx(["x"], extra={"word/document.xml": document.encode()})


def _run(text: str, properties: str = "") -> str:
    return f'<w:r>{properties}<w:t xml:space="preserve">{text}</w:t></w:r>'


_ODF_NS = (
    "xmlns:office='urn:oasis:names:tc:opendocument:xmlns:office:1.0'"
    " xmlns:text='urn:oasis:names:tc:opendocument:xmlns:text:1.0'"
    " xmlns:style='urn:oasis:names:tc:opendocument:xmlns:style:1.0'"
    " xmlns:dc='http://purl.org/dc/elements/1.1/'"
)


def _open_document(body: str, styles: str = "") -> bytes:
    content = (
        f"<office:document-content {_ODF_NS}><office:automatic-styles>{styles}"
        f"</office:automatic-styles><office:body><office:text>{body}</office:text>"
        "</office:body></office:document-content>"
    )
    return package(
        {"mimetype": b"application/vnd.oasis.opendocument.text", "content.xml": content.encode()},
        stored=("mimetype",),
    )


_SHOWN = "SSN 123-45-6789"


@pytest.mark.parametrize(
    "document",
    [
        # A digit corrected with Track Changes on: Word shows the new one.
        _word(
            "<w:p>"
            + _run("SSN 123-45-678")
            + f"<w:del {_REVISION}><w:r><w:delText>0</w:delText></w:r></w:del>"
            + f"<w:ins {_REVISION}>{_run('9')}</w:ins></w:p>"
        ),
        # A tracked deletion, and a tracked move away, between the halves.
        _word(
            "<w:p>"
            + _run("SSN 123-45-")
            + f"<w:del {_REVISION}><w:r><w:delText>x</w:delText></w:r></w:del>"
            + _run("6789")
            + "</w:p>"
        ),
        _word(
            "<w:p>"
            + _run("SSN 123-45-")
            + f"<w:moveFrom {_REVISION}>{_run('moved')}</w:moveFrom>"
            + _run("6789")
            + "</w:p>"
        ),
        # A field whose instruction sits between the halves (empty result).
        _word(
            "<w:p>"
            + _run("SSN 123-45-")
            + '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            + '<w:r><w:instrText xml:space="preserve"> QUOTE "" </w:instrText></w:r>'
            + '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            + '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            + _run("6789")
            + "</w:p>"
        ),
        # Text runs, not instrText, between a field's begin and separate are
        # its instruction too; and a field nested in a field's result.
        _word(
            "<w:p>"
            + _run("SSN 123-45-")
            + '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            + _run("QUOTE x")
            + '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            + '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            + _run("6789")
            + "</w:p>"
        ),
        _word(
            "<w:p>"
            + _run("SSN 123-")
            + '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            + _run("IF 1 = 1")
            + '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            + _run("45-")
            + '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            + _run("PAGE")
            + '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            + '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            + '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            + _run("6789")
            + "</w:p>"
        ),
        # Hyphens drawn as symbols (a symbol font's private-use code too).
        _word(
            "<w:p>"
            + _run("SSN 123")
            + '<w:r><w:sym w:font="Arial" w:char="002D"/></w:r>'
            + _run("45")
            + '<w:r><w:sym w:font="Symbol" w:char="F02D"/></w:r>'
            + _run("6789")
            + "</w:p>"
        ),
        # A hidden run, ruby text and a text box between the halves.
        _word(
            "<w:p>"
            + _run("SSN 123-45-")
            + _run("x", "<w:rPr><w:vanish/></w:rPr>")
            + _run("6789")
            + "</w:p>"
        ),
        _word(
            "<w:p>"
            + _run("SSN 123-45-")
            + f"<w:r><w:ruby><w:rt>{_run('x')}</w:rt><w:rubyBase>{_run('67')}</w:rubyBase>"
            + "</w:ruby></w:r>"
            + _run("89")
            + "</w:p>"
        ),
        _word(
            "<w:p>"
            + _run("SSN 123-45-")
            + "<w:r><mc:AlternateContent><mc:Choice Requires='wps'><w:drawing>"
            + "<w:txbxContent><w:p>"
            + _run("box")
            + "</w:p></w:txbxContent></w:drawing></mc:Choice><mc:Fallback><w:pict>"
            + _run("box")
            + "</w:pict></mc:Fallback></mc:AlternateContent></w:r>"
            + _run("6789")
            + "</w:p>"
        ),
        # A paragraph whose mark is deleted, or hidden, joins the next.
        _word(
            f"<w:p><w:pPr><w:rPr><w:del {_REVISION}/></w:rPr></w:pPr>"
            + _run("SSN 123-45-")
            + "</w:p><w:p>"
            + _run("6789")
            + "</w:p>"
        ),
        _word(
            "<w:p><w:pPr><w:rPr><w:specVanish/></w:rPr></w:pPr>"
            + _run("SSN 123-45-")
            + "</w:p><w:p>"
            + _run("6789")
            + "</w:p>"
        ),
        # A phonetic run in a spreadsheet's shared string.
        package(
            {
                "[Content_Types].xml": b"<Types/>",
                "xl/sharedStrings.xml": b"<sst><si><r><t>SSN 123-45-</t></r>"
                b'<rPh sb="0" eb="1"><t>x</t></rPh><r><t>6789</t></r></si></sst>',
            }
        ),
        # OpenDocument writes a comment inline, at the start of its range.
        _open_document(
            "<text:p>SSN 123-45-<office:annotation office:name='c1'><dc:creator>Ada"
            "</dc:creator><dc:date>2026-01-01T00:00:00</dc:date><text:p>check these digits"
            "</text:p></office:annotation>6789<office:annotation-end office:name='c1'/></text:p>"
        ),
    ],
    ids=[
        "tracked-correction",
        "tracked-deletion",
        "tracked-move",
        "field-instruction",
        "field-instruction-runs",
        "nested-field",
        "symbols",
        "hidden-run",
        "ruby",
        "text-box",
        "deleted-paragraph-mark",
        "hidden-paragraph-mark",
        "phonetic-run",
        "odf-annotation",
    ],
)
def test_a_value_is_read_as_shown_around_what_the_line_does_not_show(document: bytes) -> None:
    # The joined reading pastes the hidden text into the value; the shown
    # reading joins the text around it, as Word, Excel and LibreOffice do.
    result = _read(document)
    assert result["complete"] is True, result
    assert _SHOWN in result["text"]


def test_what_the_line_does_not_show_is_still_read_on_its_own() -> None:
    result = _read(
        _open_document(
            "<text:p>Total <office:annotation><text:p>ask jane.doe@corp.example</text:p>"
            "</office:annotation>due</text:p>"
        )
    )
    assert "Total due" in result["text"] and "\nask jane.doe@corp.example\n" in result["text"]
    # Hidden only when on: w:val="0" shows the run.
    shown = _read(
        _word(
            "<w:p>"
            + _run("SSN 123-45-")
            + _run("x", '<w:rPr><w:vanish w:val="0"/></w:rPr>')
            + _run("6789")
            + "</w:p>"
        )
    )
    assert shown["complete"] is True and _SHOWN not in shown["text"]


@pytest.mark.parametrize(
    "document",
    [
        docx(
            ["x"],
            extra={
                "word/styles.xml": f'<w:styles {_W}><w:style w:styleId="h"><w:rPr><w:vanish/>'
                "</w:rPr></w:style></w:styles>".encode()
            },
        ),
        docx(
            ["x"],
            extra={
                "word/styles.xml": f"<w:styles {_W}><w:docDefaults><w:rPrDefault><w:rPr>"
                "<w:vanish/></w:rPr></w:rPrDefault></w:docDefaults></w:styles>".encode()
            },
        ),
        _open_document(
            "<text:p>x</text:p>",
            "<style:style style:name='T1'><style:text-properties text:display='none'/>"
            "</style:style>",
        ),
    ],
    ids=["word-style", "word-defaults", "odf-style"],
)
def test_a_style_hiding_text_is_incomplete(document: bytes) -> None:
    # Which runs use the style would need style resolution: not vouched for.
    assert _read(document)["complete"] is False
    visible = _open_document(
        "<text:p>x</text:p>",
        "<style:style style:name='T1'><style:text-properties text:display='true'/></style:style>",
    )
    assert _read(visible)["complete"] is True


@pytest.mark.parametrize(
    "body",
    [
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>' + _run("x"),
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>' + _run("x"),
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:fldChar w:fldCharType="separate"/>'
        '</w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r>',
        '<w:r><w:sym w:font="Arial" w:char="zz"/></w:r>',
        '<w:r><w:sym w:font="Arial" w:char="0007"/></w:r>',
    ],
    ids=["open-field", "stray-end", "second-separate", "unreadable-symbol", "control-symbol"],
)
def test_a_field_or_symbol_not_resolved_is_incomplete(body: str) -> None:
    # A field never closed or boundaries out of order, a symbol whose
    # character cannot be read: what the line shows is not sure.
    assert _read(_word(f"<w:p>{body}</w:p>"))["complete"] is False
    # DrawingML's symbol font choice (no character) stands for nothing.
    assert _read(_word('<w:p><w:r><w:rPr><a:sym typeface="Wingdings"/></w:rPr></w:r></w:p>'))[
        "complete"
    ]


def test_spreadsheet_sheet_names_strings_and_formulas() -> None:
    result = _read(xlsx([EMAIL], sheet_name="Payroll 2026"))
    assert result["complete"] is True
    for expected in (EMAIL, "Payroll 2026", "SUM(1,2)"):
        assert expected in result["text"]


@pytest.mark.parametrize(
    ("code", "stored", "shown"),
    [
        # Excel's Special formats: Social Security Number, phone, ZIP + 4.
        ("000\\-00\\-0000", "123456789", "123-45-6789"),
        ("[<=9999999]###\\-####;\\(###\\)\\ ###\\-####", "4155550132", "(415) 555-0132"),
        ("00000\\-0000", "941051234", "94105-1234"),
        # Literal text holding digits, spaces between digit groups, colours.
        ('"123-45-"0000', "6789", "123-45-6789"),
        ('"+1 415 555 "0000', "132", "+1 415 555 0132"),
        ("0000 0000 0000 0000", "4111111111111111", "4111 1111 1111 1111"),
        ("[Red]000-00-0000", "1.23456789E8", "123-45-6789"),
        # Sections, signs, zero, decimals, percent, scaling, General.
        ("000-00-0000;(000-00-0000)", "-123456789", "(123-45-6789)"),
        ("000-00-0000", "-123456789", "-123-45-6789"),
        ('0-0;0-0;"zero" 0-0', "0", "zero 0-0"),
        ("0 0.0#", "12.5", "1 2.5"),
        ("0 0.00", "12.345", "1 2.35"),
        (".00 0", "12.5", "12.50 0"),
        ("0-0%", "0.25", "2-5%"),
        ("0-0,", "25000", "2-5"),
        ('"ID-"#-General', "7", None),
        ('"+"General', "14155550132", "+14155550132"),
        ("#-#", "0", "-"),
        ("??-??", "5", "  - 5"),
        # Conditions choose the section; none holding shows nothing.
        ("[<=9999999]###\\-####;\\(###\\)\\ ###\\-####", "5550132", "555-0132"),
        ("[>100]0-0;[<0]0 0;0.0", "5", None),
        ("[>100]0-0;[<0]0 0", "5", None),
        ("[<0]0 0;0-0", "-12", "1 2"),
    ],
)
def test_number_formats_are_rendered_as_a_spreadsheet_shows_them(
    code: str, stored: str, shown: str | None
) -> None:
    assert render_number(code, stored) == shown


@pytest.mark.parametrize(
    ("code", "verdict"),
    [
        ('"$"#,##0.00', "plain"),
        ('#,##0.00 "kr"', "plain"),
        ('_("$"* #,##0.00_);_("$"* \\(#,##0.00\\);_("$"* "-"??_);_(@_)', "plain"),
        ("0.00E+00", "plain"),
        # Dates and times: at most four digits need no rendering; written as
        # dates are (each number once, text between them) a date; digits
        # in the text, a unit twice or numbers side by side join values.
        ("yyyy-mm-dd", "date"),
        ("mmm-yy", "plain"),
        ("h:mm", "plain"),
        ("m/d/yyyy h:mm AM/PM", "date"),
        ("[$-F800]dddd, mmmm dd, yyyy", "date"),
        ("[h]:mm:ss", "date"),
        ("h:mm:ss.000", "date"),
        ("[ss].00", "date"),
        ('"123-45-"yyyy', "render"),
        ('[<=99999]"123-45-"yyyy', "render"),
        ("hhm-ss-yyyy", "render"),
        ("yyyy mmdd hhmm ssyy", "render"),
        ("yyyymmdd", "render"),
        ("[h]:mm yyyy", "render"),
        # Codes not rendered: eras, another calendar or digit script, digit
        # placeholders beside a date.
        ("e/m/d", "uncertain"),
        ("B2dd/mm/yyyy", "uncertain"),
        ("[$-2010409]yyyy-mm-dd", "uncertain"),
        ("yyyy 0000", "uncertain"),
        ("ss.0 0", "uncertain"),
        ("000-00-0000", "render"),
        ('"123-45-"@', "uncertain"),
        ("# ?/?", "uncertain"),
        ("[>100]000-00-0000", "render"),
        ("[>100]000-00-0000;0 0", "render"),
        ("[>=1E3]000-00-0000", "uncertain"),
        ("#,##0-0000", "uncertain"),
        ("000-00-yyyy", "uncertain"),
    ],
)
def test_number_format_verdicts(code: str, verdict: str) -> None:
    assert number_format(code) == verdict


# Serials of 6789-01-01 12:03:45 and 4111-11-11 11:11:11 (a hair past, so
# rounding and truncating agree), 2026-09-30 23:59:59.
_SSN_TIME = "1785673.5026041667"
_CARD_TIME = "807867.46609953704"


@pytest.mark.parametrize(
    ("code", "stored", "date1904", "shown"),
    [
        # A date code beside digits in literal text, or numbers side by side.
        ('"123-45-"yyyy', _SSN_TIME, False, ["123-45-6789"]),
        ("hhm-ss-yyyy", _SSN_TIME, False, ["123-45-6789"]),
        ("yyyy mmdd hhmm ssyy", _CARD_TIME, False, ["4111 1111 1111 1111"]),
        # Months, minutes, twelve-hour clocks, names, elapsed and fractional time.
        ("m/d/yyyy h:mm:ss AM/PM", "46295.99999", False, ["9/30/2026 11:59:59 PM"]),
        ("mm/dd/yy hh:mm a/p", "46295.25", False, ["09/30/26 06:00 a"]),
        ("dddd, mmmm d, yyyy", "46295", False, ["Wednesday, September 30, 2026"]),
        ("ddd mmm mmmmm dd yyyy", "46295", False, ["Wed Sep S 30 2026"]),
        ("[h]:mm:ss", "1.5", False, ["36:00:00"]),
        ("[mm]:ss yyyy", "1.5", False, ["2160:00 1900"]),
        ("mm:ss.00", "0.00001", False, ["00:00.86"]),
        # Rounded and truncated where they differ.
        (
            "yyyy-mm-dd hh:mm:ss",
            "46295.999999",
            False,
            ["2026-10-01 00:00:00", "2026-09-30 23:59:59"],
        ),
        # The 1900 system's 0 January and 29 February; the 1904 system.
        ("yyyy-mm-dd", "0", False, ["1900-01-00"]),
        ("yyyy-mm-dd", "1", False, ["1900-01-01"]),
        ("dddd yyyy-mm-dd", "60", False, ["Wednesday 1900-02-29"]),
        ("yyyy-mm-dd", "61", False, ["1900-03-01"]),
        ("yyyy-mm-dd", "0", True, ["1904-01-01"]),
        ("yyyy-mm-dd", "-1", True, ["-1904-01-02"]),
        # No date shown: a negative date in the 1900 system, past 9999, four
        # digits or fewer, a code not rendered.
        ("yyyy-mm-dd", "-1", False, None),
        ("yyyy-mm-dd", "2958466", False, None),
        ("mmm-yy", "46295", False, None),
        ("e/m/d", "46295", False, None),
    ],
)
def test_dates_are_rendered_as_a_spreadsheet_shows_them(
    code: str, stored: str, date1904: bool, shown: list[str] | None
) -> None:
    rendered = render_number(code, stored, date1904=date1904)
    assert (rendered.split("\n") if rendered is not None else None) == shown


@pytest.mark.parametrize(
    ("text", "date1904", "serial"),
    [
        ("2026-09-30", False, "46295"),
        ("2026-09-30T12:00:00", False, "46295.5"),
        ("1900-01-01", False, "1"),
        ("1900-02-28", False, "59"),
        ("1904-01-02", True, "1"),
        ("not a date", False, None),
    ],
)
def test_an_iso_date_cell_is_its_serial(text: str, date1904: bool, serial: str | None) -> None:
    assert iso_serial(text, date1904) == serial


def test_a_date_cell_is_read_as_its_format_shows_it() -> None:
    book = workbook(
        [(_SSN_TIME, '"123-45-"yyyy'), ("6789-01-01T12:03:45", "hhm-ss-yyyy")],
        ['"123-45-"yyyy', "hhm-ss-yyyy"],
        kinds={1: "d"},
    )
    result = _read(book)
    assert result["complete"] is True
    assert result["text"].count("123-45-6789") == 2
    # The workbook's 1904 date system, read before the sheets whatever the
    # zip order.
    in_1904 = workbook(
        [("1784211.5026041667", "hhm-ss-yyyy")],
        ["hhm-ss-yyyy"],
        workbook_pr='<workbookPr date1904="1"/>',
    )
    result = _read(in_1904)
    assert result["complete"] is True and "123-45-6789" in result["text"]


def test_a_cell_is_read_as_its_number_format_shows_it() -> None:
    # Excel shows 123-45-6789 for the stored 123456789; bare digits never
    # match, so the stored number alone scanned clean.
    book = workbook(
        [("123456789", "000\\-00\\-0000"), ("4155550132", "(000) 000-0000"), ("5", '"$"0.00')],
        ["000\\-00\\-0000", "(000) 000-0000", '"$"0.00'],
    )
    result = _read(book)
    assert result["complete"] is True
    assert "123-45-6789" in result["text"] and "(415) 555-0132" in result["text"]
    assert "$5.00" not in result["text"]  # nothing to render: not read twice
    assert render_number("0-0", "not a number") is None
    assert render_number("0-0", "1E+40") is None


@pytest.mark.parametrize(
    "book",
    [
        # A text format joining literal digits to the cell's text.
        workbook([("0", '"123-45-"@')], ['"123-45-"@']),
        # A conditional format's number format: which cells it shows is not resolved.
        workbook(
            [("123456789", None)],
            [],
            styles_extra='<dxfs><dxf><numFmt numFmtId="170" formatCode="000-00-0000"/></dxf>'
            "</dxfs>",
        ),
        # A chart showing its values in such a format.
        workbook(
            [("1", None)],
            [],
            extra={
                "xl/charts/chart1.xml": b"<c:chartSpace xmlns:c='urn:c'><c:numCache>"
                b"<c:formatCode>000-00-0000</c:formatCode><c:pt idx='0'><c:v>123456789</c:v>"
                b"</c:pt></c:numCache></c:chartSpace>"
            },
        ),
        # A styles part under another name: its formats would reach no cell.
        workbook([("123456789", "000-00-0000")], ["000-00-0000"], styles_part="xl/s.xml"),
        # A second styles part: which format a cell's style index finds is
        # not resolved (a decoy listed first shifts every index).
        workbook(
            [("123456789", "000\\-00\\-0000")],
            ["000\\-00\\-0000"],
            extra={
                "docProps/styles.xml": b"<styleSheet><cellXfs><xf numFmtId='0'/>"
                b"<xf numFmtId='0'/></cellXfs></styleSheet>"
            },
        ),
        # The date system named only after a date was shown.
        workbook(
            [(_SSN_TIME, "hhm-ss-yyyy")],
            ["hhm-ss-yyyy"],
            workbook_part="xl/zbook.xml",
            workbook_pr='<workbookPr date1904="true"/>',
        ),
        # A date cell whose value is no date; a date code not rendered.
        workbook([("soon", "yyyy-mm-dd")], ["yyyy-mm-dd"], kinds={0: "d"}),
        workbook([("46295", "e/m/d")], ["e/m/d"]),
        # A chart showing its values in a date format that joins values.
        workbook(
            [("1", None)],
            [],
            extra={
                "xl/charts/chart1.xml": b"<c:chartSpace xmlns:c='urn:c'><c:numCache>"
                b"<c:formatCode>hhm-ss-yyyy</c:formatCode></c:numCache></c:chartSpace>"
            },
        ),
    ],
    ids=[
        "text-format",
        "conditional",
        "chart",
        "relocated-styles",
        "second-styles",
        "late-date-system",
        "not-a-date",
        "era",
        "chart-date",
    ],
)
def test_a_number_format_not_rendered_is_incomplete(book: bytes) -> None:
    assert _read(book)["complete"] is False


def test_a_chart_date_written_as_dates_are_is_complete() -> None:
    chart = b"<c:chartSpace xmlns:c='urn:c'><c:formatCode>m/d/yyyy</c:formatCode></c:chartSpace>"
    assert _read(workbook([("1", None)], [], extra={"xl/charts/chart1.xml": chart}))["complete"]


def test_a_thumbnail_is_derived_but_any_other_image_is_unread() -> None:
    assert _read(pptx(["slide text"]))["complete"] is True
    assert _read(odt(["para"]))["complete"] is True
    pictured = _read(odt(["para"], picture=True))
    assert pictured["format"] == "odf" and pictured["complete"] is False
    assert "para" in pictured["text"]
    embedded = _read(docx(["x"], extra={"word/embeddings/oleObject1.bin": b"\xd0\xcf"}))
    assert embedded["complete"] is False


@pytest.mark.parametrize(
    "part",
    [
        b'<w:document xmlns:w="urn:w"><w:p><w:binData w:name="i.png">iVBORw0K</w:binData>'
        b"</w:p></w:document>",
        b'<office:document-content xmlns:office="urn:o"><office:binary-data>iVBORw0K'
        b"</office:binary-data></office:document-content>",
    ],
    ids=["wordml", "opendocument"],
)
def test_base64_data_inside_an_xml_part_is_unread(part: bytes) -> None:
    assert xml_text(part)[1] is False
    assert _read(docx(["x"], extra={"word/document.xml": part}))["complete"] is False


def test_non_breaking_hyphens_and_spaces_are_read_as_shown() -> None:
    document = (
        b'<w:document xmlns:w="urn:w"><w:body><w:p><w:r><w:t>SSN 123</w:t><w:noBreakHyphen/>'
        b"<w:t>45</w:t><w:noBreakHyphen/><w:t>6789 in</w:t><w:softHyphen/><w:t>voice</w:t>"
        b"</w:r></w:p></w:body></w:document>"
    )
    result = _read(docx(["x"], extra={"word/document.xml": document}))
    assert result["complete"] is True
    assert "SSN 123-45-6789 invoice" in result["text"]
    rtf = _read(b"{\\rtf1\\ansi SSN 123\\_45\\_6789\\~in\\-voice \\{x\\}\\\\}")
    assert "SSN 123-45-6789 invoice {x}\\" in rtf["text"]


def test_attribute_values_are_read_namespace_declarations_are_not() -> None:
    text, whole = xml_text(
        b'<r xmlns:a="urn:x"><a:link href="https://h.example/?k=v" a:t="1"/></r>'
    )
    assert "https://h.example/?k=v" in text and "urn:x" not in text and whole


def test_whitespace_elements_and_blocks() -> None:
    text, _ = xml_text(b"<d><p>a<tab/>b<s/>c</p><p>d<br/>e</p></d>")
    assert text.split("\n")[0] == "a\tb c"


def test_an_xml_part_is_collected_only_up_to_the_text_cap() -> None:
    text, whole = xml_text(b"<d><p>" + b"x" * 1000 + b"</p><p>tail</p></d>", budget=100)
    assert not whole and "tail" not in text and len(text) < 100
    result = _read(docx(["y" * 500]), max_chars=300)
    assert result["complete"] is False and len(result["text"]) <= 300


@pytest.mark.parametrize(
    "document",
    [
        b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]><r>&a;</r>',
        b'<!DOCTYPE r SYSTEM "http://evil.example/x.dtd"><r/>',
    ],
)
def test_a_document_type_declaration_is_refused(document: bytes) -> None:
    result = _read(package({"[Content_Types].xml": b"<Types/>", "word/document.xml": document}))
    assert result == {"format": "ooxml", "text": None, "complete": False, "reason": "refused"}


def test_zip_bombs_hit_the_limits() -> None:
    assert _read(zip_bomb(8 << 20))["reason"] == "limit"  # inflation ratio
    stored = package(
        {"[Content_Types].xml": b"<T/>", "word/document.xml": b"<a>" + b"x" * 5000 + b"</a>"},
        stored=("word/document.xml",),
    )
    assert _read(stored, max_inflated=1000)["reason"] == "limit"  # total inflated size
    assert _read(stored)["complete"] is True
    many = io.BytesIO()
    with zipfile.ZipFile(many, "w") as archive:
        for n in range(extract_worker.MAX_ENTRIES + 1):
            archive.writestr(f"{n}.xml", b"")
    assert _read(many.getvalue())["reason"] == "limit"


def test_encrypted_entries_are_unread() -> None:
    data = bytearray(docx(["x"]))
    # Set the "encrypted" flag on every central-directory entry.
    at = 0
    while (at := data.find(b"PK\x01\x02", at)) != -1:
        data[at + 8] |= 0x1
        at += 4
    result = _read(bytes(data))
    assert result["complete"] is False


def test_other_archives_and_disabled_formats_are_unsupported() -> None:
    plain_zip = package({"notes.txt": b"hello"})
    assert _read(plain_zip) == {
        "format": "zip",
        "text": None,
        "complete": False,
        "reason": "unsupported",
    }
    assert _read(odt(["x"]), {"ooxml"})["reason"] == "unsupported"
    assert _read(docx(["x"]), {"pdf"})["reason"] == "unsupported"
    assert _read(pdf(["x"]), {"ooxml"})["reason"] == "unsupported"
    assert _read(b"\x89PNG\r\n\x1a\n")["format"] == "other"


# --- markup and RTF ---------------------------------------------------------------------


def test_markup_in_a_declared_charset_is_complete() -> None:
    page = b"<html><meta charset='windows-1252'><body>caf\xe9 jane.doe&#64;corp.example</body>"
    result = _read(page)
    assert result["format"] == "html" and result["complete"] is True
    assert EMAIL in result["text"] and "café" in result["text"]


@pytest.mark.parametrize(
    ("data", "exact"),
    [
        ("<?xml version='1.0'?><r>é</r>".encode("utf-16-le"), True),
        ("<?xml version='1.0'?><r>é</r>".encode("utf-16-be"), True),
        (b"<html>no charset \xe9</html>", False),
        (b"<html><meta charset='klingon'>\xe9</html>", False),
        (b"<?xml version='1.0' encoding='utf-8'?><r>\xff</r>", False),
    ],
)
def test_markup_decoding(data: bytes, exact: bool) -> None:
    assert decode_markup(data)[1] is exact
    assert is_markup(data)


@pytest.mark.parametrize("charset", [b"utf-16", b"utf-32", b"cp037", b"cp1026", b"utf-7"])
def test_a_declared_charset_that_does_not_keep_ascii_is_not_exact(charset: bytes) -> None:
    page = b'<html><head><meta charset="' + charset + b'"></head><body>mail '
    page += EMAIL.encode() + b" +14155550132 caf\xe9</body></html>"
    page += b" " * (len(page) % 2)
    result = _read(page)
    assert result["complete"] is False
    assert EMAIL in result["text"] and "+14155550132" in result["text"]
    assert not keeps_ascii(charset.decode()) and not keeps_ascii("no-such-codec")
    assert keeps_ascii("shift_jis") and keeps_ascii("windows-1252")


@pytest.mark.parametrize(
    ("page", "charset", "agrees"),
    [
        (b"<?xml version='1.0' encoding='koi8-r'?><r/>", "koi8-r", True),
        (b"<html><meta charset='shift_jis'>", "shift_jis", True),
        (
            b'<html><meta http-equiv="Content-Type" content="text/html; charset=euc-jp">',
            "euc-jp",
            True,
        ),
        (b"<html><!-- charset=sjis --><meta charset='shift_jis'>", "shift_jis", True),
        # A mention that is not a declaration: never the charset.
        (b"<html><!-- <meta charset='koi8-r'> --><body>", None, False),
        (b"<html><script>var charset='koi8-r'</script><body>", None, False),
        (b" <r><?xml version='1.0' encoding='koi8-r'?></r>", None, False),
        (b"<html><body>" + b"x" * 1024 + b"<meta charset='koi8-r'>", None, False),
        # A decoy ahead of the real declaration: declared, but unsure.
        (b"<html><!-- saved as charset=koi8-r --><meta charset='shift_jis'>", "shift_jis", False),
        (b"<html><meta charset='shift_jis'><!-- charset=x-no-such-codec -->", "shift_jis", False),
    ],
)
def test_the_charset_comes_from_a_real_declaration(
    page: bytes, charset: str | None, agrees: bool
) -> None:
    assert declared_charset(page) == (charset, agrees)


def test_a_decoy_charset_is_read_as_declared_but_not_vouched_for() -> None:
    # Shift_JIS with a full-width SSN; a koi8-r mention in a comment ahead of
    # the meta tag once chose the decoding (koi8-r keeps ASCII: "exact").
    body = "SSN １２３-４５-６７８９"
    page = (
        '<html><head><!-- saved as charset=koi8-r --><meta charset="shift_jis"></head>'
        f"<body>{body}</body></html>"
    ).encode("shift_jis")
    result = _read(page)
    assert result["complete"] is False and body in result["text"]
    honest = page.replace(b"koi8-r", b"sjis")
    assert _read(honest)["complete"] is True


def test_the_raw_bytes_are_read_too() -> None:
    # A Shift_JIS lead byte swallows the address's first letter in the
    # declared decoding; the raw (Latin-1) reading keeps every ASCII byte.
    page = b"<html><meta charset='shift_jis'><body>mail \x81" + EMAIL.encode() + b"</body>"
    text, exact = decode_markup(page)
    assert exact and EMAIL not in text
    result = _read(page)
    assert result["complete"] is True and EMAIL in result["text"]
    # Plain ASCII is not read twice.
    ascii_page = b"<html><meta charset='utf-8'><body>plain</body></html>"
    assert _read(ascii_page)["text"].count("plain") == 2  # the source and its text


def test_markup_with_an_inline_image_is_incomplete() -> None:
    page = b"<html><meta charset='latin-1'><img src='data:image/png;base64,AAAA' alt='x\xe9'>"
    assert _read(page)["complete"] is False


@pytest.mark.parametrize(
    "page",
    [
        b"<html><meta charset='utf-8'><iframe src='data:text/html;base64,PGI+'></iframe>",
        b"<?xml version='1.0' encoding='utf-8'?><w:wordDocument xmlns:w='urn:w'>"
        b"<w:binData w:name='wordml://1.png'>iVBORw0K</w:binData></w:wordDocument>",
        "<html><img src='data:image/png;base64,AAAA'></html>".encode("utf-16-le"),
    ],
    ids=["data-uri", "wordml-2003", "utf-16"],
)
def test_markup_inlining_binary_data_is_incomplete(page: bytes) -> None:
    assert _read(page)["complete"] is False


def test_rtf_best_effort() -> None:
    source = b"{\\rtf1\\ansi\\ansicpg1252 caf\\'e9 \\u8364? mail " + EMAIL.encode()
    source += b"\\par\\tab x\\{y\\}}"
    result = _read(source)
    assert result["format"] == "rtf" and result["complete"] is True
    assert "café" in result["text"] and "€" in result["text"] and EMAIL in result["text"]
    assert _read(b"{\\rtf1 {\\pict\\pngblip 89504e47}}")["complete"] is False
    assert _read(b"{\\rtf1\\ansi\\ansicpg99999 text}")["complete"] is False


_POLYGLOT_DOCX = docx(["SSN 123-45-6789"])


@pytest.mark.parametrize(
    ("data", "kind", "carries"),
    [
        (pdf(["x"]), "pdf", False),
        (pdf(["x"]) + b"\r\n\x00 \n", "pdf", False),
        # A zip appended to a PDF: zip readers find its end record at the end.
        (pdf(["x"]) + _POLYGLOT_DOCX, "pdf", True),
        (pdf(["x"]) + b"<html>SSN 123-45-6789</html>", "pdf", True),
        (pdf(["x"]).replace(b"%%EOF", b"%%EOX"), "pdf", True),
        (_POLYGLOT_DOCX, "zip", False),
        (_POLYGLOT_DOCX + b"<html>x</html>", "zip", True),
        (b"PK\x03\x04 no end record", "zip", True),
        # PDF readers find a PDF by its header, or by its trailer from the
        # end whatever precedes or follows it.
        (b"<html>" + pdf(["x"]), "html", True),
        (b"<html>" + b" " * 2048 + pdf(["x"]), "html", True),
        (b"<html>" + b" " * 2048 + pdf(["x"]) + b"x" * 100_000, "html", True),
        (b"<html>" + b" " * 2048 + pdf(["x"]).replace(b"%PDF-", b"%XXX-"), "html", True),
        (b"\x89PNG\r\n\x1a\n" + bytes(4096) + b"trailer<<>>\nstartxref\n9\n", "other", True),
        (b"\x89PNG\r\n\x1a\n" + bytes(4096) + b"%%EOF", "other", True),
        (b"<html>" + _POLYGLOT_DOCX, "html", True),
        (b"<html>plain</html>", "html", False),
        (b"\x89PNG\r\n\x1a\n" + bytes(4096), "other", False),
        (b"{\\rtf1 x}\r\n", "rtf", False),
        (b"{\\rtf1 x}<html>SSN</html>", "rtf", True),
        (b"{\\rtf1 {x}", "rtf", False),
        (b"{\\rtf1 x}" + pdf(["x"]), "rtf", True),
    ],
    ids=lambda value: f"{len(value)}-bytes" if isinstance(value, bytes) else None,
)
def test_a_file_carrying_another_format_is_found(data: bytes, kind: str, carries: bool) -> None:
    assert carries_another(data, kind) is carries


def test_a_polyglot_is_read_but_not_vouched_for() -> None:
    # A clean PDF with a .docx appended: python-docx, openpyxl and Tika open
    # the docx the file's name or type points them at.
    result = _read(pdf(["quarterly figures"]) + _POLYGLOT_DOCX)
    assert result["format"] == "pdf" and result["complete"] is False
    assert "quarterly figures" in result["text"]
    trailing = _read(_POLYGLOT_DOCX + pdf(["x"]))
    assert trailing["complete"] is False and "SSN 123-45-6789" in trailing["text"]


def test_a_zip_reader_s_names_and_comments_are_read() -> None:
    commented = io.BytesIO()
    with zipfile.ZipFile(commented, "w") as archive:
        archive.comment = b"SSN 123-45-6789"
        archive.writestr("[Content_Types].xml", b"<Types/>")
        entry = zipfile.ZipInfo("word/document.xml")
        entry.comment = b"jane.doe@corp.example"
        archive.writestr(entry, b"<w:document xmlns:w='urn:w'/>")
    result = _read(commented.getvalue())
    assert result["complete"] is True
    for expected in ("SSN 123-45-6789", EMAIL, "word/document.xml"):
        assert expected in result["text"]


_RTF_SSN = "SSN 123-45-6789"


def _full_width(text: str) -> bytes:
    """``text`` in Shift_JIS as RTF escapes (every byte past ASCII)."""
    return b"".join(b"\\'%02x" % c if c > 0x7F else bytes([c]) for c in text.encode("cp932"))


@pytest.mark.parametrize(
    ("document", "shown"),
    [
        # Every reader skips the \ucN fallback characters after a \uN.
        (b"{\\rtf1\\ansi SSN \\u49?\\u50?\\u51?-\\u52?\\u53?-\\u54?\\u55?\\u56?\\u57?}", _RTF_SSN),
        (b"{\\rtf1\\ansi\\uc2 SSN \\u49\\'81\\'82\\u50\\'81\\'82\\u51??-45-6789}", _RTF_SSN),
        (b"{\\rtf1\\ansi\\uc0 SSN \\u49\\u50\\u51-45-6789}", _RTF_SSN),
        # A control word counts as one fallback character, and so does a
        # character of text.
        (b"{\\rtf1\\ansi SSN \\u49\\u50\\u51 -45-6789}", "SSN 1345-6789"),
        # A group's end ends its fallback too.
        (b"{\\rtf1\\ansi SSN 12{\\u51}-45-6789}", _RTF_SSN),
        # A double-byte code page: both bytes of a character decoded together.
        (
            b"{\\rtf1\\ansi\\ansicpg932 " + _full_width("SSN １２３-４５-６７８９") + b"}",
            "SSN １２３-４５-６７８９",
        ),
        # A font's own character set.
        (
            b"{\\rtf1\\ansi\\ansicpg1252{\\fonttbl{\\f0\\fcharset0 Arial;}"
            b"{\\f1\\fcharset128 Mincho;}}\\f1 " + _full_width("SSN １２３-４５-６７８９") + b"}",
            "SSN １２３-４５-６７８９",
        ),
        (
            b"{\\rtf1\\ansi{\\fonttbl{\\f1\\cpg932 Mincho;}}\\f1 "
            + _full_width("１２３")
            + b"-45-6789}",
            "１２３-45-6789",
        ),
        # Hidden and deleted text, a field's instruction, a footnote and a
        # shape between the halves: shown without them, the halves join.
        (b"{\\rtf1\\ansi SSN 123-45-{\\v x}6789}", _RTF_SSN),
        (b"{\\rtf1\\ansi SSN 123-45-\\v x\\v0 6789}", _RTF_SSN),
        (b"{\\rtf1\\ansi SSN 123-45-{\\deleted x}6789}", _RTF_SSN),
        (b"{\\rtf1\\ansi SSN 123-45-{\\field{\\*\\fldinst QUOTE x}{\\fldrslt }}6789}", _RTF_SSN),
        # A field's instruction is a destination with or without \\*.
        (b"{\\rtf1\\ansi SSN 123-45-{\\field{\\fldinst QUOTE}{\\fldrslt }}6789}", _RTF_SSN),
        # An index entry's text and a drawing object's text box.
        (b"{\\rtf1\\ansi SSN 123-45-{\\txe x}6789}", _RTF_SSN),
        (b"{\\rtf1\\ansi SSN 123-45-{\\do{\\dptxbxtext box}}6789}", _RTF_SSN),
        (b"{\\rtf1\\ansi SSN 123-45-{\\super\\chftn}{\\footnote note}6789}", _RTF_SSN),
        (b"{\\rtf1\\ansi SSN 123-45-{\\shp{\\shptxt box}}6789}", _RTF_SSN),
        # \\upr: a reader of Unicode shows the \\ud part.
        (b"{\\rtf1\\ansi SSN {\\upr{abc}{\\*\\ud{123}}}-45-6789}", _RTF_SSN),
    ],
    ids=[
        "uc1",
        "uc2-escapes",
        "uc0",
        "control-word-fallback",
        "group-ends-fallback",
        "cp932",
        "font-charset",
        "font-code-page",
        "hidden-group",
        "hidden-toggle",
        "deleted",
        "field-instruction",
        "field-instruction-bare",
        "index-text",
        "drawing-object",
        "footnote",
        "shape",
        "upr",
    ],
)
def test_rtf_is_read_as_a_reader_shows_it(document: bytes, shown: str) -> None:
    result = _read(document)
    assert result["format"] == "rtf" and result["complete"] is True, result
    assert shown in result["text"]


@pytest.mark.parametrize(
    "document",
    [
        # Bytes the declared double-byte code page does not decode.
        b"{\\rtf1\\ansi\\ansicpg932 \\'82\\'ff}",
        # A character standing for one that could not be mapped.
        b"{\\rtf1\\ansi x\\u65533?}",
        # A font whose character set has no code page here.
        b"{\\rtf1\\ansi{\\fonttbl{\\f1\\cpg99999 X;}}\\f1 \\'e9}",
    ],
    ids=["undecodable", "replacement", "unknown-code-page"],
)
def test_rtf_text_that_cannot_be_decoded_is_incomplete(document: bytes) -> None:
    assert _read(document)["complete"] is False


def test_detect() -> None:
    assert detect(b"%PDF-1.4") == "pdf"
    assert detect(b"PK\x05\x06") == "zip"
    assert detect(b"{\\rtf1") == "rtf"
    assert detect(b"  \n<html>") == "html"
    assert detect(b"GIF89a") is None


# --- the text cap and the process entry -----------------------------------------------------


def test_text_past_the_cap_is_cut_and_incomplete() -> None:
    result = _read(pdf(["a" * 400]), max_chars=100)
    assert result["complete"] is False and len(result["text"]) == 100


def test_main_reads_stdin_and_writes_one_json_object(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = {
        "formats": ["pdf", "ooxml"],
        "max_bytes": 1 << 20,
        "max_chars": 1000,
        "max_inflated": 1 << 20,
        "memory_bytes": 256 << 20,
        "cpu_seconds": 5,
    }
    applied: list[tuple[int, int]] = []
    monkeypatch.setattr(extract_worker, "apply_limits", lambda m, c: applied.append((m, c)))
    out = io.BytesIO()
    assert main(["w", json.dumps(settings)], io.BytesIO(pdf([EMAIL])), out) == 0
    assert applied == [(256 << 20, 5)]
    assert json.loads(out.getvalue())["text"].strip() == EMAIL
    # A file over the cap is not read at all.
    out = io.BytesIO()
    main(["w", json.dumps({**settings, "max_bytes": 10})], io.BytesIO(pdf(["x"])), out, limit=False)
    assert json.loads(out.getvalue())["reason"] == "limit"


def test_apply_limits_sets_each_cap_and_skips_refused_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    import resource

    calls: list[tuple[int, tuple[int, int]]] = []

    def setrlimit(which: int, value: tuple[int, int]) -> None:
        calls.append((which, value))
        if which == resource.RLIMIT_AS:
            raise ValueError("not on this platform")

    monkeypatch.setattr(resource, "setrlimit", setrlimit)
    apply_limits(256 << 20, 7)
    limits = dict(calls)
    assert limits[resource.RLIMIT_CPU] == (7, 7)
    assert limits[resource.RLIMIT_FSIZE] == (0, 0)
    assert limits[resource.RLIMIT_DATA] == (256 << 20, 256 << 20)
    assert resource.RLIMIT_AS in limits  # attempted, refused, skipped


def test_apply_limits_applies_none_without_resource_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    # Windows has no `resource` module: the worker starts without limits
    # (the parent bounds its input instead) rather than failing to start.
    monkeypatch.setitem(sys.modules, "resource", None)
    apply_limits(256 << 20, 7)


# --- what OCR of the rendered pages covers (EXT-1) ------------------------------------

_IMAGE = b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray"
_DRAW = b"q 100 0 0 100 0 0 cm /Im1 Do Q"


def _image_pdf(content: bytes, *, resources: bytes, extra: tuple[bytes, ...] = ()) -> bytes:
    """One page drawing ``content`` with ``resources``; object 5 is a 1x1
    image, ``extra`` objects follow from 6."""
    return pdf_objects(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
            b" /Resources << " + resources + b" >> >>",
            pdf_stream(b"", content),
            pdf_stream(_IMAGE[2:] + b" /BitsPerComponent 8", b"\x80"),
            *extra,
        ]
    )


@pytest.mark.parametrize(
    ("data", "ocr"),
    [
        # A scanned page: the image is drawn — OCR of the page reads it.
        (pdf([], image_page=True), True),
        (_image_pdf(_DRAW, resources=b"/XObject << /Im1 5 0 R >>"), True),
        # Drawn through a form the page draws.
        (
            _image_pdf(
                b"/Fm1 Do",
                resources=b"/XObject << /Fm1 6 0 R >>",
                extra=(
                    pdf_stream(
                        b"/Type /XObject /Subtype /Form /BBox [0 0 612 792]"
                        b" /Resources << /XObject << /Im1 5 0 R >> >>",
                        _DRAW,
                    ),
                ),
            ),
            True,
        ),
        # Listed in the page's resources but never drawn: no OCR sees it.
        (_image_pdf(b"", resources=b"/XObject << /Im1 5 0 R >>"), False),
        # Drawn, but the file also holds an image no page draws.
        (
            _image_pdf(
                _DRAW,
                resources=b"/XObject << /Im1 5 0 R >>",
                extra=(pdf_stream(_IMAGE[2:] + b" /BitsPerComponent 8", b"\x80"),),
            ),
            False,
        ),
    ],
    ids=["scanned", "drawn", "through-a-form", "listed-only", "one-not-drawn"],
)
def test_an_image_a_page_draws_is_an_ocr_gap(data: bytes, ocr: bool) -> None:
    result = _read(data)
    assert result["complete"] is False and result["ocr"] is ocr


def test_what_page_ocr_never_sees_stays_unread() -> None:
    scanned = pdf([], image_page=True)
    # An attachment beside a scanned page: OCR of the page never reads it.
    writer_pdf = _attached(scanned)
    assert _read(writer_pdf)["ocr"] is False
    # The text cap cut the reading: what was cut is not on a page OCR reads.
    capped = pdf(["x" * 200], image_page=True)
    assert _read(capped, max_chars=50)["ocr"] is False
    # A complete reading needs no OCR; a failure is no OCR gap.
    assert _read(pdf(["text page"]))["ocr"] is False
    assert _read(b"%PDF-1.7\nnot a pdf").get("ocr") is not True


def _attached(data: bytes) -> bytes:
    import io as _io

    from pypdf import PdfReader, PdfWriter

    writer = PdfWriter(clone_from=PdfReader(_io.BytesIO(data)))
    writer.add_attachment("secret.txt", b"employee ssn 123-45-6789")
    out = _io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_a_reading_marked_incomplete_is_never_an_ocr_gap() -> None:
    reading = Reading(100)
    reading.ocr_gap()
    assert reading.complete is False and reading.ocr_completes is True
    reading.complete = reading.complete  # False: anything marked so is unread
    assert reading.ocr_completes is False
    fresh = Reading(100)
    fresh.complete = True
    assert fresh.complete is True and fresh.ocr_completes is False


def test_image_text_free() -> None:
    from document_fixtures import jpeg, png
    from llm_redact.extract_worker import image_text_free

    assert image_text_free(png()) is True
    assert image_text_free(jpeg()) is True
    for chunk in (b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"iCCP", b"sPLT", b"prVt"):
        assert image_text_free(png((chunk, b"Comment\x00ssn 123-45-6789"))) is False, chunk
    assert image_text_free(png((b"pHYs", bytes(9)), (b"gAMA", bytes(4)))) is True
    for marker in (0xE1, 0xE2, 0xED, 0xFE, 0xF0):
        assert image_text_free(jpeg((marker, b"Exif\x00\x00 ssn"))) is False, hex(marker)
    assert image_text_free(jpeg((0xEE, b"Adobe" + bytes(7)))) is True
    assert image_text_free(jpeg(thumbnail=True)) is False
    for trailer in (b"x", b"ssn 123-45-6789"):
        assert image_text_free(png(trailer=trailer)) is False
        assert image_text_free(jpeg(trailer=trailer)) is False
    # Malformed or cut short, and every other format: not vouched for.
    whole_png, whole_jpeg = png(), jpeg()
    for cut in range(len(whole_png)):
        assert image_text_free(whole_png[:cut]) is False
    for cut in range(len(whole_jpeg)):
        assert image_text_free(whole_jpeg[:cut]) is False
    bad_app0 = whole_jpeg.replace(b"JFIF\x00", b"JFXX\x00", 1)
    assert image_text_free(bad_app0) is False
    zero_length = b"\xff\xd8\xff\xdb\x00\x01" + whole_jpeg[2:]
    assert image_text_free(zero_length) is False
    not_a_marker = b"\xff\xd8\x00" + whole_jpeg[2:]
    assert image_text_free(not_a_marker) is False
    assert image_text_free(b"BM" + bytes(64)) is False
    assert image_text_free(b"GIF89a" + bytes(32)) is False


def test_image_text_free_holds_each_header_to_its_length() -> None:
    """A picture header the allowlist names may still carry bytes past its
    defined size (an oversized tIME, an APP14 or a table segment longer than
    its tables): only a header of its specified length (or layout) is text
    free."""
    import struct

    from document_fixtures import jpeg, png
    from llm_redact.extract_worker import image_text_free

    ssn = b" ssn 123-45-6789"
    for chunk, size in (
        (b"tIME", 7),
        (b"pHYs", 9),
        (b"gAMA", 4),
        (b"sRGB", 1),
        (b"cHRM", 32),
        (b"cICP", 4),
        (b"cLLI", 8),
        (b"mDCV", 24),
    ):
        assert image_text_free(png((chunk, bytes(size)))) is True, chunk
        assert image_text_free(png((chunk, bytes(size) + ssn))) is False, chunk
    for chunk, ok, bad in (
        (b"PLTE", bytes(6), bytes(769)),
        (b"PLTE", bytes(3), bytes(4)),
        (b"tRNS", bytes(256), bytes(257)),
        (b"sBIT", bytes(4), bytes(5)),
        (b"bKGD", bytes(6), bytes(3)),
        (b"hIST", bytes(512), bytes(3)),
    ):
        assert image_text_free(png((chunk, ok))) is True, chunk
        assert image_text_free(png((chunk, bad))) is False, chunk
    # IHDR is the fixture's own: a longer one is not text free.
    whole = png()
    longer_header = whole.replace(
        struct.pack(">I", 13) + b"IHDR", struct.pack(">I", 13 + len(ssn)) + b"IHDR", 1
    )
    at = longer_header.index(b"IHDR") + 4 + 13
    assert image_text_free(longer_header[:at] + ssn + longer_header[at:]) is False
    assert image_text_free(jpeg((0xEE, b"Adobe" + bytes(7) + ssn))) is False
    assert image_text_free(jpeg((0xEE, b"Other" + bytes(7)))) is False
    # Table and frame segments hold exactly their tables and components.
    assert image_text_free(jpeg((0xDB, bytes(65) + ssn))) is False
    assert image_text_free(jpeg((0xDB, b"\x10" + bytes(128)))) is True
    assert image_text_free(jpeg((0xDB, b"\x20" + bytes(128)))) is False
    assert image_text_free(jpeg((0xC4, bytes(17) + ssn))) is False
    assert image_text_free(jpeg((0xC4, b"\x10\x01" + bytes(15) + b"A"))) is True
    assert image_text_free(jpeg((0xC4, b"\x10\x01" + bytes(15)))) is False
    assert image_text_free(jpeg((0xDD, b"\x00\x04"))) is True
    assert image_text_free(jpeg((0xDD, b"\x00\x04" + ssn))) is False
    for marker in (0xC8, 0xD8):
        assert image_text_free(jpeg((marker, ssn))) is False, hex(marker)
    whole_jpeg = jpeg()
    frame = b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    assert frame in whole_jpeg
    longer_frame = b"\xff\xc0" + struct.pack(">H", 11 + len(ssn)) + frame[4:] + ssn
    assert image_text_free(whole_jpeg.replace(frame, longer_frame)) is False
    scan = b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00"
    assert scan in whole_jpeg
    longer_scan = b"\xff\xda" + struct.pack(">H", 8 + len(ssn)) + scan[4:] + ssn
    assert image_text_free(whole_jpeg.replace(scan, longer_scan)) is False
