"""The document extraction WORKER: one hostile file in, its text out.

Run as ``python -I -m llm_redact.extract_worker CONFIG`` by the built-in
upload inspector (``extraction.py``), one process per file: the file arrives on
stdin, one JSON object leaves on stdout —
``{"format": ..., "text": str | null, "complete": bool, "reason": str}``,
plus ``"display"`` (the file's text once, as a reader shows it: what
``[extraction] convert`` sends in place of a file; null unless complete)
and ``"pages"`` (a PDF's page count, null for other formats or when the
file could not be opened: a cloud OCR reading is complete only when it
covers that many pages), and ``"ocr"``: whether what the reading left unread
is only what OCR of the rendered pages reads (``Reading.ocr_completes``) —
only then may a cloud OCR reading complete the file.
The process caps its own address space, CPU time, file writes, open files
and child processes (``apply_limits``) BEFORE it reads a byte of the file,
and the parent kills it at its wall-clock deadline, so a parser wedged or
blown up by a crafted file costs one killed process, never the proxy.

This module imports nothing else from llm-redact (only the standard library
and, for PDFs, pypdf — the ``extract`` extra) so the worker starts in
isolated mode (``-I``: no
environment variables, no user site, no current directory on the path).

``complete`` is the claim the core relies on to let a clean file go out:
EVERY text-bearing element was read. It is False for anything left unread
— a PDF page drawing an image, an embedded file or image in a document
package, a form this reader does not follow, text past the size cap, a
charset guessed rather than declared — and for any failure part way.

Formats (``FORMATS``): ``pdf`` (pypdf: every page's text layer, what every
form XObject, tiling pattern and annotation appearance draws, the text
marked content stands for — also in place of its glyphs — each line where
two drawings meet as the page shows it, annotation strings and link
targets, form field names and values, the document information and XMP
metadata, bookmarks);
``ooxml`` (docx/xlsx/pptx and every other Office Open XML package) and
``odf`` (OpenDocument): every XML part's text and attribute values, read
three ways — joined within paragraphs (a value Word splits across runs), as
a renderer SHOWS the line (tracked deletions, field instructions, hidden
runs, notes, comments and text boxes read on their own, the text around
them joined) and element by element, a spreadsheet number or date also as
its number format shows it — with zip-bomb limits on entry count, total
inflated size and inflation ratio, and any document type declaration
refused; ``html`` (markup that is not UTF-8 text: the decoded source itself
plus its text with character references resolved); ``rtf`` (best effort:
the source plus its decoded text).
"""

from __future__ import annotations

import codecs
import html.parser
import io
import json
import re
import sys
import zipfile
from collections.abc import Callable, Container
from datetime import date, datetime, timedelta
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any
from xml.parsers import expat

FORMATS = ("pdf", "ooxml", "odf", "html", "rtf")

# Zip packages: at most this many entries, this much inflated in total, and
# no entry inflating more than MAX_RATIO times its compressed size.
MAX_ENTRIES = 10_000
MAX_RATIO = 200

# Elements whose END separates text in the joined reading of an XML part
# (paragraphs, headings, table cells and rows, spreadsheet cells and shared
# strings, list items, slides' text bodies); every other element joins, as
# runs do inside a paragraph.
_BLOCKS = frozenset(
    {
        "p",
        "h",
        "tr",
        "tc",
        "c",
        "si",
        "row",
        "sheet",
        "definedName",
        "comment",
        "annotation",
        "note",
        "list-item",
        "table-row",
        "table-cell",
        "txBody",
        "cell",
        "br",
        "cr",
        "line-break",
    }
)
# Elements standing for a character inside a paragraph: whitespace (Word's
# absolute-position tab too), and Word's non-breaking hyphen (shown as a
# hyphen: "123-45-6789" typed with them is still a hyphenated number); a
# symbol (``w:sym``) is its character (``_XmlText._character``). A soft
# hyphen is shown only where a line breaks, so it stands for nothing.
_SPACES = {"tab": "\t", "ptab": "\t", "s": " ", "noBreakHyphen": "-"}
# Elements holding base64 binary data inline (WordprocessingML's pictures,
# OpenDocument's embedded images and objects): never read as text.
_INLINE_BINARY = frozenset({"binData", "binary-data"})
# Elements whose content a renderer does NOT show in the line they sit in:
# tracked deletions and moves (w:del, w:moveFrom), field instructions
# (w:instrText; the field's result is shown), phonetic runs (rPh), ruby text
# (shown above the line), an alternative's fallback (mc:Fallback), drawings,
# text boxes and frames (shown beside the text), OpenDocument's inline notes
# and comments (office:annotation) and hidden-text fields. The display
# reading leaves them out of the line — so the text around them joins as a
# renderer shows it — and reads each on its own.
_ASIDES = frozenset(
    {
        "del",
        "moveFrom",
        "instrText",
        "delInstrText",
        "rPh",
        "rt",
        "ruby-text",
        "Fallback",
        "drawing",
        "pict",
        "object",
        "frame",
        "custom-shape",
        "note",
        "annotation",
        "hidden-text",
    }
)
# Run properties that take a WordprocessingML run, or a paragraph's mark (the
# paragraph then joins the next one as shown), out of the display: hidden
# text, a tracked deletion or move of the mark.
_HIDING = frozenset({"vanish", "specVanish", "del", "moveFrom"})
_OFF = frozenset({"0", "false", "off"})

# Package entries that are derived renderings of the document's own text
# (the application writes them from what the XML parts hold): not read, and
# not counted as unread content.
_THUMBNAILS = re.compile(r"(?:docProps/thumbnail\.[a-z]+|Thumbnails/thumbnail\.png)", re.I)
_XML_PART = re.compile(r".+\.(?:xml|rels|vml|rdf)", re.I)
_ODF_MIME = b"application/vnd.oasis.opendocument."
# Package parts whose text is not the document's own for the DISPLAY
# reading (``Reading.show``): relationships, content types, properties,
# styles, settings, fonts, themes, the manifest, and slide layouts and
# masters (their placeholder prompts). The full reading still reads them.
_UNSHOWN_PART = re.compile(
    r"(?:^|/)(?:_rels/|\[Content_Types\]\.xml$|docProps/|META-INF/|theme/|customXml/"
    r"|slideLayouts/|slideMasters/|notesMasters/|handoutMasters/"
    r"|(?:styles|settings|fontTable|webSettings|meta|manifest|numbering|presProps|viewProps"
    r"|tableStyles|commentsExtended|commentsIds|people)\.(?:xml|rdf)$)",
    re.I,
)
_BLANK_LINES = re.compile(r"\n{2,}")
# Whitespace runs holding a line break (markup's indentation between blocks).
_SPACE_RUNS = re.compile(r"[ \t\r]*\n\s*")

# Printable ASCII and the whitespace controls: what a declared charset must
# decode to itself (``keeps_ascii``).
_ASCII = bytes(range(0x20, 0x7F)) + b"\t\n\r"
# Any mention of a charset (checked for agreement), the XML declaration's
# (only at the very start) and a meta tag's content attribute's.
_CHARSET = re.compile(rb"""(?:encoding|charset)\s*=\s*["']?\s*([A-Za-z0-9._:-]{1,40})""", re.I)
_XML_DECLARATION = re.compile(
    rb"""<\?xml\s[^>]*?\bencoding\s*=\s*["']([A-Za-z0-9._:-]{1,40})["']"""
)
_CONTENT_CHARSET = re.compile(r"""charset\s*=\s*["']?\s*([A-Za-z0-9._:-]{1,40})""", re.I)
# Markup inlining binary data: an image or any base64 data URI, or an XML
# binary-data element (Word 2003 XML, flat OpenDocument).
_INLINE_MEDIA = re.compile(r"data:image/|;base64,|<(?:[\w.-]+:)?(?:binData|binary-data)\b", re.I)
_RTF_TOKEN = re.compile(
    rb"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-fA-F]{2})|\\(.)|([{}])|([^\\{}]+)", re.S
)
_RTF_UNREAD = frozenset({b"pict", b"object", b"objdata", b"bin", b"shppict", b"nonshppict"})
_RTF_BREAKS = {b"par": "\n", b"line": "\n", b"sect": "\n", b"page": "\n", b"row": "\n"}
_RTF_TABS = frozenset({b"tab", b"cell"})
# Control symbols standing for a character: escaped braces and backslash, the
# non-breaking hyphen (\_) and space (\~); others (the optional hyphen \-,
# shown only at a line break) stand for nothing.
_RTF_SYMBOLS = {b"\\": "\\", b"{": "{", b"}": "}", b"_": "-", b"~": " "}
_ANSICPG = re.compile(rb"\\ansicpg(\d{1,5})")
# A font's \fcharsetN as a code page (0, 1 and 2 — ANSI, default, symbol —
# use the document's \ansicpg).
_RTF_CHARSETS = {
    77: "mac_roman",
    128: "cp932",
    129: "cp949",
    130: "cp1361",
    134: "gbk",
    136: "cp950",
    161: "cp1253",
    162: "cp1254",
    163: "cp1258",
    177: "cp1255",
    178: "cp1256",
    186: "cp1257",
    204: "cp1251",
    222: "cp874",
    238: "cp1250",
    254: "cp437",
    255: "cp850",
}
# Destinations whose text is not in the document's line of text: tables,
# document information, pictures and objects, notes, headers and footers,
# comments, shapes and drawing objects (text boxes), index, table-of-contents
# and bookmark entries, and a field's instruction (only its result is
# shown) — each with or without the ``\*`` a writer may omit for a
# destination a reader of its RTF version knows.
_RTF_ASIDES = frozenset(
    {
        b"fonttbl",
        b"filetbl",
        b"template",
        b"fldinst",
        b"colortbl",
        b"stylesheet",
        b"info",
        b"listtable",
        b"listoverridetable",
        b"revtbl",
        b"rsidtbl",
        b"generator",
        b"pict",
        b"object",
        b"footnote",
        b"header",
        b"headerl",
        b"headerr",
        b"headerf",
        b"footer",
        b"footerl",
        b"footerr",
        b"footerf",
        b"ftnsep",
        b"ftnsepc",
        b"ftncn",
        b"aftnsep",
        b"aftnsepc",
        b"aftncn",
        b"annotation",
        b"atnid",
        b"atnauthor",
        b"atntime",
        b"atndate",
        b"atnref",
        b"atnicn",
        b"shp",
        b"do",
        b"xe",
        b"txe",
        b"rxe",
        b"tc",
        b"bkmkstart",
        b"bkmkend",
    }
)


class Limit(Exception):
    """A resource limit of the extraction (size, ratio, entries) was hit."""


class Refused(Exception):
    """Input this reader refuses to parse (a document type declaration)."""


class Reading:
    """The text read so far, bounded: past ``max_chars`` it stops growing
    and the reading is incomplete.

    Beside it, the DISPLAY reading (``show``): the file's text once, as a
    reader of the document sees it — each PDF page's text, the shown text
    of a document package's content parts, markup's text without its
    tags, scripts and styles, RTF as shown — which ``[extraction] convert``
    sends in place of the file. It is bounded by the same cap; past it (or
    where a format has no faithful display reading: ``no_display``) there
    is none. The scan always covers the full reading (``text``), never
    just this one."""

    def __init__(self, max_chars: int) -> None:
        self.pieces: list[str] = []
        self.size = 0
        self.max_chars = max_chars
        # Why the reading may be incomplete: something left unread that a
        # page-rendering OCR would not read either (``_unread``: anything
        # marked ``complete = False``, by default), or only what OCR of the
        # rendered pages reads (``ocr_gap``: an image a page draws, page
        # text that reads as nothing or as unmapped characters).
        self._unread = False
        self._ocr_gap = False
        self.kind: str | None = None  # a package's kind, once known
        self.shown: list[str] = []
        self.shown_size = 0
        self.displayable = True
        self.pages: int | None = None  # a PDF's page count, once known

    @property
    def complete(self) -> bool:
        """Whether EVERY text-bearing element was read (see the module)."""
        return not (self._unread or self._ocr_gap)

    @complete.setter
    def complete(self, value: bool) -> None:
        # Marking a reading incomplete is fail closed: nothing OCR reads.
        if not value:
            self._unread = True

    def ocr_gap(self) -> None:
        """Incomplete only for what OCR of the rendered pages reads."""
        self._ocr_gap = True

    @property
    def ocr_completes(self) -> bool:
        """Whether a reading of every rendered page by OCR covers what this
        one left unread: it is incomplete only for ``ocr_gap`` reasons."""
        return self._ocr_gap and not self._unread

    def add(self, text: str) -> None:
        if not text:
            return
        room = self.max_chars - self.size
        if len(text) > room:
            text = text[: max(room, 0)]
            self.complete = False
        self.pieces.append(text)
        self.size += len(text)

    def show(self, text: str) -> None:
        """``text`` into the display reading (see the class)."""
        if not text or not self.displayable:
            return
        self.shown_size += len(text)
        if self.shown_size > self.max_chars:
            self.no_display()
            return
        self.shown.append(text)

    def no_display(self) -> None:
        self.displayable = False
        self.shown = []

    def text(self) -> str:
        return "\n".join(self.pieces)

    def display(self) -> str | None:
        """The display reading; None when there is none (see the class)."""
        if not self.displayable:
            return None
        return _BLANK_LINES.sub("\n\n", "\n\n".join(self.shown)).strip("\n")


# --- sniffing --------------------------------------------------------------------


def detect(data: bytes) -> str | None:
    """The local format of ``data`` by its bytes, or None: a PDF, a zip
    package (``ooxml``/``odf`` decided inside it), RTF, or markup."""
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        return "zip"
    if data.startswith(b"{\\rtf"):
        return "rtf"
    if is_markup(data):
        return "html"
    return None


_EOCD = b"PK\x05\x06"
# How far from its end a zip reader looks for the end record: the record's
# 22 bytes and a comment of up to 65535.
_EOCD_REACH = 22 + 0xFFFF
_PDF_TAIL = b" \t\r\n\x00\x0c"
# What a PDF reader finds a PDF by: its header, its trailer's pointer to the
# cross-reference table, its end marker.
# --- single images: what OCR of the picture does not see ---------------------------

# PNG chunks that carry no text: the picture itself and how it is shown,
# each with the payload lengths its specification allows (IDAT: any). Every
# other chunk (tEXt, zTXt, iTXt, eXIf, iCCP's profile, sPLT's name, a
# private one) may hold text no OCR of the picture reads, and so may a
# listed one longer than its defined content.
_PNG_PICTURE_CHUNKS: dict[bytes, Container[int] | None] = {
    b"IHDR": range(13, 14),
    b"PLTE": range(3, 769, 3),
    b"IDAT": None,
    b"IEND": range(0, 1),
    b"tRNS": range(1, 257),
    b"cHRM": range(32, 33),
    b"gAMA": range(4, 5),
    b"sBIT": range(1, 5),
    b"sRGB": range(1, 2),
    b"bKGD": frozenset({1, 2, 6}),
    b"hIST": range(2, 513, 2),
    b"pHYs": range(9, 10),
    b"tIME": range(7, 8),
    b"cICP": range(4, 5),
    b"mDCV": range(24, 25),
    b"cLLI": range(8, 9),
}
# JPEG markers that carry no text: frame headers (SOFn, DHP), scan headers,
# quantization and Huffman tables, arithmetic conditioning, restart
# interval, line count, expansion, APP14 (Adobe colour transform), each
# held to its layout (``_jpeg_segment_text_free``); APP0 only as a JFIF
# header without a thumbnail. Every other one — APP1 (Exif, XMP), APP2
# (ICC profile), any other APPn, COM, JPG and JPGn — may hold text.
_JPEG_FRAMES = frozenset({*range(0xC0, 0xC4), *range(0xC5, 0xC8), *range(0xC9, 0xCC)})
_JPEG_FRAMES |= frozenset({*range(0xCD, 0xD0), 0xDE})
_JPEG_FIXED = {0xDC: 4, 0xDD: 4, 0xDF: 3, 0xEE: 14}


def image_text_free(data: bytes) -> bool:
    """Whether a single picture holds nothing but the picture: a PNG made of
    ``_PNG_PICTURE_CHUNKS`` or a JPEG made of picture headers
    (``_jpeg_segment_text_free``; an APP0 only as a JFIF header without a
    thumbnail), each no longer than its specified layout, ending at its end
    marker with no byte after it. Any metadata a reader of the file sees
    but OCR of the picture does not — text chunks, Exif, XMP, an ICC
    profile, a comment, a thumbnail, bytes past a header's layout, trailing
    bytes — and any other format or a malformed file: False. Header walking
    only, never decoding."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _png_text_free(data)
    if data.startswith(b"\xff\xd8"):
        return _jpeg_text_free(data)
    return False


def _png_text_free(data: bytes) -> bool:
    pos = 8
    while pos + 12 <= len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        kind = data[pos + 4 : pos + 8]
        end = pos + 12 + length
        if kind not in _PNG_PICTURE_CHUNKS or end > len(data):
            return False
        sizes = _PNG_PICTURE_CHUNKS[kind]
        if sizes is not None and length not in sizes:
            return False
        if kind == b"IEND":
            return end == len(data)
        pos = end
    return False


def _jpeg_text_free(data: bytes) -> bool:
    pos = 2
    while pos + 2 <= len(data):
        if data[pos] != 0xFF:
            return False
        marker = data[pos + 1]
        if marker == 0xFF:  # a fill byte
            pos += 1
            continue
        if marker == 0xD9:  # EOI
            return pos + 2 == len(data)
        if 0xD0 <= marker <= 0xD7:  # a restart marker: no length
            pos += 2
            continue
        if pos + 4 > len(data):
            return False
        length = int.from_bytes(data[pos + 2 : pos + 4], "big")
        segment = data[pos + 4 : pos + 2 + length]
        if length < 2 or pos + 2 + length > len(data):
            return False
        if not _jpeg_segment_text_free(marker, segment):
            return False
        pos += 2 + length
        if marker == 0xDA:
            # Entropy-coded data up to the next marker that is neither a
            # stuffed 0xFF00 nor a restart marker.
            while True:
                pos = data.find(b"\xff", pos)
                if pos < 0 or pos + 1 >= len(data):
                    return False
                following = data[pos + 1]
                if following == 0x00 or 0xD0 <= following <= 0xD7:
                    pos += 2
                    continue
                break
    return False


def _jpeg_segment_text_free(marker: int, segment: bytes) -> bool:
    """Whether one JPEG marker segment (its payload: ``segment``) is a
    picture header of exactly its specified layout (``_JPEG_FRAMES``,
    ``_JPEG_FIXED``, the tables, a thumbnail-less JFIF APP0)."""
    size = len(segment)
    if marker == 0xE0:
        # JFIF: identifier, version, units, densities, then the thumbnail's
        # width and height — none allowed.
        return size == 14 and segment[:5] == b"JFIF\x00" and segment[12:14] == b"\x00\x00"
    if marker in _JPEG_FRAMES:
        # Precision, height, width, component count, three bytes each.
        return size >= 6 and size == 6 + 3 * segment[5]
    if marker == 0xDA:
        # Component count, two bytes each, then three bytes of selection.
        return size >= 1 and size == 4 + 2 * segment[0]
    if marker == 0xDB:
        return _jpeg_tables(segment, lambda at: 65 if segment[at] >> 4 == 0 else 129, 0x13)
    if marker == 0xC4:
        return _jpeg_tables(segment, lambda at: 17 + sum(segment[at + 1 : at + 17]), 0x13)
    if marker == 0xCC:
        # Arithmetic conditioning: two bytes per table, at most eight tables.
        return 0 < size <= 16 and size % 2 == 0
    if marker in _JPEG_FIXED:
        return size + 2 == _JPEG_FIXED[marker] and (marker != 0xEE or segment[:5] == b"Adobe")
    return False


def _jpeg_tables(segment: bytes, size: Callable[[int], int], highest: int) -> bool:
    """Whether ``segment`` is a run of whole tables — each opening with its
    class/precision and id byte (neither nibble above ``highest``'s), then
    ``size(offset)`` bytes in all — and nothing else."""
    at = 0
    while at < len(segment):
        kind = segment[at]
        if kind >> 4 > highest >> 4 or kind & 0x0F > highest & 0x0F:
            return False
        at += size(at)
    return at == len(segment) and at > 0


_PDF_MARKERS = (b"%PDF-", b"startxref", b"%%EOF")


def carries_another(data: bytes, kind: str) -> bool:
    """Whether ``data`` also holds a file a reader of ANOTHER format would
    open (a polyglot, read by whichever parser the provider picks from the
    file's name or declared type): a PDF's header or trailer ANYWHERE in a
    file that is not a PDF (a lenient PDF reader — pypdf, poppler, pdf.js,
    MuPDF — finds a PDF by its trailer from the end, whatever precedes or
    follows it, and without a header too), a zip end record near the end
    (where zip readers look) of one that is not a zip, bytes after a zip's
    end record, a PDF's last ``%%EOF`` or an RTF document's outer group."""
    if kind != "pdf" and any(marker in data for marker in _PDF_MARKERS):
        return True
    if kind == "zip":
        end = data.rfind(_EOCD)
        comment = int.from_bytes(data[end + 20 : end + 22], "little")
        return end < 0 or len(data) != end + 22 + comment
    if _EOCD in data[-_EOCD_REACH:]:
        return True
    if kind == "pdf":
        end = data.rfind(b"%%EOF")
        return end < 0 or bool(data[end + 5 :].strip(_PDF_TAIL))
    if kind == "rtf":
        return bool(data[_rtf_end(data) :].strip(_PDF_TAIL))
    return False


def _rtf_end(data: bytes) -> int:
    """Where an RTF document's outer group closes (the end of the data
    when it never does)."""
    depth = 0
    for match in _RTF_TOKEN.finditer(data):
        if match[5] == b"{":
            depth += 1
        elif match[5] == b"}":
            depth -= 1
            if depth <= 0:
                return match.end()
    return len(data)


def is_markup(data: bytes) -> bool:
    """Whether ``data`` opens like markup (``<`` after whitespace), in an
    ASCII-compatible encoding or UTF-16 without a byte-order mark."""
    head = data[:64].lstrip(b" \t\r\n")
    if head.startswith(b"<"):
        return True
    wide = data[:128].replace(b"\x00", b"").lstrip(b" \t\r\n")
    return wide.startswith(b"<") and b"\x00" in data[:4]


# --- PDF ---------------------------------------------------------------------------


def _pdf_strings(value: Any, reading: Reading, depth: int = 0) -> None:
    """Every string under a PDF object (a dictionary, an array), a few
    levels deep, and the text streams a value or rich text may be
    (``_PDF_TEXT_STREAMS``); page, parent and appearance links are not
    followed (appearances are read as drawn: ``_read_drawn``)."""
    from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, StreamObject

    if depth > 6:
        return
    if isinstance(value, IndirectObject):
        value = value.get_object()
    if isinstance(value, str):
        reading.add(value)
    elif isinstance(value, bytes):
        reading.add(value.decode("latin-1"))
    elif isinstance(value, DictionaryObject):
        for key, item in value.items():
            if key in _PDF_TEXT_STREAMS and isinstance(_resolved(item), StreamObject):
                _pdf_stream_text(_resolved(item), reading)
            elif key not in ("/P", "/Parent", "/AP", "/Popup", "/Dest", "/Font"):
                _pdf_strings(item, reading, depth + 1)
    elif isinstance(value, ArrayObject):
        for item in value:
            _pdf_strings(item, reading, depth + 1)


# Entries that may hold a text STREAM rather than a string: a field's value
# and rich value, an annotation's rich contents.
_PDF_TEXT_STREAMS = frozenset({"/V", "/RV", "/RC"})


def _pdf_value_text(value: Any, reading: Reading) -> None:
    """A form field's value: a text stream or strings."""
    from pypdf.generic import StreamObject

    value = _resolved(value)
    if isinstance(value, StreamObject):
        _pdf_stream_text(value, reading)
    else:
        _pdf_strings(value, reading)


def _pdf_stream_text(stream: Any, reading: Reading) -> None:
    """A text stream's text (UTF-16 with its byte-order mark, else UTF-8);
    not decodable, it is read as Latin-1 and the reading is incomplete."""
    data = stream.get_data()
    try:
        if data.startswith(b"\xfe\xff"):
            text = data[2:].decode("utf-16-be")
        else:
            text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text, reading.complete = data.decode("latin-1"), False
    reading.add(text)


# Keys attaching content the reader does not open: an embedded file (a file
# specification's /EF), PDF 2.0 associated files (/AF), rich media, a
# portfolio (/Collection).
_PDF_ATTACHING = frozenset({"/EF", "/AF", "/RichMediaContent", "/Collection"})
# Annotations carrying a file or media the reader does not open.
_PDF_MEDIA_ANNOTATIONS = frozenset(
    {"/FileAttachment", "/RichMedia", "/Movie", "/Sound", "/3D", "/Screen"}
)
# XObjects the reader does not read: an image (wherever it is painted from —
# a page, a form, a pattern, a glyph, an appearance) and PostScript.
_PDF_UNREAD_XOBJECTS = frozenset({"/Image", "/PS"})
# The text a marked-content sequence or a structure element stands for:
# what copy and paste, screen readers and text extraction that honours it
# read in place of the glyphs drawn.
_PDF_MARKED_TEXT = ("/ActualText", "/Alt", "/E")
# Direct nesting deeper than this is not followed, and so not vouched for.
_PDF_MAX_NESTING = 32

# Simple fonts, the named encodings pypdf maps to Unicode, and the standard
# 14 fonts (their encoding is known without an /Encoding entry).
_PDF_SIMPLE_FONTS = frozenset({"/Type1", "/MMType1", "/TrueType"})
_PDF_ENCODINGS = frozenset(
    {"/StandardEncoding", "/WinAnsiEncoding", "/MacRomanEncoding", "/PDFDocEncoding"}
)
_PDF_STANDARD_FONTS = frozenset(
    "/" + name
    for name in [
        "Times-Roman",
        "Times-Bold",
        "Times-Italic",
        "Times-BoldItalic",
        "Helvetica",
        "Helvetica-Bold",
        "Helvetica-Oblique",
        "Helvetica-BoldOblique",
        "Courier",
        "Courier-Bold",
        "Courier-Oblique",
        "Courier-BoldOblique",
        "Symbol",
        "ZapfDingbats",
    ]
)


def _operators(*names: bytes) -> re.Pattern[bytes]:
    """Content-stream operators ``names`` as whole tokens (PDF delimiters or
    whitespace around them; a name like ``/Do`` is not the operator)."""
    alternatives = b"|".join(re.escape(name) for name in names)
    return re.compile(
        rb"(?<![^\x00\s()<>\[\]{}%])(?:" + alternatives + rb")(?![^\x00\s()<>\[\]{}/%])"
    )


_INLINE_IMAGE = _operators(b"BI")
_SHOWS_TEXT = _operators(b"Tj", b"TJ", b"'", b'"')
_SHOWS_TEXT_OR_INLINE_IMAGE = _operators(b"Tj", b"TJ", b"'", b'"', b"BI")
_PAINTS = _operators(
    b"f",
    b"F",
    b"f*",
    b"B",
    b"B*",
    b"b",
    b"b*",
    b"S",
    b"s",
    b"sh",
    b"Do",
    b"Tj",
    b"TJ",
    b"'",
    b'"',
    b"BI",
)


def _resolved(value: Any) -> Any:
    from pypdf.generic import IndirectObject

    return value.get_object() if isinstance(value, IndirectObject) else value


def _streams(value: Any) -> list[Any]:
    """The streams of a dictionary's values (an appearance's states, a Type3
    font's glyph procedures), or the stream itself."""
    from pypdf.generic import DictionaryObject, StreamObject

    value = _resolved(value)
    if isinstance(value, StreamObject):
        return [value]
    if isinstance(value, DictionaryObject):
        return [item for item in map(_resolved, value.values()) if isinstance(item, StreamObject)]
    return []


def _draws(streams: list[Any], operators: re.Pattern[bytes]) -> bool:
    return any(operators.search(stream.get_data()) for stream in streams)


def _font_unmapped(font: Any) -> bool:
    """Whether the text pypdf reads for a font's glyphs may not be the text
    they show: no ToUnicode map, and a Type3 font (glyph procedures under
    names of its own), a composite font (bare glyph ids), a simple font
    with an encoding pypdf does not map (another name, a base encoding or a
    glyph name outside the Adobe glyph list — pypdf then emits the NAME),
    no encoding outside the standard 14 fonts, or an unknown kind. A
    composite font's descendant is decided by its parent."""
    from pypdf._codecs import adobe_glyphs
    from pypdf.generic import ArrayObject, DictionaryObject, NameObject

    subtype = _resolved(font.get("/Subtype"))
    if "/ToUnicode" in font or subtype in ("/CIDFontType0", "/CIDFontType2"):
        return False
    if subtype not in _PDF_SIMPLE_FONTS:
        return True
    encoding = _resolved(font.get("/Encoding"))
    if encoding is None:
        return _resolved(font.get("/BaseFont")) not in _PDF_STANDARD_FONTS
    if not isinstance(encoding, DictionaryObject):
        return encoding not in _PDF_ENCODINGS
    base = _resolved(encoding.get("/BaseEncoding"))
    differences = _resolved(encoding.get("/Differences"))
    names = differences if isinstance(differences, ArrayObject) else []
    return (base is not None and base not in _PDF_ENCODINGS) or any(
        isinstance(name, NameObject) and name not in adobe_glyphs for name in map(_resolved, names)
    )


def _pdf_font_unread(value: Any) -> bool:
    """A font this reader cannot vouch for (``_font_unmapped``), or a
    glyph procedure showing text or painting an inline image (what the
    glyph draws is not the text it maps to), in a font dictionary or a
    resource dictionary's fonts."""
    from pypdf.generic import DictionaryObject

    fonts = _resolved(value.get("/Font"))
    candidates = [value] if _resolved(value.get("/Type")) == "/Font" else []
    if isinstance(fonts, DictionaryObject):
        candidates += [
            font for font in map(_resolved, fonts.values()) if isinstance(font, DictionaryObject)
        ]
    return any(
        _font_unmapped(font)
        or _draws(_streams(font.get("/CharProcs")), _SHOWS_TEXT_OR_INLINE_IMAGE)
        for font in candidates
    )


def _pdf_dictionary_unread(value: Any) -> bool:
    """Whether one PDF dictionary (or stream) is content this reader does
    not read — apart from an image XObject, which ``_PdfWalk`` counts on
    its own (OCR of a page that draws it reads it): an attachment or media
    (``_PDF_ATTACHING``, ``_PDF_MEDIA_ANNOTATIONS``), a PostScript XObject,
    or a font it cannot vouch for (``_pdf_font_unread``). What forms,
    patterns and annotation appearances draw is read (``_read_drawn``)."""
    subtype = _resolved(value.get("/Subtype"))
    return bool(
        _PDF_ATTACHING.intersection(value.keys())
        or _resolved(value.get("/Type")) == "/EmbeddedFile"
        or subtype in _PDF_MEDIA_ANNOTATIONS
        or (subtype in _PDF_UNREAD_XOBJECTS and subtype != "/Image")
        or _pdf_font_unread(value)
    )


def _marked_strings(properties: Any) -> list[str]:
    """The replacement texts (``_PDF_MARKED_TEXT``) of a property list or
    a structure element."""
    texts = [_resolved(properties.get(key)) for key in _PDF_MARKED_TEXT]
    return [
        text.decode("latin-1") if isinstance(text, bytes) else text
        for text in texts
        if isinstance(text, (str, bytes))
    ]


class _PdfWalk:
    """What a walk over EVERY object of a PDF found (``_pdf_walk``): content
    this reader does not read (``unread``), the content streams drawn
    outside a page's own content — form XObjects, tiling patterns and
    every annotation appearance, read as drawn — the replacement texts of
    marked content and structure elements, JavaScript actions, and widgets
    drawn without an appearance (a viewer draws them from the value)."""

    def __init__(self) -> None:
        self.unread = False
        # The object numbers of the file's image XObjects (``images``), and
        # the object being walked (``number``).
        self.images: set[int] = set()
        self.number = 0
        self.drawn: dict[int, Any] = {}
        self.marked: list[str] = []
        self.actual = False
        self.scripts = False
        self.bare_widgets = False

    def dictionary(self, value: Any, *, top: bool = True) -> None:
        self.unread = self.unread or _pdf_dictionary_unread(value)
        subtype = _resolved(value.get("/Subtype"))
        if subtype == "/Image":
            # An image XObject is a stream, an object of its own: one nested
            # in another object is nothing a page draws — unread.
            if top:
                self.images.add(self.number)
            else:
                self.unread = True
        if subtype == "/Form" or _resolved(value.get("/PatternType")) == 1:
            self.drawn[id(value)] = value
        if "/AP" in value:
            appearance = _resolved(value["/AP"])
            for key in ("/N", "/R", "/D"):
                for stream in _streams(_get(appearance, key)):
                    self.drawn[id(stream)] = stream
        elif subtype == "/Widget":
            self.bare_widgets = True
        self.marked += _marked_strings(value)
        self.actual = self.actual or "/ActualText" in value
        self.scripts = self.scripts or bool(
            _resolved(value.get("/S")) == "/JavaScript" or "/JS" in value
        )

    def value(self, value: Any, depth: int = 0) -> None:
        """One object and what is nested in it DIRECTLY (a reference is
        its own object, walked on its own)."""
        from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

        if depth > _PDF_MAX_NESTING:
            self.unread = True
            return
        if isinstance(value, DictionaryObject):
            self.dictionary(value, top=depth == 0)
            items: Any = value.values()
        elif isinstance(value, ArrayObject):
            items = value
        else:
            return
        for item in items:
            if not isinstance(item, IndirectObject):
                self.value(item, depth + 1)


def _pdf_walk(reader: Any) -> _PdfWalk:
    """Walk ANY object of the file — every one the cross-reference table
    and the object streams list, reached from a page or not: walking every
    object, not the paths a viewer takes, means no key or nesting hides
    one."""
    from pypdf.generic import IndirectObject

    numbers = [
        (number, generation) for generation, table in reader.xref.items() for number in table
    ]
    numbers += [(number, 0) for number in reader.xref_objStm]
    walk = _PdfWalk()
    for number, generation in numbers:
        walk.number = number
        walk.value(reader.get_object(IndirectObject(number, generation, reader)))
    return walk


def _page_images(page: Any) -> set[int]:
    """The object numbers of the image XObjects a page DRAWS — named by a
    ``Do`` in its content stream, or in that of a form it draws (each form
    followed once) — with each such image's soft mask and stencil mask:
    what OCR of the rendered page reads. An image a resource dictionary
    merely lists, or one reached any other way (a thumbnail, a pattern, an
    annotation, an unreferenced object), is not in it."""
    from pypdf.generic import ContentStream, IndirectObject

    images: set[int] = set()
    forms: set[int] = set()
    pending = [(page.get_contents(), page.get("/Resources"))]
    while pending:
        contents, resources = pending.pop()
        xobjects = _resolved(_get(_resolved(resources), "/XObject"))
        if contents is None or not hasattr(xobjects, "raw_get"):
            continue
        if not isinstance(contents, ContentStream):
            contents = ContentStream(contents, page.pdf)
        names = {ops[0] for ops, operator in contents.operations if operator == b"Do" and ops}
        for name in names:
            ref = xobjects.raw_get(name) if name in xobjects else None
            if not isinstance(ref, IndirectObject):
                continue
            target: Any = ref.get_object()
            subtype = _resolved(_get(target, "/Subtype"))
            if subtype == "/Image":
                images.add(ref.idnum)
                for key in ("/SMask", "/Mask"):
                    mask = target.raw_get(key) if key in target else None
                    if isinstance(mask, IndirectObject):
                        images.add(mask.idnum)
            elif subtype == "/Form" and ref.idnum not in forms:
                forms.add(ref.idnum)
                pending.append((target, _get(target, "/Resources")))
    return images


def _marked_visitor(walk: _PdfWalk) -> Callable[..., None]:
    """A content-stream visitor collecting the replacement texts of
    marked-content sequences whose property list is inline (into
    ``walk``)."""
    from pypdf.generic import DictionaryObject

    def visit(operator: bytes, operands: Any, *_: Any) -> None:
        if operator in (b"BDC", b"DP") and len(operands) > 1:
            properties = _resolved(operands[1])
            if isinstance(properties, DictionaryObject):
                walk.marked.extend(_marked_strings(properties))
                walk.actual = walk.actual or "/ActualText" in properties

    return visit


def _read_drawn(drawer: Any, stream: Any, reading: Reading, visit: Callable[..., None]) -> None:
    """The text a form, pattern or appearance stream draws. Incomplete when
    it paints an inline image, shows text that reads as nothing (no
    resources to find its font by) or a character pypdf could not map."""
    from pypdf.generic import StreamObject

    if not isinstance(stream, StreamObject):
        reading.complete = False
        return
    data = stream.get_data()
    text = drawer.extract_xform_text(stream, (0, 90, 180, 270), visitor_operand_before=visit)
    if (
        _INLINE_IMAGE.search(data)
        or "\ufffd" in text
        or (not text.strip() and _SHOWS_TEXT.search(data))
    ):
        reading.complete = False
    reading.add(text)


def _outline_titles(items: Any, reading: Reading, depth: int = 0) -> None:
    for item in items:
        if isinstance(item, list):
            if depth < 16:
                _outline_titles(item, reading, depth + 1)
        else:
            reading.add(str(getattr(item, "title", "") or ""))


def _page_paints(page: Any) -> bool:
    """Whether a page's content paints anything (``_PAINTS``)."""
    contents = page.get_contents()
    return contents is not None and bool(_PAINTS.search(contents.get_data()))


# --- PDF text as laid out and as stood for ------------------------------------------

_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
# Annotation flags keeping an appearance off the page: Invisible, Hidden, NoView.
_ANNOTATION_UNSHOWN = 1 | 2 | 32
# How deep forms drawing forms are followed for the row reading, and how
# many in all per page.
_ROW_DEPTH = 8
_ROW_FORMS = 64


def _get(value: Any, key: str) -> Any:
    """A dictionary entry, or None when ``value`` is no dictionary."""
    return value.get(key) if hasattr(value, "get") else None


def _product(first: Any, then: Any) -> tuple[float, ...]:
    """The PDF matrix ``first`` followed by ``then`` (row vectors)."""
    a, b, c, d, e, f = map(float, first)
    g, h, i, j, k, m = map(float, then)
    return (
        a * g + b * i,
        a * h + b * j,
        c * g + d * i,
        c * h + d * j,
        e * g + f * i + k,
        e * h + f * j + m,
    )


def _numbers(value: Any, count: int) -> tuple[float, ...] | None:
    """A PDF array of ``count`` numbers, or None."""
    items = _resolved(value)
    try:
        numbers = tuple(float(_resolved(item)) for item in items)
    except (TypeError, ValueError):
        return None
    return numbers if len(numbers) == count else None


def _placement(annotation: Any, stream: Any) -> tuple[float, ...] | None:
    """The matrix placing an annotation's appearance on the page: its
    ``/BBox`` transformed by its ``/Matrix``, mapped onto the annotation's
    ``/Rect`` (PDF 32000-1, 12.5.5); None when degenerate."""
    rect = _numbers(annotation.get("/Rect"), 4)
    box = _numbers(stream.get("/BBox"), 4)
    matrix = _numbers(stream.get("/Matrix"), 6) or _IDENTITY
    if rect is None or box is None or rect[0] == rect[2] or rect[1] == rect[3]:
        return None
    corners = [_product((1, 0, 0, 1, x, y), matrix)[4:] for x in box[0::2] for y in box[1::2]]
    left, bottom = min(x for x, _ in corners), min(y for _, y in corners)
    width = max(x for x, _ in corners) - left
    height = max(y for _, y in corners) - bottom
    if width <= 0 or height <= 0:
        return None
    sx, sy = (rect[2] - rect[0]) / width, (rect[3] - rect[1]) / height
    return _product(
        matrix,
        (sx, 0, 0, sy, min(rect[0], rect[2]) - left * sx, min(rect[1], rect[3]) - bottom * sy),
    )


def _shown_appearance(annotation: Any) -> Any:
    """The normal appearance an annotation shows (its ``/AS`` state when
    it has several), or None (none, or not shown)."""
    from pypdf.generic import DictionaryObject, StreamObject

    flags = _resolved(_get(annotation, "/F"))
    if isinstance(flags, int) and flags & _ANNOTATION_UNSHOWN:
        return None
    appearance = _resolved(_get(_resolved(_get(annotation, "/AP")), "/N"))
    if isinstance(appearance, DictionaryObject) and not isinstance(appearance, StreamObject):
        appearance = _resolved(appearance.get(_resolved(_get(annotation, "/AS"))))
    return appearance if isinstance(appearance, StreamObject) else None


class _Rows:
    """A page's text fragments placed on the page — its own content's,
    each form it draws and each annotation appearance shown on it — so the
    text of different drawings on one line can be read joined, as the page
    SHOWS it: page text followed by a form field whose appearance draws the
    rest of a value on the same line. With ``elements`` (see
    ``_struct_actual_text``) marked content's ``/ActualText`` stands in for
    its glyphs (``_StandsFor``)."""

    def __init__(self, drawer: Any, elements: Any = None) -> None:
        self.drawer = drawer
        self.elements = elements
        self.fragments: list[tuple[float, float, float, int, str]] = []
        self.sources = 0
        self.budget = _ROW_FORMS

    def collect(
        self, content: Any, transform: tuple[float, ...], depth: int = 0, number: Any = None
    ) -> None:
        """``content`` (a page or a form stream, object ``number``) drawn
        through ``transform``: its text fragments, then those of every form
        it draws (each a source of its own; ``_ROW_DEPTH`` deep and
        ``_ROW_FORMS`` in all at most)."""
        source, self.sources = self.sources, self.sources + 1
        resources = _resolved(content.get("/Resources"))
        stands = None if self.elements is None else _StandsFor(resources, number, self.elements)
        forms: list[tuple[Any, tuple[float, ...]]] = []
        level = [0]

        def before(operator: bytes, operands: Any, cm: Any, tm: Any) -> None:
            if stands is not None:
                stands.before(operator, operands)
            if operator == b"Do":
                if not level[0] and operands:
                    forms.append((operands[0], tuple(cm)))
                level[0] += 1

        def after(operator: bytes, *_: Any) -> None:
            if stands is not None:
                stands.after(operator)
            level[0] -= operator == b"Do"

        def text(chunk: str, cm: Any, tm: Any, font: Any, size: Any) -> None:
            if level[0] or not chunk.strip():
                return
            at = _product(_product(tm, cm), transform)
            scale = abs(at[0] * at[3] - at[1] * at[2]) ** 0.5
            self.fragments.append((at[5], at[4], float(size or 0) * scale, source, chunk))

        if hasattr(content, "extract_text"):
            content.extract_text(
                orientations=(0,),
                visitor_operand_before=before,
                visitor_operand_after=after,
                visitor_text=text,
            )
        else:
            self.drawer.extract_xform_text(content, (0,), 200.0, before, after, text)
        xobjects = _resolved(_get(resources, "/XObject"))
        for name, cm in forms:
            raw = _get(xobjects, name)
            form = _resolved(raw)
            if depth < _ROW_DEPTH and self.budget and _resolved(_get(form, "/Subtype")) == "/Form":
                self.budget -= 1
                matrix = _numbers(form.get("/Matrix"), 6) or _IDENTITY
                placed = _product(_product(matrix, cm), transform)
                self.collect(form, placed, depth + 1, getattr(raw, "idnum", None))

    def lines(self) -> list[str]:
        """Each line on which text of two drawings meets, joined as shown
        (directly, and with a space for a value written in groups)."""
        rows: list[list[Any]] = []
        for fragment in sorted(self.fragments, key=lambda f: (-f[0], f[1])):
            y, _, size, _, _ = fragment
            if rows and abs(rows[-1][0] - y) <= max(rows[-1][1], size, 1.0) / 2:
                rows[-1][2].append(fragment)
            else:
                rows.append([y, size, [fragment]])
        joined: list[str] = []
        for _, _, row in rows:
            if len({fragment[3] for fragment in row}) > 1:
                texts = [f[4].strip("\n") for f in sorted(row, key=lambda f: f[1])]
                joined += ["".join(texts), " ".join(texts)]
        return joined


def _page_rows(page: Any, drawer: Any, elements: Any = None) -> list[str]:
    """``_Rows`` for a page that draws forms or shows annotation
    appearances (otherwise its text reading already is the page's)."""
    rows = _Rows(drawer, elements)
    appearances = [
        (annotation, stream)
        for annotation in map(_resolved, _resolved(page.get("/Annots")) or ())
        if (stream := _shown_appearance(annotation)) is not None
    ]
    resources = _resolved(_get(_resolved(page.get("/Resources")), "/XObject"))
    if not appearances and not resources:
        return []
    rows.collect(page, _IDENTITY, number=getattr(page.indirect_reference, "idnum", None))
    for annotation, stream in appearances:
        placement = _placement(annotation, stream)
        if placement is not None:
            rows.collect(stream, placement)
    return rows.lines()


def _struct_actual_text(reader: Any) -> dict[tuple[int, int], tuple[int, str]]:
    """The ``/ActualText`` of structure elements by the marked content
    they stand for: (object number of the page or form stream, MCID) →
    (the element, its text). An element's text stands for everything
    under it (an outer one wins)."""
    from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

    found: dict[tuple[int, int], tuple[int, str]] = {}
    root = _resolved(reader.trailer["/Root"])
    stack: list[tuple[Any, Any, tuple[int, str] | None, int]] = [
        (root.get("/StructTreeRoot"), None, None, 0)
    ]
    seen: set[int] = set()
    while stack:
        node, page, text, depth = stack.pop()
        reference = node
        node = _resolved(node)
        if depth > 64:
            continue
        if isinstance(node, int) and page is not None and text is not None:
            found[(page, int(node))] = text
        elif isinstance(node, (ArrayObject, DictionaryObject)) and id(node) in seen:
            continue
        elif isinstance(node, ArrayObject):
            seen.add(id(node))
            stack += [(item, page, text, depth + 1) for item in node]
        elif isinstance(node, DictionaryObject):
            seen.add(id(node))
            container = node.get("/Stm") or node.get("/Pg")
            if isinstance(container, IndirectObject):
                page = container.idnum
            actual = _resolved(node.get("/ActualText"))
            if text is None and isinstance(actual, (str, bytes)):
                value = actual.decode("latin-1") if isinstance(actual, bytes) else str(actual)
                text = (id(reference), value)
            if "/MCID" in node:
                stack.append((node["/MCID"], page, text, depth + 1))
            stack.append((node.get("/K"), page, text, depth + 1))
    return found


class _StandsFor:
    """A content-stream visitor showing the ``/ActualText`` of marked
    content IN PLACE of the text drawn inside it, as text extraction that
    honours it (poppler, Acrobat's copy) reads the line — the replacement
    once, where its first text is shown. A marked-content sequence's own
    property list (inline or named in the resources of the page or form
    drawing it) or a structure element standing for its MCID
    (``_struct_actual_text``) gives the text; ``used`` once it stood for
    anything."""

    def __init__(self, resources: Any, container: int | None, elements: Any) -> None:
        self.elements = elements
        self.resources: list[tuple[Any, int | None]] = [(resources, container)]
        self.marked: list[tuple[int, str] | None] = []
        self.shown: set[int] = set()
        self.used = False

    def _replacement(self, operands: Any) -> tuple[int, str] | None:
        from pypdf.generic import NameObject

        properties: Any = _resolved(operands[1]) if len(operands) > 1 else None
        resources, container = self.resources[-1]
        if isinstance(properties, NameObject):
            properties = _resolved(_get(_resolved(_get(resources, "/Properties")), properties))
        actual = _resolved(_get(properties, "/ActualText"))
        if isinstance(actual, (str, bytes)):
            return (
                id(properties),
                actual.decode("latin-1") if isinstance(actual, bytes) else actual,
            )
        mcid = _resolved(_get(properties, "/MCID"))
        return self.elements.get((container, mcid)) if isinstance(mcid, int) else None

    def _form(self, operands: Any) -> None:
        from pypdf.generic import IndirectObject

        resources, _ = self.resources[-1]
        raw = _get(_resolved(_get(resources, "/XObject")), operands[0]) if operands else None
        inner = _resolved(_get(_resolved(raw), "/Resources"))
        number = raw.idnum if isinstance(raw, IndirectObject) else None
        self.resources.append((inner or resources, number))

    def before(self, operator: bytes, operands: Any, *_: Any) -> None:
        if operator in (b"BMC", b"BDC"):
            self.marked.append(self._replacement(operands) if operator == b"BDC" else None)
        elif operator == b"EMC" and self.marked:
            self.marked.pop()
        elif operator == b"Do":
            self._form(operands)
        elif operator in _SHOWS and operands:
            self._show(operator, operands)

    def after(self, operator: bytes, *_: Any) -> None:
        if operator == b"Do" and len(self.resources) > 1:
            self.resources.pop()

    def _show(self, operator: bytes, operands: Any) -> None:
        replacement = next((mark for mark in self.marked if mark is not None), None)
        if replacement is None:
            return
        key, text = replacement
        shown = "" if key in self.shown else text
        self.shown.add(key)
        self.used = True
        at = _SHOWS[operator]
        if at < len(operands):
            operands[at] = [shown] if operator == b"TJ" else shown


# Text-showing operators and which operand holds the text they show.
_SHOWS = {b"Tj": 0, b"'": 0, b'"': 2, b"TJ": 0}


def _stood_for(page: Any, drawer: Any, elements: Any, stream: Any = None) -> str | None:
    """A page's (or a form's, ``stream``) text with marked content's
    ``/ActualText`` in place (``_StandsFor``), when any stands for text."""
    content = page if stream is None else stream
    reference = getattr(content, "indirect_reference", None)
    visitor = _StandsFor(
        _resolved(content.get("/Resources")), getattr(reference, "idnum", None), elements
    )
    if stream is None:
        text = page.extract_text(
            visitor_operand_before=visitor.before, visitor_operand_after=visitor.after
        )
    else:
        text = drawer.extract_xform_text(
            stream, (0, 90, 180, 270), 200.0, visitor.before, visitor.after
        )
    return text if visitor.used else None


def read_pdf(data: bytes, reading: Reading) -> None:
    """A PDF's text (see the module): every page's text layer, what every
    form XObject, tiling pattern and annotation appearance draws, and the
    replacement text of marked content. Incomplete on a page drawing an
    image, on anything ``_pdf_walk`` finds unread in ANY object of the file
    (an attached file or media, an image, a font whose text pypdf cannot
    map), on a page or drawing that shows text but reads as none,
    JavaScript that may redraw a form field (with NeedAppearances or a
    widget without an appearance), a document script, an XFA form, a
    character pypdf could not map (U+FFFD), or a failure.

    Only some of these are what OCR of the rendered pages reads
    (``Reading.ocr_gap``): an image a page draws (every image XObject of
    the file drawn by a page, ``_page_images``), and a page whose own text
    reads as nothing or as an unmapped character. Every other reason —
    an attachment, a script, an XFA form, an image no page draws, a font it
    cannot map, a form or appearance stream, the text cap — stays unread
    whatever an OCR service says."""
    from pypdf import PageObject, PdfReader

    reader = PdfReader(io.BytesIO(data), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        reading.complete = False
        return
    # How many pages the file has (a cloud OCR reading must cover them all).
    reading.pages = len(reader.pages)
    root: Any = reader.trailer["/Root"].get_object()
    names: Any = root.get("/Names")
    names = names.get_object() if names is not None else {}
    acroform: Any = root.get("/AcroForm")
    acroform = acroform.get_object() if acroform is not None else {}
    walk = _pdf_walk(reader)
    need_appearances = getattr(_resolved(acroform.get("/NeedAppearances")), "value", None)
    redrawn = need_appearances is True or walk.bare_widgets
    drawn_images: set[int] = set()
    for page in reader.pages:
        drawn_images |= _page_images(page)
    if (
        "/EmbeddedFiles" in names
        or "/JavaScript" in names
        or "/XFA" in acroform
        or walk.unread
        or not walk.images <= drawn_images
        or (walk.scripts and redrawn)
    ):
        reading.complete = False
    elif walk.images:
        # Every image of the file is drawn by a page: OCR of it reads them.
        reading.ocr_gap()
    visit = _marked_visitor(walk)
    for page in reader.pages:
        text = page.extract_text(visitor_operand_before=visit)
        if "\ufffd" in text or (not text.strip() and _page_paints(page)):
            # A character pypdf could not map, or a page that paints
            # something — outlines, a pattern, glyphs it could not map — yet
            # reads as no text at all: what OCR of the page reads.
            reading.ocr_gap()
        reading.add(text)
        reading.show(text)
        if len(page.images) > 0:
            reading.ocr_gap()
        for annotation in page.get("/Annots") or ():
            _pdf_strings(annotation, reading)
    drawer = reader.pages[0] if len(reader.pages) else PageObject(reader)
    for stream in walk.drawn.values():
        _read_drawn(drawer, stream, reading, visit)
    # What the pages SHOW: text of different drawings meeting on one line,
    # and marked content's /ActualText in place of the glyphs it stands for.
    elements = _struct_actual_text(reader) if walk.actual else None
    for page in reader.pages:
        for line in _page_rows(page, drawer):
            reading.add(line)
        if elements is not None:
            reading.add(_stood_for(page, drawer, elements) or "")
            for line in _page_rows(page, drawer, elements):
                reading.add(line)
    if elements is not None:
        for stream in walk.drawn.values():
            reading.add(_stood_for(drawer, drawer, elements, stream) or "")
    for name, field in (reader.get_fields() or {}).items():
        reading.add(name)
        for key in ("/V", "/RV"):
            _pdf_value_text(field.get(key), reading)
        value = _resolved(field.get("/V"))
        if isinstance(value, str) and value:
            # A filled-in form field, as the display reading shows it.
            reading.show(f"{name}: {value}")
    for text in walk.marked:
        reading.add(text)
    _pdf_strings(reader.metadata, reading)
    metadata: Any = root.get("/Metadata")
    if metadata is not None:
        reading.add(metadata.get_object().get_data().decode("utf-8", "replace"))
    _outline_titles(reader.outline, reading)


# --- spreadsheet number formats -------------------------------------------------------

_FORMAT_COLORS = frozenset({"black", "blue", "cyan", "green", "magenta", "red", "white", "yellow"})
_FORMAT_COLOR = re.compile(r"color\d{1,2}", re.I)
_FORMAT_CONDITION = re.compile(r"(<=|>=|<>|<|>|=)\s*(-?\d{1,15}(?:\.\d{1,15})?)")
_COMPARE: dict[str, Callable[[Decimal, Decimal], bool]] = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "=": lambda a, b: a == b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "<>": lambda a, b: a != b,
}
# A styles part: read first, so the cells of every other part find their
# number format (``_Formats``).
_STYLES_PART = re.compile(r"(?:.*/)?styles\.xml", re.I)
_PLACEHOLDERS = frozenset({"digit", "general", "@"})
_BLANK_DIGIT = {"0": "0", "?": " ", "#": ""}
# Values past this magnitude are not rendered (no identifier is that long).
_MAX_RENDERED = Decimal(10) ** 30
# Elapsed-time codes, and a locale tag whose high digits pick another
# calendar or digit script (``[$-2010409]``): not rendered.
_ELAPSED = re.compile(r"h+|m+|s+", re.I)
_FORMAT_SCRIPT = re.compile(r"0*[1-9A-Fa-f][0-9A-Fa-f]*[0-9A-Fa-f]{4}")
# Date and time codes by their letter; a date or time section showing at
# most this many digits needs no rendering (no identifier is that short).
_DATE_UNITS = frozenset("ymdhs")
_DATE_SHORT = 4
_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
# The date systems' day 0: in the 1900 one day 60 is the 29 February 1900
# Excel counts, so the days before it are a day behind.
_EPOCH_1900 = date(1899, 12, 30)
_EPOCH_1904 = date(1904, 1, 1)


def _format_sections(code: str) -> list[list[tuple[str, str]]]:
    """A number format code as sections (``;``) of tokens: ``lit`` (quoted,
    escaped, padding, fill and plain characters, a currency symbol),
    ``digit`` (0 # ?), ``general``, ``.``, ``,``, ``%``, ``@``, ``/``,
    ``condition``, ``elapsed`` (``[h]``, ``[mm]``, ``[ss]``) and ``bad``
    (every other letter — date and time codes are read by ``_date_parts``
    — scientific notation, a locale switching calendar or digit script, and
    any other bracketed code)."""
    sections: list[list[tuple[str, str]]] = [[]]
    tokens = sections[0]
    at = 0
    while at < len(code):
        ch = code[at]
        step = 1
        if ch == ";":
            tokens = []
            sections.append(tokens)
        elif ch == '"':
            end = code.find('"', at + 1)
            end = len(code) if end == -1 else end
            tokens.append(("lit", code[at + 1 : end]))
            step = end + 1 - at
        elif ch in "\\_*":
            # An escaped character; padding the width of one (a space); a
            # fill repeated to the cell's width (once).
            tokens.append(("lit", " " if ch == "_" else code[at + 1 : at + 2]))
            step = 2
        elif ch == "[":
            end = code.find("]", at)
            end = len(code) if end == -1 else end
            inner = code[at + 1 : end]
            if inner.startswith("$"):
                currency, _, locale = inner[1:].partition("-")
                tokens.append(("lit", currency))
                if _FORMAT_SCRIPT.fullmatch(locale):
                    tokens.append(("bad", "$"))  # another calendar or digit script
            elif _ELAPSED.fullmatch(inner):
                tokens.append(("elapsed", inner.lower()))
            elif _FORMAT_CONDITION.fullmatch(inner):
                tokens.append(("condition", inner))
            elif inner.lower() not in _FORMAT_COLORS and not _FORMAT_COLOR.fullmatch(inner):
                tokens.append(("bad", inner))
            step = end + 1 - at
        elif ch in "0#?":
            tokens.append(("digit", ch))
        elif code[at : at + 7].lower() == "general":
            tokens.append(("general", ""))
            step = 7
        elif code[at : at + 2].lower() in ("e+", "e-"):
            tokens.append(("bad", "E"))  # scientific notation (its sign included)
            step = 2
        elif ch in ".,%@/":
            tokens.append((ch, ch))
        elif ch.isalpha():
            tokens.append(("bad", ch))
        else:
            tokens.append(("lit", ch))
        at += step
    return sections


def _needs_rendering(tokens: list[tuple[str, str]]) -> bool:
    """Whether a section's literal text can join its digits into a value
    the stored number does not show: a literal between two placeholders
    (``000-00-0000``, ``0000 0000``), or one holding a digit or ``+``."""
    places = [n for n, (kind, _) in enumerate(tokens) if kind in _PLACEHOLDERS]
    if not places:
        return False
    return any(
        kind in ("lit", "/") and (places[0] < n < places[-1] or re.search(r"[0-9+]", text))
        for n, (kind, text) in enumerate(tokens)
    )


def _renderable(tokens: list[tuple[str, str]]) -> bool:
    """Whether ``_render`` shows this section as a spreadsheet would: digit
    placeholders, literals, one decimal point, thousands separators and
    percent — no date or time code, fraction, text or condition."""
    kinds = [kind for kind, _ in tokens]
    places = [n for n, kind in enumerate(kinds) if kind == "digit"]
    between = set(kinds[places[0] + 1 : places[-1]]) if places else set()
    return (
        not {"bad", "@", "/"} & set(kinds)
        and kinds.count(".") <= 1
        and not (places and "general" in kinds)
        # Thousands separators among literals between the digits: not rendered.
        and not {",", "lit"} <= between
    )


def _date_section(tokens: list[tuple[str, str]]) -> bool:
    """Whether a section shows a date or a time (a date or time code)."""
    return any(
        kind == "elapsed" or (kind == "bad" and text.lower() in _DATE_UNITS)
        for kind, text in tokens
    )


def _ampm(tokens: list[tuple[str, str]], at: int) -> str | None:
    """``AM/PM`` or ``A/P`` (as written) at token ``at``."""
    for size in (5, 3):
        window = tokens[at : at + size]
        text = "".join(t for _, t in window)
        kinds = {kind for kind, _ in window}
        if len(text) == size and text.lower() in ("am/pm", "a/p") and kinds <= {"bad", "/"}:
            return text
    return None


def _repeat(tokens: list[tuple[str, str]], at: int) -> int:
    """How many tokens from ``at`` repeat it (a letter in either case)."""
    kind, text = tokens[at]
    size = 1
    while at + size < len(tokens) and tokens[at + size][0] == kind:
        if tokens[at + size][1].lower() != text.lower():
            break
        size += 1
    return size


def _date_parts(tokens: list[tuple[str, str]]) -> list[tuple[str, Any]] | None:
    """A date or time section as parts: ``lit`` text; a unit — ``y``,
    ``mo`` (a month, or ``mi``: minutes right after hours or right before
    seconds), ``d``, ``h``, ``s`` — with how many letters wrote it;
    ``elapsed`` time (``[h]``); ``ampm``; ``frac`` (fractional seconds,
    ``ss.00``). None for what is not rendered here: eras and other
    calendars, digit placeholders but fractional seconds, text, General."""
    parts: list[tuple[str, Any]] = []
    at = 0
    while at < len(tokens):
        kind, text = tokens[at]
        marker = _ampm(tokens, at)
        step = 1
        if marker is not None:
            parts.append(("ampm", marker))
            step = len(marker)
        elif kind == "bad" and text.lower() in _DATE_UNITS:
            step = _repeat(tokens, at)
            parts.append((text.lower(), step))
        elif kind == "digit" and text == "0" and _after_seconds(parts):
            step = _repeat(tokens, at)
            parts.append(("frac", step))
        elif kind in ("lit", ",", ".", "/", "elapsed"):
            parts.append(("lit" if kind != "elapsed" else kind, text))
        elif kind != "condition":
            return None
        at += step
    return _minutes(parts)


def _after_seconds(parts: list[tuple[str, Any]]) -> bool:
    """Whether the parts end in seconds and a decimal point."""
    if parts[-1:] != [("lit", ".")] or len(parts) < 2:
        return False
    kind, count = parts[-2]
    return kind == "s" or (kind == "elapsed" and count.startswith("s"))


def _minutes(parts: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """``m``/``mm`` right after hours or right before seconds (text in
    between aside) is minutes; any other ``m`` a month."""
    units = [n for n, (kind, _) in enumerate(parts) if kind not in ("lit", "ampm", "frac")]
    out = list(parts)
    for position, n in enumerate(units):
        kind, count = parts[n]
        if kind != "m":
            continue
        before = parts[units[position - 1]] if position else ("", "")
        after = parts[units[position + 1]] if position + 1 < len(units) else ("", "")
        hours = before[0] == "h" or (before[0] == "elapsed" and before[1].startswith("h"))
        seconds = after[0] == "s" or (after[0] == "elapsed" and after[1].startswith("s"))
        out[n] = ("mi" if count <= 2 and (hours or seconds) else "mo", count)
    return out


def _date_digits(parts: list[tuple[str, Any]]) -> int:
    """How many digits a date or time section can show at most."""
    shown = {"h": 2, "mi": 2, "s": 2, "elapsed": 12}
    total = 0
    for kind, value in parts:
        if kind == "lit":
            total += sum(ch.isdigit() for ch in value)
        elif kind == "frac":
            total += value
        elif kind in ("y", "mo", "d"):
            total += 2 if value <= 2 else 4 if kind == "y" else 0
        else:
            total += shown.get(kind, 0)
    return total


def _conventional(parts: list[tuple[str, Any]]) -> bool:
    """A date or time as dates are written: no digit in its literal text,
    each unit shown as a number once (a day's or a month's name aside),
    text between every two numbers, and elapsed time only among times —
    its numbers then never join into a value."""
    numbers = [
        n
        for n, (kind, value) in enumerate(parts)
        if kind != "lit" and _date_digits([(kind, value)])
    ]
    units = [parts[n][0] for n in numbers]
    return (
        not any(kind == "lit" and re.search(r"[0-9+]", value) for kind, value in parts)
        and len(units) == len(set(units))
        and all(
            any(kind == "lit" and value for kind, value in parts[a + 1 : b])
            for a, b in zip(numbers, numbers[1:], strict=False)
        )
        and not ("elapsed" in units and {"y", "mo", "d"} & set(units))
    )


def _date_verdict(tokens: list[tuple[str, str]]) -> str:
    """A date or time section's verdict (see ``number_format``)."""
    parts = _date_parts(tokens)
    if parts is None:
        return "uncertain"
    if _date_digits(parts) <= _DATE_SHORT:
        return "plain"
    return "date" if _conventional(parts) else "render"


_VERDICTS = ("plain", "date", "render", "uncertain")


def number_format(code: str) -> str:
    """``plain`` (nothing to render: no literal joins the digits, a date
    or time showing at most ``_DATE_SHORT`` digits), ``date`` (a date or
    time written as dates are: rendered in a cell), ``render`` (literal
    text joins digits: ``render_number`` shows what a spreadsheet shows) or
    ``uncertain`` (joined in a way not rendered); the strongest of its
    sections'."""
    verdict = "plain"
    for tokens in _format_sections(code):
        if _date_section(tokens):
            found = _date_verdict(tokens)
        elif _needs_rendering(tokens):
            found = "render" if _renderable(tokens) else "uncertain"
        else:
            found = "plain"
        verdict = max(verdict, found, key=_VERDICTS.index)
    return verdict


def _unresolved(code: str) -> bool:
    """Whether a format code applied where the values it shows are not
    resolved (a conditional format, a chart) may show a value not read:
    one joining digits with literal text, or not rendered at all. A date
    written as dates are (``date``) shows no more than a date."""
    return number_format(code) in ("render", "uncertain")


def _serial_date(days: int, date1904: bool) -> tuple[int, int, int, int]:
    """Year, month, day and weekday (0: Sunday) of a serial day as Excel
    shows it: in the 1904 system, or the 1900 one with its 0 January and
    29 February 1900."""
    if date1904:
        moment = _EPOCH_1904 + timedelta(days=days)
        return moment.year, moment.month, moment.day, (moment.weekday() + 1) % 7
    if days == 0:
        return 1900, 1, 0, 6
    if days == 60:
        return 1900, 2, 29, 3
    moment = _EPOCH_1900 + timedelta(days=days + (days < 60))
    return moment.year, moment.month, moment.day, (days + 6) % 7


def _date_text(kind: str, count: Any, fields: dict[str, int]) -> str:
    """One part of a date shown (``_render_date``)."""
    if kind == "lit":
        return str(count)
    if kind == "ampm":
        half = len(count) // 2
        return str(count[:half] if fields["h"] < 12 else count[half + 1 :])
    if kind == "frac":
        return f"{fields['frac']:0{fields['places']}d}"[:count]
    if kind == "elapsed":
        return f"{fields['elapsed_' + count[0]]:0{len(count)}d}"
    value = fields["h12" if kind == "h" and fields["twelve"] else kind]
    if kind == "y":
        return f"{value % 100:02d}" if count <= 2 else f"{value:04d}"
    if count <= 2:
        return f"{value:0{count}d}"
    if kind == "mo":
        name = _MONTHS[value - 1]
        return name[:3] if count == 3 else name[:1] if count == 5 else name
    if kind == "d":
        name = _WEEKDAYS[fields["weekday"]]
        return name[:3] if count == 3 else name
    return f"{value:02d}"


def _render_date(
    parts: list[tuple[str, Any]], value: Decimal, date1904: bool, rounding: str
) -> str:
    """A serial date and time as ``parts`` show it, its time rounded to
    what the section shows by ``rounding``."""
    places = max((count for kind, count in parts if kind == "frac"), default=0)
    scale = 10**places
    ticks = int((value * 86400 * scale).quantize(Decimal(1), rounding=rounding))
    days, rest = divmod(ticks, 86400 * scale)
    seconds, fraction = divmod(rest, scale)
    year, month, day, weekday = _serial_date(days, date1904)
    hour = seconds // 3600
    fields = {
        "y": year,
        "mo": month,
        "d": day,
        "weekday": weekday,
        "h": hour,
        "h12": hour % 12 or 12,
        "mi": seconds // 60 % 60,
        "s": seconds % 60,
        "frac": fraction,
        "places": places,
        "twelve": any(kind == "ampm" for kind, _ in parts),
        "elapsed_h": ticks // (3600 * scale),
        "elapsed_m": ticks // (60 * scale),
        "elapsed_s": ticks // scale,
    }
    return "".join(_date_text(kind, count, fields) for kind, count in parts)


def render_date(tokens: list[tuple[str, str]], value: Decimal, date1904: bool) -> str | None:
    """A date or time section's rendering of ``value`` — rounded and
    truncated to what it shows, both when they differ (which one a
    spreadsheet shows at a boundary is not modelled) — or None when it
    shows no date (a negative value in the 1900 system, past 9999, or at
    most ``_DATE_SHORT`` digits). A negative value in the 1904 system is
    shown as its magnitude's date with a minus sign."""
    parts = _date_parts(tokens)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if parts is None or (sign and not date1904) or _date_digits(parts) <= _DATE_SHORT:
        return None
    try:
        shown = [
            sign + _render_date(parts, value, date1904, r) for r in (ROUND_HALF_UP, ROUND_FLOOR)
        ]
    except (OverflowError, ValueError):
        return None  # past 9999: shown as #####
    return "\n".join(dict.fromkeys(shown))


def iso_serial(text: str, date1904: bool) -> str | None:
    """A cell's ISO 8601 date (``t="d"``) as the serial number its format
    shows, or None when it does not parse."""
    try:
        moment = datetime.fromisoformat(text.strip()).replace(tzinfo=None)
    except ValueError:
        return None
    delta = moment - datetime.combine(_EPOCH_1904 if date1904 else _EPOCH_1900, datetime.min.time())
    serial = Decimal(delta.days) + Decimal(delta.seconds * 10**6 + delta.microseconds) / (
        86400 * 10**6
    )
    if not date1904 and serial < 61:
        serial -= 1  # before the 29 February 1900 Excel counts
    return str(serial)


def _render_digits(tokens: list[tuple[str, str]], value: Decimal) -> list[str]:
    """Each token's text for ``value`` (non-negative, scaled): integer
    placeholders filled from the right (the extra digits in the first),
    fraction placeholders from the left."""
    point = next((n for n, (kind, _) in enumerate(tokens) if kind == "."), len(tokens))
    whole = [n for n in range(point) if tokens[n][0] == "digit"]
    fraction = [n for n in range(point, len(tokens)) if tokens[n][0] == "digit"]
    rounded = value.quantize(Decimal(1).scaleb(-len(fraction)), rounding=ROUND_HALF_UP)
    integer, _, decimals = f"{rounded:f}".partition(".")
    integer = integer.lstrip("0")
    out = [text if kind in ("lit", "%") else "" for kind, text in tokens]
    for n in reversed(whole):
        out[n] = integer[-1:] or _BLANK_DIGIT[tokens[n][1]]
        integer = integer[:-1]
    if whole:
        out[whole[0]] = integer + out[whole[0]]
    elif point < len(tokens):
        out[point] = integer  # the integer part shows without a placeholder too
    grouped = any(tokens[n][0] == "," for n in range(whole[0], whole[-1])) if whole else False
    if grouped:
        digits = "".join(out[n] for n in whole)
        groups = f"{int(digits or 0):,}" if digits.strip() else digits
        out[whole[0]] = groups
        for n in whole[1:]:
            out[n] = ""
    kept = decimals.ljust(len(fraction), "0")
    for position, n in enumerate(fraction):
        out[n] = kept[position]
    for n in reversed(fraction):
        if out[n] != "0" or tokens[n][1] == "0":
            break
        out[n] = _BLANK_DIGIT[tokens[n][1]]
    if point < len(tokens):
        out[point] += "."
    return out


def _section_for(
    sections: list[list[tuple[str, str]]], value: Decimal
) -> tuple[str, list[tuple[str, str]]]:
    """The sign shown and the section that shows ``value``: by the
    sections' conditions (``[<=9999999]``: the first that holds, else the
    next unconditional one) or by sign (positive; negative, shown without
    its minus; zero)."""
    conditions = [
        next((text for kind, text in tokens if kind == "condition"), None) for tokens in sections
    ]
    if any(conditions):
        for condition, tokens in zip(conditions, sections, strict=True):
            match = _FORMAT_CONDITION.fullmatch(condition) if condition else None
            if match is None or _COMPARE[match[1]](value, Decimal(match[2])):
                return ("-" if value < 0 and match is None else ""), tokens
        return "", []
    if value < 0 and len(sections) > 1:
        return "", sections[1]
    if value == 0 and len(sections) > 2:
        return "", sections[2]
    return ("-" if value < 0 else ""), sections[0]


def render_number(code: str, stored: str, *, date1904: bool = False) -> str | None:
    """A numeric cell's ``stored`` value as the format ``code`` shows it,
    or None when there is nothing to render (see ``number_format``); a
    date in the workbook's date system (``date1904``)."""
    try:
        value = Decimal(stored.strip())
    except InvalidOperation:
        return None
    if not value.is_finite() or abs(value) >= _MAX_RENDERED:
        return None
    sections = _format_sections(code)[:3]
    sign, tokens = _section_for(sections, value)
    if _date_section(tokens):
        return render_date(tokens, value, date1904)
    value = abs(value)
    if not (_needs_rendering(tokens) and _renderable(tokens)):
        return None
    kinds = [kind for kind, _ in tokens]
    value = value * Decimal(100) ** kinds.count("%")
    last = max((n for n, kind in enumerate(kinds) if kind == "digit"), default=-1)
    point = kinds.index(".") if "." in kinds else len(kinds)
    value = value / Decimal(1000) ** kinds[last + 1 : point].count(",")
    if "general" in kinds:
        text = f"{value.normalize():f}" if value == value.to_integral() else f"{value:.10g}"
        out = [
            text if kind == "general" else (t if kind in ("lit", "%") else "") for kind, t in tokens
        ]
    else:
        out = _render_digits(tokens, value)
    return sign + "".join(out)


# Elements of a spreadsheet's number formatting (``_XmlText._structure``):
# the format tables, one of each per workbook.
_STRUCTURE = frozenset({"numFmt", "xf", "c", "formatCode", "numFmts", "cellXfs", "workbookPr"})
_FORMAT_TABLES = frozenset({"numFmts", "cellXfs"})
# A workbook part: read right after the styles, so every date is shown in
# the workbook's date system.
_WORKBOOK_PART = re.compile(r"(?:.*/)?workbook\.xml", re.I)


def _attribute(attributes: dict[str, str], local: str) -> str | None:
    """An attribute's value by its local name (any prefix)."""
    for key, value in attributes.items():
        if key.rpartition(":")[2] == local:
            return value
    return None


class _Formats:
    """A workbook's number formats, collected from its styles part(s)
    before its sheets are read: format codes by id, and each cell style's
    format id (``cellXfs``, indexed by a cell's ``s``). ``locked`` once the
    styles parts are read: a format table found later (a styles part under
    another name) could no longer reach the cells read before it; a second
    table of either kind (``tables``: another part named ``styles.xml``)
    could shift which format a cell finds. ``date1904``: the workbook's
    date system, which must be known before a date is shown (``dated``)."""

    def __init__(self) -> None:
        self.codes: dict[str, str] = {}
        self.styles: list[str] = []
        self.locked = False
        self.tables: set[str] = set()
        self.date1904 = False
        self.dated = False

    def code(self, style: str | None) -> str | None:
        if style is None or not style.isdigit() or int(style) >= len(self.styles):
            return None
        return self.codes.get(self.styles[int(style)])


# --- zip packages ------------------------------------------------------------------


class _XmlText:
    """One XML part read with expat: text and attribute values in document
    order, in several readings — joined within block elements (every piece
    of text where it stands), as SHOWN (joined, but without the content of
    elements a renderer keeps out of the line, ``_ASIDES`` and hidden runs,
    each read on its own instead) and split at every element boundary.
    Collecting stops past ``budget`` characters (the reading is then cut:
    ``over``), so memory follows the text cap, not the part's size. An
    element inlining binary data (``_INLINE_BINARY``) sets ``binary``: its
    content is not text. A style hiding text (``uncertain``) makes what is
    shown depend on a resolution this reader does not do.

    Spreadsheet cells: a numeric cell whose style's number format joins
    its digits with literal text (``000-00-0000``) is also read as the
    format SHOWS it (``render_number``, the ``formats`` of the workbook);
    a format joining them in a way not rendered — in a cell style, a
    conditional format or a chart — is ``uncertain``."""

    def __init__(self, budget: int, formats: _Formats | None = None) -> None:
        self.formats = formats
        self.rendered: list[str] = []
        self._cell: tuple[str | None, str | None, list[str]] | None = None
        self._code: list[str] | None = None
        self.joined: list[str] = []
        self.shown: list[str] = []
        self.asides: list[str] = []
        self.split: list[str] = []
        self.attributes: list[str] = []
        self.budget = budget
        self.over = False
        self.binary = False
        self.uncertain = False
        # Open elements: [local name, opened an aside, joins the next (a
        # paragraph whose mark is hidden)]; open complex fields (True while
        # in their instruction).
        self._open: list[list[Any]] = []
        self._hidden = 0
        self.fields: list[bool] = []

    def _spend(self, size: int) -> bool:
        self.budget -= size
        self.over = self.over or self.budget < 0
        return not self.over

    def _out(self) -> list[str]:
        return self.asides if self._hidden else self.shown

    def _hides(self, local: str, attributes: dict[str, str]) -> None:
        """A run property hiding text: the run (or the paragraph's mark),
        or — in a style or the document defaults — whatever uses it."""
        names = [frame[0] for frame in self._open]
        on = not any(
            key.rpartition(":")[2] == "val" and value.lower() in _OFF
            for key, value in attributes.items()
        )
        if not on or names[-1:] != ["rPr"]:
            return
        if "style" in names or "docDefaults" in names:
            self.uncertain = self.uncertain or local in ("vanish", "specVanish")
        elif names[-2:-1] == ["pPr"]:
            paragraphs = [frame for frame in self._open if frame[0] == "p"]
            if paragraphs:
                paragraphs[-1][2] = True
        elif names[-2:-1] == ["r"] and local == "vanish" and not self._open[-2][1]:
            self._open[-2][1] = True
            self._hidden += 1
            self.asides.append("\n")

    def _number_format(self, parent: str, attributes: dict[str, str]) -> None:
        """A format code: a cell style's (``numFmts``, collected while the
        styles are read) or one applied elsewhere (a conditional format, a
        chart), which cannot be matched to the values it shows."""
        code = _attribute(attributes, "formatCode") or ""
        if parent == "numFmts" and self.formats is not None and not self.formats.locked:
            self.formats.codes[_attribute(attributes, "numFmtId") or ""] = code
        elif parent in ("numFmts", "cellXfs") or _unresolved(code):
            self.uncertain = True

    def _cell_end(self) -> None:
        """A cell read: its value as its number format shows it, when the
        format joins the digits with literal text or shows a date; a date
        cell (``t="d"``) holds its value as an ISO 8601 date."""
        assert self._cell is not None
        style, kind, value = self._cell
        self._cell = None
        formats = self.formats
        code = formats.code(style) if formats is not None else None
        verdict = "plain" if code is None else number_format(code)
        if verdict == "uncertain":
            self.uncertain = True
        elif formats is not None and code is not None and verdict != "plain":
            if kind not in (None, "n", "d"):
                return
            formats.dated = True
            text = "".join(value)
            stored = iso_serial(text, formats.date1904) if kind == "d" else text
            shown = render_number(code, stored, date1904=formats.date1904) if stored else None
            self.uncertain = self.uncertain or stored is None
            if shown and self._spend(len(shown)):
                self.rendered.append(shown)

    def _structure(self, local: str, parent: str, attributes: dict[str, str]) -> None:
        """Spreadsheet structure: number formats, cell styles, cells, the
        format tables and the date system."""
        formats = self.formats
        if local in _FORMAT_TABLES and formats is not None:
            self.uncertain = self.uncertain or local in formats.tables
            formats.tables.add(local)
        elif local == "workbookPr" and formats is not None:
            if (_attribute(attributes, "date1904") or "").lower() in ("1", "true"):
                self.uncertain = self.uncertain or formats.dated
                formats.date1904 = True
        elif local == "numFmt":
            self._number_format(parent, attributes)
        elif local == "xf" and parent == "cellXfs":
            if self.formats is None or self.formats.locked:
                self.uncertain = True
            else:
                self.formats.styles.append(_attribute(attributes, "numFmtId") or "0")
        elif local == "c":
            self._cell = (_attribute(attributes, "s"), _attribute(attributes, "t"), [])
        elif local == "formatCode":
            self._code = []

    def start(self, name: str, attributes: dict[str, str]) -> None:
        local = name.rpartition(":")[2]
        if local in _STRUCTURE:
            self._structure(local, self._open[-1][0] if self._open else "", attributes)
        self.binary = self.binary or local in _INLINE_BINARY
        values = [value for key, value in attributes.items() if not key.startswith("xmlns")]
        if local in _HIDING:
            self._hides(local, attributes)
        if local == "text-properties" and any(
            key.rpartition(":")[2] == "display" and value != "true"
            for key, value in attributes.items()
        ):
            self.uncertain = True  # an OpenDocument style hiding text
        aside = local in _ASIDES
        self._open.append([local, aside, False])
        if aside:
            self._hidden += 1
            self.asides.append("\n")
        if self._spend(sum(map(len, values)) + 3):
            space = _SPACES.get(local) or self._character(local, attributes)
            self.joined.append(space)
            self._out().append(space)
            self.split.append("\n")
            self.attributes.extend(values)

    def _character(self, local: str, attributes: dict[str, str]) -> str:
        """What a WordprocessingML run element stands for in the line: a
        symbol (``w:sym``: its character, a symbol font's private-use code
        as its low byte — an unreadable one is ``uncertain``), or nothing;
        a complex field's boundary (``w:fldChar``) opens or closes its
        instruction."""
        parent = self._open[-2][0] if len(self._open) > 1 else ""
        if local == "fldChar":
            self._field_boundary(_attribute(attributes, "fldCharType"))
        elif local == "sym" and parent == "r":
            try:
                point = int(_attribute(attributes, "char") or "", 16)
            except ValueError:
                point = -1
            point -= 0xF000 if 0xF000 <= point <= 0xF0FF else 0
            if 0x20 <= point < 0xD800 or 0xE000 <= point <= 0x10FFFF:
                return chr(point)
            self.uncertain = True
        return ""

    def _field_boundary(self, kind: str | None) -> None:
        """A complex field's ``begin``, ``separate`` or ``end``: everything
        between its begin and its separate (or end) is the instruction — a
        renderer shows the result — read on its own, like ``instrText``.
        Boundaries out of order are ``uncertain``."""
        if kind == "begin":
            self.fields.append(True)
            self._hidden += 1
        elif kind == "separate" and self.fields and self.fields[-1]:
            self.fields[-1] = False
            self._hidden -= 1
        elif kind == "end" and self.fields:
            self._hidden -= self.fields.pop()
        else:
            self.uncertain = True
        self.asides.append("\n")

    def end(self, name: str) -> None:
        local, aside, joins = self._open.pop()  # expat's events are balanced
        if local == "c" and self._cell is not None:
            self._cell_end()
        elif local == "formatCode" and self._code is not None:
            self.uncertain = self.uncertain or _unresolved("".join(self._code))
            self._code = None
        if self._spend(3):
            if local in _BLOCKS:
                self.joined.append("\n")
                if not joins:
                    self._out().append("\n")
            self.split.append("\n")
        if aside:
            self._hidden -= 1
            self.asides.append("\n")

    def data(self, text: str) -> None:
        if self._spend(3 * len(text)):
            self.joined.append(text)
            self._out().append(text)
            self.split.append(text)
            if self._code is not None:
                self._code.append(text)
            elif self._cell is not None and [f[0] for f in self._open[-2:]] == ["c", "v"]:
                self._cell[2].append(text)

    def text(self) -> str:
        """Every reading, joined; the shown one only where it differs."""
        joined, shown = "".join(self.joined), "".join(self.shown)
        readings = [joined, shown if shown != joined else "", "".join(self.asides)]
        text = "\n".join((*readings, "".join(self.split), *self.attributes, *self.rendered))
        return _BLANK_LINES.sub("\n", text).strip("\n")


def _refuse_doctype(*_: Any) -> None:
    raise Refused("document type declaration")


def xml_text(
    data: bytes, budget: int = 1 << 62, formats: _Formats | None = None
) -> tuple[str, bool]:
    """The text of one XML part (see ``_XmlText``; ``formats``: its
    workbook's number formats) and whether all of it was read: it fit
    ``budget`` characters, inlines no binary data, and hides no text or
    shows no number in a way not read. A document type declaration (and
    with it every entity declaration) is refused."""
    text, whole, _ = xml_reading(data, budget, formats)
    return text, whole


def xml_reading(
    data: bytes, budget: int = 1 << 62, formats: _Formats | None = None
) -> tuple[str, bool, str]:
    """``xml_text`` and the part's text as SHOWN (its display reading)."""
    reader = _XmlText(budget, formats)
    parser = expat.ParserCreate()
    parser.StartDoctypeDeclHandler = _refuse_doctype
    parser.EntityDeclHandler = _refuse_doctype
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.StartElementHandler = reader.start
    parser.EndElementHandler = reader.end
    parser.CharacterDataHandler = reader.data
    parser.Parse(data, True)
    unread = reader.over or reader.binary or reader.uncertain or bool(reader.fields)
    shown = _BLANK_LINES.sub("\n", "".join(reader.shown)).strip("\n")
    return reader.text(), not unread, shown


def _inflate(archive: zipfile.ZipFile, info: zipfile.ZipInfo, budget: int) -> bytes:
    """One entry's bytes, read no further than the package's remaining
    ``budget`` and ``MAX_RATIO`` times its compressed size."""
    limit = min(budget, max(info.compress_size, 1) * MAX_RATIO)
    with archive.open(info) as entry:
        data = entry.read(limit + 1)
    if len(data) > limit:
        raise Limit("inflated size")
    return data


def package_kind(archive: zipfile.ZipFile, names: set[str]) -> str | None:
    """``odf`` (a ``mimetype`` entry naming an OpenDocument type),
    ``ooxml`` (a ``[Content_Types].xml`` part), else None."""
    if "mimetype" in names:
        with archive.open("mimetype") as entry:
            if entry.read(128).startswith(_ODF_MIME):
                return "odf"
    if "[Content_Types].xml" in names:
        return "ooxml"
    return None


def read_package(data: bytes, reading: Reading, formats: set[str], max_inflated: int) -> str:
    """An OOXML/ODF package's text; returns its kind — ``zip`` for another
    archive, which (like a kind not in ``formats``) is left unread. Every
    XML part is read; any other entry but the ``mimetype`` marker and a
    thumbnail makes the reading incomplete."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_ENTRIES:
            raise Limit("entries")
        names = {info.filename for info in infos}
        kind = reading.kind = package_kind(archive, names) or "zip"
        if kind not in formats:
            return kind
        # What a zip reader lists besides the parts: entry names and comments.
        reading.add(archive.comment.decode("latin-1"))
        for info in infos:
            reading.add(info.filename)
            reading.add(info.comment.decode("latin-1"))
        budget = max_inflated
        # The styles first — every sheet's cells then find their number
        # format — then the workbook (its date system).
        workbook = _Formats() if kind == "ooxml" else None
        infos.sort(
            key=lambda info: (
                not _STYLES_PART.fullmatch(info.filename),
                not _WORKBOOK_PART.fullmatch(info.filename),
            )
        )
        for info in infos:
            name = info.filename
            if workbook is not None and not _STYLES_PART.fullmatch(name):
                workbook.locked = True
            if info.is_dir() or name == "mimetype" or _THUMBNAILS.fullmatch(name):
                continue
            if info.flag_bits & 0x1 or not _XML_PART.fullmatch(name):
                reading.complete = False  # an encrypted entry, an image, an embedded file
                continue
            part = _inflate(archive, info, budget)
            budget -= len(part)
            text, whole, shown = xml_reading(part, reading.max_chars - reading.size, workbook)
            reading.complete = reading.complete and whole
            reading.add(text)
            if not _UNSHOWN_PART.search(name):
                reading.show(shown)
        if any(_WORKBOOK_PART.fullmatch(name) for name in names):
            # A workbook's cells cite their strings by number and hold values
            # a number format shows: its parts' text is no faithful display.
            reading.no_display()
        return kind


# --- markup and RTF ----------------------------------------------------------------


# Elements whose content a browser does not show as text.
_MARKUP_UNSHOWN = frozenset({"script", "style", "template"})


class _MarkupText(html.parser.HTMLParser):
    """Markup's text and attribute values, character references resolved
    (``pieces``) — and its text as a browser shows it (``shown``: no tags,
    attributes, comments, scripts or styles)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pieces: list[str] = []
        self.shown: list[str] = []
        self._unshown = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.pieces.extend(value for _, value in attrs if value)
        self._unshown += tag in _MARKUP_UNSHOWN

    def handle_endtag(self, tag: str) -> None:
        if tag in _MARKUP_UNSHOWN and self._unshown:
            self._unshown -= 1

    def handle_data(self, data: str) -> None:
        self.pieces.append(data)
        if not self._unshown:
            self.shown.append(data)

    def handle_comment(self, data: str) -> None:
        self.pieces.append(data)


def keeps_ascii(codec: str) -> bool:
    """Whether ``codec`` decodes every printable ASCII byte (and tab, line
    feed, carriage return) to itself — as a charset a document can declare
    in ASCII must. UTF-16/32, EBCDIC code pages and UTF-7 do not: a page
    written in ASCII that declares one decodes to something else."""
    try:
        return _ASCII.decode(codec) == _ASCII.decode("ascii")
    except (LookupError, UnicodeDecodeError):
        return False


class _MetaCharset(html.parser.HTMLParser):
    """The first ``<meta charset>`` or ``<meta http-equiv=Content-Type
    content="…; charset=…">`` of markup — a real tag, not text in a
    comment, a script or an attribute value (the prescan browsers do)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.charset: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value for key, value in attrs if value is not None}
        if tag != "meta" or self.charset is not None:
            return
        if "charset" in values:
            self.charset = values["charset"].strip()
        elif values.get("http-equiv", "").lower() == "content-type":
            match = _CONTENT_CHARSET.search(values.get("content", ""))
            self.charset = match[1] if match else None


def declared_charset(data: bytes) -> tuple[str | None, bool]:
    """The charset markup declares — an XML declaration's at the very
    start, else the first real meta tag's within the first 1024 bytes
    (where browsers look), never a mention elsewhere (a comment, a script)
    — and whether every charset mentioned in the first 2048 bytes names
    the same codec (a decoy mention makes the decoding unsure)."""
    xml = _XML_DECLARATION.match(data)
    if xml is not None:
        declared: str | None = xml[1].decode("ascii")
    else:
        meta = _MetaCharset()
        meta.feed(data[:1024].decode("latin-1"))
        declared = meta.charset
    mentions = {_codec_name(m[1].decode("ascii")) for m in _CHARSET.finditer(data[:2048])}
    return declared, declared is not None and not mentions - {_codec_name(declared)}


def _codec_name(name: str) -> str | None:
    try:
        return codecs.lookup(name).name
    except LookupError:
        return None


def decode_markup(data: bytes) -> tuple[str, bool]:
    """Markup decoded by the charset it declares (``declared_charset``: one
    that keeps ASCII, ``keeps_ascii``, since the declaration itself was
    read as ASCII) or its UTF-16 shape, strictly — ``exact`` True — or else
    as Windows-1252 with replacements (every ASCII byte kept; not exact).
    Charset mentions that disagree with the declaration leave it decoded as
    declared, but not exact."""
    codec, agrees = None, True
    if data[:4] in (b"<\x00?\x00", b"<\x00!\x00", b"<\x00h\x00", b"<\x00H\x00"):
        codec = "utf-16-le"
    elif data[:4] in (b"\x00<\x00?", b"\x00<\x00!", b"\x00<\x00h", b"\x00<\x00H"):
        codec = "utf-16-be"
    else:
        codec, agrees = declared_charset(data)
        codec = codec if codec is not None and keeps_ascii(codec) else None
    if codec is not None:
        try:
            return data.decode(codec), agrees
        except (LookupError, UnicodeDecodeError):
            pass
    return data.decode("cp1252", "replace"), False


def read_markup(data: bytes, reading: Reading) -> None:
    """Markup (see ``decode_markup``): its decoded source — tags,
    attributes, comments, scripts — its text with character references
    resolved, that text as a browser shows it, and, as the RTF reader does,
    its raw bytes as Latin-1 (every ASCII byte where it stands, whatever a
    multi-byte charset made of the bytes around it). Incomplete when not decoded exactly or when it
    inlines binary data."""
    text, exact = decode_markup(data)
    raw = data.decode("latin-1")
    inline = _INLINE_MEDIA.search(text) or _INLINE_MEDIA.search(raw)
    reading.complete = reading.complete and exact and not inline
    reading.add(text)
    parser = _MarkupText()
    parser.feed(text)
    parser.close()
    pieces = "".join(parser.pieces)
    reading.add(pieces)
    # The text as a browser shows it: a value split by a script, a style or
    # a comment is whole only here (the scan covers every reading).
    shown = "".join(parser.shown)
    if shown not in pieces:
        reading.add(shown)
    reading.show(_SPACE_RUNS.sub("\n", shown).strip())
    if raw != text:
        reading.add(raw)


class _RtfState:
    """One RTF group's state (copied on ``{``, restored on ``}``)."""

    def __init__(self) -> None:
        self.uc = 1  # fallback characters after each \\uN
        self.font: int | None = None
        self.aside = False  # a destination outside the line of text
        self.hidden = False  # \\v (hidden text) or \\deleted (a tracked deletion)
        self.table = False  # inside the font table
        self.defining: int | None = None  # the font a table entry defines
        self.upr = False  # a \\upr group: only its \\ud part is shown

    def copy(self) -> _RtfState:
        state = _RtfState()
        state.__dict__.update(self.__dict__)
        state.aside = self.aside or self.upr
        state.upr = False
        return state


class _RtfText:
    """RTF decoded as a reader shows it: ``\\'hh`` bytes and 8-bit text in
    the code page of the current font (``\\fcharset``/``\\cpg`` in the font
    table) or the document's (``\\ansicpg``), consecutive bytes decoded
    together (a double-byte character is two escapes), ``\\uN`` with its
    ``\\ucN`` fallback characters skipped. ``full`` is every decoded
    character; ``shown`` leaves out hidden and deleted text and the
    destinations outside the line (``_RTF_ASIDES``, ``\\*``), so the text
    around them joins. ``complete`` turns False on bytes the code page
    does not decode, U+FFFD, and pictures, objects or binary data."""

    def __init__(self, codec: str) -> None:
        self.codec = codec
        self.fonts: dict[int, str] = {}
        self.full: list[str] = []
        self.shown: list[str] = []
        self.complete = True
        self.state = _RtfState()
        self._saved: list[_RtfState] = []
        self._pending = bytearray()
        self._skip = 0

    def _emit(self, text: str) -> None:
        self.complete = self.complete and "\ufffd" not in text
        self.full.append(text)
        if not (self.state.aside or self.state.hidden):
            self.shown.append(text)

    def flush(self) -> None:
        """The bytes collected so far, decoded together."""
        if not self._pending:
            return
        data, font = bytes(self._pending), self.state.font
        self._pending.clear()
        codec = self.fonts.get(font, self.codec) if font is not None else self.codec
        try:
            text = data.decode(codec)
        except (LookupError, UnicodeDecodeError):
            text, self.complete = data.decode("latin-1"), False
        self._emit(text)

    def _skipped(self, kind: int, text: bytes) -> bytes | None:
        """A \\uN's fallback characters taken off the token stream: each
        byte of text, escape, control word or symbol counts one; a brace
        ends them. The rest of a text token, else None when it was all
        fallback."""
        if kind == 5:
            self._skip = 0
            return text
        if kind == 6:
            taken = min(self._skip, len(text))
            self._skip -= taken
            return text[taken:] or None
        self._skip -= 1
        return None

    def token(self, match: re.Match[bytes]) -> None:
        """One token of ``_RTF_TOKEN``: 1-2 a control word (and its
        number), 3 an escaped byte, 4 a control symbol, 5 a brace, 6 text
        (line breaks in it are not text)."""
        kind = match.lastindex or 0
        text: bytes | None = match[kind]
        if kind == 6:
            text = match[6].replace(b"\r", b"").replace(b"\n", b"")
        if self._skip and text:
            text = self._skipped(kind, text)
        if not text:
            return
        if kind in (3, 6):
            self._pending += bytes([int(text, 16)]) if kind == 3 else text
            return
        self.flush()
        if kind == 5:
            self._brace(text)
        elif kind == 4:
            self._symbol(text)
        else:
            self._word(match[1], match[2])

    def _brace(self, brace: bytes) -> None:
        if brace == b"{":
            self._saved.append(self.state)
            self.state = self.state.copy()
        elif self._saved:
            self.state = self._saved.pop()

    def _symbol(self, symbol: bytes) -> None:
        if symbol == b"*":
            self.state.aside = True  # an ignorable destination
        else:
            self._emit(_RTF_SYMBOLS.get(symbol, ""))

    def _word(self, word: bytes, number: bytes | None) -> None:
        value = int(number) if number else None
        state = self.state
        if word == b"u" and value is not None:
            self._emit(chr(value % 65536))
            self._skip = state.uc
        elif word == b"uc" and value is not None:
            state.uc = max(value, 0)
        elif word == b"f" and value is not None:
            if state.table:
                state.defining = value
            else:
                state.font = value
        elif word in (b"fcharset", b"cpg") and value is not None and state.defining is not None:
            codec = f"cp{value}" if word == b"cpg" else _RTF_CHARSETS.get(value)
            if codec is not None:
                self.fonts[state.defining] = codec
        elif word in (b"v", b"deleted"):
            state.hidden = value != 0
        elif word == b"plain":
            state.hidden = False
        elif word == b"upr":
            state.upr = True
        elif word == b"ud":
            state.aside = self._saved[-1].aside if self._saved else False
        else:
            self._layout(word)

    def _layout(self, word: bytes) -> None:
        """Destinations, breaks and what the reader does not read."""
        if word in _RTF_UNREAD:
            self.complete = False
        if word in _RTF_ASIDES or word in _RTF_UNREAD:
            self.state.aside = True
            self.state.table = self.state.table or word == b"fonttbl"
        elif word in _RTF_TABS:
            self._emit("\t")
        else:
            self._emit(_RTF_BREAKS.get(word, ""))


def read_rtf(data: bytes, reading: Reading) -> None:
    """RTF, best effort: the source (Latin-1, every ASCII byte kept), its
    decoded text and the text as shown (``_RtfText``); incomplete with a
    picture, an embedded object, binary data, bytes its code pages do not
    decode or a character standing for one that could not be mapped."""
    page = _ANSICPG.search(data[:4096])
    codec = f"cp{int(page.group(1))}" if page else "cp1252"
    try:
        "".encode(codec)
    except LookupError:
        codec, reading.complete = "cp1252", False
    text = _RtfText(codec)
    for match in _RTF_TOKEN.finditer(data):
        text.token(match)
    text.flush()
    reading.complete = reading.complete and text.complete
    full, shown = "".join(text.full), "".join(text.shown)
    reading.show(shown)
    reading.add(data.decode("latin-1"))
    reading.add(full)
    if shown != full:
        reading.add(shown)


# --- the worker ---------------------------------------------------------------------


def extract(data: bytes, *, formats: set[str], max_chars: int, max_inflated: int) -> dict[str, Any]:
    """``data`` read by the local extractor its bytes select, if enabled
    (see the module); the JSON-ready result the parent reads."""
    detected = kind = detect(data) or "other"
    reading = Reading(max_chars)
    readers: dict[str, Callable[[bytes, Reading], None]] = {
        "pdf": read_pdf,
        "html": read_markup,
        "rtf": read_rtf,
    }
    try:
        if kind == "zip" and formats & {"ooxml", "odf"}:
            kind = read_package(data, reading, formats, max_inflated)
        elif kind in readers and kind in formats:
            readers[kind](data, reading)
        if kind not in formats:
            return {"format": kind, "text": None, "complete": False, "reason": "unsupported"}
        if carries_another(data, detected):
            reading.complete = False
    except Limit:
        return {"format": reading.kind or kind, "text": None, "complete": False, "reason": "limit"}
    except Refused:
        return {
            "format": reading.kind or kind,
            "text": None,
            "complete": False,
            "reason": "refused",
        }
    except Exception:  # noqa: BLE001 — a malformed file: what was read, incomplete
        reading.complete = False
        return {
            "format": reading.kind or kind,
            "text": reading.text() or None,
            "complete": False,
            "reason": "error",
            "pages": reading.pages,
        }
    return {
        "format": kind,
        "text": reading.text(),
        "complete": reading.complete,
        "reason": "ok",
        "display": reading.display() if reading.complete else None,
        "pages": reading.pages,
        # Whether OCR of every rendered page covers what was left unread.
        "ocr": reading.ocr_completes,
    }


def apply_limits(memory_bytes: int, cpu_seconds: int) -> None:
    """Cap this process before it reads the file: address space (where the
    platform enforces it — not macOS), CPU seconds, files written (none),
    open files and child processes. A limit the platform refuses is skipped
    (the parent's wall-clock kill still holds), and a platform without
    resource limits (Windows) applies none: the parent bounds the input
    there instead (``extraction.MEMORY_LIMIT_ENFORCED``)."""
    try:
        import resource
    except ImportError:
        return

    for name, value in (
        ("RLIMIT_AS", memory_bytes),
        ("RLIMIT_DATA", memory_bytes),
        ("RLIMIT_CPU", cpu_seconds),
        ("RLIMIT_FSIZE", 0),
        ("RLIMIT_CORE", 0),
        ("RLIMIT_NOFILE", 16),
        ("RLIMIT_NPROC", 0),
    ):
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            continue


def read_input(stdin: Any, max_bytes: int) -> bytes | None:
    """The file from ``stdin`` in chunks (never one allocation sized by
    the cap), or None past ``max_bytes``."""
    buffer = bytearray()
    while chunk := stdin.read(1 << 20):
        buffer += chunk
        if len(buffer) > max_bytes:
            return None
    return bytes(buffer)


def main(argv: list[str], stdin: Any, stdout: Any, *, limit: bool = True) -> int:
    """The worker process: ``argv[1]`` is its JSON settings (formats, text
    and inflation caps, memory and CPU limits, the file-size cap)."""
    settings = json.loads(argv[1])
    formats = set(settings["formats"])
    if "pdf" in formats:
        import pypdf  # noqa: F401 — imported before the limits apply

    if limit:
        apply_limits(int(settings["memory_bytes"]), int(settings["cpu_seconds"]))
    data = read_input(stdin, int(settings["max_bytes"]))
    if data is None:
        result = {"format": None, "text": None, "complete": False, "reason": "limit"}
    else:
        result = extract(
            data,
            formats=formats,
            max_chars=int(settings["max_chars"]),
            max_inflated=int(settings["max_inflated"]),
        )
    stdout.write(json.dumps(result).encode("ascii"))
    stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover — the subprocess entry
    sys.exit(main(sys.argv, sys.stdin.buffer, sys.stdout.buffer))
