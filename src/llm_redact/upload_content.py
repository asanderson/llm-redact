"""What an uploaded file IS, decided by its content: JSONL, text or binary.

A file upload (OpenAI/Azure/custom ``/files``; any provider's upload route
that reuses this) carries no reliable statement of its own type — the
declared Content-Type and file name are the client's guesses — so the proxy
reads the bytes. ``classify_file`` is the one decision every reader of an
upload shares (the redaction loop, the token-floor scan, the download's
restoration), so a part is always read as the same thing:

- ``"jsonl"``: UTF-8 text whose every non-blank line parses as a JSON
  object (at least one): batch input and fine-tuning files, redacted line
  by line as JSON.
- ``"text"``: any other text — strict UTF-8 (a UTF-8 byte-order mark kept
  as sent) or UTF-16/UTF-32 opened by its byte-order mark — redacted as ONE
  text and re-encoded exactly as it came (``FileContent.encode``). A file
  mixing JSON-object lines with other lines is text (no JSONL reader
  accepts it). An empty or whitespace-only file is text.
- ``"binary"``: everything else — bytes that do not decode, text holding a
  NUL character, and any file opening with one of the few binary
  signatures listed in ``_SIGNATURES`` even when it would decode (an
  all-ASCII PDF is binary: rewriting it would break its cross-reference
  byte offsets). Never read, never rewritten.

Signatures that are themselves invalid UTF-8 — PNG (``\\x89PNG``), JPEG
(``\\xff\\xd8\\xff``), gzip (``\\x1f\\x8b``), OLE2/legacy Office
(``\\xd0\\xcf\\x11\\xe0``), MPEG audio frames (``\\xff\\xfb``), 7-Zip, xz, zstd
— need no entry in ``_SIGNATURES``: the strict decode already makes them
binary. So do the formats whose signature is a printable word — GIF, RIFF
(WebP/WAV/AVI), ID3, Ogg, FLAC, WOFF/WOFF2 and the ISO media ``ftyp`` box
(MP4/MOV/HEIC/AVIF): each real file carries a NUL or a non-UTF-8 byte in
its fixed header, while a text file opening with the same word (a CSV row
``ID3,name,email``) is text and is redacted. Listed are only the
signatures a strict decode would accept and no text plausibly opens with:
``%PDF-`` and the control-byte ones.

Usage (a provider's upload hook)::

    content = classify_file(part.content, charge=redactor.charge)
    if content.kind == "text":
        redacted = redactor.redact_text(content.text)
        part.content = content.encode(redacted)

``charge`` bounds the per-line JSONL check: it is called with the file's
line count right before that check walks the lines — only for a file whose
first non-blank line is a JSON object, so a text file costs one string and
a binary file none (``max_body_strings``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal, NamedTuple

FileKind = Literal["jsonl", "text", "binary"]

# The binary formats a strict text decode could accept and that must never
# be rewritten. Only signatures real text cannot plausibly open with are
# listed: a printable word (``ID3``, ``RIFF``, ``GIF89a`` ...) opens ordinary
# CSV rows and prose too, and every real file of those formats already fails
# the strict decode or carries a NUL in its fixed header, so a word-only
# signature would only ever turn genuine text into an unscanned "binary".
_SIGNATURES = (
    b"%PDF-",  # PDF (an all-ASCII PDF decodes; its xref offsets must not move)
    b"PK\x03\x04",  # ZIP local file: OOXML (docx/xlsx/pptx), ODF, EPUB, JAR
    b"PK\x05\x06",  # an empty ZIP archive
    b"Rar!\x1a\x07",
    b"\x1a\x45\xdf\xa3",  # Matroska / WebM
    b"\x7fELF",
)

# Byte-order marks, longest first: the UTF-32LE mark begins with UTF-16LE's.
_BOMS = (
    (b"\xef\xbb\xbf", "utf-8"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

# The first line holding a byte that is not blank as ``bytes.strip`` reads
# blank (ASCII whitespace) — the line-level reading the JSONL redaction uses.
_FIRST_NON_BLANK_LINE = re.compile(rb"^[^\n]*?[^ \t\n\r\x0b\x0c][^\n]*", re.MULTILINE)


class FileContent(NamedTuple):
    """One uploaded file as ``classify_file`` reads it: its ``kind``, and
    for text (``"jsonl"`` or ``"text"``) the decoded ``text``, the
    ``codec`` it decoded with and the byte-order mark it opened with."""

    kind: FileKind
    text: str = ""
    codec: str = ""
    bom: bytes = b""

    def encode(self, text: str) -> bytes:
        """``text`` in this file's own encoding, behind its own byte-order
        mark: a redacted text goes out exactly as the original was
        encoded."""
        return self.bom + text.encode(self.codec)


BINARY = FileContent("binary")


def decode_text(content: bytes) -> FileContent | None:
    """``content`` decoded as text (kind ``"text"``), or None when it is
    binary: a known binary signature, bytes that do not decode strictly
    (UTF-8, or the UTF-16/32 its byte-order mark names), or a NUL character
    in the decoded text."""
    if content.startswith(_SIGNATURES):
        return None
    bom, codec = b"", "utf-8"
    for mark, name in _BOMS:
        if content.startswith(mark):
            bom, codec = mark, name
            break
    try:
        text = content[len(bom) :].decode(codec)
    except UnicodeDecodeError:
        return None
    if "\x00" in text:
        return None
    return FileContent("text", text, codec, bom)


def classify_file(content: bytes, *, charge: Callable[[int], None] | None = None) -> FileContent:
    """What the uploaded file ``content`` is (see the module): ``"jsonl"``,
    ``"text"`` or ``"binary"``. ``charge`` is called with the line count
    before the per-line JSONL check runs (a bound on per-line work)."""
    decoded = decode_text(content)
    if decoded is None:
        return BINARY
    if decoded.codec == "utf-8" and _is_jsonl(content, charge):
        return decoded._replace(kind="jsonl")
    return decoded


def _is_jsonl(content: bytes, charge: Callable[[int], None] | None) -> bool:
    """Whether every non-blank line of ``content`` is a JSON object (at
    least one): the first is tested before any line is walked, so a file of
    prose costs one parse."""
    first = _FIRST_NON_BLANK_LINE.search(content)
    if first is None or not _is_object_line(first.group()):
        return False
    if charge is not None:
        charge(content.count(b"\n") + 1)
    return all(_is_object_line(line) for line in content.split(b"\n") if line.strip())


def _is_object_line(line: bytes) -> bool:
    """Whether a non-blank line is a JSON object. JSON nested past what the
    parser reads counts as one: the file stays on the JSONL path, whose
    bounded reading refuses it, rather than being redacted as text an
    upstream JSONL reader would decode differently."""
    try:
        return isinstance(json.loads(line), dict)
    except RecursionError:
        return True
    except ValueError:
        return False
