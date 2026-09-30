"""What a multipart/form-data upload cites, as the stored-object check reads it.

The session router's ownership check (``plugin_api.SessionRouter
.object_access_refusal``; llm-redact-pro's named users) reads the PARSED
request body. An upload is not JSON, yet it can cite stored objects the
provider will act on: the lines of an uploaded batch input file are requests
the provider runs later — with the credential the upload is sent with — and
a form field can name a file (``file_id``, ``file_ids[]``). This module reads
an upload for that check ONLY: the proxy forwards the body as the route
forwards it (redacted by the adapter, or byte for byte), independently of
this reading, so ``[providers.NAME] detection = false`` routes and
pass-through routes are checked exactly like the redacted ones.

The view (``UploadView.cited``) is a list of JSON values:

- every line of every FILE part (a part with a file name) that is UTF-8
  text parsing as a JSON object — as the provider reads it: a repeated
  key's LAST occurrence, with ``normalized`` holding the body with each
  such line re-serialized, for a caller that must forward exactly what was
  checked — only in a UTF-8 file (``upload_content.classify_file``: JSONL
  or UTF-8 text); a UTF-16/32 text or a binary file is cited but never
  rewritten;
- every FORM FIELD as an object nested along its name's bracket path, the
  form encoders' convention (``file_id`` → ``{"file_id": v}``,
  ``file_ids[]`` → ``{"file_ids": [v]}``, ``a[b][0]`` → ``{"a": {"b":
  [v]}}``), its value parsed when it is a JSON object or array.

``problem`` names why an upload cannot be checked (the construct only, never
content): outside the canonical grammar, a part header without one reading,
a Content-Transfer-Encoding, a form field that is not UTF-8 text or declares
another charset (its own, or the RFC 7578 ``_charset_`` field), a JSON form
field repeating a key, a line or JSON form field nesting deeper than
``MAX_JSON_DEPTH`` (JSON the provider may read, but no walk can).
``oversized``: the JSON and field text it carries
exceed what the check reads (``max_json_bytes``). ``too_many_lines``: its
file parts hold more lines that could be JSON objects (a line whose first
byte past whitespace and a byte-order mark is ``{``, the only lines the
check parses) than it parses (``max_lines``; the proxy's
``max_body_strings``) — every other line costs a split and a byte test,
never a parse. What each means is the caller's decision (the proxy
refuses them when its own credential is spent).
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

from llm_redact import multipart
from llm_redact.jsonwalk import MAX_JSON_DEPTH, JsonTooDeep, json_bytes, loads_request
from llm_redact.upload_content import classify_file

# What the proxy reads as a part's plain bytes, and as a field's text: any
# other transfer encoding or charset is decoded by the upstream, never here.
PLAIN_TRANSFER_ENCODINGS = frozenset({b"7bit", b"8bit", b"binary"})
PLAIN_CHARSETS = frozenset({"utf-8", "us-ascii"})

OUTSIDE_GRAMMAR = "the multipart body is outside the canonical form llm-redact can check"
AMBIGUOUS = "a multipart part header cannot be parsed unambiguously"
TRANSFER_ENCODED = (
    "a multipart part declares a Content-Transfer-Encoding llm-redact does not decode"
)
CHARSET = "a multipart part declares a charset llm-redact does not decode"
NOT_TEXT = "a multipart form field is not UTF-8 text"
REPEATED_KEY = "a multipart form field repeats a JSON key"
TOO_DEEP = f"a multipart part nests JSON deeper than {MAX_JSON_DEPTH} levels"

# What json.loads drops from the head of a UTF-8 document it is given as bytes.
_UTF8_BOM = b"\xef\xbb\xbf"
# A line that can hold a JSON object: past the whitespace ``bytes.strip``
# drops and at most one byte-order mark (then JSON's own whitespace), a
# "{" — the match is the whole line. Anchored at a line's start; never
# crosses a newline. Possessive
# runs: a long run of whitespace is never backtracked into (linear).
_OBJECT_LINE = re.compile(rb"^[ \t\r\x0b\x0c]*+(?:\xef\xbb\xbf[ \t\r]*+)?\{[^\n]*", re.MULTILINE)

# A field name's bracket path: the base, then zero or more "[segment]"s.
_BRACKETED = re.compile(r"([^\[\]]*)((?:\[[^\[\]]*\])+)")
_SEGMENT = re.compile(r"\[([^\[\]]*)\]")


class UploadView(NamedTuple):
    """One upload as the stored-object check reads it (see the module)."""

    cited: list[Any]
    problem: str | None = None
    oversized: bool = False
    normalized: bytes | None = None
    too_many_lines: bool = False


class _Unreadable(Exception):
    """A part the check cannot read (the message names the construct)."""


class _Oversized(Exception):
    """More JSON and field text than the check reads."""


class _TooManyLines(Exception):
    """More candidate JSON lines than the check parses."""


def read_upload(body: bytes, boundary: bytes, *, max_json_bytes: int, max_lines: int) -> UploadView:
    """The upload ``body`` (delimited by ``boundary``) as the stored-object
    check reads it; see the module for what is read and what is refused."""
    parsed = multipart.parse(body, boundary)
    if parsed is None:
        return UploadView([], problem=OUTSIDE_GRAMMAR)
    reader = _Reader(max_json_bytes, max_lines)
    rewrote = False
    try:
        for part in parsed.parts:
            rewrote |= reader.read(part)
    except _Unreadable as exc:
        return UploadView([], problem=str(exc))
    except _Oversized:
        return UploadView([], oversized=True)
    except _TooManyLines:
        return UploadView([], too_many_lines=True)
    return UploadView(reader.cited, normalized=parsed.serialize() if rewrote else None)


class _Reader:
    """Reads parts one by one into ``cited``, within a byte budget and a
    budget of lines to parse."""

    def __init__(self, budget: int, lines: int) -> None:
        self.cited: list[Any] = []
        self._budget = budget
        self._lines = lines

    def _spend(self, size: int) -> None:
        self._budget -= size
        if self._budget < 0:
            raise _Oversized

    def _parse_line(self) -> None:
        """Count one more line to parse, before it is parsed."""
        self._lines -= 1
        if self._lines < 0:
            raise _TooManyLines

    def read(self, part: multipart.MultipartPart) -> bool:
        """Read one part into ``cited``; True when its content was rewritten
        (a file line repeating a key, re-serialized as the provider reads
        it)."""
        if part.headers is None:
            return False  # no header block: nothing a server reads as a named part
        try:
            encoding = part.header("content-transfer-encoding")
            disposition = part.params("content-disposition") or {}
            content_type = part.params("content-type") or {}
        except multipart.AmbiguousHeaders:
            raise _Unreadable(AMBIGUOUS) from None
        if encoding is not None and encoding.lower() not in PLAIN_TRANSFER_ENCODINGS:
            raise _Unreadable(TRANSFER_ENCODED)
        if "filename" in disposition or "filename*" in disposition:
            return self._read_file(part)
        name = disposition.get("name")
        if name is not None:  # an unnamed part is no form field
            self._read_field(name.value, part, content_type)
        return False

    def _read_file(self, part: multipart.MultipartPart) -> bool:
        """Every line that parses as a JSON object (a batch input file's
        requests; other files have none), read as the adapters' JSONL
        redaction reads a line: UTF-8 text (a leading byte-order mark
        dropped, as ``json.loads`` drops it from bytes) — never UTF-16/32,
        which a bytes parse would guess from a line's first bytes. A line
        repeating a key is re-serialized only in a UTF-8 file
        (``upload_content.classify_file``: JSONL, or UTF-8 text), where an
        object line re-serialized as UTF-8 keeps the file what it was; a
        UTF-16/32 text or a binary file is never rewritten (spliced UTF-8
        bytes would turn it into something else — a "binary" forwarded
        unscanned). True when a line was.

        Only a line that can be a JSON object is parsed (``_OBJECT_LINE``:
        its first byte past whitespace and one byte-order mark is ``{``),
        found by one scan of the part, and each is counted before it is
        parsed (``_parse_line``): blank lines, prose or CSV rows cost that
        scan, never a parse or a step of this loop."""
        content = part.content
        repeated: list[tuple[int, int, Any]] = []
        for match in _OBJECT_LINE.finditer(content):
            text = match.group().strip().removeprefix(_UTF8_BOM)
            self._parse_line()
            try:
                obj, duplicate_keys = loads_request(text.decode())
            except JsonTooDeep:
                raise _Unreadable(TOO_DEEP) from None
            except ValueError:  # UnicodeDecodeError included
                continue
            # A value that parses from a line starting "{" is an object.
            self._spend(len(text))
            self.cited.append(obj)
            if duplicate_keys:
                repeated.append((*match.span(), obj))
        if not repeated or classify_file(content).codec != "utf-8":
            return False
        # Each rewritten line, and the bytes since the previous one's end.
        ends = [0, *(end for _, end, _ in repeated)]
        pieces: list[bytes] = []
        for (start, _, obj), previous in zip(repeated, ends, strict=False):
            pieces += (content[previous:start], json_bytes(obj))
        pieces.append(content[ends[-1] :])
        part.content = b"".join(pieces)
        return True

    def _read_field(
        self, name: str, part: multipart.MultipartPart, content_type: dict[str, multipart.Param]
    ) -> None:
        charset = content_type.get("charset")
        if charset is not None and charset.value.lower() not in PLAIN_CHARSETS:
            raise _Unreadable(CHARSET)
        try:
            text = part.content.decode()  # strict UTF-8
        except UnicodeDecodeError:
            raise _Unreadable(NOT_TEXT) from None
        if name == "_charset_" and text.strip().lower() not in PLAIN_CHARSETS:
            raise _Unreadable(CHARSET)  # RFC 7578 §4.6: every field's default charset
        self._spend(len(part.content))
        value: Any = text
        stripped = text.strip()
        if stripped[:1] in ("{", "["):
            try:
                value, duplicate_keys = loads_request(stripped)
            except JsonTooDeep:
                raise _Unreadable(TOO_DEEP) from None
            except ValueError:
                value = text
            else:
                if duplicate_keys:
                    raise _Unreadable(REPEATED_KEY)
        self.cited.append(_nested(name, value))


def _nested(name: str, value: Any) -> dict[str, Any]:
    """``{name: value}``, nested along a bracketed name's path: an empty or
    numeric segment is a list position, any other a key."""
    match = _BRACKETED.fullmatch(name)
    if match is None:
        return {name: value}
    node = value
    for segment in reversed(_SEGMENT.findall(match.group(2))):
        node = [node] if segment == "" or segment.isdigit() else {segment: node}
    return {match.group(1): node}
