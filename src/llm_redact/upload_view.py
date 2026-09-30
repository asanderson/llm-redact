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
content): outside the canonical grammar, a part header without one reading
(a header line carrying a bare CR or LF or another control included: a
lenient reader ends the header block there), a part with neither a header
block nor an empty one (``_require_header_block``), a
Content-Transfer-Encoding, a form field that is not UTF-8 text or declares
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

A SINGLE-REQUEST upload whose first part is the created file's JSON
metadata (the Gemini API's ``multipart/related`` upload: the metadata,
then the media; its parts carry no Content-Disposition, so none is a form
field) is read by ``read_upload_metadata`` instead: ``cited`` is that
metadata object — exactly what the provider reads as the create's body
(``{"file": {"name": …, "displayName": …}}``), the file's own name
included — ``{}`` when the part is blank or the body has no part. It is
read like a JSON request body: the part's content when it declares
``application/json`` or no type at all (the provider finds the metadata by
position; a multipart/related naming another root part with ``start`` is
no body llm-redact reads), strict UTF-8 (one
leading byte-order mark dropped), ``jsonwalk.loads_request`` (a repeated
key's LAST occurrence, with ``normalized`` holding the body with the part
re-serialized, for a caller that must forward exactly what was checked),
nesting at most ``MAX_JSON_DEPTH``. ``problem`` names why it cannot be:
outside the canonical grammar, a part header without one reading or a part
without a header block (as for a form upload), a
Content-Transfer-Encoding or a charset the proxy does not decode, another
declared type, content that is not UTF-8 text or not a JSON object, or
JSON nested too deep. The
media part is not read here (a file's content is not a request body).
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
METADATA_NOT_TEXT = "the upload's first part, the file's metadata, is not UTF-8 text"
METADATA_NOT_OBJECT = (
    "the upload's first part, the file's metadata, is not a JSON object llm-redact can read"
)
METADATA_TYPE = "the upload's first part, the file's metadata, is not declared application/json"
# The part headers the check reads (MultipartPart matches names
# case-insensitively).
_TRANSFER_ENCODING = "content-transfer-encoding"
_CONTENT_DISPOSITION = "content-disposition"
_CONTENT_TYPE = "content-type"
_JSON = b"application/json"

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
    """One upload as the stored-object check reads it (see the module):
    ``cited`` is the list of what a form upload cites, or the metadata
    object of a metadata-first upload (``read_upload_metadata``)."""

    cited: Any
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


def read_upload_metadata(body: bytes, boundary: bytes) -> UploadView:
    """The single-request upload ``body`` (delimited by ``boundary``) as the
    stored-object check reads it: its FIRST part's JSON metadata object,
    read like a JSON request body (see the module)."""
    parsed = multipart.parse(body, boundary)
    if parsed is None:
        return UploadView(None, problem=OUTSIDE_GRAMMAR)
    if not parsed.parts:
        return UploadView({})  # no part at all: no metadata
    part = parsed.parts[0]
    try:
        metadata, repeated = _read_metadata(part)
    except _Unreadable as exc:
        return UploadView(None, problem=str(exc))
    if repeated is None:
        return UploadView(metadata)
    # A repeated key: the part is sent as it was read (its LAST occurrence),
    # never with an earlier one a first-wins provider would act on. Only the
    # JSON text changes: the bytes around it (an empty header block's CRLF,
    # whitespace, a byte-order mark) stay, so the part keeps its shape.
    start, end = repeated
    part.content = part.content[:start] + json_bytes(metadata) + part.content[end:]
    return UploadView(metadata, normalized=parsed.serialize())


def _read_metadata(
    part: multipart.MultipartPart,
) -> tuple[dict[str, Any], tuple[int, int] | None]:
    """The metadata object ``part`` holds (``{}`` when it is blank) and,
    when it repeats a key, the span of its JSON text in the part's content
    (else None). Raises _Unreadable for what the check cannot read. A part
    with an empty header block is read too (see ``_require_header_block``)."""
    _require_header_block(part)
    if part.headers is not None:
        try:
            encoding = part.header(_TRANSFER_ENCODING)
            media = part.header(_CONTENT_TYPE)
            content_type = part.params(_CONTENT_TYPE) or {}
        except multipart.AmbiguousHeaders:
            raise _Unreadable(AMBIGUOUS) from None
        if encoding is not None and encoding.lower() not in PLAIN_TRANSFER_ENCODINGS:
            raise _Unreadable(TRANSFER_ENCODED)
        if media is not None and media.partition(b";")[0].strip().lower() != _JSON:
            # Read as JSON only when it says so (or says nothing): a server
            # parsing by the declared type reads other text another way
            # (a form-encoded "x&file.name=…&y" inside a JSON string).
            raise _Unreadable(METADATA_TYPE)
        charset = content_type.get("charset")
        if charset is not None and charset.value.lower() not in PLAIN_CHARSETS:
            raise _Unreadable(CHARSET)
    # The JSON text: past the whitespace ``bytes.strip`` drops and at most
    # one byte-order mark.
    raw = part.content
    start = len(raw) - len(raw.lstrip())
    if raw.startswith(_UTF8_BOM, start):
        start += len(_UTF8_BOM)
    end = len(raw.rstrip())
    content = raw[start:end]
    if not content:
        return {}, None
    try:
        text = content.decode()  # strict UTF-8
    except UnicodeDecodeError:
        raise _Unreadable(METADATA_NOT_TEXT) from None
    try:
        metadata, duplicate_keys = loads_request(text)
    except JsonTooDeep:
        raise _Unreadable(TOO_DEEP) from None
    except ValueError:
        raise _Unreadable(METADATA_NOT_OBJECT) from None
    if not isinstance(metadata, dict):
        raise _Unreadable(METADATA_NOT_OBJECT)
    return metadata, (start, end) if duplicate_keys else None


def _require_header_block(part: multipart.MultipartPart) -> None:
    """Raise _Unreadable when every reader may not find ``part``'s header
    block where the check does: a part without a header/body separator
    has one reading only when it is empty or opens with CRLF (an empty
    header block). Otherwise a strict reader takes its first lines for
    headers, and one accepting a bare LF as a line break ends them at a
    bare-LF blank line, reading what follows as the part's content — none
    of it where the check reads it."""
    if part.headers is None and part.content and not part.content.startswith(b"\r\n"):
        raise _Unreadable(AMBIGUOUS)


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
        _require_header_block(part)
        if part.headers is None:
            return False  # an empty header block: nothing a server reads as a named part
        try:
            encoding = part.header(_TRANSFER_ENCODING)
            disposition = part.params(_CONTENT_DISPOSITION) or {}
            content_type = part.params(_CONTENT_TYPE) or {}
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
        for index, (start, _, obj) in enumerate(repeated):
            pieces += (content[ends[index] : start], json_bytes(obj))
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
