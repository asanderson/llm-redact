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

- every line of every FILE part (a part with a file name) that parses as a
  JSON object — as the provider reads it: a repeated key's LAST occurrence,
  with ``normalized`` holding the body with each such line re-serialized,
  for a caller that must forward exactly what was checked;
- every FORM FIELD as an object nested along its name's bracket path, the
  form encoders' convention (``file_id`` → ``{"file_id": v}``,
  ``file_ids[]`` → ``{"file_ids": [v]}``, ``a[b][0]`` → ``{"a": {"b":
  [v]}}``), its value parsed when it is a JSON object or array.

``problem`` names why an upload cannot be checked (the construct only, never
content): outside the canonical grammar, a part header without one reading,
a Content-Transfer-Encoding, a form field that is not UTF-8 text or declares
another charset (its own, or the RFC 7578 ``_charset_`` field), a JSON form
field repeating a key. ``oversized``: the JSON and field text it carries
exceed what the check reads (``max_json_bytes``). What each means is the
caller's decision (the proxy refuses them when its own credential is spent).
"""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple

from llm_redact import multipart
from llm_redact.jsonwalk import loads_request

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

# A field name's bracket path: the base, then zero or more "[segment]"s.
_BRACKETED = re.compile(r"([^\[\]]*)((?:\[[^\[\]]*\])+)")
_SEGMENT = re.compile(r"\[([^\[\]]*)\]")


def json_line(value: Any) -> bytes:
    """``value`` re-serialized as one JSON line of UTF-8, non-ASCII kept as
    is — how an uploaded line the proxy rewrites (a redacted value, a
    repeated key) is sent. A lone surrogate, which only a ``\\ud800``-style
    escape can carry, has no UTF-8 form: that value is written with every
    non-ASCII character escaped instead — the same JSON value, never a
    failed request."""
    try:
        return json.dumps(value, ensure_ascii=False).encode()
    except UnicodeEncodeError:
        return json.dumps(value).encode()  # ensure_ascii: pure ASCII


class UploadView(NamedTuple):
    """One upload as the stored-object check reads it (see the module)."""

    cited: list[Any]
    problem: str | None = None
    oversized: bool = False
    normalized: bytes | None = None


class _Unreadable(Exception):
    """A part the check cannot read (the message names the construct)."""


class _Oversized(Exception):
    """More JSON and field text than the check reads."""


def read_upload(body: bytes, boundary: bytes, *, max_json_bytes: int) -> UploadView:
    """The upload ``body`` (delimited by ``boundary``) as the stored-object
    check reads it; see the module for what is read and what is refused."""
    parsed = multipart.parse(body, boundary)
    if parsed is None:
        return UploadView([], problem=OUTSIDE_GRAMMAR)
    reader = _Reader(max_json_bytes)
    rewrote = False
    try:
        for part in parsed.parts:
            rewrote |= reader.read(part)
    except _Unreadable as exc:
        return UploadView([], problem=str(exc))
    except _Oversized:
        return UploadView([], oversized=True)
    return UploadView(reader.cited, normalized=parsed.serialize() if rewrote else None)


class _Reader:
    """Reads parts one by one into ``cited``, within a byte budget."""

    def __init__(self, budget: int) -> None:
        self.cited: list[Any] = []
        self._budget = budget

    def _spend(self, size: int) -> None:
        self._budget -= size
        if self._budget < 0:
            raise _Oversized

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
        requests; other files have none), re-serialized where it repeats a
        key — the reading the adapters' JSONL redaction shares. True when a
        line was."""
        lines = part.content.split(b"\n")
        rewritten = False
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj, duplicate_keys = loads_request(stripped)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            self._spend(len(stripped))
            self.cited.append(obj)
            if duplicate_keys:
                lines[index] = json_line(obj)
                rewritten = True
        if rewritten:
            part.content = b"\n".join(lines)
        return rewritten

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
