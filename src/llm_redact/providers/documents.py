"""Uploaded DOCUMENTS: files a provider stores and reads later (a text file,
a JSON or JSONL file, a PDF, an image) — the content policy shared by the
Gemini API's single-request (multipart/related) Files upload and
Anthropic's Files API upload, and the restoration of what their downloads
serve back.

- TEXT (strict UTF-8 without a NUL byte) is redacted: a JSON Lines file
  object by object and a JSON document as the value it holds (each
  re-serialized only where something was redacted, or where an object
  repeats a key — the provider reads its LAST occurrence, which is what
  the walk saw), anything else as text.
- BINARY content (everything else) is never read. It is forwarded as sent
  only with ``forward_binary`` (the client's own provider key: the
  provider authorizes the client, and the documented media non-goal
  applies); otherwise — a credential the PROXY holds — the whole upload is
  refused (``UnredactableRequest``): the proxy never sends content it did
  not read under its own credential.

This is the local stand-in for the shared upload content classifier: the
policy is the OpenAI Files upload's (a text file is redacted as text; a
binary one is forwarded unscanned only under the client's own key).

Every part's file name (``filename`` / ``filename*``) is redacted too;
with ``require_scanned`` a part declaring a transfer encoding or a charset
the proxy does not decode, or a header without one reading, refuses the
upload, as do bytes outside every part.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from llm_redact import multipart
from llm_redact.jsonwalk import json_text, loads_request
from llm_redact.placeholders import may_carry_tokens, merge_floors, token_floors
from llm_redact.redactor import Redactor, UnredactableRequest
from llm_redact.rehydrate import Rehydrator
from llm_redact.upload_view import PLAIN_CHARSETS, PLAIN_TRANSFER_ENCODINGS

NOT_TEXT = (
    "the uploaded file is not UTF-8 text llm-redact can redact (a binary file is forwarded"
    " unscanned only with the client's own provider key)"
)
_OUTSIDE_PARTS = "the multipart body carries a preamble or epilogue llm-redact does not redact"
_TRANSFER_ENCODED = (
    "a multipart part declares a Content-Transfer-Encoding llm-redact does not decode"
)
_CHARSET = "a multipart part declares a charset llm-redact does not decode"


def text_of(content: bytes) -> str | None:
    """``content`` as text when it is TEXT (strict UTF-8, no NUL), else None."""
    if b"\x00" in content:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


class _Line(NamedTuple):
    text: str
    obj: Any  # None: a blank line
    repeats: bool


def _json_lines(text: str) -> list[_Line] | None:
    """The lines of a JSON Lines text, each with its object (None for a
    blank line) — or None when a non-blank line is not a JSON object, or
    no line is one."""
    lines: list[_Line] = []
    for line in text.split("\n"):
        if not line.strip():
            lines.append(_Line(line, None, False))
            continue
        try:
            obj, repeats = loads_request(line)
        except ValueError:
            return None
        if not isinstance(obj, dict):
            return None
        lines.append(_Line(line, obj, repeats))
    return lines if any(line.obj is not None for line in lines) else None


def redact_text_document(text: str, redactor: Redactor) -> str:
    """A text document redacted (see the module)."""
    lines = _json_lines(text)
    if lines is not None:
        out: list[str] = []
        for line in lines:
            if line.obj is None:
                out.append(line.text)
                continue
            redacted = redactor.redact_json(line.obj)
            out.append(json_text(redacted) if redacted != line.obj or line.repeats else line.text)
        return "\n".join(out)
    try:
        value, repeats = loads_request(text)
    except ValueError:
        return redactor.redact_text(text)
    redacted = redactor.redact_json(value)
    return json_text(redacted) if redacted != value or repeats else text


def rehydrate_document(raw: bytes, rehydrator: Rehydrator) -> bytes | None:
    """A downloaded document restored, or None (left untouched): BINARY
    content is never read; a JSON document or JSON Lines file is restored
    as JSON source (restored values re-escaped, so it stays valid JSON,
    its formatting kept); any other text as text."""
    text = text_of(raw)
    if text is None:
        return None
    if _json_lines(text) is not None or _is_json(text):
        restored = rehydrator.rehydrate_json_source_text(text)
    else:
        restored = rehydrator.rehydrate_text(text)
    return restored.encode("utf-8") if restored != text else None


def _is_json(text: str) -> bool:
    try:
        loads_request(text)
    except ValueError:
        return False
    return True


def _require_plain(part: multipart.MultipartPart, *, text: bool) -> None:
    """Refuse a part the proxy could not read as plain bytes: any transfer
    encoding but 7bit/8bit/binary, and — on a part read as text — a
    declared charset other than UTF-8/US-ASCII. AmbiguousHeaders
    propagates."""
    encoding = part.header("content-transfer-encoding")
    if encoding is not None and encoding.lower() not in PLAIN_TRANSFER_ENCODINGS:
        raise UnredactableRequest(_TRANSFER_ENCODED)
    charset = (part.params("content-type") or {}).get("charset")
    if text and charset is not None and charset.value.lower() not in PLAIN_CHARSETS:
        raise UnredactableRequest(_CHARSET)


def _floors(parsed: multipart.Multipart) -> dict[str, int]:
    """The token floors of an upload: every file name and every TEXT part
    (binary parts are never read)."""
    floors: dict[str, int] = {}

    def observe(text: str) -> str:
        merge_floors(floors, token_floors(text))
        return text

    for part in parsed.parts:
        part.redact_filenames(observe, strict=False)  # returns every name unchanged
        text = text_of(part.content)
        if text is not None:
            observe(text)
    return floors


def redact_document_upload(
    body: bytes,
    boundary: bytes,
    redactor: Redactor,
    *,
    require_scanned: bool,
    forward_binary: bool,
) -> bytes | None:
    """A multipart upload of documents redacted part by part (see the
    module): None when nothing changed (or, leniently, when the body is
    outside the canonical grammar). Raises UnredactableRequest for what it
    cannot read where that refuses, and BlockedRequest anywhere."""
    parsed = multipart.parse(body, boundary)
    if parsed is None:
        if require_scanned:
            raise UnredactableRequest(
                "the multipart body is outside the canonical form llm-redact can redact"
            )
        return None
    if require_scanned and (parsed.preamble.strip() or parsed.epilogue.strip()):
        raise UnredactableRequest(_OUTSIDE_PARTS)
    # Every line of a text part is a string to redact: counted against
    # max_body_strings before anything splits it.
    redactor.charge(sum(part.content.count(b"\n") + 1 for part in parsed.parts))
    if may_carry_tokens(body):
        # A token anywhere in the upload bounds the numbers new values take.
        redactor = redactor.with_floors(_floors(parsed))
    changed = False
    try:
        for part in parsed.parts:
            changed |= _redact_part(
                part, redactor, require_scanned=require_scanned, forward_binary=forward_binary
            )
    except multipart.AmbiguousHeaders as exc:
        raise UnredactableRequest(str(exc)) from None
    return parsed.serialize() if changed else None


def _redact_part(
    part: multipart.MultipartPart,
    redactor: Redactor,
    *,
    require_scanned: bool,
    forward_binary: bool,
) -> bool:
    changed = part.redact_filenames(redactor.redact_text, strict=require_scanned)
    text = text_of(part.content)
    if require_scanned:
        _require_plain(part, text=text is not None)
    if text is None:
        if require_scanned and not forward_binary:
            raise UnredactableRequest(NOT_TEXT)
        return changed  # binary, never read: as sent
    redacted = redact_text_document(text, redactor)
    if redacted != text:
        part.content = redacted.encode("utf-8")
        changed = True
    return changed
