"""File uploads and downloads for the providers whose upload is NOT the
OpenAI Files form — only what is genuinely provider-specific lives here;
the content policy itself is the OpenAI Files upload's, shared:

- ``redact_related_upload``: the Gemini API's single-request upload, a
  ``multipart/related`` body — the file's JSON metadata, then its media —
  whose parts carry no Content-Disposition, so none is a named form field:
  EVERY part is read as a file by its content
  (``upload_content.classify_file``: JSONL, text or binary) and redacted by
  the OpenAI upload's own part loop (JSONL line by line as JSON, text as
  one text re-encoded as it came, a binary part refused unless
  ``forward_binary`` — the proxy's choice for the client's own credential
  under ``[detection] binary_uploads = "forward"`` — which is told how
  many went out unscanned). Anthropic's Files upload is multipart/form-data
  and uses the OpenAI upload unchanged (``redact_files_upload``).
- ``rehydrate_download``: a downloaded file restored like an OpenAI file
  download (``openai.rehydrate_text_file``), each JSON line through the
  plain JSON walk (no OpenAI ``arguments`` source override).
"""

from __future__ import annotations

from collections.abc import Callable

from llm_redact import multipart
from llm_redact.placeholders import may_carry_tokens
from llm_redact.providers.openai import (
    OpenAIAdapter,
    _multipart_floors,
    _Reading,
    rehydrate_text_file,
)
from llm_redact.redactor import Redactor, UnredactableRequest
from llm_redact.rehydrate import Rehydrator
from llm_redact.upload_content import classify_file

# The OpenAI Files upload: its part loop and whole-upload handling are the
# one implementation of the upload content policy. Its system note is never
# asked for here (``inject_note=False``: these routes carry none).
_FILES = OpenAIAdapter()
_FILES_PATH = "/v1/files"
_OUTSIDE_PARTS = "the multipart body carries a preamble or epilogue llm-redact does not redact"


def redact_files_upload(
    body: bytes,
    boundary: bytes,
    redactor: Redactor,
    *,
    require_scanned: bool,
    forward_binary: Callable[[int], None] | None,
) -> bytes | None:
    """A multipart/form-data file upload (Anthropic's Files API) redacted
    exactly as the OpenAI Files upload is."""
    return _FILES.redact_multipart(
        _FILES_PATH,
        body,
        boundary,
        redactor,
        inject_note=False,
        require_scanned=require_scanned,
        forward_binary=forward_binary,
    )


def _file_reading(part: multipart.MultipartPart, charge: Callable[[int], None]) -> _Reading:
    """A related part read as a FILE, by its content."""
    content = classify_file(part.content, charge=charge)
    return _Reading("document" if content.kind == "text" else content.kind, content)


def redact_related_upload(
    body: bytes,
    boundary: bytes,
    redactor: Redactor,
    *,
    require_scanned: bool,
    forward_binary: Callable[[int], None] | None,
) -> bytes | None:
    """A multipart/related upload redacted part by part (see the module):
    None when nothing changed (or, leniently, when the body is outside the
    canonical grammar). Raises UnredactableRequest for what it cannot read
    where that refuses, and BlockedRequest anywhere."""
    parsed = multipart.parse(body, boundary)
    if parsed is None:
        if require_scanned:
            raise UnredactableRequest(
                "the multipart body is outside the canonical form llm-redact can redact"
            )
        return None
    if require_scanned and (parsed.preamble.strip() or parsed.epilogue.strip()):
        raise UnredactableRequest(_OUTSIDE_PARTS)
    # One reading per part (a JSONL part's lines charged against
    # max_body_strings before anything walks them), shared by the floor
    # scan and the part loop.
    readings = [_file_reading(part, redactor.charge) for part in parsed.parts]
    if may_carry_tokens(body):
        redactor = redactor.with_floors(_multipart_floors(parsed, readings))
    changed = False
    try:
        for part, reading in zip(parsed.parts, readings, strict=True):
            changed |= _FILES._redact_part(
                part,
                reading,
                redactor,
                inject_note=False,
                require_scanned=require_scanned,
                forward_binary=forward_binary is not None,
            )
    except multipart.AmbiguousHeaders as exc:
        raise UnredactableRequest(str(exc)) from None
    binary = sum(reading.kind == "binary" for reading in readings)
    if binary and forward_binary is not None:
        forward_binary(binary)  # every piece was read or allowed
    return parsed.serialize() if changed else None


def rehydrate_download(raw: bytes, rehydrator: Rehydrator) -> bytes | None:
    """A downloaded file restored (None: untouched): the OpenAI file
    download's reading, JSON lines through the plain JSON walk."""
    return rehydrate_text_file(raw, rehydrator, rehydrator.rehydrate_json)
