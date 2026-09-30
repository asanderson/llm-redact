"""File uploads and downloads for the providers whose upload is NOT the
OpenAI Files form — only what is genuinely provider-specific lives here;
the content policy itself is the OpenAI Files upload's, shared:

- ``redact_related_upload``: the Gemini API's single-request upload, a
  ``multipart/related`` body — the file's JSON metadata, then its media —
  whose parts carry no Content-Disposition, so none is a named form field.
  The FIRST part is the metadata, what the provider reads as the create's
  body: read exactly as the stored-object check reads it
  (``upload_view.read_metadata_part``: a JSON object, escapes resolved —
  pretty-printed or on one line alike) and redacted as that value, every
  string of it; metadata it cannot read is refused where every piece must
  be scanned. Every OTHER part is read as a file by its content
  (``upload_content.classify_file``: JSONL, text or binary) and redacted by
  the OpenAI upload's own part loop (JSONL line by line as JSON, text as
  one text re-encoded as it came, a binary part refused unless
  ``forward_binary`` — the proxy's choice for the client's own credential
  under ``[detection] binary_uploads = "forward"`` — which is told how
  many went out unscanned — or one the proxy cleared through its extracted
  text, ``InspectedUpload``). Anthropic's Files upload is
  multipart/form-data and uses the OpenAI upload unchanged
  (``redact_files_upload``). ``read_*_upload`` read either upload once for
  the proxy's inspection of its binary parts (``read_multipart``).
- ``rehydrate_download``: a downloaded file restored like an OpenAI file
  download (``openai.rehydrate_text_file``), every string of a JSON line
  restored with no skip set (no OpenAI ``arguments`` source override).
"""

from __future__ import annotations

from collections.abc import Callable

from llm_redact import multipart
from llm_redact.jsonwalk import transform_all_strings
from llm_redact.placeholders import may_carry_tokens
from llm_redact.providers.base import InspectedUpload
from llm_redact.providers.openai import (
    OpenAIAdapter,
    PartsReading,
    _multipart_floors,
    _Reading,
    checked_reading,
    read_file_part,
    reading_of,
    record_raw_texts,
    rehydrate_text_file,
    uncleared_binaries,
)
from llm_redact.redactor import Redactor, UnredactableRequest
from llm_redact.rehydrate import Rehydrator
from llm_redact.upload_view import MetadataPart, read_metadata_part

# The OpenAI Files upload: its part loop and whole-upload handling are the
# one implementation of the upload content policy. Its system note is never
# asked for here (``inject_note=False``: these routes carry none).
_FILES = OpenAIAdapter()
_FILES_PATH = "/v1/files"
_OUTSIDE_PARTS = "the multipart body carries a preamble or epilogue llm-redact does not redact"


def read_files_upload(
    body: bytes, boundary: bytes, charge: Callable[[int], None]
) -> PartsReading | None:
    """A multipart/form-data file upload read as ``redact_files_upload``
    reads it, every piece required scanned, its part headers checked
    (``read_multipart``)."""
    return checked_reading(
        _FILES.read_form_upload(_FILES_PATH, body, boundary, charge, require_scanned=True)
    )


def redact_files_upload(
    body: bytes,
    boundary: bytes,
    redactor: Redactor,
    *,
    require_scanned: bool,
    forward_binary: Callable[[int], None] | None,
    inspected: InspectedUpload | None = None,
) -> bytes | None:
    """A multipart/form-data file upload (Anthropic's Files API) redacted
    exactly as the OpenAI Files upload is — every JSONL line as data: no
    field of this upload makes its lines requests the provider runs."""
    return _FILES.redact_form_upload(
        _FILES_PATH,
        body,
        boundary,
        redactor,
        inject_note=False,
        require_scanned=require_scanned,
        forward_binary=forward_binary,
        request_purposes=False,
        inspected=inspected,
    )


def read_related_upload(
    body: bytes, boundary: bytes, charge: Callable[[int], None], *, require_scanned: bool = True
) -> PartsReading | None:
    """A multipart/related upload read as ``redact_related_upload`` reads
    it: the first part the file's metadata (``_metadata_reading``), every
    other part a file, read by its content (a JSONL part's lines charged
    against max_body_strings before anything walks them). None for a body
    outside the canonical grammar read leniently."""
    parsed = multipart.parse(body, boundary)
    if parsed is None:
        if require_scanned:
            raise UnredactableRequest(
                "the multipart body is outside the canonical form llm-redact can redact"
            )
        return None
    if require_scanned and (parsed.preamble.strip() or parsed.epilogue.strip()):
        raise UnredactableRequest(_OUTSIDE_PARTS)
    readings = [
        read_file_part(part.content, charge)
        if index
        else _metadata_reading(part, charge, require_scanned=require_scanned)
        for index, part in enumerate(parsed.parts)
    ]
    return PartsReading(parsed, readings)


def _metadata_reading(
    part: multipart.MultipartPart, charge: Callable[[int], None], *, require_scanned: bool
) -> _Reading:
    """The upload's FIRST part, the file's metadata, read as the JSON value
    the stored-object check reads (``upload_view.read_metadata_part``) —
    one reading of it, so a pretty-printed part is redacted with its
    escapes resolved like a one-line one. Metadata that reading refuses is
    unredactable where every piece must be scanned (the problem names the
    construct only); leniently, the part is read as a file."""
    read = read_metadata_part(part)
    if isinstance(read, MetadataPart):
        return _Reading("metadata", metadata=read)
    if require_scanned:
        raise UnredactableRequest(read)
    return read_file_part(part.content, charge)


def _redact_metadata(
    part: multipart.MultipartPart, read: MetadataPart, redactor: Redactor, *, strict: bool
) -> bool:
    """The metadata part redacted as the JSON value it is: every string
    (keys never), as a data line of an upload is — the provider reads it as
    the create's body. Only the span of its JSON text is rewritten, and only
    when a value changed or a key repeats (exactly what was read then goes
    out); its file names are redacted like any part's. True when it
    changed."""
    changed = part.redact_filenames(redactor.redact_text, strict=strict)
    redacted = transform_all_strings(read.metadata, redactor.redact_text)
    if redacted == read.metadata and not read.repeated:
        return changed
    part.content = read.splice(part.content, redacted)
    return True


def redact_related_upload(
    body: bytes,
    boundary: bytes,
    redactor: Redactor,
    *,
    require_scanned: bool,
    forward_binary: Callable[[int], None] | None,
    inspected: InspectedUpload | None = None,
) -> bytes | None:
    """A multipart/related upload redacted part by part (see the module):
    None when nothing changed (or, leniently, when the body is outside the
    canonical grammar). Raises UnredactableRequest for what it cannot read
    where that refuses, and BlockedRequest anywhere. ``inspected``: the
    proxy's reading of this same body and the binary parts it cleared
    through their extracted text (``ProviderAdapter.redact_multipart``)."""
    reading = reading_of(inspected) or read_related_upload(
        body, boundary, redactor.charge, require_scanned=require_scanned
    )
    if reading is None:
        return None
    # One reading per part, shared by the floor scan and the part loop.
    parsed, readings = reading
    cleared = inspected.cleared if inspected is not None else frozenset()
    if may_carry_tokens(body):
        redactor = redactor.with_floors(_multipart_floors(parsed, readings))
    changed = False
    originals = [part.content for part in parsed.parts]
    try:
        for index, (part, part_reading) in enumerate(zip(parsed.parts, readings, strict=True)):
            if part_reading.metadata is not None:
                changed |= _redact_metadata(
                    part, part_reading.metadata, redactor, strict=require_scanned
                )
                continue
            changed |= _FILES._redact_part(
                part,
                part_reading,
                redactor,
                inject_note=False,
                require_scanned=require_scanned,
                forward_binary=forward_binary is not None or index in cleared,
            )
    except multipart.AmbiguousHeaders as exc:
        raise UnredactableRequest(str(exc)) from None
    record_raw_texts(parsed.parts, originals, readings)
    binary = uncleared_binaries(readings, cleared)
    if binary and forward_binary is not None:
        forward_binary(binary)  # every piece was read or allowed
    return parsed.serialize() if changed else None


def rehydrate_download(raw: bytes, rehydrator: Rehydrator) -> bytes | None:
    """A downloaded file restored (None: untouched): the OpenAI file
    download's reading, every string of a JSON line restored with NO skip
    set (the upload redacts a data line's every value)."""
    return rehydrate_text_file(
        raw, rehydrator, lambda value: transform_all_strings(value, rehydrator.rehydrate_text)
    )
