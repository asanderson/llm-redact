"""Google Gemini API adapter (generateContent / streamGenerateContent).

Event model notes (pinned by hand-authored fixtures; the live drift test
compares observed key sets against the KNOWN_* frozensets below — Gemini SSE
events carry no event name, so key-set drift is the analogue of the
Responses adapter's KNOWN_EVENT_TYPES):

- ``:streamGenerateContent?alt=sse`` emits data-only SSE events, each a full
  GenerateContentResponse chunk with partial ``candidates[].content.parts``.
  A «TOKEN» can split across two chunks' text parts, so text flows through
  RehydratorPool channels keyed ``(candidate_index, "text"|"thought")`` —
  never per-event flushing, which would leak the partial prefix.
- There is no terminal [DONE] sentinel: ``finishReason`` on a candidate is
  the only flush point. Leftovers are appended to that candidate's last text
  part (creating ``content.parts`` if the finish chunk carried none) because
  ``_stream_rehydrated`` discards anything still held at stream end.
- ``functionCall.args`` is a parsed JSON *object* (not JSON source like the
  OpenAI ``arguments`` string) and arrives complete in one event: whole-string
  restoration is correct there. Every chunk is walked from its ROOT — like the
  buffered response — so args keep their opaque position (a tool parameter
  named ``id`` or ``data`` is restored) and code, grounding and citation
  metadata are restored too; only the text parts are held out of that walk
  and fed to their channels instead (streaming == buffered).
- ``:streamGenerateContent`` WITHOUT ``alt=sse`` returns one JSON *array* of
  those same chunks; its elements split tokens exactly like the SSE form, so
  rehydrate_body runs per-candidate streaming channels across elements.
"""

import re
import urllib.parse
from collections.abc import Callable, Mapping
from typing import Any

from llm_redact.jsonwalk import json_text, loads_bounded, transform_strings
from llm_redact.multipart import parse_boundary as parse_multipart_boundary
from llm_redact.providers.base import SYSTEM_NOTE, InspectedUpload, ProviderAdapter, RouteKind
from llm_redact.providers.documents import (
    read_related_upload,
    redact_related_upload,
    rehydrate_download,
)
from llm_redact.providers.openai import PartsReading, checked_reading
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator, RehydratorPool, StreamingRehydrator
from llm_redact.sse import SSEEvent

_GEMINI_PATH = re.compile(
    r"/(?:v1|v1beta)/(?:models|tunedModels)/[^/:]+:"
    r"(generateContent|streamGenerateContent|countTokens|embedContent"
    r"|batchEmbedContents|batchGenerateContent|asyncBatchEmbedContent|predict"
    r"|predictLongRunning)"
)
# Verbs that answer with a long-running operation whose results are read
# back by its name later (the Gemini API's batch mode — generation and
# async embeddings — and Veo).
_OPERATION_VERBS = frozenset(
    {"batchGenerateContent", "asyncBatchEmbedContent", "predictLongRunning"}
)
# Verbs whose answer carries nothing to restore (a count, vectors, image
# bytes, an operation name).
_REDACT_ONLY_VERBS = frozenset(
    {
        "countTokens",
        "embedContent",
        "batchEmbedContents",
        "batchGenerateContent",
        "asyncBatchEmbedContent",
        "predict",
        "predictLongRunning",
    }
)
# Context caching: only the create (POST /…/cachedContents) carries content to
# redact. The per-cache GET/PATCH/DELETE and list return metadata (name, model,
# token counts, expiry) — never the cached content — so they pass through.
_GEMINI_CACHED_CREATE = re.compile(r"/(?:v1|v1beta)/cachedContents")
# The Files API. A file is created by the media upload
# (``/upload/v1beta/files``: the multipart protocol's one request, or a
# resumable upload's finalizing chunk when it is sent through the proxy),
# the metadata-only create, or ``files:register`` (Cloud Storage objects);
# each answer names the file(s) as ``files/<id>``, which later bodies cite
# (``fileData.fileUri``, a batch's input ``fileName``) and paths read.
_GEMINI_FILE_CREATE = re.compile(r"(?:/upload)?/v1beta/files(?::register)?")
# Recognized (redacted and restored, so a credential the proxy holds may
# reach them): the list and a file's metadata (``{"files": [...]}`` and the
# File, each echoing its ``displayName``), delete (the name only), the
# download (``:download``, and the ``/download/v1beta/`` media form a batch's
# output file is fetched from: text restored, binary never read), the
# metadata-only create and the single-request media upload (see
# ``_upload_protocol``). ``files:register`` names Cloud Storage objects the
# provider reads as the caller: pass-through.
_GEMINI_FILES = "/v1beta/files"
_GEMINI_FILE = re.compile(r"/v1beta/files/[^/:]+")
_GEMINI_FILE_DOWNLOAD = re.compile(r"(?:/download)?/v1beta/files/[^/:]+:download")
_GEMINI_UPLOAD = "/upload/v1beta/files"
# Resumable upload: the start request carries the file's metadata (JSON) and
# its answer an ``X-Goog-Upload-URL`` — an upload session on Google's host,
# minted for the request's credential — to which the client sends the data
# chunks DIRECTLY, never through the proxy. Under the client's own key that
# is the client's session (its data is the documented media gap); under a
# credential the proxy holds it would let the client store unread bytes as
# the proxy's principal, so the start is refused there (only the
# single-request multipart protocol is served) and the session header is
# never relayed.
_UPLOAD_SESSION_HEADERS = frozenset({"x-goog-upload-url", "x-goog-upload-control-url"})
_RESUMABLE_REFUSAL = (
    "the Gemini API's resumable upload hands the client an upload URL whose data never passes"
    " through llm-redact; upload the file in one request with the multipart protocol"
    " (X-Goog-Upload-Protocol: multipart) instead"
)
# Batch Mode's jobs, by the name the create answers with (``batches/<id>``:
# what the SDKs poll). A batch's status (the operation read back by name)
# echoes its display name and, once it finished, its INLINED responses —
# model output carrying placeholders — or names its output FILE (the
# creator's: only the creator's own read of the batch reaches the provider
# and gets here unsealed). The list (``{"operations": [...]}``) echoes every
# batch; cancel and delete carry the name only.
_GEMINI_BATCHES = "/v1beta/batches"
_GEMINI_BATCH_STATUS = re.compile(r"/v1beta/batches/[^/:]+")
_GEMINI_BATCH_CANCEL = re.compile(r"/v1beta/batches/[^/:]+:cancel")
# The model listing and one model's metadata (no :verb): recognized,
# redact-only — a body-less no-op, like Vertex's model metadata.
_GEMINI_MODELS = re.compile(r"/v1beta/models(?:/[^/:]+)?")
_FILE_PREFIX = "files/"

# Live drift detector reference sets (tests/test_live.py): observed keys must
# be subsets of these, or the API shape moved under us.
KNOWN_CHUNK_KEYS = frozenset(
    {"candidates", "usageMetadata", "promptFeedback", "modelVersion", "responseId", "createTime"}
)
KNOWN_CANDIDATE_KEYS = frozenset(
    {
        "content",
        "finishReason",
        "index",
        "safetyRatings",
        "citationMetadata",
        "groundingMetadata",
        "avgLogprobs",
        "logprobsResult",
        "tokenCount",
        "finishMessage",
    }
)
KNOWN_PART_KEYS = frozenset(
    {
        "text",
        "thought",
        "thoughtSignature",
        "functionCall",
        "functionResponse",
        "inlineData",
        "fileData",
        "executableCode",
        "codeExecutionResult",
    }
)

_ChannelKey = tuple[int, str]

_CACHE_PREFIX = "cachedContents/"


def cache_object_ids(body: Any) -> tuple[str, ...]:
    """The context cache a cache-create response names, as
    ``cachedContents/<id>``; any other ``name`` (a long-running operation's)
    as it is.

    The Gemini API answers with exactly that; Vertex answers with the full
    resource name (``projects/{p}/locations/{l}/cachedContents/<id>``).
    Both are reported in the Gemini form, which is how a session router
    reads the ``cachedContent`` a later generateContent body cites — so one
    cache has one id whichever form the client uses.
    """
    if not isinstance(body, dict):
        return ()
    name = body.get("name")
    if not isinstance(name, str) or not name:
        return ()
    at = name.rfind(_CACHE_PREFIX)
    return (name[at:] if at >= 0 else name,)


def _file_name(value: Any) -> str | None:
    """A Files API file's name (``files/<id>``) from a File object."""
    name = value.get("name") if isinstance(value, dict) else None
    return name if isinstance(name, str) and name.startswith(_FILE_PREFIX) else None


def file_object_ids(body: Any, *, register: bool = False) -> tuple[str, ...]:
    """The files a Files API create answers with: the upload's (and the
    metadata-only create's) ``{"file": File}`` — or, ``register``, only
    ``files:register``'s ``{"files": [File, …]}``. A create's answer holding
    a ``files`` array is a LISTING (an upstream that ran another method),
    never files this request created."""
    if not isinstance(body, dict):
        return ()
    if not register:
        name = _file_name(body.get("file"))
        return () if name is None else (name,)
    listed = body.get("files")
    files = listed if isinstance(listed, list) else ()
    return tuple(name for name in map(_file_name, files) if name is not None)


def batch_output_file_ids(body: Any) -> tuple[str, ...]:
    """The output file a finished batch's status names: ``response``'s (and
    the metadata's ``output``) ``responsesFile``; none while it runs, or for
    results inlined in the operation."""
    if not isinstance(body, dict):
        return ()
    metadata = body.get("metadata")
    outputs = (body.get("response"), metadata.get("output") if isinstance(metadata, dict) else None)
    found = (
        output.get("responsesFile") if isinstance(output, dict) else None for output in outputs
    )
    return tuple(
        dict.fromkeys(
            name for name in found if isinstance(name, str) and name.startswith(_FILE_PREFIX)
        )
    )


def _header_values(headers: "Mapping[str, str] | None", name: str) -> list[str]:
    """Every value of header ``name`` (comma lists split, lowercased)."""
    if headers is None:
        return []
    getlist = getattr(headers, "getlist", None)
    raw = (
        getlist(name)
        if callable(getlist)
        else [value for key, value in headers.items() if key.lower() == name]
    )
    return [item.strip().lower() for value in raw for item in value.split(",") if item.strip()]


def _upload_query(query: str) -> list[tuple[str, str]]:
    """The query parameters an upload reads its protocol from: every one
    whose (decoded, lowercased) name starts ``upload`` — ``uploadType``,
    ``upload_id``, ``upload_protocol`` — with its lowercased value."""
    found = []
    for piece in query.split("&"):
        name, _, value = piece.partition("=")
        name = urllib.parse.unquote_plus(name).strip().lower()
        if name.startswith("upload"):
            found.append((name, urllib.parse.unquote_plus(value).strip().lower()))
    return found


def _upload_forwarded_unread(headers: "Mapping[str, str] | None", query: str) -> bool:
    """Whether a POST to the upload path carries file DATA the proxy
    forwards unread: a resumable session's data chunk (an ``upload_id``, or
    an ``X-Goog-Upload-Command`` other than ``start``) or the raw protocol
    (``X-Goog-Upload-Protocol: raw``, ``uploadType=media``). Such a request
    is no route llm-redact recognizes: pass-through with the client's own
    key (the media gap), refused under a credential the proxy holds."""
    params = _upload_query(query)
    commands = _header_values(headers, "x-goog-upload-command")
    protocols = [*_header_values(headers, "x-goog-upload-protocol")]
    protocols += [value for name, value in params if name in ("uploadtype", "upload_protocol")]
    return (
        any(name == "upload_id" for name, _ in params)
        or any(command != "start" for command in commands)
        or any(protocol in ("raw", "media") for protocol in protocols)
    )


def _single_request_upload(headers: "Mapping[str, str] | None", query: str) -> bool:
    """Whether an upload request asks for nothing but the single-request
    (multipart) protocol: no upload command, no ``upload_id``, and every
    protocol it names — header or query — ``multipart``."""
    params = _upload_query(query)
    protocols = [*_header_values(headers, "x-goog-upload-protocol")]
    for name, value in params:
        if name not in ("uploadtype", "upload_protocol"):
            return False
        protocols.append(value)
    return not _header_values(headers, "x-goog-upload-command") and all(
        protocol == "multipart" for protocol in protocols
    )


# RFC 2387's "start" names the ROOT part by its Content-ID: with it the
# file's metadata need not be the FIRST part — the one llm-redact reads as
# the metadata (upload_view.read_upload_metadata).
_ROOT_ELSEWHERE = frozenset({"start"})


def _related_boundary(content_type: str) -> bytes | None:
    """The boundary of a multipart/related content type (the Gemini API's
    single-request upload: the file's JSON metadata, then its media); None
    also when the content type has more than one reading
    (``multipart.parse_boundary``) or names a root part (``start``)."""
    return parse_multipart_boundary(content_type, "multipart/related", refuse=_ROOT_ELSEWHERE)


class GeminiAdapter(ProviderAdapter):
    name = "gemini"
    capability_response_headers = _UPLOAD_SESSION_HEADERS

    def matches(self, method: str, path: str) -> RouteKind:
        if method == "GET" and _GEMINI_MODELS.fullmatch(path):
            return RouteKind.REDACT_ONLY
        if method == "GET" and (
            path == _GEMINI_FILES
            or _GEMINI_FILE.fullmatch(path)
            or _GEMINI_FILE_DOWNLOAD.fullmatch(path)
        ):
            # A file's metadata and the list echo display names; a download
            # serves the file back (text restored, binary untouched).
            return RouteKind.CHAT
        if method == "DELETE" and _GEMINI_FILE.fullmatch(path):
            return RouteKind.REDACT_ONLY  # the name only
        if method == "POST" and path in (_GEMINI_FILES, _GEMINI_UPLOAD):
            # The metadata-only create and the media upload: the metadata
            # (display name) and a text file redacted, the File restored.
            return RouteKind.CHAT
        if method == "GET" and (path == _GEMINI_BATCHES or _GEMINI_BATCH_STATUS.fullmatch(path)):
            # A batch's status and the list: display names and inlined
            # responses restored (the static session batches use).
            return RouteKind.CHAT
        if (method == "DELETE" and _GEMINI_BATCH_STATUS.fullmatch(path)) or (
            method == "POST" and _GEMINI_BATCH_CANCEL.fullmatch(path)
        ):
            # The name only, either way: recognized, redact-only.
            return RouteKind.REDACT_ONLY
        if method != "POST":
            return RouteKind.NONE
        # Cache-create carries contents + systemInstruction to redact; the
        # create response is metadata only, so there is nothing to rehydrate.
        if _GEMINI_CACHED_CREATE.fullmatch(path):
            return RouteKind.REDACT_ONLY
        match = _GEMINI_PATH.fullmatch(path)
        if match is None:
            return RouteKind.NONE
        # countTokens sees full message content but returns only a count;
        # embeddings responses are vectors; batchGenerateContent and
        # asyncBatchEmbedContent return a long-running operation NAME (the
        # results are read later through the batch's status) — none has
        # content to rehydrate on this response. predict (Imagen) and
        # predictLongRunning (Veo) carry the prompt in instances[] but
        # answer with image bytes / an operation name.
        if match.group(1) in _REDACT_ONLY_VERBS:
            return RouteKind.REDACT_ONLY
        return RouteKind.CHAT

    def tracks_object_ids(self, method: str, path: str, body: Any = None) -> bool:
        # A context cache: later generateContent requests name it in
        # `cachedContent`, and the model echoes its (redacted) content. The
        # long-running jobs whose results are read back by their operation
        # NAME: a batch (`batches/<id>`) and a Veo video
        # (`models/<m>/operations/<id>`). A Files API file (every create
        # form), and a finished batch's output file, named on its status.
        if method == "GET":
            return _GEMINI_BATCH_STATUS.fullmatch(path) is not None
        if method != "POST":
            return False
        if _GEMINI_CACHED_CREATE.fullmatch(path) or _GEMINI_FILE_CREATE.fullmatch(path):
            return True
        match = _GEMINI_PATH.fullmatch(path)
        return match is not None and match.group(1) in _OPERATION_VERBS

    def object_ids_from_body(self, method: str, path: str, body: Any) -> tuple[str, ...]:
        if _GEMINI_FILE_CREATE.fullmatch(path):
            return file_object_ids(body, register=path.endswith(":register"))
        if method == "GET":
            return batch_output_file_ids(body)
        return cache_object_ids(body)

    def matches_request(
        self,
        method: str,
        path: str,
        headers: "Mapping[str, str] | None" = None,
        query: str = "",
    ) -> RouteKind:
        # An upload request carrying data the proxy would forward unread (a
        # resumable data chunk, the raw protocol) is not recognized.
        if method == "POST" and path == _GEMINI_UPLOAD and _upload_forwarded_unread(headers, query):
            return RouteKind.NONE
        return self.matches(method, path)

    def proxy_credential_refusal(
        self,
        method: str,
        path: str,
        headers: "Mapping[str, str]",
        query: str,
    ) -> str | None:
        if (
            method == "POST"
            and path == _GEMINI_UPLOAD
            and not _single_request_upload(headers, query)
        ):
            return _RESUMABLE_REFUSAL
        return None

    def multipart_boundary(self, path: str, content_type: str) -> bytes | None:
        if path == _GEMINI_UPLOAD:
            return _related_boundary(content_type)
        return super().multipart_boundary(path, content_type)

    def upload_metadata_boundary(self, path: str, content_type: str) -> bytes | None:
        # The single-request upload's first part is the file's metadata
        # ({"file": {"name", "displayName", ...}}): what the stored-object
        # check reads, as the metadata-only create's JSON body is read.
        return _related_boundary(content_type) if path == _GEMINI_UPLOAD else None

    def redacts_multipart(self, path: str) -> bool:
        return path == _GEMINI_UPLOAD

    def read_multipart(
        self, path: str, body: bytes, boundary: bytes, charge: Callable[[int], None]
    ) -> PartsReading | None:
        return checked_reading(read_related_upload(body, boundary, charge))

    def redact_multipart(
        self,
        path: str,
        body: bytes,
        boundary: bytes,
        redactor: Redactor,
        *,
        inject_note: bool,
        require_scanned: bool = False,
        forward_binary: Callable[[int], None] | None = None,
        inspected: InspectedUpload | None = None,
    ) -> bytes | None:
        # The single-request upload: the metadata part (JSON: the display
        # name) and the media part, each read as a FILE by the shared upload
        # policy (JSONL/text redacted; binary forwarded only when the proxy
        # passes forward_binary — the client's own key, binary_uploads =
        # "forward" — else refused).
        return redact_related_upload(
            body,
            boundary,
            redactor,
            require_scanned=require_scanned,
            forward_binary=forward_binary,
            inspected=inspected,
        )

    def restores_file_download(self, method: str, path: str) -> bool:
        return method == "GET" and _GEMINI_FILE_DOWNLOAD.fullmatch(path) is not None

    def rehydrate_raw_body(self, path: str, raw: bytes, rehydrator: Rehydrator) -> bytes | None:
        if _GEMINI_FILE_DOWNLOAD.fullmatch(path) is None:
            return None
        return rehydrate_download(raw, rehydrator)

    def lists_objects(self, method: str, path: str) -> bool:
        return method == "GET" and path in (_GEMINI_BATCHES, _GEMINI_FILES)

    def listing_items(self, body: Any) -> list[Any] | None:
        # The batch list answers ``{"operations": [...], "nextPageToken"}``,
        # the file list ``{"files": [...], "nextPageToken"}``.
        if not isinstance(body, dict):
            return None
        for key in ("operations", "files"):
            items = body.get(key)
            if isinstance(items, list):
                return items
        return None

    def listing_item_id(self, item: Any) -> str | None:
        # Listed by name (``batches/<id>``, ``files/<id>``), as the create
        # reported it.
        value = item.get("name") if isinstance(item, dict) else None
        return value if isinstance(value, str) else None

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        # countTokens bodies carry the same systemInstruction schema as the
        # chat request they mirror, so the note belongs in the count;
        # embed* bodies have no such field and must stay untouched, and
        # neither has a batch's status or list (no generate body at all).
        return (
            kind is RouteKind.CHAT and _GEMINI_PATH.fullmatch(path) is not None
        ) or path.endswith(":countTokens")

    def error_body(self, message: str, *, status: int = 413) -> dict[str, Any]:
        grpc_status = "FAILED_PRECONDITION" if status == 502 else "INVALID_ARGUMENT"
        return {"error": {"code": status, "message": message, "status": grpc_status}}

    def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        # REST uses camelCase; accept the snake_case alias some SDKs emit.
        key = "system_instruction" if "system_instruction" in body else "systemInstruction"
        existing = body.get(key)
        if existing is None:
            body[key] = {"parts": [{"text": SYSTEM_NOTE}]}
        elif isinstance(existing, dict):
            existing = dict(existing)
            existing["parts"] = [*(existing.get("parts") or []), {"text": SYSTEM_NOTE}]
            body[key] = existing
        elif isinstance(existing, str):
            body[key] = {"parts": [{"text": existing}, {"text": SYSTEM_NOTE}]}
        return body

    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        if not event.data:
            return [event]
        try:
            payload = loads_bounded(event.data)
        except ValueError:
            return [event]
        rehydrated = _rehydrate_chunk(
            payload,
            feed=lambda key, text: pool.get(key).feed(text),
            flush=lambda key: pool.flush(key),
            whole=pool.rehydrate_whole,
        )
        if rehydrated != payload:  # else the provider's own bytes go out
            event.data = json_text(rehydrated)
        return [event]

    def rehydrate_body(self, body: Any, rehydrator: Rehydrator) -> Any:
        if isinstance(body, list):
            return _rehydrate_chunk_list(body, rehydrator)
        return rehydrator.rehydrate_json(body)


class StreamedText:
    """A text value held out of a whole-value walk (jsonwalk returns a
    non-JSON object untouched): it streams through its channel instead.
    Shared with the Gemini Live adapter (realtime.py)."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def _hold_text_parts(chunk: dict[str, Any]) -> dict[str, Any]:
    """``chunk`` with every candidate text part's text held as ``StreamedText``,
    copied along that path only — the caller's tree is never modified."""
    candidates = []
    for candidate in chunk["candidates"]:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        if isinstance(content, dict) and isinstance(content.get("parts"), list):
            held = [
                {**part, "text": StreamedText(part["text"])}
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                else part
                for part in content["parts"]
            ]
            candidate = {**candidate, "content": {**content, "parts": held}}
        candidates.append(candidate)
    return {**chunk, "candidates": candidates}


def _rehydrate_chunk(
    chunk: Any,
    *,
    feed: Callable[[_ChannelKey, str], str],
    flush: Callable[[_ChannelKey], str],
    whole: Callable[[str], str],
) -> Any:
    """One stream chunk, restored exactly as the buffered response walk
    restores it — function-call args at their opaque position (a tool's own
    parameter named ``id`` or ``data``), generated code, grounding and
    citation metadata, usage-only chunks — except the candidates' text
    parts, which stream through per-(candidate, text|thought) channels (a
    «TOKEN» can straddle chunks) and flush on finishReason. Returns a new
    tree; ``chunk`` is not modified."""
    if not isinstance(chunk, dict) or not isinstance(chunk.get("candidates"), list):
        return transform_strings(chunk, whole)
    walked = transform_strings(_hold_text_parts(chunk), whole)
    for candidate in walked["candidates"]:
        if not isinstance(candidate, dict):
            continue
        index = candidate.get("index", 0)
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        for part in parts if isinstance(parts, list) else ():
            if isinstance(part, dict) and isinstance(part.get("text"), StreamedText):
                kind = "thought" if part.get("thought") else "text"
                part["text"] = feed((index, kind), part["text"].text)
        if candidate.get("finishReason"):
            for kind in ("text", "thought"):
                leftover = flush((index, kind))
                if leftover:
                    _append_text(candidate, kind, leftover)
    return walked


def _append_text(candidate: dict[str, Any], kind: str, leftover: str) -> None:
    """Attach flushed leftover to the candidate's last matching text part."""
    content = candidate.get("content")
    if not isinstance(content, dict):
        content = {}
        candidate["content"] = content
    parts = content.get("parts")
    if not isinstance(parts, list):
        parts = []
        content["parts"] = parts
    wanted_thought = kind == "thought"
    for part in reversed(parts):
        if (
            isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and bool(part.get("thought")) == wanted_thought
        ):
            part["text"] += leftover
            return
    new_part: dict[str, Any] = {"text": leftover}
    if wanted_thought:
        new_part["thought"] = True
    parts.append(new_part)


def _rehydrate_chunk_list(chunks: list[Any], rehydrator: Rehydrator) -> list[Any]:
    """The non-SSE streamGenerateContent array: same split-token hazard as
    the SSE stream, handled with per-candidate streaming channels, and every
    element restored like the SSE chunk it would have been."""
    channels: dict[_ChannelKey, StreamingRehydrator] = {}

    def feed(key: _ChannelKey, text: str) -> str:
        channel = channels.get(key)
        if channel is None:
            channel = rehydrator.streaming_channel()
            channels[key] = channel
        return channel.feed(text)

    def flush(key: _ChannelKey) -> str:
        channel = channels.pop(key, None)
        return channel.flush() if channel is not None else ""

    out: list[Any] = []
    last_candidate: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        walked = _rehydrate_chunk(chunk, feed=feed, flush=flush, whole=rehydrator.rehydrate_text)
        out.append(walked)
        candidates = walked.get("candidates") if isinstance(walked, dict) else None
        for candidate in candidates if isinstance(candidates, list) else ():
            if isinstance(candidate, dict):
                last_candidate[candidate.get("index", 0)] = candidate
    # A stream that never carried finishReason still must not drop text.
    for (index, kind), channel in list(channels.items()):
        leftover = channel.flush()
        if leftover and index in last_candidate:
            _append_text(last_candidate[index], kind, leftover)
    channels.clear()
    return out
