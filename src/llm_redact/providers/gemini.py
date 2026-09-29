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
from collections.abc import Callable
from typing import Any

from llm_redact.jsonwalk import json_text, loads_bounded, transform_strings
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.rehydrate import Rehydrator, RehydratorPool, StreamingRehydrator
from llm_redact.sse import SSEEvent

_GEMINI_PATH = re.compile(
    r"/(?:v1|v1beta)/(?:models|tunedModels)/[^/:]+:"
    r"(generateContent|streamGenerateContent|countTokens|embedContent"
    r"|batchEmbedContents|batchGenerateContent|predict|predictLongRunning)"
)
# Verbs that answer with a long-running operation whose results are read
# back by its name later (the Gemini API's batch mode and Veo).
_OPERATION_VERBS = frozenset({"batchGenerateContent", "predictLongRunning"})
# Context caching: only the create (POST /…/cachedContents) carries content to
# redact. The per-cache GET/PATCH/DELETE and list return metadata (name, model,
# token counts, expiry) — never the cached content — so they pass through.
_GEMINI_CACHED_CREATE = re.compile(r"/(?:v1|v1beta)/cachedContents")
# The Files API (media — the documented non-goal — so every route passes
# through, never redacted or restored). A file is created by the media
# upload (``/upload/v1beta/files``: the multipart protocol's one request, or
# a resumable upload's finalizing chunk when it is sent through the proxy),
# the metadata-only create, or ``files:register`` (Cloud Storage objects);
# each answer names the file(s) as ``files/<id>``, which later bodies cite
# (``fileData.fileUri``, a batch's input ``fileName``) and paths read.
_GEMINI_FILE_CREATE = re.compile(r"(?:/upload)?/v1beta/files(?::register)?")
# A batch's status (the operation read back by name): once it finished it
# names the batch's output FILE — the creator's (only the creator's own read
# of the batch reaches the provider and gets here unsealed).
_GEMINI_BATCH_STATUS = re.compile(r"/v1beta/batches/[^/:]+")
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


def file_object_ids(body: Any) -> tuple[str, ...]:
    """The files a Files API create answers with: the upload's (and the
    metadata-only create's) ``{"file": File}``, or ``files:register``'s
    ``{"files": [File, …]}``."""
    if not isinstance(body, dict):
        return ()
    listed = body.get("files")
    files = [body.get("file"), *(listed if isinstance(listed, list) else ())]
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


class GeminiAdapter(ProviderAdapter):
    name = "gemini"

    def matches(self, method: str, path: str) -> RouteKind:
        if method == "GET" and _GEMINI_MODELS.fullmatch(path):
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
        # embeddings responses are vectors; batchGenerateContent returns a
        # long-running operation NAME (the generated content is fetched later
        # via the operation) — none has content to rehydrate on this response.
        # predict (Imagen) and predictLongRunning (Veo) carry the prompt in
        # instances[] but answer with image bytes / an operation name.
        if match.group(1) in (
            "countTokens",
            "embedContent",
            "batchEmbedContents",
            "batchGenerateContent",
            "predict",
            "predictLongRunning",
        ):
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
            return file_object_ids(body)
        if method == "GET":
            return batch_output_file_ids(body)
        return cache_object_ids(body)

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        # countTokens bodies carry the same systemInstruction schema as the
        # chat request they mirror, so the note belongs in the count;
        # embed* bodies have no such field and must stay untouched.
        return kind is RouteKind.CHAT or path.endswith(":countTokens")

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
