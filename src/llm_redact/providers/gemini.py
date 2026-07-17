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
  OpenAI ``arguments`` string) and arrives complete in one event: a plain
  jsonwalk with whole-string restoration is correct there.
- ``:streamGenerateContent`` WITHOUT ``alt=sse`` returns one JSON *array* of
  those same chunks; its elements split tokens exactly like the SSE form, so
  rehydrate_body runs per-candidate streaming channels across elements.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from llm_redact.jsonwalk import transform_strings
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.rehydrate import Rehydrator, RehydratorPool, StreamingRehydrator
from llm_redact.sse import SSEEvent

_GEMINI_PATH = re.compile(
    r"/(?:v1|v1beta)/(?:models|tunedModels)/[^/:]+:"
    r"(generateContent|streamGenerateContent|countTokens|embedContent"
    r"|batchEmbedContents|batchGenerateContent|predict|predictLongRunning)"
)
# Context caching: only the create (POST /…/cachedContents) carries content to
# redact. The per-cache GET/PATCH/DELETE and list return metadata (name, model,
# token counts, expiry) — never the cached content — so they pass through.
_GEMINI_CACHED_CREATE = re.compile(r"/(?:v1|v1beta)/cachedContents")

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


class GeminiAdapter(ProviderAdapter):
    name = "gemini"

    def matches(self, method: str, path: str) -> RouteKind:
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
            payload = json.loads(event.data)
        except ValueError:
            return [event]
        if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
            return [event]  # usageMetadata / promptFeedback-only chunks
        _process_candidates(
            payload["candidates"],
            feed=lambda key, text: pool.get(key).feed(text),
            flush=lambda key: pool.flush(key),
            whole=pool.rehydrate_whole,
        )
        event.data = json.dumps(payload, ensure_ascii=False)
        return [event]

    def rehydrate_body(self, body: Any, rehydrator: Rehydrator) -> Any:
        if isinstance(body, list):
            return _rehydrate_chunk_list(body, rehydrator)
        return rehydrator.rehydrate_json(body)


def _process_candidates(
    candidates: list[Any],
    *,
    feed: Callable[[_ChannelKey, str], str],
    flush: Callable[[_ChannelKey], str],
    whole: Callable[[str], str],
) -> None:
    """Rewrite one chunk's candidates in place; flush on finishReason."""
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        index = candidate.get("index", 0)
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    kind = "thought" if part.get("thought") else "text"
                    part["text"] = feed((index, kind), part["text"])
                elif isinstance(part.get("functionCall"), dict):
                    # args is a parsed object arriving complete: walk its
                    # string values (skip_keys protects "name").
                    call = part["functionCall"]
                    if isinstance(call.get("args"), dict):
                        call["args"] = transform_strings(call["args"], whole)
        if candidate.get("finishReason"):
            for kind in ("text", "thought"):
                leftover = flush((index, kind))
                if leftover:
                    _append_text(candidate, kind, leftover)


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
    the SSE stream, handled with per-candidate streaming channels."""
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

    last_candidate: dict[int, dict[str, Any]] = {}
    for chunk in chunks:
        if isinstance(chunk, dict) and isinstance(chunk.get("candidates"), list):
            _process_candidates(
                chunk["candidates"],
                feed=feed,
                flush=flush,
                whole=rehydrator.rehydrate_text,
            )
            for candidate in chunk["candidates"]:
                if isinstance(candidate, dict):
                    last_candidate[candidate.get("index", 0)] = candidate
        else:
            rehydrator.rehydrate_json(chunk)
    # A stream that never carried finishReason still must not drop text.
    for (index, kind), channel in list(channels.items()):
        leftover = channel.flush()
        if leftover and index in last_candidate:
            _append_text(last_candidate[index], kind, leftover)
    channels.clear()
    return chunks
