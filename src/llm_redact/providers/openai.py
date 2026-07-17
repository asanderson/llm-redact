"""OpenAI Chat Completions adapter (/v1/chat/completions, data: {json} SSE).

Also covers the Files + Batches surfaces (verified against
platform.openai.com docs, 2026-07): ``POST /v1/files`` is
multipart/form-data whose file part is JSONL — batch input lines
({custom_id, method, url, body}) and fine-tuning lines ({messages: [...]})
both carry user content, so every line that parses as a JSON object is
redacted (and chat-shaped ones get the system note); anything else in the
upload — form fields, binary documents, unparseable lines — is preserved
byte-identically. ``GET /v1/files/{id}/content`` rehydrates batch OUTPUT
files the same way, line by line. ``/v1/batches`` itself carries only file
ids and processing metadata: deliberate pass-through, pinned by test.
Batch flows use the static vault session (an async fetch has no
conversation anchor — the realtime WS stance).
"""

import json
import re
from collections.abc import Hashable, Mapping
from typing import Any

from llm_redact import multipart
from llm_redact.jsonwalk import transform_strings
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent

_FILE_CONTENT_RE = re.compile(r"/v1/files/[^/]+/content")
_STORED_COMPLETION_RE = re.compile(r"/v1/chat/completions/[^/]+")

# Multipart endpoints whose TEXT FORM FIELDS are the content (their file
# parts are media — the non-goal). Everything else keeps the /v1/files
# JSONL-file-part handling.
_PROMPT_FIELD_PATH_SUFFIXES = ("/images/edits", "/videos")
_PROMPT_FIELDS = frozenset({"prompt"})

# Sora video jobs: list/create, item retrieve/delete, and remix. The
# binary /content download deliberately does NOT match (media
# pass-through) — only single-segment ids and the /remix action do.
_VIDEO_ROUTE_RE = re.compile(r"/v1/videos(?:/[^/]+(?:/remix)?)?")


def _parse_object_line(line: bytes) -> dict[str, Any] | None:
    """The line's JSON object, or None for blank/unparseable/non-object."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


# Delta fields that carry reasoning-model chain-of-thought as a string,
# rehydrated on their own per-choice channels exactly like `content`.
# `reasoning_content` is DeepSeek/vLLM/Groq/xAI; `reasoning` is OpenRouter's
# unified field. Non-streaming responses carry these too, but the generic
# jsonwalk already rehydrates those — only the streaming deltas need this.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def _leftover_to_delta(key: Hashable, text: str) -> tuple[int, dict[str, Any]] | None:
    """Map a flushed channel key to (choice_index, delta payload)."""
    if not (isinstance(key, tuple) and text):
        return None
    if len(key) == 2 and key[1] == "content":
        return key[0], {"content": text}
    if len(key) == 2 and key[1] in _REASONING_FIELDS:
        # Reasoning-model streams (DeepSeek reasoner, Groq, xAI, OpenRouter)
        # carry chain-of-thought in a sibling delta field.
        return key[0], {key[1]: text}
    if len(key) == 3 and key[1] == "tool":
        return key[0], {"tool_calls": [{"index": key[2], "function": {"arguments": text}}]}
    return None


def _synthetic_chunk(index: int, delta: dict[str, Any]) -> SSEEvent:
    return SSEEvent(
        data=json.dumps(
            {
                "object": "chat.completion.chunk",
                "choices": [{"index": index, "delta": delta, "finish_reason": None}],
            },
            ensure_ascii=False,
        )
    )


class OpenAIAdapter(ProviderAdapter):
    name = "openai"

    def matches(self, method: str, path: str) -> RouteKind:
        if method == "POST" and path == "/v1/chat/completions":
            return RouteKind.CHAT
        if method == "GET" and _STORED_COMPLETION_RE.fullmatch(path):
            # Stored-completion retrieval: the saved content may carry
            # placeholders that were sent upstream — restore them.
            return RouteKind.CHAT
        if method == "POST" and path == "/v1/completions":
            # Legacy text completions: deprecated, but still a content
            # leak for old tools. prompt redacted; choices[].text
            # rehydrated (streaming included). No system note — the body
            # has no messages field to carry one.
            return RouteKind.CHAT
        if method == "POST" and path == "/v1/embeddings":
            # Input text is redactable; the response is vectors — nothing
            # to rehydrate. (Embedded placeholders shift the vectors, but
            # forwarding raw secrets is never acceptable.)
            return RouteKind.REDACT_ONLY
        if method == "POST" and path == "/v1/files":
            # Multipart upload whose JSONL file part carries user content;
            # redacted via redact_multipart. The response is file metadata.
            return RouteKind.REDACT_ONLY
        if method == "POST" and path == "/v1/images/generations":
            # The OUTPUT is media (the non-goal) but the prompt is plain
            # text that must not reach the provider in the clear. Response
            # media (b64_json/url) has nothing to restore; a dall-e-3
            # revised_prompt echo may carry placeholder tokens — the
            # fail-safe direction, documented in api-coverage.md.
            return RouteKind.REDACT_ONLY
        if method == "POST" and path == "/v1/images/edits":
            # Multipart: the prompt rides a plain form FIELD next to the
            # image/mask file parts; only named text fields are scanned
            # (the file parts are media). /v1/images/variations carries no
            # text at all and stays pass-through, pinned by test.
            return RouteKind.REDACT_ONLY
        if method == "POST" and path == "/v1/audio/speech":
            # Text-to-speech: the `input` is user text; the response is
            # audio bytes forwarded verbatim (non-JSON branch).
            # transcriptions/translations upload AUDIO — the media
            # non-goal — and stay pass-through, pinned by test.
            return RouteKind.REDACT_ONLY
        if _VIDEO_ROUTE_RE.fullmatch(path):
            # Sora video jobs: create/remix prompts are text, and the job
            # object ECHOES the prompt — so create/remix/list/retrieve are
            # all CHAT (request redacted where present, echoed prompt
            # restored). The binary /content download and delete stay
            # pass-through.
            if method in ("POST", "GET"):
                return RouteKind.CHAT
            return RouteKind.NONE
        if method == "GET" and _FILE_CONTENT_RE.fullmatch(path):
            # Batch output downloads: JSONL rehydrated line by line via
            # rehydrate_raw_body (no request body — redaction no-ops).
            return RouteKind.CHAT
        if path.startswith("/v1/conversations"):
            # Stateful item store paired with the Responses API. Item content
            # (message text) rode through UNREDACTED before this. POST create /
            # add-items redact the request items AND rehydrate the echoed
            # response; GET retrieve / list-items rehydrate the stored content.
            # DELETE carries ids only. Conversations use the STATIC vault
            # session (async reads have no anchor — the batch/realtime stance,
            # enforced in sessions.py), so redact and rehydrate always agree.
            if method in ("POST", "GET"):
                return RouteKind.CHAT
            return RouteKind.NONE
        # /v1/batches and file list/metadata/delete carry ids and
        # processing metadata only: deliberate pass-through, pinned by test.
        return RouteKind.NONE

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        # File uploads inject per JSONL line (into chat-shaped bodies)
        # inside redact_multipart; this gate just allows that to happen.
        # Legacy completions have no messages field — a note would corrupt
        # the body shape.
        if path == "/v1/completions":
            return False
        if path.startswith("/v1/conversations"):
            # Item bodies carry `items`, not `messages`; injecting the note
            # would graft a spurious `messages` field and corrupt the request.
            return False
        if path.startswith("/v1/videos"):
            # Video job bodies have no messages field either — a note would
            # graft one and corrupt the create/remix request.
            return False
        return kind is RouteKind.CHAT or path == "/v1/files"

    def matches_request(
        self, method: str, path: str, headers: "Mapping[str, str] | None" = None
    ) -> RouteKind:
        # /v1/files and /v1/batches are shared with Anthropic's beta Files
        # API; an anthropic-version header marks that traffic, which is
        # not ours (it passes through to the anthropic upstream).
        if (
            headers is not None
            and "anthropic-version" in headers
            and path.startswith(("/v1/files", "/v1/batches"))
        ):
            return RouteKind.NONE
        return self.matches(method, path)

    def error_body(self, message: str, *, status: int = 413) -> dict[str, Any]:
        return {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "request_too_large" if status == 413 else None,
            }
        }

    def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
        body = dict(body)
        messages = list(body.get("messages", []))
        messages.insert(0, {"role": "system", "content": SYSTEM_NOTE})
        body["messages"] = messages
        return body

    def rehydrate_body(self, body: Any, rehydrator: Rehydrator) -> Any:
        # `message.tool_calls[].function.arguments` is raw JSON *source*, not
        # a parsed object: restored originals must be re-escaped there or a
        # value containing quotes/newlines corrupts the arguments string.
        return transform_strings(
            body,
            rehydrator.rehydrate_text,
            key_overrides={"arguments": rehydrator.rehydrate_json_source_text},
        )

    def redact_multipart(
        self, path: str, body: bytes, boundary: bytes, redactor: Redactor, *, inject_note: bool
    ) -> bytes | None:
        parsed = multipart.parse(body, boundary)
        if parsed is None:
            return None  # outside the canonical grammar: forward verbatim
        changed = False
        if path.endswith(_PROMPT_FIELD_PATH_SUFFIXES):
            # Media endpoints: the file parts ARE the media (non-goal); the
            # user text rides named form fields. Suffix match so the Azure
            # subclass's /openai/... path shapes reuse this unchanged.
            # Matched by NAME regardless of a filename attribute: a prompt
            # part dressed up as a file upload must not slip past the scan
            # (fail closed) — binary content still skips via the decode.
            for part in parsed.parts:
                if part.name not in _PROMPT_FIELDS:
                    continue
                try:
                    text = part.content.decode("utf-8")
                except UnicodeDecodeError:
                    continue  # not text: leave the bytes alone
                redacted = redactor.redact_text(text)
                if redacted != text:
                    part.content = redacted.encode("utf-8")
                    changed = True
            return parsed.serialize() if changed else None
        for part in parsed.parts:
            if part.filename is None:
                continue  # plain form fields (purpose, ...) are not content
            new_content = self._redact_jsonl(part.content, redactor, inject_note=inject_note)
            if new_content != part.content:
                part.content = new_content
                changed = True
        return parsed.serialize() if changed else None

    def _redact_jsonl(self, data: bytes, redactor: Redactor, *, inject_note: bool) -> bytes:
        out: list[bytes] = []
        for line in data.split(b"\n"):
            obj = _parse_object_line(line)
            if obj is None:
                out.append(line)  # blank/binary/unparseable: byte-identical
                continue
            redacted = redactor.redact_json(obj)
            if redacted == obj:
                out.append(line)
                continue
            if inject_note:
                body_obj = redacted.get("body")
                if isinstance(body_obj, dict) and isinstance(body_obj.get("messages"), list):
                    # Batch input line: {custom_id, method, url, body}.
                    redacted = {**redacted, "body": self.inject_system_note(body_obj)}
                elif isinstance(redacted.get("messages"), list):
                    # Fine-tuning line: a bare chat example.
                    redacted = self.inject_system_note(redacted)
            out.append(json.dumps(redacted, ensure_ascii=False).encode("utf-8"))
        return b"\n".join(out)

    def rehydrate_raw_body(self, path: str, raw: bytes, rehydrator: Rehydrator) -> bytes | None:
        if not _FILE_CONTENT_RE.fullmatch(path):
            return None
        out: list[bytes] = []
        changed = False
        for line in raw.split(b"\n"):
            obj = _parse_object_line(line)
            if obj is None:
                out.append(line)  # non-JSONL file contents stay untouched
                continue
            hydrated = self.rehydrate_body(obj, rehydrator)
            if hydrated == obj:
                out.append(line)
            else:
                out.append(json.dumps(hydrated, ensure_ascii=False).encode("utf-8"))
                changed = True
        return b"\n".join(out) if changed else None

    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        if not event.data:
            return [event]
        if event.data.strip() == "[DONE]":
            # Flush everything still buffered before the terminal sentinel.
            return [*self._flush_to_events(pool.flush_all()), event]
        try:
            payload = json.loads(event.data)
        except ValueError:
            return [event]

        changed = False
        finished: list[int] = []
        for choice in payload.get("choices", []):
            index = choice.get("index", 0)
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    delta["content"] = pool.get((index, "content")).feed(content)
                    changed = True
                for field in _REASONING_FIELDS:
                    value = delta.get(field)
                    if isinstance(value, str):
                        delta[field] = pool.get((index, field)).feed(value)
                        changed = True
                for tool_call in delta.get("tool_calls") or []:
                    function = tool_call.get("function")
                    if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                        function["arguments"] = pool.get(
                            (index, "tool", tool_call.get("index", 0)), json_source=True
                        ).feed(function["arguments"])
                        changed = True
            text_value = choice.get("text")
            if isinstance(text_value, str):
                # Legacy /v1/completions chunks carry text directly.
                choice["text"] = pool.get((index, "legacy")).feed(text_value)
                changed = True
            if choice.get("finish_reason"):
                finished.append(index)

        # A finished choice can still have held-back text: emit it as
        # synthetic chunks ordered before the finish_reason chunk.
        synthetic: list[SSEEvent] = []
        for index in finished:

            def _for_choice(key: Hashable, i: int = index) -> bool:
                return isinstance(key, tuple) and key[0] == i

            synthetic.extend(self._flush_to_events(pool.flush_matching(_for_choice)))

        if changed:
            event.data = json.dumps(payload, ensure_ascii=False)
        return [*synthetic, event]

    @staticmethod
    def _flush_to_events(leftovers: dict[Hashable, str]) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        for key, text in leftovers.items():
            if isinstance(key, tuple) and text and len(key) == 2 and key[1] == "legacy":
                # Legacy completions leftover: a text_completion-shaped chunk.
                events.append(
                    SSEEvent(
                        data=json.dumps(
                            {
                                "object": "text_completion",
                                "choices": [{"index": key[0], "text": text, "finish_reason": None}],
                            },
                            ensure_ascii=False,
                        )
                    )
                )
                continue
            mapped = _leftover_to_delta(key, text)
            if mapped is not None:
                events.append(_synthetic_chunk(mapped[0], mapped[1]))
        return events
