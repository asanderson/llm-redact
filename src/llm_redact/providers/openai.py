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
conversation anchor — the realtime WS stance); a user-scoping session
router (llm-redact-pro's named users) makes that the user's own copy.
"""

import json
import re
from collections.abc import Hashable, Mapping
from typing import Any

from llm_redact import multipart
from llm_redact.jsonwalk import loads_request, transform_strings
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.redactor import Redactor, UnredactableRequest
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent

_FILE_CONTENT_RE = re.compile(r"/v1/files/[^/]+/content")
# The file list and one file's object: both echo each upload's filename.
_FILE_OBJECT_RE = re.compile(r"/v1/files(?:/[^/]+)?")
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


def _parse_request_line(line: bytes) -> tuple[dict[str, Any] | None, bool]:
    """``_parse_object_line`` for an uploaded REQUEST line, plus whether an
    object in it repeats a key (then the line must be re-serialized)."""
    stripped = line.strip()
    if not stripped:
        return None, False
    try:
        obj, duplicate_keys = loads_request(stripped)
    except ValueError:
        return None, False
    return (obj, duplicate_keys) if isinstance(obj, dict) else (None, False)


def _redact_text_part(
    part: multipart.MultipartPart, redactor: Redactor, *, require_scanned: bool
) -> bool:
    """Redact a form part's content as UTF-8 text, in place; True when it
    changed. Content that is not UTF-8 is left alone — or, under identity
    auth (``require_scanned``), refused as unscannable."""
    try:
        text = part.content.decode("utf-8")
    except UnicodeDecodeError:
        if require_scanned:
            raise UnredactableRequest(
                "a multipart form field is not UTF-8 text llm-redact can redact"
            ) from None
        return False
    redacted = redactor.redact_text(text)
    if redacted == text:
        return False
    part.content = redacted.encode("utf-8")
    return True


# What an identity-authorized upload may declare: a part the proxy cannot
# read as its plain bytes is never signed.
_PLAIN_TRANSFER_ENCODINGS = frozenset({b"7bit", b"8bit", b"binary"})
_PLAIN_CHARSETS = frozenset({"utf-8", "us-ascii"})


def _require_plain_encoding(part: multipart.MultipartPart, *, scanned: bool) -> None:
    """Identity auth: refuse a part the proxy could not read as plain bytes
    (the body-part twin of the request Content-Encoding rule): any
    Content-Transfer-Encoding but 7bit/8bit/binary (RFC 7578 deprecates
    them), and — on a part whose content is scanned — a declared charset
    other than UTF-8/US-ASCII, whether its own Content-Type parameter or
    the RFC 7578 §4.6 ``_charset_`` field. AmbiguousHeaders propagates."""
    encoding = part.header("content-transfer-encoding")
    if encoding is not None and encoding.lower() not in _PLAIN_TRANSFER_ENCODINGS:
        raise UnredactableRequest(
            "a multipart part declares a Content-Transfer-Encoding llm-redact does not decode"
        )
    if not scanned:
        return
    charset = (part.params("content-type") or {}).get("charset")
    declared = [] if charset is None else [charset.value]
    if part.name == "_charset_":
        declared.append(part.content.decode("latin-1").strip())
    if any(name.lower() not in _PLAIN_CHARSETS for name in declared):
        raise UnredactableRequest("a multipart part declares a charset llm-redact does not decode")


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


# Responses naming objects the provider stores for later reads: an uploaded
# file, a batch (whose output/error files appear on its status), a stored
# conversation. Matched on the path's tail, so the Azure and custom-provider
# prefixes need no override.
_BATCH_OBJECT_RE = re.compile(r"(?:^|/)batches/[^/]+(?:/cancel)?")
_BATCH_FILE_KEYS = ("output_file_id", "error_file_id")


def _string_ids(body: Any, keys: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(body, dict):
        return ()
    return tuple(str(body[k]) for k in keys if isinstance(body.get(k), str) and body[k])


# Sora video jobs: create (POST …/videos) and remix (POST …/videos/{id}/remix,
# which creates a NEW video) answer with the stored video object — later
# read by id (GET …/videos/{id}). Tail-anchored like the batch pattern.
_VIDEO_CREATE_RE = re.compile(r"(?:^|/)videos(?:/[^/]+/remix)?$")


def _stored_completion_create(path: str, body: Any) -> bool:
    """A chat completion created with ``store: true``: OpenAI keeps it for
    later reads (GET …/chat/completions/{id}); without the flag nothing is
    stored and nothing is reported (one durable row per chat request would
    grow the map without bound)."""
    return (
        path.rstrip("/").endswith("/chat/completions")
        and isinstance(body, dict)
        and body.get("store") is True
    )


# Collection reads whose `{"object": "list", "data": [...]}` answer lists
# stored objects by id: files, batches, video jobs and stored chat
# completions. Tail-anchored, so the Azure and custom-provider prefixes
# need no override.
_LISTING_RE = re.compile(r"(?:^|/)(?:files|batches|videos|chat/completions)$")


def _tail_is_create(path: str) -> bool:
    """POST to the collection itself (``…/files``, ``…/batches``,
    ``…/conversations``), not to a member or sub-resource."""
    tail = path.rstrip("/").rsplit("/", 1)[-1]
    return tail in ("files", "batches", "conversations")


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
            # Multipart upload whose JSONL file part (and file NAME) carries
            # user content; redacted via redact_multipart. The response is
            # the file object, echoing the redacted filename: restored.
            return RouteKind.CHAT
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
        if method == "GET" and _FILE_OBJECT_RE.fullmatch(path):
            # The file list and a file's metadata echo the filename the
            # upload redacted: restored in the request's own session (a
            # user-scoping router answers listings and foreign reads from
            # an empty one). DELETE carries ids only and passes through.
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
        # the body shape — and file downloads/objects are body-less GETs.
        if path == "/v1/completions" or path.startswith("/v1/files/"):
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

    def tracks_object_ids(self, method: str, path: str, body: Any = None) -> bool:
        if method == "POST" and (
            _tail_is_create(path)
            or _VIDEO_CREATE_RE.search(path.rstrip("/")) is not None
            or _stored_completion_create(path, body)
        ):
            return True
        return method in ("GET", "POST") and _BATCH_OBJECT_RE.search(path) is not None

    def object_ids_from_body(self, method: str, path: str, body: Any) -> tuple[str, ...]:
        if _BATCH_OBJECT_RE.search(path) is not None:
            return _string_ids(body, _BATCH_FILE_KEYS)
        if path.rstrip("/").endswith("/batches"):
            return _string_ids(body, ("id", *_BATCH_FILE_KEYS))
        return _string_ids(body, ("id",))

    def lists_objects(self, method: str, path: str) -> bool:
        return method == "GET" and _LISTING_RE.search(path.rstrip("/")) is not None

    def listing_items(self, body: Any) -> list[Any] | None:
        if not isinstance(body, dict) or body.get("object") != "list":
            return None
        data = body.get("data")
        return data if isinstance(data, list) else None

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

    def redacts_multipart(self, path: str) -> bool:
        # The Files upload (JSONL file parts) and the prompt-field media
        # routes; suffix match so Azure's /openai/... shapes reuse it.
        return path.endswith(("/files", *_PROMPT_FIELD_PATH_SUFFIXES))

    def redact_multipart(
        self,
        path: str,
        body: bytes,
        boundary: bytes,
        redactor: Redactor,
        *,
        inject_note: bool,
        require_scanned: bool = False,
    ) -> bytes | None:
        parsed = multipart.parse(body, boundary)
        if parsed is None:
            return None  # outside the canonical grammar: forward verbatim
        if require_scanned and (parsed.preamble.strip() or parsed.epilogue.strip()):
            # Bytes outside every part: servers ignore them, but they still
            # leave the machine under the proxy's identity, unscanned.
            raise UnredactableRequest(
                "the multipart body carries a preamble or epilogue llm-redact does not redact"
            )
        # Media endpoints: the file parts ARE the media (the documented
        # non-goal); the user text rides named form fields. Suffix match so
        # the Azure subclass's /openai/... path shapes reuse this unchanged.
        media = path.endswith(_PROMPT_FIELD_PATH_SUFFIXES)
        changed = False
        try:
            for part in parsed.parts:
                changed |= self._redact_part(
                    part,
                    redactor,
                    media=media,
                    inject_note=inject_note,
                    require_scanned=require_scanned,
                )
        except multipart.AmbiguousHeaders as exc:
            # Only reachable under identity auth (require_scanned): a part
            # header without a single reading is never signed.
            raise UnredactableRequest(str(exc)) from None
        return parsed.serialize() if changed else None

    def _redact_part(
        self,
        part: multipart.MultipartPart,
        redactor: Redactor,
        *,
        media: bool,
        inject_note: bool,
        require_scanned: bool,
    ) -> bool:
        # The upload's file name is user content on every route (the part
        # name is structural, like a JSON key). Strict under identity auth,
        # so the routing reads below always see the one reading.
        changed = part.redact_filenames(redactor.redact_text, strict=require_scanned)
        if media and part.name in _PROMPT_FIELDS:
            # Matched by NAME regardless of a filename attribute: a prompt
            # part dressed up as a file upload must not slip past the scan
            # (fail closed) — binary content skips via the decode (refused
            # under identity).
            kind = "text"
        elif part.filename is not None:
            kind = "media" if media else "jsonl"  # media: never scanned
        else:
            # Plain form fields (purpose, model, size, user, ...) are
            # forwarded as-is by default; the proxy's own identity signs
            # them only once scanned as text.
            kind = "text" if require_scanned else "field"
        if require_scanned:
            _require_plain_encoding(part, scanned=kind != "media")
        if kind == "jsonl":
            new_content = self._redact_jsonl(
                part.content, redactor, inject_note=inject_note, require_scanned=require_scanned
            )
            if new_content != part.content:
                part.content = new_content
                changed = True
        elif kind == "text":
            changed |= _redact_text_part(part, redactor, require_scanned=require_scanned)
        return changed

    def _redact_jsonl(
        self, data: bytes, redactor: Redactor, *, inject_note: bool, require_scanned: bool = False
    ) -> bytes:
        out: list[bytes] = []
        for line in data.split(b"\n"):
            obj, duplicate_keys = _parse_request_line(line)
            if obj is None:
                if require_scanned and line.strip():
                    raise UnredactableRequest(
                        "an uploaded JSONL line is not a JSON object llm-redact can redact"
                    )
                out.append(line)  # blank/binary/unparseable: byte-identical
                continue
            redacted = redactor.redact_json(obj)
            changed = redacted != obj
            if not changed and not duplicate_keys:
                # Unchanged — and no repeated key whose earlier occurrence
                # the walk never saw — so the original line is safe.
                out.append(line)
                continue
            if inject_note and changed:
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
