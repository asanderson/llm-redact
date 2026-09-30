"""OpenAI Chat Completions adapter (/v1/chat/completions, data: {json} SSE).

Also covers the Files + Batches surfaces (verified against
platform.openai.com docs, 2026-07): ``POST /v1/files`` is
multipart/form-data whose file part is read BY ITS CONTENT
(``upload_content.classify_file``): a JSONL file — batch input lines
({custom_id, method, url, body}) and fine-tuning lines ({messages: [...]})
both carry user content — has every line redacted as JSON (and chat-shaped
ones get the system note); any other text file (UTF-8, or UTF-16/32 with a
byte-order mark) is redacted as one text and re-encoded as it came; a
binary file (a PDF, an image, an archive, text that does not decode) is
never read. Every part's ``filename`` is redacted too, and the file object
the provider echoes (the upload response, the file list, one file's
metadata) is restored. Plain form fields are scanned as UTF-8 text like the
JSON strings they mirror (a ``user`` field as a chat body's ``user``). The
proxy requires every piece of an upload scanned (``require_scanned``, the
scanned-body rule): anything it cannot read — a field that is not UTF-8, a
JSONL line nesting too deep, a transfer encoding — refuses the upload. A
binary file part is the one piece that may go out unscanned, and only when
the caller says so (``forward_binary``: the proxy's choice when the request
is sent with the client's own credential and ``[detection] binary_uploads``
is "forward"); otherwise it refuses the upload too. A caller that does not
require scanning (``require_scanned=False``) gets the lenient reading: the
structural fields — enums, sizes, counts, the model — as sent, and what
cannot be read preserved byte-identically.
``GET /v1/files/{id}/content`` restores a downloaded file by the same
reading: a text file (batch OUTPUT JSONL, a results CSV) line by line — a
JSON line as JSON, any other line as text — and a binary file untouched.
``/v1/batches`` carries file ids,
processing state and the caller's own ``metadata`` (free-form strings the
batch object echoes on every read): create is redacted and every echo —
a batch's GET, its cancel, and the LIST — restored in the request's own
session (Azure's handling). With llm-redact-pro's named users a listing
resolves to an empty session and the session router attributes each
listed batch to the session that created it (``listing_item_session``).
Batch flows use the static vault session (an async fetch has no
conversation anchor — the realtime WS stance); a user-scoping session
router (llm-redact-pro's named users) makes that the user's own copy.
"""

import re
from collections.abc import Callable, Hashable, Mapping
from typing import Any, NamedTuple

from llm_redact import multipart
from llm_redact.jsonwalk import (
    JsonTooDeep,
    json_bytes,
    json_text,
    loads_bounded,
    loads_request,
    transform_all_strings,
    transform_strings,
)
from llm_redact.placeholders import json_floors, may_carry_tokens, merge_floors, token_floors
from llm_redact.providers.attribution import provider_markers
from llm_redact.providers.base import SYSTEM_NOTE, ProviderAdapter, RouteKind
from llm_redact.redactor import Redactor, UnredactableRequest
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent
from llm_redact.upload_content import BINARY, FileContent, classify_file
from llm_redact.upload_view import PLAIN_CHARSETS, PLAIN_TRANSFER_ENCODINGS

_FILE_CONTENT_RE = re.compile(r"/v1/files/[^/]+/content")
# A code interpreter container's file, downloaded (uploaded, or written by
# the code): restored like a Files API download (text line by line, a
# binary file untouched).
_CONTAINER_FILE_CONTENT_RE = re.compile(r"/v1/containers/[^/]+/files/[^/]+/content")
# The file list and one file's object: both echo each upload's filename.
_FILE_OBJECT_RE = re.compile(r"/v1/files(?:/[^/]+)?")
_STORED_COMPLETION_RE = re.compile(r"/v1/chat/completions/[^/]+")

# Multipart endpoints whose TEXT FORM FIELDS are the content (their file
# parts are media — the non-goal). Everything else keeps the /v1/files
# JSONL-file-part handling.
_PROMPT_FIELD_PATH_SUFFIXES = ("/images/edits", "/videos")
_PROMPT_FIELDS = frozenset({"prompt"})

# Plain form fields whose values are protocol, not content — the multipart
# twin of jsonwalk.STRUCTURAL_KEYS: enums, sizes, counts and the model name
# of the multipart routes redact_multipart handles (verified against the
# OpenAI/Azure Files upload, Images edit and Videos create schemas), plus
# RFC 7578's ``_charset_`` declaration. The lenient reading
# (``require_scanned=False``) forwards them as sent;
# every other plain field (``user``, ``prompt``, anything unknown) is user
# content and is redacted as text, as its JSON twin is. The proxy's own
# identity scans them all (it signs only what it read).
_STRUCTURAL_FORM_FIELDS = frozenset(
    {
        "_charset_",
        "background",
        "expires_after[anchor]",
        "expires_after[seconds]",
        "input_fidelity",
        "model",
        "moderation",
        "n",
        "output_compression",
        "output_format",
        "partial_images",
        "purpose",
        "quality",
        "response_format",
        "seconds",
        "size",
        "stream",
    }
)

# Sora video jobs: list/create, item retrieve/delete, and remix. The
# binary /content download is matched on its own (_VIDEO_CONTENT_RE).
_VIDEO_ROUTE_RE = re.compile(r"/v1/videos(?:/[^/]+(?:/remix)?)?")
_VIDEO_ITEM_RE = re.compile(r"/v1/videos/[^/]+")
_VIDEO_CONTENT_RE = re.compile(r"/v1/videos/[^/]+/content")
# The model listing and one model (shared with Anthropic, the Gemini API's
# v1 surface and Cohere: matches_request leaves their marked requests alone).
_MODELS_RE = re.compile(r"/v1/models(?:/[^/]+)?")
# Deleting a stored object carries its id only.
_DELETE_RE = re.compile(r"/v1/files/[^/]+|/v1/conversations/[^/]+(?:/items/[^/]+)?")

# Fine-tuning jobs: create and the list, one job, its cancel/pause/resume
# (each answering with the job, which echoes the caller's `metadata`), its
# events (provider messages) and its checkpoints (metadata only).
_FINE_TUNING_JOBS = re.compile(r"/v1/fine_tuning/jobs(?:/[^/]+)?")
_FINE_TUNING_ACTION = re.compile(r"/v1/fine_tuning/jobs/[^/]+/(?:cancel|pause|resume)")
_FINE_TUNING_EVENTS = re.compile(r"/v1/fine_tuning/jobs/[^/]+/events")
_FINE_TUNING_CHECKPOINTS = re.compile(r"/v1/fine_tuning/jobs/[^/]+/checkpoints")
# A job create's fields the provider uses exactly as sent (``verbatim_fields``):
# the training and validation FILE ids; the `suffix`, which becomes part of
# the fine-tuned model's name — the name every later request cites in its
# `model`, structural and forwarded as sent, so a placeholder in it would name
# a model the client can never address; and the `integrations` (a Weights &
# Biases project, entity, run name and tags the provider logs to).
_FINE_TUNING_VERBATIM = (
    ("suffix",),
    ("training_file",),
    ("validation_file",),
    ("integrations",),
)

# Vector stores (the file_search index): the store (create, list, read,
# modify, delete), its search, its files (attach, list, read, update, detach,
# parsed content) and its file batches (create, read, cancel, list files).
_VECTOR_STORE_POSTS = re.compile(
    r"/v1/vector_stores(?:/[^/]+(?:/search|/files(?:/[^/]+)?|/file_batches(?:/[^/]+/cancel)?)?)?"
)
_VECTOR_STORE_GETS = re.compile(
    r"/v1/vector_stores(?:/[^/]+(?:/files(?:/[^/]+(?:/content)?)?|/file_batches/[^/]+(?:/files)?)?)?"
)
_VECTOR_STORE_DELETES = re.compile(r"/v1/vector_stores/[^/]+(?:/files/[^/]+)?")
# Code interpreter containers: the container (create, list, read, delete)
# and its files (upload — multipart, or JSON naming a stored file — list,
# read, download, delete).
_CONTAINER_POSTS = re.compile(r"/v1/containers(?:/[^/]+/files)?")
_CONTAINER_GETS = re.compile(r"/v1/containers(?:/[^/]+(?:/files(?:/[^/]+(?:/content)?)?)?)?")
_CONTAINER_DELETES = re.compile(r"/v1/containers/[^/]+(?:/files/[^/]+)?")
# Vector store and container verbatim fields (``verbatim_fields``), by the
# POST's tail: the FILE ids a store, an attach, a file batch or a container
# names; a search's attribute filters name attribute KEYS (JSON keys, never
# rewritten where they were set).
_STORED_OBJECT_VERBATIM: tuple[tuple[re.Pattern[str], tuple[tuple[str, ...], ...]], ...] = (
    (re.compile(r"(?:^|/)vector_stores$"), (("file_ids",),)),
    (re.compile(r"(?:^|/)vector_stores/[^/]+/files$"), (("file_id",),)),
    (
        re.compile(r"(?:^|/)vector_stores/[^/]+/file_batches$"),
        (("file_ids",), ("files", "*", "file_id")),
    ),
    (re.compile(r"(?:^|/)vector_stores/[^/]+/search$"), (("filters", "**", "key"),)),
    # The stored files a container starts with, and the stored file a JSON
    # container-file create copies in.
    (re.compile(r"(?:^|/)containers$"), (("file_ids",),)),
    (re.compile(r"(?:^|/)containers/[^/]+/files$"), (("file_id",),)),
)
# A store's or container's `name` is a plain LABEL (the object is addressed
# by its id): user text under a key the walk treats as structural, so it is
# redacted explicitly (``label_fields``: a store's create and modify, a
# container's create) and restored on every object that echoes it — a
# vector store or container object, alone or listed (``_restore_labels``,
# by the object's own `object` type, so a listing item restored on its own
# is covered too).
_LABEL_POSTS = re.compile(r"(?:^|/)(?:vector_stores(?:/[^/]+)?|containers)$")
_LABELLED_OBJECTS = frozenset({"vector_store", "container"})
# A fine-tuning job's reinforcement GRADERS carry a `name` each — a label the
# user writes (a multi-grader nests named graders under `graders`, at any
# depth): redacted on the job create (``label_fields``) and restored in
# every echo of the job — a `fine_tuning.job` object, alone or listed.
_GRADER_NAMES = ("method", "**", "name")
_FINE_TUNING_JOB = "fine_tuning.job"


def _restore_labels(node: Any, rehydrator: Rehydrator) -> Any:
    """``node`` with the `name` of a vector store or container object, and
    every `name` under a fine-tuning job's `method` (its graders'), restored
    (the node itself, or each item of a list envelope)."""
    if not isinstance(node, dict):
        return node
    name = node.get("name")
    if node.get("object") in _LABELLED_OBJECTS and isinstance(name, str):
        node = {**node, "name": rehydrator.rehydrate_text(name)}
    if node.get("object") == _FINE_TUNING_JOB and "method" in node:
        node = {**node, "method": _restore_names(node["method"], rehydrator)}
    data = node.get("data")
    if node.get("object") == "list" and isinstance(data, list):
        node = {**node, "data": [_restore_labels(item, rehydrator) for item in data]}
    return node


def _restore_names(node: Any, rehydrator: Rehydrator) -> Any:
    """``node`` with every string `name` in it, at any depth, restored."""
    if isinstance(node, list):
        return [_restore_names(item, rehydrator) for item in node]
    if not isinstance(node, dict):
        return node
    return {
        key: rehydrator.rehydrate_text(value)
        if key == "name" and isinstance(value, str)
        else _restore_names(value, rehydrator)
        for key, value in node.items()
    }


# Batches whose request or response carries the caller's `metadata`:
# create (POST /v1/batches) and cancel, a batch's GET and the list (every
# answer is a batch object, or a list of them, echoing that metadata).
_BATCH_POST_RE = re.compile(r"/v1/batches|/v1/batches/[^/]+/cancel")
_BATCH_GET_RE = re.compile(r"/v1/batches(?:/[^/]+)?")


def _parse_request_line(line: bytes) -> tuple[dict[str, Any] | None, bool]:
    """An uploaded REQUEST line's JSON object (None for a blank, unparseable
    or non-object line, or one nesting too deep), plus whether an object in
    it repeats a key (then the line must be re-serialized)."""
    stripped = line.strip()
    if not stripped:
        return None, False
    try:
        obj, duplicate_keys = loads_request(stripped)
    except ValueError:
        return None, False
    return (obj, duplicate_keys) if isinstance(obj, dict) else (None, False)


# The Files upload purposes whose JSONL lines the provider RUNS as requests,
# with the top-level keys of a line that ARE a request body — read with the
# request-body structural keys (a message's `role`, a tool call's `id` and
# function `name`, a content part's `type` must reach the provider as sent):
# a batch input line's `body` ({custom_id, method, url, body}), and a
# fine-tuning example's conversation (chat, preference and reinforcement
# formats). Everything else in such a line (the batch envelope, a
# reinforcement example's grader fields) and EVERY line of any other upload
# — another purpose, a container's file, Anthropic's Files, the Gemini
# upload — is the caller's DATA, every string of it redacted with no skip
# set: a data file's `id`, `name`, `type` or `data` is content.
_REQUEST_LINE_KEYS: Mapping[str, frozenset[str]] = {
    "batch": frozenset({"body"}),
    "fine-tune": frozenset(
        {
            "messages",
            "tools",
            "functions",
            "parallel_tool_calls",
            "input",
            "preferred_output",
            "non_preferred_output",
        }
    ),
}
# A Files upload names its purpose in a form field; a container's file
# upload (``/containers/{id}/files``) has none that makes its lines requests.
_CONTAINER_FILES_RE = re.compile(r"/containers/[^/]+/files$")


def _request_line_keys(path: str, parsed: multipart.Multipart) -> frozenset[str]:
    """The top-level keys an upload's JSONL lines hold requests under
    (``_REQUEST_LINE_KEYS``): by its ``purpose`` form field, exactly as
    sent, when every one the upload carries names the same request purpose
    on a Files upload route; else none (every line is data)."""
    if _CONTAINER_FILES_RE.search(path):
        return frozenset()
    purposes = {
        part.content for part in parsed.parts if part.name == "purpose" and part.filename is None
    }
    if len(purposes) != 1:
        return frozenset()
    return _REQUEST_LINE_KEYS.get(purposes.pop().decode("latin-1"), frozenset())


def _redact_line(obj: dict[str, Any], redactor: Redactor, request_keys: frozenset[str]) -> Any:
    """One uploaded JSONL line redacted: the values under ``request_keys``
    as the request bodies they are, every other value as data (no skip
    set). Keys are never touched."""
    return {
        key: redactor.redact_json({key: value})[key]
        if key in request_keys
        else transform_all_strings(value, redactor.redact_text)
        for key, value in obj.items()
    }


def _redact_text_part(
    part: multipart.MultipartPart, redactor: Redactor, *, require_scanned: bool
) -> bool:
    """Redact a form part's content as UTF-8 text, in place; True when it
    changed. Content that is not UTF-8 is left alone — or, when every piece
    must be scanned (``require_scanned``), refused as unscannable."""
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


def _part_kind(part: multipart.MultipartPart, *, media: bool, require_scanned: bool) -> str:
    """How the upload's part loop reads a part: "file" (a file part of an
    upload route, read by its content: ``_read_part``), "text" (UTF-8
    text), "media" (a media file part, never read) or "field" (a structural
    form field, forwarded as-is)."""
    if media and part.name in _PROMPT_FIELDS:
        # Matched by NAME regardless of a filename attribute: a prompt
        # part dressed up as a file upload must not slip past the scan
        # (fail closed) — binary content skips via the decode (refused
        # under identity).
        return "text"
    if part.filename is not None:
        return "media" if media else "file"  # media: never scanned
    # A plain form field is user content (`user`, anything unknown — a part
    # with no name included) unless structural (`purpose`, `model`, `size`
    # ...); the proxy's own identity signs even those only once scanned.
    if require_scanned or part.name not in _STRUCTURAL_FORM_FIELDS:
        return "text"
    return "field"


class _Reading(NamedTuple):
    """How the upload's part loop reads one part: ``_part_kind``'s kind,
    with a "file" part replaced by what its content is — "jsonl", "document"
    (any other text file) or "binary" — and that ``content``."""

    kind: str
    content: FileContent = BINARY  # read only for "document"/"jsonl"


def _read_part(
    part: multipart.MultipartPart,
    *,
    media: bool,
    require_scanned: bool,
    charge: Callable[[int], None],
) -> _Reading:
    """``part`` as the part loop, the floor scan and the refusals all read
    it: one decision per part (``charge`` bounds the JSONL check's
    per-line work)."""
    kind = _part_kind(part, media=media, require_scanned=require_scanned)
    if kind != "file":
        return _Reading(kind)
    content = classify_file(part.content, charge=charge)
    return _Reading("document" if content.kind == "text" else content.kind, content)


def _multipart_floors(parsed: multipart.Multipart, readings: list[_Reading]) -> dict[str, int]:
    """The token floors of an upload, read the way its part loop reads it
    (``readings``): every file name (decoded, ``filename*`` included), every
    JSONL line as the JSON it parses to (escapes resolved) or else as UTF-8
    text, a text file as the text it decodes to, and every other part
    except media files as UTF-8 text — plain form fields too, since the
    upstream reads them, and binary files, which the upstream may read as
    text. Media file parts are never read (the documented non-goal)."""
    floors: dict[str, int] = {}

    def observe(text: str) -> str:
        merge_floors(floors, token_floors(text))
        return text

    for part, reading in zip(parsed.parts, readings, strict=True):
        part.redact_filenames(observe, strict=False)  # returns every name unchanged
        if reading.kind == "document":
            observe(reading.content.text)
        elif reading.kind == "jsonl":
            for line in part.content.split(b"\n"):
                if may_carry_tokens(line):
                    obj, _ = _parse_request_line(line)
                    merge_floors(
                        floors,
                        json_floors(obj)
                        if obj is not None
                        else token_floors(line.decode("utf-8", "replace")),
                    )
        elif reading.kind != "media":
            # A form field ("text"/"field": a structural field is read too,
            # the upstream reads it) or a binary file.
            observe(part.content.decode("utf-8", "replace"))
    return floors


# The charsets a text file part may declare: the ones naming what its
# content decoded as (a UTF-16/32 file by its byte-order mark).
_DECLARABLE_CHARSETS = {
    "utf-8": PLAIN_CHARSETS,
    "utf-16-le": frozenset({"utf-16", "utf-16le"}),
    "utf-16-be": frozenset({"utf-16", "utf-16be"}),
    "utf-32-le": frozenset({"utf-32", "utf-32le"}),
    "utf-32-be": frozenset({"utf-32", "utf-32be"}),
}


def _require_plain_encoding(
    part: multipart.MultipartPart, *, scanned: bool, codec: str = "utf-8"
) -> None:
    """Identity auth: refuse a part the proxy could not read as plain bytes
    (the body-part twin of the request Content-Encoding rule): any
    Content-Transfer-Encoding but 7bit/8bit/binary (RFC 7578 deprecates
    them) — on EVERY part, a forwarded binary file included: the upstream
    would decode bytes the proxy never read or classified — and, on a part
    whose content is scanned, a declared charset other than the one it
    decoded as (``codec``: UTF-8/US-ASCII, or a text file's UTF-16/32),
    whether its own Content-Type parameter or the RFC 7578 §4.6
    ``_charset_`` field. AmbiguousHeaders propagates."""
    encoding = part.header("content-transfer-encoding")
    if encoding is not None and encoding.lower() not in PLAIN_TRANSFER_ENCODINGS:
        raise UnredactableRequest(
            "a multipart part declares a Content-Transfer-Encoding llm-redact does not decode"
        )
    if not scanned:
        return
    charset = (part.params("content-type") or {}).get("charset")
    declared = [] if charset is None else [charset.value]
    if part.name == "_charset_":
        declared.append(part.content.decode("latin-1").strip())
    if any(name.lower() not in _DECLARABLE_CHARSETS[codec] for name in declared):
        raise UnredactableRequest("a multipart part declares a charset llm-redact does not decode")


BINARY_FILE = "an uploaded file is binary (not text llm-redact can redact)"


def rehydrate_text_file(
    raw: bytes, rehydrator: Rehydrator, restore: Callable[[Any], Any]
) -> bytes | None:
    """A downloaded file restored (None: left untouched), read the way an
    uploaded one is (``upload_content.classify_file``), per FILE: a binary
    file stays untouched; a JSONL file (batch OUTPUT files, a JSONL upload
    redacted line by line) is restored line by line as JSON; any other text
    (a results CSV, a text file uploaded redacted as ONE text) is restored
    as one text — a value lands exactly as it was redacted, never
    JSON-escaped because a line happens to parse as JSON once it holds a
    placeholder. Re-encoded as it came. ``restore`` restores one parsed
    JSON value (the adapter's non-streaming transform). Shared by every
    provider's file download (OpenAI/Azure/custom, the Gemini API's,
    Anthropic's)."""
    content = classify_file(raw)
    if content.kind == "binary" or not may_carry_tokens(content.text):
        return None
    if content.kind == "text":
        text = rehydrator.rehydrate_text(content.text)
        return content.encode(text) if text != content.text else None
    lines = content.text.split("\n")
    restored = [_rehydrate_file_line(line, rehydrator, restore) for line in lines]
    return content.encode("\n".join(restored)) if restored != lines else None


def _rehydrate_file_line(line: str, rehydrator: Rehydrator, restore: Callable[[Any], Any]) -> str:
    """One line of a downloaded JSONL file, restored: a line holding a JSON
    value as JSON (a restored value is escaped where it lands; a token the
    provider wrote escaped is found), any other line as text (tokens never
    span lines), a line nesting JSON too deep untouched. The line's own
    surrounding whitespace (a CRLF's CR) is kept."""
    if not may_carry_tokens(line):
        return line
    try:
        value = loads_bounded(line)
    except JsonTooDeep:
        return line
    except ValueError:
        return rehydrator.rehydrate_text(line)
    hydrated = restore(value)
    if hydrated == value:
        return line
    start, end = len(line) - len(line.lstrip()), len(line.rstrip())
    return line[:start] + json_text(hydrated) + line[end:]


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
        data=json_text(
            {
                "object": "chat.completion.chunk",
                "choices": [{"index": index, "delta": delta, "finish_reason": None}],
            }
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
_LISTING_RE = re.compile(
    r"(?:^|/)(?:files|batches|videos|chat/completions|fine_tuning/jobs|vector_stores|containers)$"
)
# A collection BELOW a stored object (a vector store's or a container's
# files): its items are that object's, read in the session the object's own
# read resolves to — not a listing of stored objects a router attributes one
# by one.
_NESTED_LISTING_RE = re.compile(
    r"(?:^|/)(?:vector_stores/[^/]+/(?:file_batches/[^/]+/)?|containers/[^/]+/)files$"
)


# The Uploads API (large files in parts): completing an upload creates the
# stored FILE, named in the answer's nested `file` object.
_UPLOAD_COMPLETE_RE = re.compile(r"(?:^|/)uploads/[^/]+/complete$")

# Fine-tuning jobs: the create answers with the job,
# later read by id; a job read (and its cancel/pause/resume, each answering
# with the job) names the files the job WROTE once it finished
# (`result_files`) — reported like a batch's output files on its status, as
# the reader's; the session router attributes them to the job's recorded
# creator only. Tail-anchored: the Azure and custom-provider prefixes match.
_FINE_TUNING_CREATE_RE = re.compile(r"(?:^|/)fine_tuning/jobs$")
_FINE_TUNING_JOB_RE = re.compile(r"(?:^|/)fine_tuning/jobs/[^/]+(?:/cancel|/pause|/resume)?$")


def _result_files(body: Any) -> tuple[str, ...]:
    """The files a fine-tuning job's ``result_files`` names."""
    files = body.get("result_files") if isinstance(body, dict) else None
    if not isinstance(files, list):
        return ()
    return tuple(dict.fromkeys(f for f in files if isinstance(f, str) and f))


def _completed_upload_file(body: Any) -> tuple[str, ...]:
    """The file a completed Upload created — unless it holds requests the
    provider will run (``purpose`` batch), or its purpose is not stated:
    the Uploads API's parts are forwarded unread (an opaque byte range can
    split a line), so the stored objects those requests cite were never
    checked, and the file is not reported as its uploader's (it stays one
    nobody is recorded creating)."""
    if not isinstance(body, dict):
        return ()
    file = body.get("file")
    if not isinstance(file, dict):
        return ()
    purposes = [purpose for purpose in (body.get("purpose"), file.get("purpose")) if purpose]
    if not purposes or any(
        not isinstance(purpose, str) or purpose.lower() == "batch" for purpose in purposes
    ):
        return ()
    return _string_ids(file, ("id",))


def _tail_is_create(path: str) -> bool:
    """POST to the collection itself (``…/files``, ``…/batches``,
    ``…/conversations``), not to a member or sub-resource."""
    tail = path.rstrip("/").rsplit("/", 1)[-1]
    return tail in ("files", "batches", "conversations", "vector_stores", "containers")


def _match_fine_tuning(method: str, path: str) -> RouteKind:
    """Fine-tuning jobs (``/v1/fine_tuning/...``). A job create carries the
    caller's free-form `metadata` (redacted out, like a batch's) and the
    fields the provider keeps as sent (``_FINE_TUNING_VERBATIM``: scanned,
    never rewritten); the training data itself is the FILE, redacted at its
    upload (``/v1/files``). Every answer naming a job (the create, the list,
    a job, its cancel/pause/resume) echoes that metadata, and a job's events
    are the provider's messages about it: all restored (CHAT). Checkpoints
    carry metadata only. Checkpoint PERMISSIONS (an admin key's sharing of a
    checkpoint across projects) and the alpha graders stay pass-through."""
    if method == "POST" and (path == "/v1/fine_tuning/jobs" or _FINE_TUNING_ACTION.fullmatch(path)):
        return RouteKind.CHAT
    if method == "GET" and (
        _FINE_TUNING_JOBS.fullmatch(path) or _FINE_TUNING_EVENTS.fullmatch(path)
    ):
        return RouteKind.CHAT
    if method == "GET" and _FINE_TUNING_CHECKPOINTS.fullmatch(path):
        return RouteKind.REDACT_ONLY
    return RouteKind.NONE


def _match_vector_stores(method: str, path: str) -> RouteKind:
    """Vector stores (``/v1/vector_stores...``). What the caller writes —
    a store's `description` and `metadata`, a file's `attributes` (values
    under keys the caller chooses: walked like `metadata`), a search's
    `query` and attribute-filter values — is redacted, and every answer that
    echoes it or carries the stored FILES' content (search results, a
    file's parsed content, filenames) is restored, in the request's own
    session: the static one, where the files were uploaded and the
    attributes redacted, so a filter value's placeholder is the stored
    attribute's (the vault is deterministic). A store's `name` is a label,
    redacted (``label_fields``) and restored on every echo; file ids are
    verbatim (``_STORED_OBJECT_VERBATIM``). A delete carries ids only."""
    if method == "POST" and _VECTOR_STORE_POSTS.fullmatch(path):
        return RouteKind.CHAT
    if method == "GET" and _VECTOR_STORE_GETS.fullmatch(path):
        return RouteKind.CHAT
    if method == "DELETE" and _VECTOR_STORE_DELETES.fullmatch(path):
        return RouteKind.REDACT_ONLY
    return RouteKind.NONE


def _match_containers(method: str, path: str) -> RouteKind:
    """Code interpreter containers (``/v1/containers...``). A container file
    upload is multipart (``redact_multipart``: the file part and its
    filename, as a ``/v1/files`` upload) or JSON naming a stored file
    (verbatim); a container file's object echoes its filename in `path`,
    and its content — uploaded, or written by the code — is restored like a
    Files API download (``rehydrate_raw_body``). A container's `name` is a
    label (redacted, restored on every echo); its starting `file_ids` are
    verbatim. Deletes carry ids only."""
    if method == "POST" and _CONTAINER_POSTS.fullmatch(path):
        return RouteKind.CHAT
    if method == "GET" and _CONTAINER_GETS.fullmatch(path):
        return RouteKind.CHAT
    if method == "DELETE" and _CONTAINER_DELETES.fullmatch(path):
        return RouteKind.REDACT_ONLY
    return RouteKind.NONE


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
            # restored). A job's delete carries its id only.
            if method in ("POST", "GET"):
                return RouteKind.CHAT
            if method == "DELETE" and _VIDEO_ITEM_RE.fullmatch(path):
                return RouteKind.REDACT_ONLY
            return RouteKind.NONE
        if method == "GET" and _VIDEO_CONTENT_RE.fullmatch(path):
            # The rendered video: media bytes, nothing to restore.
            return RouteKind.REDACT_ONLY
        if method == "GET" and _FILE_CONTENT_RE.fullmatch(path):
            # Batch output downloads: JSONL rehydrated line by line via
            # rehydrate_raw_body (no request body — redaction no-ops).
            return RouteKind.CHAT
        if method == "GET" and _FILE_OBJECT_RE.fullmatch(path):
            # The file list and a file's metadata echo the filename the
            # upload redacted: restored in the request's own session (a
            # user-scoping router answers listings and foreign reads from
            # an empty one).
            return RouteKind.CHAT
        if method == "GET" and _MODELS_RE.fullmatch(path):
            # The model listing: metadata only. REDACT_ONLY on a body-less
            # request is a no-op that makes the route RECOGNIZED — the id and
            # metadata routes below are too (the Azure stance), so a routed
            # request that spends a key the proxy holds may still reach them.
            return RouteKind.REDACT_ONLY
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
            if method == "DELETE" and _DELETE_RE.fullmatch(path):
                return RouteKind.REDACT_ONLY
            return RouteKind.NONE
        if method == "DELETE" and _DELETE_RE.fullmatch(path):
            # A file's delete: its id only.
            return RouteKind.REDACT_ONLY
        if path.startswith("/v1/fine_tuning/"):
            return _match_fine_tuning(method, path)
        if path.startswith("/v1/vector_stores"):
            return _match_vector_stores(method, path)
        if path.startswith("/v1/containers"):
            return _match_containers(method, path)
        if (method == "POST" and _BATCH_POST_RE.fullmatch(path)) or (
            method == "GET" and _BATCH_GET_RE.fullmatch(path)
        ):
            # Batches: create carries the caller's free-form `metadata`
            # (redacted out), and create/retrieve/cancel/list all answer
            # with batch objects echoing it (restored back, in the request's
            # own session — llm-redact-pro reads a named user's listing in
            # an empty session and restores only that user's own items).
            # Structural fields — input_file_id, endpoint,
            # completion_window, ids, status, counts — carry nothing a
            # detector matches (pinned by test).
            return RouteKind.CHAT
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
        if path.startswith(
            ("/v1/videos", "/v1/batches", "/v1/fine_tuning/", "/v1/vector_stores", "/v1/containers")
        ):
            # Video job, batch, fine-tuning job, vector store and container
            # bodies have no messages field either — a note would graft one
            # and corrupt the request (a container file is a file, never a
            # chat example).
            return False
        return kind is RouteKind.CHAT or path == "/v1/files"

    def tracks_object_ids(self, method: str, path: str, body: Any = None) -> bool:
        tail = path.rstrip("/")
        if method == "POST" and (
            _tail_is_create(path)
            or _VIDEO_CREATE_RE.search(tail) is not None
            or _UPLOAD_COMPLETE_RE.search(tail) is not None
            or _FINE_TUNING_CREATE_RE.search(tail) is not None
            or _stored_completion_create(path, body)
        ):
            return True
        return method in ("GET", "POST") and (
            _BATCH_OBJECT_RE.search(path) is not None
            or _FINE_TUNING_JOB_RE.search(tail) is not None
        )

    def object_ids_from_body(self, method: str, path: str, body: Any) -> tuple[str, ...]:
        tail = path.rstrip("/")
        if _BATCH_OBJECT_RE.search(path) is not None:
            return _string_ids(body, _BATCH_FILE_KEYS)
        if _UPLOAD_COMPLETE_RE.search(tail) is not None:
            return _completed_upload_file(body)
        if tail.endswith("/batches"):
            return _string_ids(body, ("id", *_BATCH_FILE_KEYS))
        if _FINE_TUNING_JOB_RE.search(tail) is not None:
            return _result_files(body)  # never the job again: a read
        if _FINE_TUNING_CREATE_RE.search(tail) is not None:
            return (*_string_ids(body, ("id",)), *_result_files(body))
        return _string_ids(body, ("id",))

    def verbatim_fields(self, method: str, path: str) -> tuple[tuple[str, ...], ...]:
        # Tail-anchored like the tracking patterns: the Azure and
        # custom-provider prefixes need no override.
        tail = path.rstrip("/")
        if method != "POST":
            return ()
        if _FINE_TUNING_CREATE_RE.search(tail) is not None:
            return _FINE_TUNING_VERBATIM
        for pattern, positions in _STORED_OBJECT_VERBATIM:
            if pattern.search(tail) is not None:
                return positions
        return ()

    def label_fields(self, method: str, path: str) -> tuple[tuple[str, ...], ...]:
        if method != "POST":
            return ()
        tail = path.rstrip("/")
        if _LABEL_POSTS.search(tail) is not None:
            return (("name",),)
        if _FINE_TUNING_CREATE_RE.search(tail) is not None:
            return (_GRADER_NAMES,)
        return ()

    def lists_objects(self, method: str, path: str) -> bool:
        tail = path.rstrip("/")
        return (
            method == "GET"
            and _LISTING_RE.search(tail) is not None
            and _NESTED_LISTING_RE.search(tail) is None
        )

    def listing_items(self, body: Any) -> list[Any] | None:
        if not isinstance(body, dict) or body.get("object") != "list":
            return None
        data = body.get("data")
        return data if isinstance(data, list) else None

    def matches_request(
        self,
        method: str,
        path: str,
        headers: "Mapping[str, str] | None" = None,
        query: str = "",
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
        # The model listing is every provider's: a request carrying another
        # provider's marker (anthropic-version, a Google key, the Cohere
        # SDK's header) is that provider's, never the OpenAI upstream's.
        if _MODELS_RE.fullmatch(path) and provider_markers(headers, query) - {"openai"}:
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
        restored = transform_strings(
            body,
            rehydrator.rehydrate_text,
            key_overrides={"arguments": rehydrator.rehydrate_json_source_text},
        )
        # A vector store's or container's `name`, and a fine-tuning job's
        # grader names, are labels the request redacted (``label_fields``),
        # under a key the walk skips.
        return _restore_labels(restored, rehydrator)

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
        forward_binary: Callable[[int], None] | None = None,
    ) -> bytes | None:
        return self.redact_form_upload(
            path,
            body,
            boundary,
            redactor,
            inject_note=inject_note,
            require_scanned=require_scanned,
            forward_binary=forward_binary,
            request_purposes=True,
        )

    def redact_form_upload(
        self,
        path: str,
        body: bytes,
        boundary: bytes,
        redactor: Redactor,
        *,
        inject_note: bool,
        require_scanned: bool,
        forward_binary: Callable[[int], None] | None,
        request_purposes: bool,
    ) -> bytes | None:
        """``redact_multipart``'s form upload, shared with the other
        providers' multipart/form-data Files upload: ``request_purposes``
        says whether a ``purpose`` field can make JSONL lines requests
        (``_REQUEST_LINE_KEYS``: the OpenAI Files API's only)."""
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
        # One reading per part, shared by the floor scan, the part loop and
        # its refusals. A JSONL file's lines are strings to redact: counted
        # against max_body_strings BEFORE anything walks them (the JSONL
        # check, then the floor scan) — millions of empty lines cost per
        # line, not per byte. A text file is one string (redact_text counts
        # it), a binary file none.
        readings = [
            _read_part(part, media=media, require_scanned=require_scanned, charge=redactor.charge)
            for part in parsed.parts
        ]
        if may_carry_tokens(body):
            # Token floors from the WHOLE upload before any part is redacted:
            # a token in a later line bounds the numbers an earlier line's
            # values take (one batch file, one session, one output file).
            redactor = redactor.with_floors(_multipart_floors(parsed, readings))
        request_keys = _request_line_keys(path, parsed) if request_purposes else frozenset()
        changed = False
        try:
            for part, reading in zip(parsed.parts, readings, strict=True):
                changed |= self._redact_part(
                    part,
                    reading,
                    redactor,
                    inject_note=inject_note,
                    require_scanned=require_scanned,
                    forward_binary=forward_binary is not None,
                    request_keys=request_keys,
                )
        except multipart.AmbiguousHeaders as exc:
            # Only reachable with require_scanned (strict header reads): a part
            # header without a single reading is never signed.
            raise UnredactableRequest(str(exc)) from None
        binary = sum(reading.kind == "binary" for reading in readings)
        if binary and forward_binary is not None:
            # Every piece was read or allowed: these go out unscanned.
            forward_binary(binary)
        return parsed.serialize() if changed else None

    def _redact_part(
        self,
        part: multipart.MultipartPart,
        reading: _Reading,
        redactor: Redactor,
        *,
        inject_note: bool,
        require_scanned: bool,
        forward_binary: bool,
        request_keys: frozenset[str] = frozenset(),
    ) -> bool:
        # The upload's file name is user content on every route (the part
        # name is structural, like a JSON key) — a binary file's too. Strict
        # with require_scanned, so the routing reads always see one reading.
        changed = part.redact_filenames(redactor.redact_text, strict=require_scanned)
        kind = reading.kind
        if require_scanned:
            _require_plain_encoding(
                part,
                scanned=kind not in ("media", "binary"),
                codec=reading.content.codec or "utf-8",
            )
        if kind == "jsonl":
            new_content = self._redact_jsonl(
                part.content,
                redactor,
                request_keys,
                inject_note=inject_note,
                require_scanned=require_scanned,
            )
            if new_content != part.content:
                part.content = new_content
                changed = True
        elif kind == "document":
            # A text file: ONE text (a value may span its lines), re-encoded
            # exactly as it came — its own encoding, its own byte-order mark.
            text = reading.content.text
            redacted = redactor.redact_text(text)
            if redacted != text:
                part.content = reading.content.encode(redacted)
                changed = True
        elif kind == "text":
            changed |= _redact_text_part(part, redactor, require_scanned=require_scanned)
        elif kind == "binary" and require_scanned and not forward_binary:
            raise UnredactableRequest(BINARY_FILE)
        return changed

    def _redact_jsonl(
        self,
        data: bytes,
        redactor: Redactor,
        request_keys: frozenset[str],
        *,
        inject_note: bool,
        require_scanned: bool = False,
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
            redacted = _redact_line(obj, redactor, request_keys)
            changed = redacted != obj
            if not changed and not duplicate_keys:
                # Unchanged — and no repeated key whose earlier occurrence
                # the walk never saw — so the original line is safe.
                out.append(line)
                continue
            if inject_note and changed:
                # Only a request the provider runs carries the note; a data
                # line is the caller's file, never rewritten beyond redaction.
                body_obj = redacted.get("body")
                if (
                    "body" in request_keys
                    and isinstance(body_obj, dict)
                    and isinstance(body_obj.get("messages"), list)
                ):
                    # Batch input line: {custom_id, method, url, body}.
                    redacted = {**redacted, "body": self.inject_system_note(body_obj)}
                elif "messages" in request_keys and isinstance(redacted.get("messages"), list):
                    # Fine-tuning line: a bare chat example.
                    redacted = self.inject_system_note(redacted)
            out.append(json_bytes(redacted))
        return b"\n".join(out)

    def rehydrate_raw_body(self, path: str, raw: bytes, rehydrator: Rehydrator) -> bytes | None:
        if not (_FILE_CONTENT_RE.fullmatch(path) or _CONTAINER_FILE_CONTENT_RE.fullmatch(path)):
            return None
        return rehydrate_text_file(
            raw, rehydrator, lambda value: self.rehydrate_body(value, rehydrator)
        )

    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        if not event.data:
            return [event]
        if event.data.strip() == "[DONE]":
            # Flush everything still buffered before the terminal sentinel.
            return [*self._flush_to_events(pool.flush_all()), event]
        try:
            payload = loads_bounded(event.data)
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
            event.data = json_text(payload)
        return [*synthetic, event]

    @staticmethod
    def _flush_to_events(leftovers: dict[Hashable, str]) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        for key, text in leftovers.items():
            if isinstance(key, tuple) and text and len(key) == 2 and key[1] == "legacy":
                # Legacy completions leftover: a text_completion-shaped chunk.
                events.append(
                    SSEEvent(
                        data=json_text(
                            {
                                "object": "text_completion",
                                "choices": [{"index": key[0], "text": text, "finish_reason": None}],
                            }
                        )
                    )
                )
                continue
            mapped = _leftover_to_delta(key, text)
            if mapped is not None:
                events.append(_synthetic_chunk(mapped[0], mapped[1]))
        return events
