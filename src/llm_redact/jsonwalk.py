"""Generic recursive transformation of string values in a parsed JSON tree.

Walking every string value (never keys) is what lets redaction and
rehydration cover system prompts, nested content blocks, and tool results
for any provider without hardcoding body shapes.

Two rules keep that walk from corrupting protocol fields without letting
user data slip past it:

* a STRUCTURAL key is skipped only when its value is a scalar. The skip set
  guards enum/identifier strings (``model``, ``role``, ``type``, ``id``,
  base64 ``data``…), as do the base64 media strings at their known
  positions (Bedrock ``source.bytes``); an object or array under one of
  those names is always walked, because user JSON (a tool result, a
  grounding document) can use the same names for its own keys.
* an OPAQUE position — where the value is caller-supplied JSON of arbitrary
  shape (tool-call arguments echoed in history, tool results, documents)
  or a map whose keys the caller chooses (``metadata``, prompt-template
  ``variables``, ``:predict`` instances) — is walked with NO skip set at
  all: a user key named ``id`` or ``name`` there is data, not protocol.

Every walk recurses once or twice per level of nesting, so every document
the proxy reads — a request's or an answer's — is parsed through
``loads_bounded``/``loads_request``, which refuse one nesting deeper than
``MAX_JSON_DEPTH``: no walk over a parsed document can exhaust the stack.
"""

import json
import re
from collections.abc import Callable
from itertools import chain
from typing import Any

# Enum-like or structural fields whose SCALAR values must never be rewritten:
# doing so would corrupt the request (model routing, role dispatch,
# content-block typing) — and a `data` string carries base64 media, which
# detectors must not chew on. Objects/arrays under these names are walked.
STRUCTURAL_KEYS = frozenset(
    {
        "model",
        "role",
        "type",
        "id",
        "name",
        "tool_use_id",
        "tool_call_id",
        "call_id",
        "previous_response_id",
        "media_type",
        "stop_reason",
        "stop_sequence",
        "finish_reason",
        "data",
        "signature",
    }
)

# Opaque user-JSON positions as (parent key, key): the parent key is the key
# of the enclosing object, and a list's items inherit the list's key. Every
# string anywhere below such a value is transformed, structural names
# included. Verified against each provider's request/response schema:
OPAQUE_POSITIONS = frozenset(
    {
        # Gemini / Vertex (HTTP): parts[].functionCall.args (model calls
        # echoed in history) and parts[].functionResponse.response (tool
        # results); the proto JSON mapping also accepts snake_case.
        ("functionCall", "args"),
        ("function_call", "args"),
        ("functionResponse", "response"),
        ("function_response", "response"),
        # Gemini Live / Vertex Live: toolResponse.functionResponses[].response
        # outbound, toolCall.functionCalls[].args inbound.
        ("functionResponses", "response"),
        ("function_responses", "response"),
        ("functionCalls", "args"),
        ("function_calls", "args"),
        # Anthropic Messages (also Bedrock invoke, Claude on Vertex):
        # content[].input of tool_use / server_tool_use / mcp_tool_use.
        ("content", "input"),
        # Bedrock Converse: toolUse.input and toolResult.content[].json.
        ("toolUse", "input"),
        ("content", "json"),
        # Cohere v2 tool results: {"type": "document", "document": {"data"}}.
        ("document", "data"),
        # Cohere v1: tool_results[].outputs, tool_results[].call.parameters,
        # and response tool_calls[].parameters.
        ("tool_results", "outputs"),
        ("call", "parameters"),
        ("tool_calls", "parameters"),
        # Ollama native: tool_calls[].function.arguments is a parsed object
        # (OpenAI's is a JSON-source string, handled by the string rules).
        ("function", "arguments"),
        # Maps whose KEYS the caller chooses: OpenAI Responses / Realtime GA
        # prompt-template variables (`{{name}}`, `{{data}}` are ordinary
        # variable names) …
        ("prompt", "variables"),
        # … and a `:predict` body's top level (parent None): `instances` is
        # model input of arbitrary shape (a custom Vertex endpoint's feature
        # columns named `id`, `name`, `type`), `parameters` its settings.
        # Google's publisher models carry media in `bytesBase64Encoded`,
        # which no skip set ever guarded.
        (None, "instances"),
        (None, "parameters"),
    }
)
# Keys whose value is opaque wherever they appear: Cohere `documents` (v1
# dicts of arbitrary string fields, v2 {"id", "data"} or strings, rerank
# strings) is grounding content, never protocol; `metadata` (OpenAI chat,
# Responses, batches, conversations and Realtime `response.metadata` /
# `tracing.metadata`, on OpenAI and Azure; Anthropic Messages) and Bedrock
# Converse `requestMetadata` are string maps keyed by the caller, and so are
# OpenAI vector-store file `attributes` (and the search results and
# file_search results that echo them).
OPAQUE_ANYWHERE = frozenset({"attributes", "documents", "metadata", "requestMetadata"})
_OPAQUE_KEYS = frozenset(key for _, key in OPAQUE_POSITIONS) | OPAQUE_ANYWHERE

# Enum ARRAYS at their known schema positions, (parent key, key): skipped
# when every element is a scalar (the scalar-only rule would otherwise walk
# them, since the array itself is not a scalar). OpenAI Realtime
# session/response modalities (beta and GA spellings) and Gemini/Vertex
# generationConfig.responseModalities (proto JSON: either spelling on
# either side).
ENUM_LIST_POSITIONS = frozenset(
    {
        ("session", "modalities"),
        ("response", "modalities"),
        ("session", "output_modalities"),
        ("response", "output_modalities"),
        ("generationConfig", "responseModalities"),
        ("generationConfig", "response_modalities"),
        ("generation_config", "responseModalities"),
        ("generation_config", "response_modalities"),
    }
)
_ENUM_LIST_KEYS = frozenset(key for _, key in ENUM_LIST_POSITIONS)

# Base64 media STRINGS at their known positions, (parent key, key), skipped
# like base64 `data`: Bedrock carries image/document/video blocks as
# {"source": {"bytes": <base64>}} (Converse, ApplyGuardrail, CountTokens,
# Nova invoke bodies). Plaintext inside base64 is the media non-goal, so
# scanning it finds nothing real — it only burns event-loop CPU and, when a
# token-shaped run occurs inside the blob, rewrites it (corrupting the media).
MEDIA_POSITIONS = frozenset({("source", "bytes")})
_MEDIA_KEYS = frozenset(key for _, key in MEDIA_POSITIONS)


def _is_opaque(parent: str | None, key: str) -> bool:
    return key in _OPAQUE_KEYS and (key in OPAQUE_ANYWHERE or (parent, key) in OPAQUE_POSITIONS)


def _is_media(parent: str | None, key: str, value: Any) -> bool:
    return key in _MEDIA_KEYS and (parent, key) in MEDIA_POSITIONS and isinstance(value, str)


def _is_enum_list(parent: str | None, key: str, value: Any) -> bool:
    return (
        key in _ENUM_LIST_KEYS
        and (parent, key) in ENUM_LIST_POSITIONS
        and isinstance(value, list)
        and not any(isinstance(item, dict | list) for item in value)
    )


def _walk_opaque(obj: Any, fn: Callable[[str], str]) -> Any:
    """Every string below an opaque position (keys are never touched)."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [_walk_opaque(item, fn) for item in obj]
    if isinstance(obj, dict):
        return {key: _walk_opaque(value, fn) for key, value in obj.items()}
    return obj


def transform_all_strings(obj: Any, fn: Callable[[str], str]) -> Any:
    """Apply ``fn`` to every string value in a JSON tree with NO skip set:
    the reading of a document that is the caller's DATA, not a request (an
    uploaded JSONL data file's lines), where a key named ``id``, ``name``,
    ``type`` or ``data`` is content. Keys are never touched."""
    return _walk_opaque(obj, fn)


def transform_strings(
    obj: Any,
    fn: Callable[[str], str],
    *,
    skip_keys: frozenset[str] = STRUCTURAL_KEYS,
    key_overrides: dict[str, Callable[[str], str]] | None = None,
) -> Any:
    """Apply ``fn`` to every string value in a JSON tree.

    ``skip_keys`` names keys whose SCALAR values are left alone (objects and
    arrays under them are still walked). ``key_overrides`` maps specific
    keys to a different transform for their string values — used for fields
    that carry raw JSON source (tool-call ``arguments``) and need
    escape-aware handling.
    """
    return _walk(obj, fn, skip_keys, key_overrides, None)


def _walk(
    obj: Any,
    fn: Callable[[str], str],
    skip_keys: frozenset[str],
    key_overrides: dict[str, Callable[[str], str]] | None,
    parent: str | None,
) -> Any:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [_walk(item, fn, skip_keys, key_overrides, parent) for item in obj]
    if isinstance(obj, dict):
        # `data` is skipped because it normally carries base64 media — but
        # Anthropic plaintext documents ride in {"type": "text", "data":
        # "<full document>"} sources, which absolutely must be redacted.
        plaintext_source = obj.get("type") == "text"
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key_overrides and key in key_overrides and isinstance(value, str):
                out[key] = key_overrides[key](value)
            elif _is_opaque(parent, key):
                out[key] = _walk_opaque(value, fn)
            elif key == "data" and plaintext_source and isinstance(value, str):
                out[key] = fn(value)
            elif (
                (key in skip_keys and not isinstance(value, dict | list))
                or _is_enum_list(parent, key, value)
                or _is_media(parent, key, value)
            ):
                # Scalars only: an object/array under a structural name (the
                # OpenAI {"object": "list", "data": [...]} envelope, a user
                # key inside a tool result) is content and is walked below.
                out[key] = value
            else:
                out[key] = _walk(value, fn, skip_keys, key_overrides, key)
        return out
    return obj


class _DuplicateKey(Exception):
    """Internal signal: an object in the document repeats a key."""


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise _DuplicateKey
    return obj


# The proxy's one JSON serializer: exactly ``json.dumps(value,
# ensure_ascii=False)`` (the settings json.dumps builds its encoder from),
# non-ASCII kept as is.
_ENCODE = json.JSONEncoder(ensure_ascii=False).encode
# A lone UTF-16 surrogate has no UTF-8 form, yet JSON can carry one: a
# ``\ud800``-style escape, or (``json.loads`` of bytes decodes with
# surrogatepass) surrogate-encoded bytes. Written raw by that serializer,
# it fails the UTF-8 encode every body, SSE event, NDJSON line and
# WebSocket frame goes through.
_SURROGATE = re.compile("[\ud800-\udfff]")


def _escape_surrogate(match: re.Match[str]) -> str:
    return f"\\u{ord(match.group()):04x}"


def json_text(value: Any) -> str:
    """``value`` re-serialized the proxy's one way, and always
    UTF-8-encodable: each lone surrogate (only ever inside a string literal,
    never part of an escape the encoder wrote) is written back as its
    ``\\udXXX`` escape — the same JSON value. Every other character, and so
    every value without a lone surrogate, comes out exactly as
    ``json.dumps(value, ensure_ascii=False)`` has it."""
    text = _ENCODE(value)
    if text.isascii():
        return text
    try:
        text.encode()
    except UnicodeEncodeError:
        return _SURROGATE.sub(_escape_surrogate, text)
    return text


def json_bytes(value: Any) -> bytes:
    """``json_text(value)`` as UTF-8, encoded once."""
    text = _ENCODE(value)
    try:
        return text.encode()
    except UnicodeEncodeError:
        return _SURROGATE.sub(_escape_surrogate, text).encode()


# How deeply a document the proxy reads may nest objects and arrays. The
# parser alone reads thousands of levels (3.12+), while a walk recurses once
# per level (twice for an array on 3.11) under an interpreter recursion limit
# of 1000 frames: a deeper body was a bare 500, a deeper answer a 502 or a
# cut stream. 128 is serde_json's default and past protobuf's 100 (the Gemini
# and Vertex APIs' own limit); a walk this deep stays hundreds of frames
# below the limit, even with every call wrapped (mutation testing's
# trampolines triple the frames a level costs).
MAX_JSON_DEPTH = 128
_CONTAINERS = frozenset({dict, list})


class JsonTooDeep(ValueError):
    """A document nests objects and arrays deeper than MAX_JSON_DEPTH (or
    than the parser itself could read). A ValueError, so every caller that
    treats an unparseable document as such treats this one the same way: a
    client's is refused, an upstream's forwarded as it came."""

    def __init__(self) -> None:
        super().__init__(f"the JSON document nests deeper than {MAX_JSON_DEPTH} levels")


def _too_deep(value: Any) -> bool:
    """Whether a parsed value nests deeper than MAX_JSON_DEPTH: its
    containers, level by level — never recursively."""
    level = [value]
    for _ in range(MAX_JSON_DEPTH):
        level = [
            child
            for child in chain.from_iterable(
                node.values() if type(node) is dict else node
                for node in level
                if type(node) in _CONTAINERS
            )
            if type(child) in _CONTAINERS
        ]
        if not level:
            return False
    return True


def loads_bounded(data: bytes | str) -> Any:
    """``json.loads(data)`` of a document the proxy walks, refused
    (JsonTooDeep) when it nests deeper than MAX_JSON_DEPTH, however deep
    the parser could go."""
    try:
        value = json.loads(data)
    except RecursionError:
        raise JsonTooDeep from None
    if _too_deep(value):
        raise JsonTooDeep
    return value


def loads_request(data: bytes | str) -> tuple[Any, bool]:
    """``loads_bounded(data)`` plus whether ANY object in it repeats a key.

    The parsed value keeps the LAST occurrence (Python's rule); an earlier
    one is never walked. A caller that would forward the ORIGINAL bytes
    when the walk changed nothing must re-serialize instead when this is
    True — an upstream parser may keep the first occurrence, which the
    redactor never saw. Raises ValueError like ``json.loads`` (JsonTooDeep
    for a document nesting too deep). The hook runs per object (exact,
    nested ones included); the second, plain parse happens only for a
    duplicate-bearing document."""
    try:
        value = json.loads(data, object_pairs_hook=_unique_pairs)
    except _DuplicateKey:
        return loads_bounded(data), True
    except RecursionError:
        raise JsonTooDeep from None
    if _too_deep(value):
        raise JsonTooDeep
    return value, False
