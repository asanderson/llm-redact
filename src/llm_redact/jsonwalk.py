"""Generic recursive transformation of string values in a parsed JSON tree.

Walking every string value (never keys) is what lets redaction and
rehydration cover system prompts, nested content blocks, and tool results
for any provider without hardcoding body shapes.
"""

from collections.abc import Callable
from typing import Any

# Enum-like or structural fields that must never be rewritten: doing so would
# corrupt the request (model routing, role dispatch, content-block typing) —
# and `data` carries base64 media, which detectors must not chew on.
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


def transform_strings(
    obj: Any,
    fn: Callable[[str], str],
    *,
    skip_keys: frozenset[str] = STRUCTURAL_KEYS,
    key_overrides: dict[str, Callable[[str], str]] | None = None,
) -> Any:
    """Apply ``fn`` to every string value in a JSON tree.

    ``key_overrides`` maps specific keys to a different transform for their
    string values — used for fields that carry raw JSON source (tool-call
    ``arguments``) and need escape-aware handling.
    """
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [
            transform_strings(item, fn, skip_keys=skip_keys, key_overrides=key_overrides)
            for item in obj
        ]
    if isinstance(obj, dict):
        # `data` is skipped because it normally carries base64 media — but
        # Anthropic plaintext documents ride in {"type": "text", "data":
        # "<full document>"} sources, which absolutely must be redacted.
        plaintext_source = obj.get("type") == "text"
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key_overrides and key in key_overrides and isinstance(value, str):
                out[key] = key_overrides[key](value)
            elif key == "data" and plaintext_source and isinstance(value, str):
                out[key] = fn(value)
            elif key == "data" and obj.get("object") == "list":
                # The OpenAI list envelope {"object": "list", "data": [...]}
                # carries content items (Conversations items, etc.), NOT base64
                # media, so its elements must be walked and redacted/rehydrated.
                # (Image responses have no "object": "list", so their b64_json
                # data stays skipped.)
                out[key] = transform_strings(
                    value, fn, skip_keys=skip_keys, key_overrides=key_overrides
                )
            elif key in skip_keys:
                out[key] = value
            else:
                out[key] = transform_strings(
                    value, fn, skip_keys=skip_keys, key_overrides=key_overrides
                )
        return out
    return obj
