from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum
from typing import Any

from llm_redact.eventstream import EventStreamMessage
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent

SYSTEM_NOTE = (
    "Some values in this conversation have been replaced with privacy tokens "
    "of the form «TYPE_NNN» (for example «EMAIL_001»). Treat each token as an "
    "opaque identifier for the real value and reproduce every token exactly, "
    "character for character, whenever you refer to it."
)


_MCP_TOOL_SENTINEL: dict[str, Any] = {"type": "mcp"}


def strip_mcp_tools(node: Any) -> Any:
    """Replace tools[].type == "mcp" entries with a bare sentinel.

    MCP connector blocks (server_url, headers, authorization) are
    provider-directed CONFIGURATION: the provider must receive the real
    credential to call the MCP server on the model's behalf, so redacting
    them breaks the feature — and they are addressed to the trusted
    provider, not conversation content. Stripping BEFORE redaction (rather
    than restoring after) keeps detection counts and note-injection
    decisions honest: nothing in the block is counted as redacted.
    """
    if isinstance(node, dict):
        return {
            key: (
                [
                    _MCP_TOOL_SENTINEL
                    if isinstance(tool, dict) and tool.get("type") == "mcp"
                    else strip_mcp_tools(tool)
                    for tool in value
                ]
                if key == "tools" and isinstance(value, list)
                else strip_mcp_tools(value)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [strip_mcp_tools(item) for item in node]
    return node


def restore_mcp_tools(original: Any, redacted: Any) -> Any:
    """Put the original MCP tool entries back after redaction (the inverse
    of strip_mcp_tools; positions are unchanged because stripping keeps a
    sentinel in each slot)."""
    if isinstance(original, dict) and isinstance(redacted, dict):
        out: dict[str, Any] = {}
        for key, red_val in redacted.items():
            orig_val = original.get(key)
            if (
                key == "tools"
                and isinstance(orig_val, list)
                and isinstance(red_val, list)
                and len(orig_val) == len(red_val)
            ):
                out[key] = [
                    orig
                    if isinstance(orig, dict) and orig.get("type") == "mcp"
                    else restore_mcp_tools(orig, red)
                    for orig, red in zip(orig_val, red_val, strict=True)
                ]
            else:
                out[key] = restore_mcp_tools(orig_val, red_val)
        return out
    if isinstance(original, list) and isinstance(redacted, list) and len(original) == len(redacted):
        return [restore_mcp_tools(o, r) for o, r in zip(original, redacted, strict=True)]
    return redacted


_EXEMPT_STASH_SENTINEL: dict[str, Any] = {"type": "mcp_exempt_stash"}


def _is_exempt_mcp_block(node: dict[str, Any], exempt: frozenset[str], ids: frozenset[str]) -> bool:
    """An MCP content block addressed to an exempt server.

    Anthropic blocks (mcp_tool_use/mcp_tool_result) and OpenAI Responses
    items (mcp_call/mcp_list_tools/mcp_approval_request) all carry a type
    starting with "mcp"; the server identifier is server_name (Anthropic)
    or server_label (OpenAI). Anthropic mcp_tool_result blocks name no
    server — they are exempt only when their tool_use_id provably points
    at an exempt mcp_tool_use in the same body (fail-closed: an
    uncorrelatable result block is redacted normally).
    """
    block_type = node.get("type")
    if not isinstance(block_type, str) or not block_type.startswith("mcp"):
        return False
    server = node.get("server_name") or node.get("server_label")
    if isinstance(server, str) and server in exempt:
        return True
    tool_use_id = node.get("tool_use_id")
    return isinstance(tool_use_id, str) and tool_use_id in ids


def _exempt_mcp_use_ids(node: Any, exempt: frozenset[str]) -> set[str]:
    ids: set[str] = set()
    if isinstance(node, dict):
        block_type = node.get("type")
        server = node.get("server_name") or node.get("server_label")
        if (
            isinstance(block_type, str)
            and block_type.startswith("mcp")
            and isinstance(server, str)
            and server in exempt
            and isinstance(node.get("id"), str)
        ):
            ids.add(node["id"])
        for value in node.values():
            ids |= _exempt_mcp_use_ids(value, exempt)
    elif isinstance(node, list):
        for item in node:
            ids |= _exempt_mcp_use_ids(item, exempt)
    return ids


def stash_exempt_mcp_blocks(node: Any, exempt: frozenset[str]) -> Any:
    """Replace exempt-server MCP content blocks with an inert sentinel.

    Per-server opt-out of detection ([detection.mcp] exempt_servers): the
    strip_mcp_tools mechanism generalized to CONTENT blocks. Stashing
    before redaction (not restoring values after) keeps detection counts
    and note-injection decisions honest — nothing in an exempt block is
    counted. restore_exempt_mcp_blocks puts the originals back by
    position, which is sound because redact_json preserves structure.
    """
    ids = frozenset(_exempt_mcp_use_ids(node, exempt))

    def stash(item: Any) -> Any:
        if isinstance(item, dict):
            if _is_exempt_mcp_block(item, exempt, ids):
                return dict(_EXEMPT_STASH_SENTINEL)
            return {key: stash(value) for key, value in item.items()}
        if isinstance(item, list):
            return [stash(entry) for entry in item]
        return item

    return stash(node)


def restore_exempt_mcp_blocks(original: Any, redacted: Any, exempt: frozenset[str]) -> Any:
    """Put the original exempt MCP blocks back after redaction.

    Positions are decided on the ORIGINAL (the same predicate the stash
    used), never by matching sentinel contents — a request body that
    happens to contain sentinel-shaped data cannot confuse it.
    """
    ids = frozenset(_exempt_mcp_use_ids(original, exempt))

    def restore(orig: Any, red: Any) -> Any:
        if isinstance(orig, dict) and _is_exempt_mcp_block(orig, exempt, ids):
            return orig
        if isinstance(orig, dict) and isinstance(red, dict):
            return {key: restore(orig.get(key), value) for key, value in red.items()}
        if isinstance(orig, list) and isinstance(red, list) and len(orig) == len(red):
            return [restore(o, r) for o, r in zip(orig, red, strict=True)]
        return red

    return restore(original, redacted)


class RouteKind(Enum):
    CHAT = "chat"  # redact request + rehydrate response (incl. streaming)
    REDACT_ONLY = "redact_only"  # redact request, pass response through
    NONE = "none"  # not this adapter's route


class ProviderAdapter(ABC):
    name: str
    # Adapters whose streaming responses are AWS binary event streams
    # (application/vnd.amazon.eventstream) rather than SSE set this; the
    # proxy then routes such responses through rehydrate_eventstream_message.
    handles_eventstream: bool = False
    # Likewise for NDJSON streams (application/x-ndjson, Ollama): routed
    # through rehydrate_ndjson_line.
    handles_ndjson: bool = False

    @abstractmethod
    def matches(self, method: str, path: str) -> RouteKind: ...

    def matches_request(
        self, method: str, path: str, headers: "Mapping[str, str] | None" = None
    ) -> RouteKind:
        """Header-aware routing hook; the default ignores headers.

        Used where a path is shared between providers (OpenAI and
        Anthropic both use /v1/files) and only a header disambiguates.
        """
        return self.matches(method, path)

    @abstractmethod
    def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
        """Add the token-preservation note to the request body."""

    def wants_system_note(self, kind: RouteKind, path: str) -> bool:
        """Whether this route's body can carry the token-preservation note.

        Chat bodies can; embeddings bodies have no system field and would be
        corrupted by one. Token-counting routes vary by provider (the note
        is part of what the chat request will carry, so counting it is
        correct where the schema allows) — adapters override as needed.
        """
        return kind is RouteKind.CHAT

    def error_body(self, message: str, *, status: int = 413) -> dict[str, Any]:
        """Provider-shaped error payload for proxy-generated errors.

        ``status`` is the HTTP status the caller will send — adapters use it
        to pick the provider's matching error type/code so SDKs classify the
        failure correctly (413 oversized, 400 blocked, 502 unconfigured).
        """
        return {"error": {"message": message, "type": "invalid_request_error"}}

    def prepare_request(
        self,
        body: dict[str, Any],
        redactor: Redactor,
        *,
        inject_note: bool,
        mcp_exempt: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        before = len(redactor.counts)
        total_before = sum(redactor.counts.values())
        # Exempt MCP blocks are stashed around redaction only (restored
        # BEFORE note injection, which may restructure lists and break the
        # position-based restore).
        target = stash_exempt_mcp_blocks(body, mcp_exempt) if mcp_exempt else body
        redacted = redactor.redact_json(target)
        if mcp_exempt:
            redacted = restore_exempt_mcp_blocks(body, redacted, mcp_exempt)
        changed = sum(redactor.counts.values()) != total_before or len(redactor.counts) != before
        if inject_note and changed:
            redacted = self.inject_system_note(redacted)
        return redacted  # type: ignore[no-any-return]

    def rehydrate_body(self, body: Any, rehydrator: Rehydrator) -> Any:
        return rehydrator.rehydrate_json(body)

    def response_id_from_body(self, body: Any) -> str | None:
        """Provider response id for conversation-chain tracking, if any."""
        return None

    def response_id_from_event(self, event: SSEEvent) -> str | None:
        """Response id observed on a stream, if this event carries one."""
        return None

    @abstractmethod
    def rehydrate_event(self, event: SSEEvent, pool: RehydratorPool) -> list[SSEEvent]:
        """Rewrite one SSE event; may inject synthetic flush events."""

    def rehydrate_eventstream_message(
        self, message: EventStreamMessage, pool: RehydratorPool
    ) -> list[EventStreamMessage]:
        """Rewrite one binary event-stream message; may inject synthetic
        flush frames. Only consulted when ``handles_eventstream`` is set."""
        return [message]

    def rehydrate_ndjson_line(self, line: bytes, pool: RehydratorPool) -> bytes:
        """Rewrite one NDJSON line (no trailing newline). Only consulted
        when ``handles_ndjson`` is set."""
        return line

    def redact_multipart(
        self, path: str, body: bytes, boundary: bytes, redactor: Redactor, *, inject_note: bool
    ) -> bytes | None:
        """Rewrite a multipart/form-data request body for ``path``.

        None means "leave it alone" — the proxy forwards the original
        bytes verbatim (matching the non-JSON-body default). Raising
        BlockedRequest rejects the whole request: one leaking line in an
        uploaded file is a leak.
        """
        return None

    def rehydrate_raw_body(self, path: str, raw: bytes, rehydrator: Rehydrator) -> bytes | None:
        """Rehydrate a buffered non-JSON response body (None = untouched).

        Consulted on CHAT routes whose response is not application/json —
        e.g. an OpenAI batch output file download, which is JSONL served
        as a file.
        """
        return None
