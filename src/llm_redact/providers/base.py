from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import count
from typing import Any, NamedTuple, Protocol

from llm_redact.eventstream import EventStreamMessage
from llm_redact.jsonwalk import loads_bounded
from llm_redact.multipart import MultipartPart
from llm_redact.multipart import parse_boundary as parse_multipart_boundary
from llm_redact.redactor import Redactor, UnredactableRequest
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent

# The example token is number 000, which the vault never issues (numbering
# starts at 001), so the note can never name a real value or raise a token
# floor.
SYSTEM_NOTE = (
    "Some values in this conversation have been replaced with privacy tokens "
    "of the form «TYPE_NNN» (for example «EMAIL_000»). Treat each token as an "
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


class VerbatimFieldRedacted(UnredactableRequest):
    """A request field the provider uses EXACTLY as sent — an identifier or
    a name it keeps (a fine-tuned model's name suffix, a file id, a W&B
    project) — holds a value llm-redact redacts. Such a
    field is scanned but never rewritten: a placeholder there would name a
    model, file or project that does not exist (and the client, which gets
    the real value back, would cite a name the provider never saw). So the
    request is refused (400) instead of forwarded with the value. The
    message names the field only, never its content."""


# A verbatim field's position (``ProviderAdapter.verbatim_fields``): keys from
# the body's root; "*" steps into every item of a list, "**" into every
# object and list below (any depth, the current level included).
VerbatimPosition = tuple[str, ...]
_EVERY_ITEM = "*"
_ANY_DEPTH = "**"


class _Held(NamedTuple):
    """A field at one slot of a ``_Slots`` trie: its place in the order
    fields were held (``order``), the ``value`` it carries and the
    ``position`` that named it."""

    order: int
    value: Any
    position: VerbatimPosition


# A trie of slots (the concrete keys and list indexes a position names): each
# step maps to the trie below it, or to the _Held field at that slot (nothing
# below a held field is kept: it goes back with it).
_Slots = dict[str | int, "_Slots | _Held"]


class _Cursor:
    """Where a traversal of the body stands: one step below its parent's.
    Its trie node is made (or found) only when a field is held at or below
    it, and then kept — so holding a field costs O(1) beyond the traversal
    that reached it, whatever its depth (a slot is never built as a tuple
    of its steps and walked again). None inside a field already held."""

    __slots__ = ("_node", "_parent", "_resolved", "_step")

    def __init__(self, parent: "_Cursor | None", step: str | int, node: _Slots | None) -> None:
        self._parent = parent
        self._step = step
        self._node = node
        self._resolved = parent is None  # the root is the trie itself

    def child(self, step: str | int) -> "_Cursor":
        return _Cursor(self, step, None)

    def node(self) -> _Slots | None:
        if not self._resolved:
            self._resolved = True
            parent = self._parent.node() if self._parent is not None else None
            child = None if parent is None else parent.setdefault(self._step, {})
            self._node = child if isinstance(child, dict) else None
        return self._node

    def hold(self, held: _Held) -> None:
        """Hold ``held`` here unless a field at or around this slot is held
        already; a field held inside it goes back with it (dropped)."""
        parent = self._parent.node() if self._parent is not None else None
        if parent is not None and not isinstance(parent.get(self._step), _Held):
            parent[self._step] = held
            self._resolved, self._node = True, None


def _each_slot(
    node: Any,
    position: VerbatimPosition,
    cursor: _Cursor,
    found: Callable[[Any, _Cursor], None],
) -> None:
    """Call ``found(value, cursor)`` for every slot ``position`` names in
    ``node``, in document order: one visit per node the position reaches."""
    if not position:
        found(node, cursor)
        return
    head, rest = position[0], position[1:]
    if head == _ANY_DEPTH:
        _each_slot(node, rest, cursor, found)
        children: Any = (
            node.items()
            if isinstance(node, dict)
            else enumerate(node)
            if isinstance(node, list)
            else ()
        )
        for key, child in children:
            _each_slot(child, position, cursor.child(key), found)
    elif head == _EVERY_ITEM:
        if isinstance(node, list):
            for index, item in enumerate(node):
                _each_slot(item, rest, cursor.child(index), found)
    elif isinstance(node, dict) and head in node:
        _each_slot(node[head], rest, cursor.child(head), found)


def _held_fields(trie: _Slots) -> list[_Held]:
    """Every field held in ``trie``, in the order it was held."""
    fields: list[_Held] = []
    pending = [trie]
    while pending:
        for child in pending.pop().values():
            if isinstance(child, _Held):
                fields.append(child)
            else:
                pending.append(child)
    return sorted(fields)


def _rebuilt(node: Any, trie: _Slots, replace: Callable[[_Held], Any]) -> Any:
    """``node`` with each held slot's value replaced by ``replace(held)`` —
    one pass, copying only the containers along the held paths (the
    caller's body is never changed)."""
    copy: Any = dict(node) if isinstance(node, dict) else list(node)
    for step, below in trie.items():
        copy[step] = (
            replace(below) if isinstance(below, _Held) else _rebuilt(node[step], below, replace)
        )
    return copy


def _strings_in(value: Any) -> list[str]:
    """Every string value in ``value`` (keys are never read)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return []
    return [text for item in value for text in _strings_in(item)]


def prepare_route_request(
    adapter: "ProviderAdapter",
    method: str,
    path: str,
    body: dict[str, Any],
    redactor: Redactor,
    *,
    inject_note: bool,
    mcp_exempt: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """``adapter.prepare_request`` for a request to ``path`` — the proxy's
    entry point, with the route in view. The route's VERBATIM fields
    (``verbatim_fields``) are held out of the walk and scanned on their own:
    a value llm-redact would redact there refuses the request
    (``VerbatimFieldRedacted``; a block-mode value raises BlockedRequest, a
    warn-mode one is counted and forwarded, as anywhere), and otherwise they
    go back exactly as sent. The route's LABEL fields (``label_fields``) are
    redacted after the walk, which skips them as structural. Linear in the
    body: the fields are held in a trie of their slots, built as each
    traversal reaches them (``_Cursor``: no field costs its depth), taken
    out in one copy and put back in another; each field found is counted
    against the redactor's string budget at once (a verbatim field when
    found, a label when redacted), so a body of too many is refused before
    they are all collected."""
    held: _Slots = {}
    order = count()
    for position in adapter.verbatim_fields(method, path):

        def hold(value: Any, cursor: _Cursor, position: VerbatimPosition = position) -> None:
            # Counted as found (max_body_strings): a body of more fields than
            # the budget is refused before they are all collected. Each is
            # counted again per string when scanned below.
            redactor.charge(1)
            cursor.hold(_Held(next(order), value, position))

        _each_slot(body, position, _Cursor(None, "", held), hold)
    fields = _held_fields(held)
    for field in fields:
        for text in _strings_in(field.value):
            if redactor.redact_text(text) != text:
                label = ".".join(
                    key for key in field.position if key not in (_EVERY_ITEM, _ANY_DEPTH)
                )
                raise VerbatimFieldRedacted(
                    f"llm-redact: the request field `{label}` holds a value llm-redact redacts,"
                    " but the provider uses that field exactly as sent (an identifier or a"
                    " name it keeps), so it cannot carry a placeholder; the request was not"
                    " forwarded"
                )
    target = _rebuilt(body, held, lambda field: None) if fields else body
    prepared = adapter.prepare_request(
        target, redactor, inject_note=inject_note, mcp_exempt=mcp_exempt
    )
    if fields:
        prepared = _rebuilt(prepared, held, lambda field: field.value)
    # LABELS: user text under a key the walk skips as structural (a vector
    # store's `name`), redacted here like any other text — in slot order.
    labels: _Slots = {}
    for position in adapter.label_fields(method, path):

        def redact_label(
            value: Any, cursor: _Cursor, position: VerbatimPosition = position
        ) -> None:
            if isinstance(value, str):
                cursor.hold(_Held(0, redactor.redact_text(value), position))

        _each_slot(prepared, position, _Cursor(None, "", labels), redact_label)
    return _rebuilt(prepared, labels, lambda field: field.value) if labels else prepared


class RouteKind(Enum):
    CHAT = "chat"  # redact request + rehydrate response (incl. streaming)
    REDACT_ONLY = "redact_only"  # redact request, pass response through
    NONE = "none"  # not this adapter's route


class UploadReading(Protocol):
    """An upload as its adapter reads it (``ProviderAdapter.read_multipart``):
    what the proxy needs of it before redaction."""

    def binary_parts(self) -> list[tuple[int, MultipartPart]]:
        """Each BINARY file part, with its position among the parts."""
        ...


@dataclass(frozen=True)
class InspectedUpload:
    """What the proxy hands ``redact_multipart`` after inspecting an
    upload's binary file parts: the adapter's own ``reading`` of the SAME
    body (``read_multipart``), and the positions of the binary parts it
    ``cleared`` to go out byte-identical (their extracted text was complete
    and scanned clean)."""

    reading: UploadReading
    cleared: frozenset[int] = frozenset()


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
        self,
        method: str,
        path: str,
        headers: "Mapping[str, str] | None" = None,
        query: str = "",
    ) -> RouteKind:
        """Header-aware routing hook; the default ignores headers and query.

        Used where a path is shared between providers (OpenAI and
        Anthropic both use /v1/files and /v1/models) and only a provider's
        marker — a header, or a Google ``key=`` query parameter — says whose
        request it is (``providers.attribution.provider_markers``).
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

    def verbatim_fields(self, method: str, path: str) -> tuple[VerbatimPosition, ...]:
        """The request fields of this route the provider uses EXACTLY as sent
        (identifiers and names it keeps: a file id, a fine-tuned model's
        suffix, a W&B project), as positions (``VerbatimPosition``).
        They are scanned but never rewritten: ``prepare_route_request``
        refuses a request whose verbatim field holds a value it would redact.
        None by default."""
        return ()

    def label_fields(self, method: str, path: str) -> tuple[VerbatimPosition, ...]:
        """The request fields of this route that are user text although
        their key is one the walk treats as structural (a vector store's
        `name`: a label, the object is addressed by id), as positions. They
        are redacted like any other text (``prepare_route_request``); the
        adapter restores them wherever an answer echoes them. None by
        default."""
        return ()

    def rehydrate_body(self, body: Any, rehydrator: Rehydrator) -> Any:
        return rehydrator.rehydrate_json(body)

    def response_id_from_body(self, body: Any) -> str | None:
        """Provider response id for conversation-chain tracking, if any."""
        return None

    def response_id_from_event(self, event: SSEEvent) -> str | None:
        """Response id observed on a stream, if this event carries one."""
        return None

    def tracks_object_ids(self, method: str, path: str, body: Any = None) -> bool:
        """Whether a 2xx JSON response to this request names objects the
        provider STORES for later reads (an uploaded file, a batch, a stored
        conversation) — only then is the body parsed for
        ``object_ids_from_body``. ``body`` is the parsed REQUEST body (None
        when there is none or it is not JSON), for objects stored only on a
        request flag (OpenAI chat completions with ``store: true``). A
        streamed response is read event by event (``object_ids_from_event``,
        ``reports_object_ids_once``). Only objects the answer to THIS
        request created are reported: the proxy drops any id the request
        body itself carries (an existing object the answer echoes)."""
        return False

    def object_ids_from_body(self, method: str, path: str, body: Any) -> tuple[str, ...]:
        """The stored-object ids a tracked response names (the created
        object's id, plus a batch's output and error file ids, or the files
        a tool run generated)."""
        return ()

    def object_ids_from_event(self, method: str, path: str, event: SSEEvent) -> tuple[str, ...]:
        """The stored-object ids one event of a tracked STREAM names: by
        default its data parsed as the body ``object_ids_from_body`` reads
        (a stored chat completion's chunks carry its id). An adapter whose
        streams name objects on a few events only overrides this with a
        cheap test first: it runs for every event of a tracked stream."""
        try:
            payload = loads_bounded(event.data)
        except ValueError:
            return ()  # [DONE], keep-alives, anything not JSON
        return self.object_ids_from_body(method, path, payload)

    def reports_object_ids_once(self, method: str, path: str) -> bool:
        """Whether a tracked stream names ALL its stored objects on the
        first event naming any (a stored chat completion's id rides every
        chunk): the proxy then stops reading the stream for ids. False — a
        stream naming generated files on the events that carry them — has
        every event read, each id reported once."""
        return True

    def lists_objects(self, method: str, path: str) -> bool:
        """Whether this request reads a COLLECTION of provider-stored objects
        (a listing whose items a session router may attribute to their
        creator's session) — only then is a 2xx JSON response parsed for
        ``listing_items``."""
        return False

    def listing_items(self, body: Any) -> list[Any] | None:
        """The listing's item array (each item a stored object whose string
        ``id`` names it), or None when ``body`` is not this adapter's
        listing shape. The returned list belongs to ``body``: the proxy
        replaces items in it by position."""
        return None

    def listing_item_id(self, item: Any) -> str | None:
        """The stored-object id one listed item names — as this adapter's
        ``object_ids_from_body`` reports the object when it is created — or
        None (the item then stays as the listing's own session delivers
        it). OpenAI-shaped listings name it ``id``; the Gemini API's name
        it ``name`` (``files/<id>``, ``batches/<id>``)."""
        value = item.get("id") if isinstance(item, dict) else None
        return value if isinstance(value, str) else None

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

    def read_multipart(
        self, path: str, body: bytes, boundary: bytes, charge: Callable[[int], None]
    ) -> "UploadReading | None":
        """The upload on ``path`` as ``redact_multipart`` reads it with
        every piece required scanned — parsed once, each part classified
        once (``charge`` bounds the per-line JSONL check, as in
        redaction), every part's headers checked as the redaction would
        check them and, when it has a binary part, every scanned part's
        format (what the redaction refuses before looking at a value) —
        so the proxy can inspect its BINARY file parts
        (``plugin_api.UploadInspector``) before redaction and hand the same
        reading back (``redact_multipart(inspected=...)``). None when the
        route reads no file parts by their content (this base) or the body
        is outside the canonical grammar; raises what ``redact_multipart``
        would raise for the body as a whole (UnredactableRequest,
        TooManyStrings)."""
        return None

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
        inspected: "InspectedUpload | None" = None,
    ) -> bytes | None:
        """Rewrite a multipart request body for ``path`` (delimited by the
        ``boundary`` ``multipart_boundary`` read).

        None means "nothing changed" — the proxy forwards the original
        bytes. Raising BlockedRequest rejects the whole request: one
        leaking line in an uploaded file is a leak. ``require_scanned``
        makes every piece the adapter would forward unscanned an
        ``UnredactableRequest`` naming its kind; the proxy always passes it
        (the scanned-body rule: a recognized route forwards only what the
        proxy read — under its own identity, and under the client's own key
        wherever redaction applies). The one exception is
        ``forward_binary``: when given, a BINARY file part
        (``upload_content.classify_file``) is forwarded unscanned, byte
        for byte, and the callable is told how many were once the whole
        upload was read — the proxy passes it only when the request goes
        out with the client's own credential and ``[detection]
        binary_uploads`` is "forward". ``inspected`` (from ``read_multipart``
        on the same body): the reading to use instead of reading the body
        again, and the binary file parts the proxy CLEARED — their extracted
        text scanned clean and complete — which go out byte-identical, not
        refused and not counted as unscanned. This base scans nothing.
        (What an upload cites for the stored-object check is read
        separately, before redaction: ``upload_view.read_upload``.)

        The proxy cannot see inside the parts, so an adapter that redacts
        them first raises ``redactor``'s token floors (``with_floors``) to
        every token the upload carries as it reads it — or a value could be
        numbered onto a token a later part already holds.
        """
        if require_scanned:
            raise UnredactableRequest("this route's multipart body is not one llm-redact redacts")
        return None

    def redacts_multipart(self, path: str) -> bool:
        """Whether ``path`` is a multipart route whose body ``redact_multipart``
        scans. Consulted by the scanned-body rule: a multipart body on any
        other route is refused rather than forwarded unscanned."""
        return False

    def multipart_boundary(self, path: str, content_type: str) -> bytes | None:
        """The multipart boundary of a request body on ``path`` with this
        ``content_type``, or None when the body is not multipart as this
        route reads it. multipart/form-data everywhere; an adapter whose
        upload route speaks another multipart type (the Gemini API's
        multipart/related upload) accepts it on that route only."""
        return parse_multipart_boundary(content_type)

    def upload_metadata_boundary(self, path: str, content_type: str) -> bytes | None:
        """The boundary of a SINGLE-REQUEST upload on ``path`` whose first
        part is the created file's JSON metadata — what the provider reads
        as the create's body (the Gemini API's multipart/related upload) —
        or None. The stored-object check reads such an upload's metadata
        object (``upload_view.read_upload_metadata``) instead of form data
        (``upload_view.read_upload``)."""
        return None

    def proxy_credential_refusal(
        self,
        method: str,
        path: str,
        headers: "Mapping[str, str]",
        query: str,
    ) -> str | None:
        """Why this RECOGNIZED request must not be sent with a credential
        the PROXY holds (its cloud identity, a routed operator key), or
        None. The proxy answers such a request with a recorded 403 before
        it redacts or sends anything — for a protocol whose answer would
        hand the client a capability minted under the proxy's credential
        (the Gemini API's resumable upload URL, whose data chunks go
        straight to the provider, unread). The message names the protocol,
        never a value."""
        return None

    # Response headers carrying a capability the provider minted for the
    # request's credential (an upload session URL): never relayed to a
    # client when that credential is the proxy's.
    capability_response_headers: frozenset[str] = frozenset()

    def rehydrate_raw_body(self, path: str, raw: bytes, rehydrator: Rehydrator) -> bytes | None:
        """Restore a downloaded FILE's bytes (None = untouched): consulted,
        buffered, for every CHAT answer ``restores_file_download`` names —
        whatever its Content-Type (a file is served with its own: JSON Lines,
        CSV, even ``application/json``)."""
        return None

    def restores_file_download(self, method: str, path: str) -> bool:
        """Whether a CHAT answer to ``method path`` is a downloaded FILE
        (an OpenAI file's or container file's content, Anthropic Files
        content, a Gemini ``:download``): restored by ``rehydrate_raw_body``
        — per file, as it was redacted on upload — never by the whole-body
        JSON walk or a streaming reading its Content-Type would pick."""
        return False
