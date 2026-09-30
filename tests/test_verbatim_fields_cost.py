"""Holding a route's verbatim fields out of the walk costs time linear in the body.

``providers.base.prepare_route_request`` runs on the event loop before any
``max_body_strings`` charge, for any admitted client (its own key is
enough). It used to scan every earlier held field and copy the enclosing
list once per field: a vector store search with 16,000 filters (640 KB) or
a file batch with 16,000 files (288 KB) blocked the proxy for ~30 s. The
fields are now held in a trie of their slots, taken out in one copy and put
back in another — the same fields held, in the same order, as before.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import llm_redact.providers.base as base
from llm_redact.config import Config
from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.providers.base import VerbatimFieldRedacted, prepare_route_request
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor, TooManyStrings
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"
COUNT = 16_000
# The hold-out may cost at most this multiple of the walk it wraps (the
# same body walked with no verbatim field): about 1-2x now; the quadratic
# hold-out cost ~150x (~30 s against ~0.2 s). Relative, so an instrumented
# run (coverage, mutmut's trampolines) slows both sides alike.
RATIO_BOUND = 6.0
# Below this, timer noise dominates: never a failure.
FLOOR_SECONDS = 0.5

SHAPES: dict[str, tuple[str, dict[str, Any]]] = {
    "search-filters": (
        "/v1/vector_stores/vs_1/search",
        {
            "query": f"mail {EMAIL}",
            "filters": {
                "type": "or",
                "filters": [{"key": f"k{i}", "type": "eq", "value": i} for i in range(COUNT)],
            },
        },
    ),
    "file-batch-files": (
        "/v1/vector_stores/vs_1/file_batches",
        {
            "files": [{"file_id": f"file-{i}", "attributes": {"n": str(i)}} for i in range(COUNT)],
            "attributes": {"to": EMAIL},
        },
    ),
}


def _redactor() -> Redactor:
    config = DetectionConfig()
    return Redactor(build_detectors(config), InMemoryVault(), build_allowlist(config))


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_many_verbatim_fields_are_held_in_linear_time(
    shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, body = SHAPES[shape]
    copies = 0
    rebuilt = base._rebuilt

    def counting(node: Any, trie: Any, replace: Any) -> Any:
        nonlocal copies
        copies += 1
        return rebuilt(node, trie, replace)

    started = time.perf_counter()
    prepare_route_request(_Adapter(()), "POST", path, body, _redactor(), inject_note=False)
    walk = time.perf_counter() - started
    monkeypatch.setattr(base, "_rebuilt", counting)
    started = time.perf_counter()
    prepared = prepare_route_request(
        OpenAIAdapter(), "POST", path, body, _redactor(), inject_note=False
    )
    elapsed = time.perf_counter() - started
    assert elapsed < max(RATIO_BOUND * walk, FLOOR_SECONDS), (
        f"{shape}: {elapsed:.2f}s / {walk:.2f}s"
    )
    # The deterministic half: each container on a held path is copied once
    # per pass (out, then back) — never once per field.
    assert copies <= 2 * (COUNT + 3)
    # Held fields went back exactly as sent; everything else was walked.
    assert EMAIL not in str(prepared)
    items = prepared["filters"]["filters"] if "filters" in prepared else prepared["files"]
    assert [item.get("key", item.get("file_id")) for item in items[:2]] in (
        ["k0", "k1"],
        ["file-0", "file-1"],
    )


async def test_a_large_search_goes_through_the_proxy() -> None:
    upstream: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream.append(request)
        return httpx.Response(200, json={"object": "vector_store.search_results.page", "data": []})

    app = create_app(Config(), upstream_transport=httpx.MockTransport(handler))
    path, body = SHAPES["search-filters"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        reply = await client.post(path, json=body, headers={"authorization": "Bearer sk-own"})
    assert reply.status_code == 200
    (sent,) = upstream
    assert EMAIL.encode() not in sent.content and b'"key": "k15999"' in sent.content


# --- the same fields held, in the same order ------------------------------------------


class _Adapter(OpenAIAdapter):
    """Scripted verbatim/label positions over the OpenAI walk."""

    def __init__(self, verbatim: tuple[tuple[str, ...], ...], labels: Any = ()) -> None:
        self._verbatim = verbatim
        self._labels = labels

    def verbatim_fields(self, method: str, path: str) -> tuple[tuple[str, ...], ...]:
        return self._verbatim

    def label_fields(self, method: str, path: str) -> tuple[tuple[str, ...], ...]:
        return self._labels


@pytest.mark.parametrize(
    "verbatim",
    [
        # The outer field first, then one inside it: the inner goes with it.
        (("a",), ("a", "b")),
        # The inner field first: the outer one takes it back with it.
        (("a", "b"), ("a",)),
        # Any depth: a match inside a matched node.
        (("**", "a"),),
    ],
    ids=["outer-first", "inner-first", "nested-any-depth"],
)
def test_a_field_inside_a_held_field_goes_back_with_it(verbatim: Any) -> None:
    body = {"a": {"b": "id-1", "a": {"x": "id-2"}}, "text": f"mail {EMAIL}"}
    prepared = prepare_route_request(
        _Adapter(verbatim), "POST", "/v1/x", body, _redactor(), inject_note=False
    )
    assert prepared == {"a": body["a"], "text": "mail «EMAIL_001»"}
    assert body["text"] == f"mail {EMAIL}"  # the caller's body is never changed
    with pytest.raises(VerbatimFieldRedacted, match=r"`a`"):
        prepare_route_request(
            _Adapter(verbatim),
            "POST",
            "/v1/x",
            {"a": {"b": EMAIL}},
            _redactor(),
            inject_note=False,
        )


def test_the_first_held_field_that_would_be_redacted_is_named() -> None:
    adapter = _Adapter((("second",), ("first",), ("second",)))
    body = {"first": EMAIL, "second": EMAIL}
    with pytest.raises(VerbatimFieldRedacted, match=r"`second`"):
        prepare_route_request(adapter, "POST", "/v1/x", body, _redactor(), inject_note=False)


def test_labels_are_redacted_in_slot_order_and_held_once() -> None:
    adapter = _Adapter((), labels=(("items", "*", "name"), ("name",), ("items", "*", "name")))
    body = {
        "items": [{"name": "a@corp.example"}, {"name": "b@corp.example"}, {"name": 3}],
        "name": "c@corp.example",
    }
    prepared = prepare_route_request(adapter, "POST", "/v1/x", body, _redactor(), inject_note=False)
    assert prepared == {
        "items": [{"name": "«EMAIL_001»"}, {"name": "«EMAIL_002»"}, {"name": 3}],
        "name": "«EMAIL_003»",
    }
    unchanged = {"items": [{"name": "plain"}]}
    assert (
        prepare_route_request(adapter, "POST", "/v1/x", unchanged, _redactor(), inject_note=False)
        == unchanged
    )


# --- no field costs its depth; too many are refused as found ---------------------------


def _deep_search(depth: int, count: int) -> bytes:
    """A search body whose filters nest ``depth`` deep around ``count``
    filter keys (the `("filters", "**", "key")` verbatim position)."""
    text = "[" + ",".join(['{"key":"k"}'] * count) + "]"
    for _ in range(depth):
        text = '{"f":' + text + "}"
    return ('{"query":"q","filters":' + text + "}").encode()


async def _refusal_seconds(path: str, raw: bytes) -> float:
    upstream: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        upstream.append(request)
        return httpx.Response(200, json={"data": []})

    app = create_app(Config(), upstream_transport=httpx.MockTransport(handler))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", timeout=None
    ) as client:
        started = time.perf_counter()
        reply = await client.post(
            path,
            content=raw,
            headers={"authorization": "Bearer sk-own", "content-type": "application/json"},
        )
        elapsed = time.perf_counter() - started
    assert reply.status_code == 413 and "max_body_strings" in reply.text
    assert upstream == []
    return elapsed


async def test_a_deep_body_of_too_many_verbatim_fields_is_refused_as_fast_as_a_chat() -> None:
    # About 10 MiB, 120 deep, 800,000 filter keys: over max_body_strings. The
    # hold-out once collected every (depth-long) slot before the first
    # charge: 22 s against 1.3 s for the same body on chat completions.
    raw = _deep_search(120, 800_000)
    chat = await _refusal_seconds("/v1/chat/completions", raw)
    search = await _refusal_seconds("/v1/vector_stores/vs_1/search", raw)
    assert search < max(3 * chat, FLOOR_SECONDS), f"{search:.2f}s / {chat:.2f}s"


def test_verbatim_fields_are_counted_as_found(monkeypatch: pytest.MonkeyPatch) -> None:
    # The budget refuses before the fields are all collected: the collection
    # stops at the first field over it.
    body = {"filters": {"filters": [{"key": f"k{i}"} for i in range(100)]}}
    found = 0
    cursor_hold = base._Cursor.hold

    def counting(self: Any, held: Any) -> None:
        nonlocal found
        found += 1
        cursor_hold(self, held)

    monkeypatch.setattr(base._Cursor, "hold", counting)
    with pytest.raises(TooManyStrings):
        prepare_route_request(
            OpenAIAdapter(),
            "POST",
            "/v1/vector_stores/vs_1/search",
            body,
            _redactor().with_budget(10),
            inject_note=False,
        )
    assert found == 10


# --- the same fields held as the slot-by-slot reference -------------------------------
#
# The reference below is the earlier implementation (every slot built as a
# tuple, held by walking the trie from the root, its value read by walking
# the body again): the trie-as-you-go hold-out must hold exactly the same
# fields, drop the same inner ones and name the same first refused field.


def _reference_slots(node: Any, position: tuple[str, ...], at: tuple[Any, ...]) -> list[Any]:
    if not position:
        return [at]
    head, rest = position[0], position[1:]
    found: list[Any] = []
    if head == "**":
        found += _reference_slots(node, rest, at)
        children: Any = (
            node.items()
            if isinstance(node, dict)
            else enumerate(node)
            if isinstance(node, list)
            else ()
        )
        for key, child in children:
            found += _reference_slots(child, position, (*at, key))
    elif head == "*":
        if isinstance(node, list):
            for index, item in enumerate(node):
                found += _reference_slots(item, rest, (*at, index))
    elif isinstance(node, dict) and head in node:
        found += _reference_slots(node[head], rest, (*at, head))
    return found


def _reference_hold(trie: dict[Any, Any], slot: tuple[Any, ...], held: Any) -> None:
    node = trie
    for step in slot[:-1]:
        child = node.setdefault(step, {})
        if isinstance(child, base._Held):
            return
        node = child
    if not isinstance(node.get(slot[-1]), base._Held):
        node[slot[-1]] = held


def _reference_prepare(adapter: _Adapter, body: dict[str, Any], redactor: Redactor) -> Any:
    held: dict[Any, Any] = {}
    order = 0
    for position in adapter.verbatim_fields("POST", "/v1/x"):
        for slot in _reference_slots(body, position, ()):
            value = body
            for step in slot:
                value = value[step]
            _reference_hold(held, slot, base._Held(order, value, position))
            order += 1
    fields = base._held_fields(held)
    for field in fields:
        for text in base._strings_in(field.value):
            if redactor.redact_text(text) != text:
                label = ".".join(key for key in field.position if key not in ("*", "**"))
                raise VerbatimFieldRedacted(label)
    target = base._rebuilt(body, held, lambda field: None) if fields else body
    prepared = adapter.prepare_request(target, redactor, inject_note=False)
    if fields:
        prepared = base._rebuilt(prepared, held, lambda field: field.value)
    labels: dict[Any, Any] = {}
    for position in adapter.label_fields("POST", "/v1/x"):
        for slot in _reference_slots(prepared, position, ()):
            value = prepared
            for step in slot:
                value = value[step]
            if isinstance(value, str):
                _reference_hold(labels, slot, base._Held(0, redactor.redact_text(value), position))
    return base._rebuilt(prepared, labels, lambda field: field.value) if labels else prepared


_KEYS = st.sampled_from(["a", "b", "name"])
_LEAVES = st.sampled_from(["x", EMAIL, 1, None])
_BODIES = st.recursive(
    _LEAVES,
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(_KEYS, inner, max_size=3),
    max_leaves=12,
)
_POSITIONS = st.lists(
    st.lists(st.sampled_from(["a", "b", "name", "*", "**"]), min_size=1, max_size=3)
    .map(tuple)
    .filter(lambda p: p[-1] not in ("*", "**") and p.count("**") <= 1),
    max_size=3,
).map(tuple)


def _outcome(run: Any) -> Any:
    try:
        return ("ok", run())
    except VerbatimFieldRedacted as exc:
        return ("refused", str(exc))


@settings(deadline=None, max_examples=300)
@given(body=st.dictionaries(_KEYS, _BODIES, max_size=3), verbatim=_POSITIONS, labels=_POSITIONS)
def test_the_hold_out_holds_what_the_reference_holds(
    body: dict[str, Any], verbatim: Any, labels: Any
) -> None:
    adapter = _Adapter(verbatim, labels)
    new = _outcome(
        lambda: prepare_route_request(
            adapter, "POST", "/v1/x", body, _redactor(), inject_note=False
        )
    )
    old = _outcome(lambda: _reference_prepare(adapter, body, _redactor()))
    if new[0] == "refused":
        # The message names the same field.
        assert old[0] == "refused" and f"`{old[1]}`" in new[1]
    else:
        assert new == old
