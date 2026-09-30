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

import llm_redact.providers.base as base
from llm_redact.config import Config
from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.providers.base import VerbatimFieldRedacted, prepare_route_request
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
from llm_redact.redactor import Redactor
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
