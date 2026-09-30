"""OpenAI vector stores, recognized so a credential the proxy holds may reach
them (a routed operator key, Azure's identity auth).

What the caller writes — a store's ``description`` and ``metadata``, a
file's ``attributes`` (a map keyed by the caller: a key named ``name`` or
``id`` is data), a search's ``query`` and attribute-filter values — is
redacted; every answer echoing it, or carrying the stored files' content
(search results, a file's parsed content), is restored. A search's filter
VALUE is redacted in the same (static) session the attributes were, so its
placeholder is the stored attribute's and the filter still matches. A
store's ``name`` is a label: redacted, and restored on every echo. File ids
and filter keys are verbatim (scanned, never rewritten).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llm_redact.jsonwalk import transform_strings
from llm_redact.providers.openai import OpenAIAdapter
from stored_objects import EMAIL, TOKEN, Routed

FILE = "file-XjGxS3KTG0uNmNOK362iJua3"
OTHER = "file-Yz0123456789abcdefABCD"
STORE = "vs_1"


class VectorStoreHost:
    """A provider that keeps stores, their files' attributes, and each
    file's content — as it holds a file uploaded through the proxy: with
    the placeholders the upload was redacted to."""

    def __init__(self) -> None:
        self.stores: dict[str, dict[str, Any]] = {}
        self.files: dict[str, dict[str, Any]] = {}
        self.chunks = {FILE: f"Contact {TOKEN} about the renewal.", OTHER: "nothing here"}

    def _file(self, file_id: str) -> dict[str, Any]:
        return self.files[file_id]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        segments = request.url.path.rstrip("/").split("/")[2:]  # after /v1
        body = json.loads(request.content) if request.content else {}
        if segments == ["vector_stores"] and request.method == "POST":
            store = {"id": STORE, "object": "vector_store", **body}
            self.stores[STORE] = store
            return httpx.Response(200, json=store)
        if segments == ["vector_stores"]:
            data = list(self.stores.values())
            return httpx.Response(200, json={"object": "list", "data": data, "has_more": False})
        tail = segments[2:]
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": segments[-1], "deleted": True})
        if tail == ["search"]:
            return self._search(body)
        if tail == ["files"] and request.method == "POST":
            return httpx.Response(200, json=self._attach(body))
        if tail in (["files"], ["file_batches", "vsfb_1", "files"]):
            data = list(self.files.values())
            return httpx.Response(200, json={"object": "list", "data": data, "has_more": False})
        if tail[:1] == ["files"] and tail[2:] == ["content"]:
            item = self._file(tail[1])
            return httpx.Response(
                200,
                json={
                    "file_id": item["id"],
                    "filename": f"notes-{EMAIL}.txt".replace(EMAIL, TOKEN),
                    "attributes": item["attributes"],
                    "content": [{"type": "text", "text": self.chunks[item["id"]]}],
                },
            )
        if tail[:1] == ["files"]:
            item = self._file(tail[1])
            if request.method == "POST":
                item["attributes"] = body["attributes"]
            return httpx.Response(200, json=item)
        if tail == ["file_batches"]:
            for entry in body.get("files", []):
                self._attach(entry)
            for file_id in body.get("file_ids", []):
                self._attach({"file_id": file_id, "attributes": body.get("attributes")})
            return httpx.Response(200, json=self._batch("in_progress"))
        if tail[:1] == ["file_batches"]:
            return httpx.Response(
                200, json=self._batch("cancelled" if tail[-1] == "cancel" else "completed")
            )
        store = self.stores[STORE]
        if request.method == "POST":
            store.update(body)
        return httpx.Response(200, json=store)

    def _attach(self, body: dict[str, Any]) -> dict[str, Any]:
        item = {
            "id": body["file_id"],
            "object": "vector_store.file",
            "vector_store_id": STORE,
            "attributes": body.get("attributes"),
            "status": "completed",
        }
        self.files[item["id"]] = item
        return item

    @staticmethod
    def _batch(status: str) -> dict[str, Any]:
        return {"id": "vsfb_1", "object": "vector_store.file_batch", "status": status}

    def _search(self, body: dict[str, Any]) -> httpx.Response:
        # Exact attribute match, as the provider compares: the filter value
        # must equal the STORED attribute (both placeholders, same session).
        clauses = body.get("filters", {}).get("filters", [])
        hits = [
            item
            for item in self.files.values()
            if all((item["attributes"] or {}).get(c["key"]) == c["value"] for c in clauses)
        ]
        return httpx.Response(
            200,
            json={
                "object": "vector_store.search_results.page",
                "search_query": [body["query"]],
                "data": [
                    {
                        "file_id": item["id"],
                        "filename": f"notes-{TOKEN}.txt",
                        "score": 0.9,
                        "attributes": item["attributes"],
                        "content": [{"type": "text", "text": self.chunks[item["id"]]}],
                    }
                    for item in hits
                ],
                "has_more": False,
            },
        )


ATTRIBUTES = {"owner": EMAIL, "name": EMAIL, "tier": "gold", "pages": 3}


async def test_vector_stores_are_served_under_an_operator_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = VectorStoreHost()
    base = f"/v1/vector_stores/{STORE}"
    paths = [
        "/v1/vector_stores",
        base,
        f"{base}/files",
        f"{base}/files/{FILE}",
        f"{base}/files/{FILE}/content",
        f"{base}/search",
        f"{base}/file_batches",
        f"{base}/file_batches/vsfb_1",
        f"{base}/file_batches/vsfb_1/cancel",
        f"{base}/file_batches/vsfb_1/files",
    ]
    routed = Routed(monkeypatch, host, paths)
    create = {
        "name": f"kb for {EMAIL}",
        "description": f"notes kept for {EMAIL}",
        "file_ids": [FILE],
        "chunking_strategy": {
            "type": "static",
            "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 400},
        },
        "expires_after": {"anchor": "last_active_at", "days": 7},
        "metadata": {"owner": EMAIL},
    }
    created = await routed.send("POST", "/v1/vector_stores", create)
    assert created.status_code == 200, created.text  # forwarded, not the 403
    sent = routed.provider.last_json()
    assert sent["description"] == f"notes kept for {TOKEN}"
    assert sent["metadata"] == {"owner": TOKEN}
    # The name is a label (the store is addressed by id): redacted like any
    # text, although the walk skips `name` as structural elsewhere.
    assert sent["name"] == f"kb for {TOKEN}"
    for key in ("file_ids", "chunking_strategy", "expires_after"):
        assert sent[key] == create[key]
    assert created.json()["description"] == create["description"]
    assert created.json()["name"] == create["name"]
    assert routed.sessions.objects == [(STORE, "default")]

    attached = await routed.send(
        "POST", f"{base}/files", {"file_id": FILE, "attributes": ATTRIBUTES}
    )
    assert attached.status_code == 200
    stored = routed.provider.last_json()["attributes"]
    # A caller attribute keyed `name` is data: redacted like the others.
    assert stored == {"owner": TOKEN, "name": TOKEN, "tier": "gold", "pages": 3}
    assert attached.json()["attributes"] == ATTRIBUTES
    # The vector-store file is the attached FILE (the body cites it): never
    # reported as created by this request.
    assert routed.sessions.objects == [(STORE, "default")]

    query = {
        "query": f"what did {EMAIL} ask",
        "filters": {"type": "and", "filters": [{"type": "eq", "key": "owner", "value": EMAIL}]},
        "max_num_results": 5,
        "ranking_options": {"ranker": "auto", "score_threshold": 0.2},
    }
    found = await routed.send("POST", f"{base}/search", query)
    assert found.status_code == 200
    sent = routed.provider.last_json()
    assert sent["query"] == f"what did {TOKEN} ask"
    assert sent["filters"]["filters"][0] == {"type": "eq", "key": "owner", "value": TOKEN}
    page = found.json()
    assert page["search_query"] == [query["query"]]
    (hit,) = page["data"]  # the filter matched the stored attribute
    assert hit["content"][0]["text"] == f"Contact {EMAIL} about the renewal."
    assert hit["filename"] == f"notes-{EMAIL}.txt" and hit["attributes"] == ATTRIBUTES

    content = await routed.send("GET", f"{base}/files/{FILE}/content")
    assert content.json()["content"][0]["text"] == f"Contact {EMAIL} about the renewal."
    assert content.json()["filename"] == f"notes-{EMAIL}.txt"

    updated = await routed.send(
        "POST", f"{base}/files/{FILE}", {"attributes": {**ATTRIBUTES, "tier": EMAIL}}
    )
    assert routed.provider.last_json()["attributes"]["tier"] == TOKEN
    assert updated.json()["attributes"]["tier"] == EMAIL

    listing = await routed.send("GET", "/v1/vector_stores")
    assert listing.json()["data"][0]["metadata"] == {"owner": EMAIL}
    assert listing.json()["data"][0]["name"] == create["name"]  # restored in the list
    one = await routed.send("GET", base)
    assert one.json()["name"] == create["name"]
    assert routed.sessions.listed == [STORE]  # a listing the router attributes
    files = await routed.send("GET", f"{base}/files")
    assert files.json()["data"][0]["attributes"]["owner"] == EMAIL
    assert routed.sessions.listed == [STORE]  # a store's own files: not attributed

    batch = await routed.send(
        "POST",
        f"{base}/file_batches",
        {"files": [{"file_id": OTHER, "attributes": {"owner": EMAIL}}]},
    )
    assert batch.status_code == 200
    assert routed.provider.last_json()["files"] == [
        {"file_id": OTHER, "attributes": {"owner": TOKEN}}
    ]
    assert (await routed.send("GET", f"{base}/file_batches/vsfb_1")).status_code == 200
    batch_files = await routed.send("GET", f"{base}/file_batches/vsfb_1/files")
    assert {item["attributes"]["owner"] for item in batch_files.json()["data"]} == {EMAIL}
    cancelled = await routed.send("POST", f"{base}/file_batches/vsfb_1/cancel")
    assert cancelled.json()["status"] == "cancelled"

    renamed = await routed.send("POST", base, {"name": EMAIL, "metadata": {"owner": EMAIL}})
    assert routed.provider.last_json() == {"name": TOKEN, "metadata": {"owner": TOKEN}}
    assert renamed.json()["metadata"] == {"owner": EMAIL}
    assert renamed.json()["name"] == EMAIL
    assert (await routed.send("DELETE", f"{base}/files/{FILE}")).status_code == 200
    assert (await routed.send("DELETE", base)).status_code == 200

    everything = b"".join(request.content for request in routed.provider.requests)
    assert EMAIL.encode() not in everything
    assert b"character for character" not in everything  # never a system note
    assert all(identity for *_, identity in routed.sessions.checks)


@pytest.mark.parametrize(
    ("path", "body", "field"),
    [
        ("/v1/vector_stores", {"file_ids": [FILE, EMAIL]}, "file_ids"),
        ("/v1/vector_stores/vs_1/files", {"file_id": EMAIL}, "file_id"),
        ("/v1/vector_stores/vs_1/file_batches", {"files": [{"file_id": EMAIL}]}, "files.file_id"),
        (
            "/v1/vector_stores/vs_1/search",
            {"query": "q", "filters": {"type": "eq", "key": EMAIL, "value": "x"}},
            "filters.key",
        ),
    ],
)
async def test_a_value_in_a_verbatim_field_refuses_the_request(
    path: str, body: dict[str, Any], field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    routed = Routed(monkeypatch, VectorStoreHost(), [path])
    response = await routed.send("POST", path, body)
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert f"`{field}`" in message and EMAIL not in message
    assert routed.provider.requests == [] and routed.sessions.objects == []


def test_attributes_are_a_caller_keyed_map_in_both_directions() -> None:
    """``attributes`` anywhere is walked with no structural skips — the
    request's attach, a search result's echo — like ``metadata``."""
    body = {"data": [{"id": "file-1", "attributes": {"id": "x", "type": "y", "name": "z"}}]}
    walked = transform_strings(body, str.upper)
    assert walked["data"][0]["id"] == "file-1"  # structural outside the map
    assert walked["data"][0]["attributes"] == {"id": "X", "type": "Y", "name": "Z"}


def test_a_vector_store_tracks_its_create_only() -> None:
    adapter = OpenAIAdapter()
    assert adapter.tracks_object_ids("POST", "/v1/vector_stores")
    assert adapter.object_ids_from_body("POST", "/v1/vector_stores", {"id": STORE}) == (STORE,)
    assert not adapter.tracks_object_ids("POST", f"/v1/vector_stores/{STORE}/search")
    assert not adapter.tracks_object_ids("POST", f"/v1/vector_stores/{STORE}/file_batches")
    assert adapter.lists_objects("GET", "/openai/v1/vector_stores")
    assert not adapter.lists_objects("GET", f"/v1/vector_stores/{STORE}/files")
    assert not adapter.lists_objects("GET", f"/v1/vector_stores/{STORE}/file_batches/b/files")


def test_labels_are_redacted_and_restored_only_where_they_are_labels() -> None:
    from llm_redact.rehydrate import Rehydrator
    from llm_redact.vault import InMemoryVault

    adapter = OpenAIAdapter()
    for path in ("/v1/vector_stores", "/v1/vector_stores/vs_1", "/openai/v1/containers"):
        assert adapter.label_fields("POST", path) == (("name",),)
    for method, path in (
        ("GET", "/v1/vector_stores"),
        ("POST", "/v1/vector_stores/vs_1/search"),
        ("POST", "/v1/fine_tuning/jobs"),
        ("POST", "/v1/chat/completions"),
    ):
        assert adapter.label_fields(method, path) == ()
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    rehydrator = Rehydrator(vault)
    listed = {
        "object": "list",
        "data": [
            {"object": "vector_store", "name": token},
            {"object": "container", "name": token},
            {"object": "file", "name": token},  # another object's `name`: structural
            "odd",
        ],
    }
    out = adapter.rehydrate_body(listed, rehydrator)
    assert [item["name"] for item in out["data"][:3]] == [EMAIL, EMAIL, token]
    assert adapter.rehydrate_body({"object": "vector_store", "name": 3}, rehydrator)["name"] == 3
