"""The Gemini API's Batch Mode beyond the create: a batch's status, the list,
cancel and delete, and async batch embeddings (``:asyncBatchEmbedContent``).

A batch is created with ``models/{m}:batchGenerateContent`` (or, for
embeddings, ``:asyncBatchEmbedContent``) and answers with a long-running
operation named ``batches/<id>`` — the name the SDKs poll with
``GET /v1beta/batches/<id>``, cancel with ``POST …:cancel`` and delete. The
status echoes the batch's display name (redacted at create) and, once it
finished, its INLINED responses: model output carrying the placeholders the
inlined requests were sent with — restored in the request's own session (the
static session batches use). The list (``{"operations": [...]}``) echoes
every batch; a session router attributing listed items restores only the
reader's own (by the operation's ``name``).

All of it is RECOGNIZED, so a credential the proxy holds (a routed operator
key) may reach it: forwarded redacted, restored, the created batch reported
to the session router — never refused as an unrecognized route.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from lent_routes import SESSION, Owners, client, lent_app
from llm_redact.providers.base import RouteKind
from llm_redact.providers.gemini import GeminiAdapter

EMAIL = "jane.doe@corp.example"
OTHER = "sam.roe@corp.example"
TOKEN = "«EMAIL_001»"
GEMINI = "https://gemini.test"
MODEL = "/v1beta/models/gemini-2.5-flash"
KEY = {"x-goog-api-key": "AIza-client"}


def test_batch_routes_are_recognized() -> None:
    adapter = GeminiAdapter()
    assert adapter.matches("GET", "/v1beta/batches") is RouteKind.CHAT
    assert adapter.matches("GET", "/v1beta/batches/b1") is RouteKind.CHAT
    assert adapter.matches("POST", "/v1beta/batches/b1:cancel") is RouteKind.REDACT_ONLY
    assert adapter.matches("DELETE", "/v1beta/batches/b1") is RouteKind.REDACT_ONLY
    assert adapter.matches("POST", f"{MODEL}:asyncBatchEmbedContent") is RouteKind.REDACT_ONLY
    for method, path in (
        ("POST", "/v1beta/batches"),
        ("POST", "/v1beta/batches/b1"),
        ("GET", "/v1beta/batches/b1:cancel"),
        ("DELETE", "/v1beta/batches"),
        ("PATCH", "/v1beta/batches/b1:updateGenerateContentBatch"),
        ("GET", "/v1beta/batches/b1/x"),
        ("GET", f"{MODEL}:asyncBatchEmbedContent"),
    ):
        assert adapter.matches(method, path) is RouteKind.NONE, (method, path)
    # No generate body on any of them: the note is never injected.
    for kind, path in (
        (RouteKind.CHAT, "/v1beta/batches/b1"),
        (RouteKind.CHAT, "/v1beta/batches"),
        (RouteKind.REDACT_ONLY, f"{MODEL}:asyncBatchEmbedContent"),
        (RouteKind.REDACT_ONLY, "/v1beta/batches/b1:cancel"),
    ):
        assert not adapter.wants_system_note(kind, path), path
    assert adapter.wants_system_note(RouteKind.CHAT, f"{MODEL}:generateContent")
    assert adapter.wants_system_note(RouteKind.REDACT_ONLY, f"{MODEL}:countTokens")


def test_async_batch_embeddings_are_tracked_by_their_operation_name() -> None:
    adapter = GeminiAdapter()
    path = f"{MODEL}:asyncBatchEmbedContent"
    assert adapter.tracks_object_ids("POST", path)
    assert adapter.object_ids_from_body("POST", path, {"name": "batches/e1"}) == ("batches/e1",)
    # The list never records ownership (a read), its items are named by name.
    assert not adapter.tracks_object_ids("GET", "/v1beta/batches")
    assert adapter.lists_objects("GET", "/v1beta/batches")
    assert not adapter.lists_objects("GET", "/v1beta/batches/b1")
    assert not adapter.lists_objects("POST", "/v1beta/batches")
    listing = {"operations": [{"name": "batches/b1"}, {"done": True}, "x"]}
    items = adapter.listing_items(listing)
    assert items is listing["operations"]
    assert [adapter.listing_item_id(item) for item in items] == ["batches/b1", None, None]
    assert adapter.listing_items({"operations": "x"}) is None
    assert adapter.listing_items(["x"]) is None


class BatchMode:
    """A Gemini API that stores each batch as sent and, once asked to
    finish it, answers the status with its inlined responses echoing the
    first inlined request's text (the model repeats the placeholder)."""

    def __init__(self) -> None:
        self.batches: dict[str, dict[str, Any]] = {}
        self.received: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.received.append(request)
        path = request.url.path
        if path.endswith((":batchGenerateContent", ":asyncBatchEmbedContent")):
            batch = json.loads(request.content)["batch"]
            name = f"batches/b{len(self.batches) + 1}"
            self.batches[name] = {
                "name": name,
                "metadata": {
                    "@type": "type.googleapis.com/google.ai.generativelanguage.v1main."
                    "GenerateContentBatch",
                    "name": name,
                    "model": path.split("/")[-1].split(":")[0],
                    "displayName": batch.get("display_name", ""),
                    "state": "BATCH_STATE_PENDING",
                },
                "_requests": batch["input_config"]["requests"]["requests"],
            }
            return httpx.Response(200, json=_public(self.batches[name]))
        name = path.removeprefix("/v1beta/").removesuffix(":cancel")
        if request.method == "GET" and path == "/v1beta/batches":
            listed = [_public(batch) for batch in self.batches.values()]
            return httpx.Response(200, json={"operations": listed, "nextPageToken": ""})
        if name not in self.batches:
            return httpx.Response(404, json={"error": {"code": 404, "message": "not found"}})
        if request.method in ("DELETE", "POST"):
            return httpx.Response(200, json={})
        return httpx.Response(200, json=_public(self.batches[name]))

    def finish(self, name: str) -> None:
        batch = self.batches[name]
        text = batch["_requests"][0]["request"]["contents"][0]["parts"][0]["text"]
        reply = {"candidates": [{"content": {"parts": [{"text": f"re: {text}"}], "role": "model"}}]}
        batch["done"] = True
        batch["metadata"]["state"] = "BATCH_STATE_SUCCEEDED"
        batch["response"] = {
            "@type": "type.googleapis.com/google.ai.generativelanguage.v1main."
            "GenerateContentBatchOutput",
            "inlinedResponses": {"inlinedResponses": [{"response": reply, "metadata": {}}]},
        }


def _public(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in batch.items() if not key.startswith("_")}


def _batch_body(email: str, *, embed: bool = False) -> dict[str, Any]:
    request: dict[str, Any] = (
        {"model": "models/embedding-001", "content": {"parts": [{"text": f"mail {email}"}]}}
        if embed
        else {"contents": [{"role": "user", "parts": [{"text": f"mail {email}"}]}]}
    )
    return {
        "batch": {
            "display_name": f"nightly for {email}",
            "input_config": {"requests": {"requests": [{"request": request, "metadata": {}}]}},
        }
    }


@pytest.mark.parametrize("proxy_credential", [True, False], ids=["operator-key", "own-key"])
async def test_a_batch_is_served_end_to_end_under_a_routed_credential(
    proxy_credential: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    gemini = BatchMode()
    owners = Owners(listings=["/v1beta/batches"])
    app, router = lent_app(
        monkeypatch, "gemini", GEMINI, gemini, owners=owners, proxy_credential=proxy_credential
    )
    async with client(app) as http:
        created = await http.post(
            f"{MODEL}:batchGenerateContent", json=_batch_body(EMAIL), headers=KEY
        )
        embedded = await http.post(
            f"{MODEL}:asyncBatchEmbedContent", json=_batch_body(OTHER, embed=True), headers=KEY
        )
        assert created.status_code == 200, created.text
        assert embedded.status_code == 200, embedded.text
        name = created.json()["name"]
        gemini.finish(name)
        status = await http.get(f"/v1beta/{name}", headers=KEY)
        listed = await http.get("/v1beta/batches", headers=KEY)
        cancelled = await http.post(f"/v1beta/{name}:cancel", headers=KEY)
        deleted = await http.delete(f"/v1beta/{name}", headers=KEY)

    # Every route reached Google, redacted: no address in any request, and
    # no system note grafted onto a batch body.
    assert [r.url.path for r in gemini.received] == [
        f"{MODEL}:batchGenerateContent",
        f"{MODEL}:asyncBatchEmbedContent",
        f"/v1beta/{name}",
        "/v1beta/batches",
        f"/v1beta/{name}:cancel",
        f"/v1beta/{name}",
    ]
    assert all(b"@corp.example" not in r.content for r in gemini.received)
    sent = json.loads(gemini.received[0].content)
    assert sent["batch"]["display_name"] == f"nightly for {TOKEN}"
    assert "systemInstruction" not in json.dumps(sent)
    assert all(plan.begun for plan in router.plans)  # routed, never refused

    # The create is redact-only (its echo keeps the placeholder, the
    # fail-safe direction); the finished batch's status restores the
    # display name and the inlined model output.
    assert created.json()["metadata"]["displayName"] == f"nightly for {TOKEN}"
    body = status.json()
    assert body["metadata"]["displayName"] == f"nightly for {EMAIL}"
    (inlined,) = body["response"]["inlinedResponses"]["inlinedResponses"]
    assert inlined["response"]["candidates"][0]["content"]["parts"][0]["text"] == (
        f"re: mail {EMAIL}"
    )
    assert cancelled.status_code == 200 and deleted.status_code == 200

    # Both creates are reported as the creator's; the status read and the
    # list are reads (a finished inline batch names no output file).
    assert owners.objects == [("batches/b1", SESSION), ("batches/b2", SESSION)]
    # The list is read in an EMPTY session and each item restored in its
    # creator's — the reader's own here — by the operation's name.
    operations = listed.json()["operations"]
    assert [op["metadata"]["displayName"] for op in operations] == [
        f"nightly for {EMAIL}",
        f"nightly for {OTHER}",
    ]
    assert listed.json()["nextPageToken"] == ""


async def test_a_listed_batch_nobody_is_recorded_creating_keeps_its_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini = BatchMode()
    owners = Owners(listings=["/v1beta/batches"])
    app, _ = lent_app(monkeypatch, "gemini", GEMINI, gemini, owners=owners)
    async with client(app) as http:
        created = await http.post(
            f"{MODEL}:batchGenerateContent", json=_batch_body(EMAIL), headers=KEY
        )
        assert created.status_code == 200
        owners.objects.clear()  # as if another user had created it
        listed = await http.get("/v1beta/batches", headers=KEY)
    (operation,) = listed.json()["operations"]
    # Read in the (empty) listing session: the placeholder stays in place.
    assert operation["metadata"]["displayName"] == f"nightly for {TOKEN}"
