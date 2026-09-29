"""OpenAI Batches carry the caller's free-form ``metadata``: batch create is
redacted, and every batch object the provider answers with (create,
retrieve, cancel) echoes it back restored — on OpenAI's ``/v1``, a custom
provider's ``/custom/<name>/`` prefix, and Azure's ``/openai/v1`` (whose
handling this mirrors). Structural fields travel byte-identical, no system
note is ever injected, stored-object tracking keeps working, and the batch
LIST stays pass-through: only a session router's per-item attribution
restores it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.proxy import create_app
from llm_redact.registry import Registry

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"
FILE_ID = "file-XjGxS3KTG0uNmNOK362iJua3"
UPSTREAM = "https://upstream.test"

PREFIXES = [
    pytest.param("/v1", "openai", id="openai"),
    pytest.param("/custom/lm/v1", "custom:lm", id="custom"),
    pytest.param("/openai/v1", "azure", id="azure-v1"),
]


class BatchStore:
    """A provider that stores each batch exactly as it was sent."""

    def __init__(self) -> None:
        self.batches: dict[str, dict[str, Any]] = {}
        self.received: list[tuple[str, str, bytes]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.received.append((request.method, request.url.path, request.content))
        segments = request.url.path.rstrip("/").split("/")
        if request.method == "POST" and segments[-1] == "batches":
            body = json.loads(request.content)
            batch_id = f"batch_{len(self.batches) + 1}"
            self.batches[batch_id] = {
                "id": batch_id,
                "object": "batch",
                "endpoint": body["endpoint"],
                "errors": None,
                "input_file_id": body["input_file_id"],
                "completion_window": body["completion_window"],
                "status": "validating",
                "output_file_id": "file-out-1",
                "error_file_id": None,
                "created_at": 1714508499,
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
                "metadata": body.get("metadata"),
            }
            return httpx.Response(200, json=self.batches[batch_id])
        if request.method == "GET" and segments[-1] == "batches":
            return httpx.Response(
                200,
                json={"object": "list", "data": list(self.batches.values()), "has_more": False},
            )
        batch_id = segments[-2] if segments[-1] == "cancel" else segments[-1]
        batch = self.batches[batch_id]
        if segments[-1] == "cancel":
            batch["status"] = "cancelling"
        return httpx.Response(200, json=batch)


class ListingRouter:
    """A router (not static: static mode reports no creators) that resolves
    to one session, tracks creators and attributes listed batches to the
    session that created them."""

    def __init__(self, session: str) -> None:
        self.mode = "per-user"
        self.session = session
        self.objects: list[tuple[str, str]] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return self.session

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def record_object_id(self, object_id: str, session_id: str) -> bool:
        self.objects.append((object_id, session_id))
        return True

    def listing_item_session(self, object_id: str) -> str | None:
        return next((s for o, s in self.objects if o == object_id), None)


def _create_body() -> dict[str, Any]:
    return {
        "input_file_id": FILE_ID,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {"owner": EMAIL, "purpose": "nightly eval"},
    }


def _app(provider: str, store: BatchStore, **config: Any) -> Any:
    providers = {**Config().providers, provider: ProviderConfig(UPSTREAM)}
    return create_app(
        Config(providers=providers, **config), upstream_transport=httpx.MockTransport(store)
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


@pytest.mark.parametrize(("prefix", "provider"), PREFIXES)
async def test_batch_metadata_is_redacted_out_and_restored_in_every_echo(
    prefix: str, provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    store = BatchStore()
    app = _app(provider, store)
    async with _client(app) as client:
        created = await client.post(f"{prefix}/batches", json=_create_body())
        batch_id = created.json()["id"]
        fetched = await client.get(f"{prefix}/batches/{batch_id}")
        cancelled = await client.post(f"{prefix}/batches/{batch_id}/cancel")
        listed = await client.get(f"{prefix}/batches")

    # The provider saw only the placeholder; every structural field arrived
    # exactly as sent, and no system note was grafted on.
    method, _, sent = store.received[0]
    upstream_body = json.loads(sent)
    assert method == "POST" and EMAIL.encode() not in sent
    assert upstream_body == {
        **_create_body(),
        "metadata": {"owner": TOKEN, "purpose": "nightly eval"},
    }
    assert "messages" not in upstream_body
    assert all(EMAIL.encode() not in content for _, _, content in store.received)

    # Create, retrieve and cancel all answer with the value restored and
    # the batch object otherwise untouched.
    for response in (created, fetched, cancelled):
        assert response.status_code == 200
        body = response.json()
        assert body["metadata"] == {"owner": EMAIL, "purpose": "nightly eval"}
        expected = {**store.batches[batch_id], "metadata": body["metadata"]}
        if response is not cancelled:
            expected["status"] = body["status"]
        assert body == expected
    assert cancelled.json()["status"] == "cancelling"

    # The LIST stays unrestored without a router that attributes its items:
    # it spans every batch the credential can see.
    assert listed.status_code == 200
    assert listed.json()["data"][0]["metadata"]["owner"] == TOKEN


@pytest.mark.parametrize(("prefix", "provider"), PREFIXES)
async def test_the_creator_is_recorded_and_the_list_restores_attributed_items(
    prefix: str, provider: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = ListingRouter("main")
    reg = Registry()
    reg.build_session_router = lambda config, **kw: router
    monkeypatch.setattr(registry_mod, "_registry", reg)
    store = BatchStore()
    app = _app(
        provider,
        store,
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"), session="main"),
    )
    async with _client(app) as client:
        created = await client.post(f"{prefix}/batches", json=_create_body())
        batch_id = created.json()["id"]
        store.batches[batch_id]["error_file_id"] = "file-err-1"
        await client.get(f"{prefix}/batches/{batch_id}")
        listed = await client.get(f"{prefix}/batches")
    # Create reports the batch and its output file; a status read reports
    # the output and error files — after the kind change as before it.
    assert router.objects == [
        (batch_id, "main"),
        ("file-out-1", "main"),
        ("file-out-1", "main"),
        ("file-err-1", "main"),
    ]
    (item,) = listed.json()["data"]
    assert item["metadata"]["owner"] == EMAIL


def test_batch_structural_fields_carry_nothing_a_detector_matches() -> None:
    """Every built-in rule (the default detection config) leaves realistic
    batch structural values alone, so a batch create/echo walked whole stays
    byte-identical outside `metadata`."""
    state = create_app(Config()).state.proxy
    structural = {
        "input_file_id": [FILE_ID, "file-abc123", "file-6F2ksmvXxt4VdoqmHRw6kL"],
        "endpoint": [
            "/v1/responses",
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/completions",
            "/v1/moderations",
        ],
        "completion_window": ["24h"],
        "id": ["batch_abc123", "batch_6720d7b1a2d88190b1b0bd4c3a1fa3b7"],
        "object": ["batch"],
        "status": [
            "validating",
            "failed",
            "in_progress",
            "finalizing",
            "completed",
            "expired",
            "cancelling",
            "cancelled",
        ],
        "output_file_id": ["file-cvaTdG", "file-HfHq2bDGsHhRGvpCcxYDgnnh"],
    }
    for key, values in structural.items():
        for value in values:
            body = {key: value, "created_at": 1714508499, "request_counts": {"total": 100}}
            assert state.redactor.redact_json(body) == body, (key, value)
    assert sum(state.detection_counts.values()) == 0
