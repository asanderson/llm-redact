"""OpenAI fine-tuning jobs, recognized so a credential the proxy holds may
reach them (a routed operator key, Azure's identity auth).

A job create carries the caller's free-form ``metadata`` (redacted out and
restored in every echo of the job) and fields the provider keeps EXACTLY as
sent — the ``suffix`` (part of the fine-tuned model's name, which later
requests cite in their structural ``model``), the training/validation file
ids and the W&B ``integrations``. Those are scanned but never rewritten
(``verbatim_fields``): a value llm-redact would redact there refuses the
request. The created job and, on its reads, its ``result_files`` are
reported to the session router; the list is a listing it attributes.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.providers.base import (
    ProviderAdapter,
    RouteKind,
    VerbatimFieldRedacted,
    prepare_route_request,
)
from llm_redact.providers.custom import CustomOpenAIAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.redactor import BlockedRequest, Redactor
from llm_redact.vault import InMemoryVault
from stored_objects import EMAIL, TOKEN, Routed

JOB = "ftjob-abc123"
FILE_ID = "file-XjGxS3KTG0uNmNOK362iJua3"


class JobStore:
    """A provider that keeps each job as it was created."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        segments = request.url.path.rstrip("/").split("/")
        if request.method == "POST" and segments[-1] == "jobs":
            body = json.loads(request.content)
            job = {
                "id": JOB,
                "object": "fine_tuning.job",
                "model": body["model"],
                "training_file": body["training_file"],
                "validation_file": body.get("validation_file"),
                "suffix": body.get("suffix"),
                "fine_tuned_model": None,
                "status": "validating_files",
                "result_files": [],
                "integrations": body.get("integrations", []),
                "metadata": body.get("metadata"),
            }
            self.jobs[JOB] = job
            return httpx.Response(200, json=job)
        if request.method == "GET" and segments[-1] == "jobs":
            return httpx.Response(
                200, json={"object": "list", "data": list(self.jobs.values()), "has_more": False}
            )
        if segments[-1] == "events":
            note = self.jobs[JOB]["metadata"]["owner"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "fine_tuning.job.event", "id": "ftevent-1", "message": note}
                    ],
                    "has_more": False,
                },
            )
        job = self.jobs[JOB]
        if segments[-1] == "cancel":
            job["status"] = "cancelled"
        else:
            job["status"] = "succeeded"
            job["result_files"] = ["file-result-1"]
            job["fine_tuned_model"] = f"ft:gpt-4o-mini:org:{job['suffix']}:abc"
        return httpx.Response(200, json=job)


CREATE = {
    "model": "gpt-4o-mini-2024-07-18",
    "training_file": FILE_ID,
    "suffix": "support-bot",
    "hyperparameters": {"n_epochs": "auto"},
    "method": {"type": "supervised", "supervised": {"hyperparameters": {"batch_size": 8}}},
    "integrations": [{"type": "wandb", "wandb": {"project": "tuning", "tags": ["v1"]}}],
    "metadata": {"owner": f"requested by {EMAIL}", "id": EMAIL},
    "seed": 42,
}


async def test_fine_tuning_is_served_under_an_operator_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore()
    paths = [
        "/v1/fine_tuning/jobs",
        f"/v1/fine_tuning/jobs/{JOB}",
        f"/v1/fine_tuning/jobs/{JOB}/events",
        f"/v1/fine_tuning/jobs/{JOB}/cancel",
        f"/v1/fine_tuning/jobs/{JOB}/checkpoints",
    ]
    routed = Routed(monkeypatch, store, paths)

    created = await routed.send("POST", "/v1/fine_tuning/jobs", CREATE)
    assert created.status_code == 200, created.text  # forwarded, not the 403
    sent = routed.provider.last_json()
    # The caller's metadata (a caller key named `id` included) left redacted;
    # every verbatim and structural field byte-identical; no system note.
    assert sent["metadata"] == {"owner": f"requested by {TOKEN}", "id": TOKEN}
    assert {k: v for k, v in sent.items() if k != "metadata"} == {
        k: v for k, v in CREATE.items() if k != "metadata"
    }
    assert EMAIL not in routed.provider.requests[-1].content.decode()
    # The echo restored; the created job reported as the requester's.
    assert created.json()["metadata"] == CREATE["metadata"]
    assert routed.sessions.objects == [(JOB, "default")]
    ((adapter_name, _, _, _, identity),) = routed.sessions.checks
    assert adapter_name == "openai" and identity is True

    listing = await routed.send("GET", "/v1/fine_tuning/jobs")
    assert listing.json()["data"][0]["metadata"] == CREATE["metadata"]
    assert routed.sessions.listed == [JOB]  # a listing the router attributes

    read = await routed.send("GET", f"/v1/fine_tuning/jobs/{JOB}")
    assert read.json()["metadata"] == CREATE["metadata"]
    assert read.json()["fine_tuned_model"] == "ft:gpt-4o-mini:org:support-bot:abc"
    assert routed.sessions.objects[-1] == ("file-result-1", "default")

    events = await routed.send("GET", f"/v1/fine_tuning/jobs/{JOB}/events")
    assert events.json()["data"][0]["message"] == f"requested by {EMAIL}"

    cancelled = await routed.send("POST", f"/v1/fine_tuning/jobs/{JOB}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["metadata"] == CREATE["metadata"]

    checkpoints = await routed.send("GET", f"/v1/fine_tuning/jobs/{JOB}/checkpoints")
    assert checkpoints.status_code == 200
    assert b"character for character" not in b"".join(r.content for r in routed.provider.requests)


@pytest.mark.parametrize(
    ("field", "body"),
    [
        ("suffix", {**CREATE, "suffix": EMAIL}),
        ("training_file", {**CREATE, "training_file": EMAIL}),
        ("validation_file", {**CREATE, "validation_file": f"x {EMAIL}"}),
        (
            "integrations",
            {**CREATE, "integrations": [{"type": "wandb", "wandb": {"entity": EMAIL}}]},
        ),
    ],
)
async def test_a_value_in_a_verbatim_field_refuses_the_job(
    field: str, body: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    routed = Routed(monkeypatch, JobStore(), ["/v1/fine_tuning/jobs"])
    response = await routed.send("POST", "/v1/fine_tuning/jobs", body)
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert f"`{field}`" in message and EMAIL not in message
    assert routed.provider.requests == []
    assert routed.sessions.objects == []
    (row,) = routed.app.state.proxy.recent
    assert row["status"] == 400


# --- prepare_route_request, unit --------------------------------------------------------


class _Positions(OpenAIAdapter):
    def __init__(self, *positions: tuple[str, ...]) -> None:
        self.positions = positions

    def verbatim_fields(self, method: str, path: str) -> tuple[tuple[str, ...], ...]:
        return self.positions


def _redactor(**kwargs: Any) -> Redactor:
    config = DetectionConfig()
    return Redactor(build_detectors(config), InMemoryVault(), build_allowlist(config), **kwargs)


def test_verbatim_positions_walk_lists_depths_and_skip_nested_fields() -> None:
    body = {
        "files": [{"file_id": "file-1", "note": EMAIL}, "loose", {"file_id": "file-2"}],
        "filters": {
            "type": "and",
            "filters": [{"type": "eq", "key": "tier", "value": EMAIL}, {"key": "k2"}],
        },
        "scalar": 3,
        "name": "store",
    }
    adapter = _Positions(
        ("files", "*", "file_id"),
        ("filters", "**", "key"),
        ("filters",),  # holds the whole subtree: the keys above are inside it
        ("scalar", "**", "key"),  # a scalar has nothing below
        ("missing", "*"),
        ("name", "*"),  # not a list
        ("name",),
    )
    with pytest.raises(VerbatimFieldRedacted, match="`filters`"):
        prepare_route_request(adapter, "POST", "/p", body, _redactor(), inject_note=False)
    clean = {**body, "filters": {"type": "eq", "key": "tier", "value": "gold"}}
    prepared = prepare_route_request(adapter, "POST", "/p", clean, _redactor(), inject_note=False)
    assert prepared["files"][0] == {"file_id": "file-1", "note": TOKEN}  # walked
    assert prepared["filters"] == clean["filters"] and prepared["name"] == "store"
    assert body["files"][0]["note"] == EMAIL  # the caller's body untouched


def test_a_verbatim_value_is_blocked_or_forwarded_by_its_mode() -> None:
    adapter = _Positions(("suffix",))
    with pytest.raises(BlockedRequest):
        prepare_route_request(
            adapter,
            "POST",
            "/p",
            {"suffix": EMAIL},
            _redactor(modes={"EMAIL": "block"}),
            inject_note=False,
        )
    redactor = _redactor(modes={"EMAIL": "warn"})
    prepared = prepare_route_request(
        adapter, "POST", "/p", {"suffix": EMAIL, "x": EMAIL}, redactor, inject_note=False
    )
    assert prepared == {"suffix": EMAIL, "x": EMAIL}  # warn: observed, forwarded
    assert redactor.warn_counts["EMAIL"] == 2  # each value counted once


def test_the_default_adapter_has_no_verbatim_fields() -> None:
    class Plain(ProviderAdapter):
        name = "plain"

        def matches(self, method: str, path: str) -> RouteKind:
            return RouteKind.CHAT

        def inject_system_note(self, body: dict[str, Any]) -> dict[str, Any]:
            return body

        def rehydrate_event(self, event: Any, pool: Any) -> list[Any]:
            return [event]

    assert Plain().verbatim_fields("POST", "/v1/fine_tuning/jobs") == ()


@pytest.mark.parametrize(
    "path",
    [
        "/v1/fine_tuning/jobs",
        "/openai/v1/fine_tuning/jobs",
        "/openai/fine_tuning/jobs",
        "/custom/lm/api/v1/fine_tuning/jobs",
    ],
)
def test_the_verbatim_fields_hold_on_every_prefix(path: str) -> None:
    from llm_redact.providers.azure_openai import AzureOpenAIAdapter

    adapter = (
        AzureOpenAIAdapter()
        if path.startswith("/openai")
        else CustomOpenAIAdapter("lm")
        if path.startswith("/custom")
        else OpenAIAdapter()
    )
    assert ("suffix",) in adapter.verbatim_fields("POST", path)
    assert adapter.verbatim_fields("GET", path) == ()
    assert adapter.verbatim_fields("POST", path + "/ftjob-1/cancel") == ()
    assert not adapter.wants_system_note(RouteKind.CHAT, path)
    assert adapter.lists_objects("GET", path)
