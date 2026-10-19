"""The explicit ciphertext opt-out follows the protocol walk, never user maps."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
from llm_redact.jsonwalk import transform_all_strings, transform_strings
from llm_redact.ner_prefetch import CollectingRedactor, collect_request_strings
from llm_redact.providers.openai import _redact_line
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.realtime import GeminiLiveWs, OpenAIRealtimeWs
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent, SSEParser, serialize
from llm_redact.vault import InMemoryVault

BAIT = "gAAAAABq0eYc-AKIAIOSFODNN7EXAMPLE-opaque"


def _redactor(vault: InMemoryVault, enabled: bool = False) -> Redactor:
    config = DetectionConfig()
    return Redactor(
        build_detectors(config),
        vault,
        build_allowlist(config),
        encrypted_content_passthrough=enabled,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_protocol_scalar_and_siblings(vault: InMemoryVault, enabled: bool) -> None:
    redactor = _redactor(vault, enabled)
    body = {
        "input": [
            {
                "type": "reasoning",
                "encrypted_content": BAIT,
                "summary": [{"text": BAIT}],
                "nested": {"encrypted_content": BAIT},
            }
        ]
    }
    result = redactor.redact_json(body)["input"][0]
    assert (result["encrypted_content"] == BAIT) is enabled
    assert (result["nested"]["encrypted_content"] == BAIT) is enabled
    assert "AKIAIOSFODNN7EXAMPLE" not in result["summary"][0]["text"]
    assert (
        BAIT
        in collect_request_strings(
            OpenAIResponsesAdapter(),
            "POST",
            "/v1/responses",
            {"input": [{"encrypted_content": BAIT}]},
            limit=100,
            encrypted_content_passthrough=enabled,
        )
    ) is not enabled


@pytest.mark.parametrize("container", ["metadata", "documents", "attributes", "requestMetadata"])
def test_opaque_maps_always_scan(vault: InMemoryVault, container: str) -> None:
    redactor = _redactor(vault, True)
    body = {container: {"nested": {"encrypted_content": BAIT}}}
    assert "AKIAIOSFODNN7EXAMPLE" not in str(redactor.redact_json(body))
    collector = CollectingRedactor(100, encrypted_content_passthrough=True)
    collector.redact_json(body)
    assert BAIT in collector.strings


@pytest.mark.parametrize(
    "body",
    [
        {"content": [{"type": "tool_use", "input": {"encrypted_content": BAIT}}]},
        {"parts": [{"functionCall": {"args": {"encrypted_content": BAIT}}}]},
        {"encrypted_content": [BAIT, {"text": BAIT}]},
        {"encrypted_content": {"text": BAIT}},
    ],
)
def test_tool_inputs_and_container_values_scan(vault: InMemoryVault, body: Any) -> None:
    assert "AKIAIOSFODNN7EXAMPLE" not in str(_redactor(vault, True).redact_json(body))


def test_instance_policy_and_copies(vault: InMemoryVault) -> None:
    enabled = _redactor(vault, True)
    disabled = _redactor(vault)
    for redactor in [
        enabled,
        enabled.with_budget(100),
        enabled.with_floors({"AWS_ACCESS_KEY": 4}),
        enabled.with_precomputed({}, enabled.plan),
    ]:
        assert redactor.redact_json({"encrypted_content": BAIT})["encrypted_content"] == BAIT
    assert disabled.redact_json({"encrypted_content": BAIT})["encrypted_content"] != BAIT
    collector = CollectingRedactor(100, encrypted_content_passthrough=True).with_budget(100)
    collector.redact_json({"encrypted_content": BAIT, "text": "ordinary"})
    assert collector.strings == ["ordinary"]
    assert (
        transform_all_strings({"encrypted_content": BAIT}, str.upper)["encrypted_content"]
        == BAIT.upper()
    )
    assert (
        transform_strings(
            {"encrypted_content": BAIT}, str.upper, encrypted_content_passthrough=True
        )["encrypted_content"]
        == BAIT
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_responses_json_and_sse_split_sweep(vault: InMemoryVault, enabled: bool) -> None:
    token = _redactor(vault).redact_text("jane@corp.example")
    adapter = OpenAIResponsesAdapter()
    item = {
        "type": "reasoning",
        "encrypted_content": token,
        "summary": [{"text": token}],
        "metadata": {"encrypted_content": token},
    }
    expected = {
        "type": "reasoning",
        "encrypted_content": token if enabled else "jane@corp.example",
        "summary": [{"text": "jane@corp.example"}],
        "metadata": {"encrypted_content": "jane@corp.example"},
    }
    assert (
        adapter.rehydrate_body(item, Rehydrator(vault, encrypted_content_passthrough=enabled))
        == expected
    )
    raw = serialize(
        SSEEvent(
            event="response.output_item.done",
            data=json.dumps({"type": "response.output_item.done", "item": item}),
        )
    )
    for cut in range(len(raw) + 1):
        parser = SSEParser()
        pool = RehydratorPool(vault, encrypted_content_passthrough=enabled)
        events = parser.feed(raw[:cut]) + parser.feed(raw[cut:]) + parser.close()
        restored = [out for event in events for out in adapter.rehydrate_event(event, pool)]
        assert json.loads(restored[0].data)["item"] == expected, cut


@pytest.mark.parametrize("adapter", [OpenAIRealtimeWs(), GeminiLiveWs()])
@pytest.mark.parametrize("enabled", [False, True])
def test_realtime_policy_matches_collector_and_restore(
    vault: InMemoryVault, adapter: Any, enabled: bool
) -> None:
    payload = {
        "type": "response.create",
        "response": {
            "encrypted_content": BAIT,
            "instructions": BAIT,
            "metadata": {"encrypted_content": BAIT},
        },
    }
    ctx = SimpleNamespace(redactor=_redactor(vault, enabled))
    redacted = json.loads(adapter.redact_message(json.dumps(payload), ctx))
    assert (redacted["response"]["encrypted_content"] == BAIT) is enabled
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted["response"]["instructions"]
    collector = CollectingRedactor(100, encrypted_content_passthrough=enabled)
    adapter.redact_message(json.dumps(payload), SimpleNamespace(redactor=collector))
    assert collector.strings.count(BAIT) == (2 if enabled else 3)
    token = ctx.redactor.redact_text("jane@corp.example")
    embedded = {"encrypted_content": token, "text": token, "metadata": {"encrypted_content": token}}
    if isinstance(adapter, OpenAIRealtimeWs):
        frame = {"type": "response.output_item.done", "item": embedded}
        result_key = "item"
    else:
        frame = {"serverContent": embedded}
        result_key = "serverContent"
    pool = RehydratorPool(vault, encrypted_content_passthrough=enabled)
    result = json.loads(adapter.rehydrate_message(json.dumps(frame), pool)[0])[result_key]
    assert result["encrypted_content"] == (token if enabled else "jane@corp.example")
    assert result["text"] == "jane@corp.example"
    assert result["metadata"]["encrypted_content"] == "jane@corp.example"


def test_uploaded_data_lines_keep_scanning(vault: InMemoryVault) -> None:
    redactor = _redactor(vault, True)
    result = _redact_line(
        {"body": {"encrypted_content": BAIT}, "encrypted_content": BAIT},
        redactor.redact_text,
        frozenset({"body"}),
        encrypted_content_passthrough=True,
    )
    assert result["body"]["encrypted_content"] == BAIT
    assert "AKIAIOSFODNN7EXAMPLE" not in result["encrypted_content"]
