"""OpenAI Responses API adapter tests.

The event shapes here are hand-authored from the public API documentation
(not recorded transcripts); they pin the shape this adapter assumes.
"""

import json
from typing import Any

from llm_redact.providers.base import SYSTEM_NOTE, RouteKind
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent
from llm_redact.vault import InMemoryVault


def _event(payload: dict[str, Any]) -> SSEEvent:
    return SSEEvent(event=payload["type"], data=json.dumps(payload, ensure_ascii=False))


def _text_delta(item_id: str, delta: str, content_index: int = 0) -> SSEEvent:
    return _event(
        {
            "type": "response.output_text.delta",
            "item_id": item_id,
            "output_index": 0,
            "content_index": content_index,
            "delta": delta,
        }
    )


def _args_delta(item_id: str, delta: str) -> SSEEvent:
    return _event(
        {
            "type": "response.function_call_arguments.delta",
            "item_id": item_id,
            "output_index": 1,
            "delta": delta,
        }
    )


def _run(events: list[SSEEvent], vault: InMemoryVault) -> list[SSEEvent]:
    adapter = OpenAIResponsesAdapter()
    pool = RehydratorPool(vault)
    out: list[SSEEvent] = []
    for event in events:
        out.extend(adapter.rehydrate_event(event, pool))
    return out


def _deltas_of(events: list[SSEEvent], event_type: str) -> str:
    parts = []
    for e in events:
        if not e.data or e.data == "[DONE]":
            continue
        payload = json.loads(e.data)
        if payload.get("type") == event_type:
            parts.append(payload["delta"])
    return "".join(parts)


def test_routing() -> None:
    adapter = OpenAIResponsesAdapter()
    assert adapter.matches("POST", "/v1/responses") is RouteKind.CHAT
    assert adapter.matches("GET", "/v1/responses/resp_123") is RouteKind.CHAT
    assert adapter.matches("POST", "/v1/responses/resp_123/cancel") is RouteKind.NONE
    assert adapter.matches("POST", "/v1/chat/completions") is RouteKind.NONE


def test_system_note_into_instructions() -> None:
    adapter = OpenAIResponsesAdapter()
    assert adapter.inject_system_note({})["instructions"] == SYSTEM_NOTE
    merged = adapter.inject_system_note({"instructions": "be brief"})
    assert merged["instructions"].startswith("be brief")
    assert SYSTEM_NOTE in merged["instructions"]


def test_output_text_split_across_deltas(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    events = [
        _text_delta("item_1", "mail «EMA"),
        _text_delta("item_1", "IL_001» ok"),
        _event(
            {
                "type": "response.output_text.done",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "text": "mail «EMAIL_001» ok",
            }
        ),
    ]
    out = _run(events, vault)
    assert _deltas_of(out, "response.output_text.delta") == "mail jane@corp.example ok"
    done = json.loads(out[-1].data)
    # The repeated full text on the done event is rehydrated wholesale.
    assert done["text"] == "mail jane@corp.example ok"


def test_leftover_flushed_before_done(vault: InMemoryVault) -> None:
    events = [
        _text_delta("item_1", "dangling «EMAIL_"),
        _event(
            {
                "type": "response.output_text.done",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "text": "dangling «EMAIL_",
            }
        ),
    ]
    out = _run(events, vault)
    types = [json.loads(e.data)["type"] for e in out]
    assert types == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
    ]
    assert _deltas_of(out, "response.output_text.delta") == "dangling «EMAIL_"


def test_function_call_arguments_json_source(vault: InMemoryVault) -> None:
    vault.placeholder_for("SECRET", 'pa"ss\nwor\\d')
    events = [
        _args_delta("item_2", '{"secret": "\\u00abSEC'),
        _args_delta("item_2", 'RET_001\\u00bb"}'),
        _event(
            {
                "type": "response.function_call_arguments.done",
                "item_id": "item_2",
                "output_index": 1,
                "arguments": '{"secret": "\\u00abSECRET_001\\u00bb"}',
            }
        ),
    ]
    out = _run(events, vault)
    reassembled = _deltas_of(out, "response.function_call_arguments.delta")
    assert json.loads(reassembled) == {"secret": 'pa"ss\nwor\\d'}
    done = json.loads(out[-1].data)
    assert json.loads(done["arguments"]) == {"secret": 'pa"ss\nwor\\d'}


def test_output_item_done_flushes_and_rehydrates(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    events = [
        _text_delta("item_1", "held «EMAIL_0"),
        _event(
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "item_1",
                    "type": "message",
                    "content": [{"type": "output_text", "text": "held «EMAIL_001»"}],
                },
            }
        ),
    ]
    out = _run(events, vault)
    # Leftover flushed as a synthetic delta before the done event.
    assert _deltas_of(out, "response.output_text.delta") == "held «EMAIL_0"
    done = json.loads(out[-1].data)
    assert done["item"]["content"][0]["text"] == "held jane@corp.example"


def test_completed_flushes_all_and_rehydrates_response(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    events = [
        _text_delta("item_1", "tail «EMAIL_0"),
        _event(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "output": [
                        {
                            "id": "item_1",
                            "type": "message",
                            "content": [{"type": "output_text", "text": "tail «EMAIL_001» done"}],
                        },
                        {
                            "id": "item_2",
                            "type": "function_call",
                            "call_id": "call_9",
                            "arguments": '{"to": "«EMAIL_001»"}',
                        },
                    ],
                },
            }
        ),
    ]
    out = _run(events, vault)
    assert _deltas_of(out, "response.output_text.delta") == "tail «EMAIL_0"
    completed = json.loads(out[-1].data)
    output = completed["response"]["output"]
    assert output[0]["content"][0]["text"] == "tail jane@corp.example done"
    assert json.loads(output[1]["arguments"]) == {"to": "jane@corp.example"}
    assert output[1]["call_id"] == "call_9"  # structural key untouched


def test_passthrough_events(vault: InMemoryVault) -> None:
    created = _event({"type": "response.created", "response": {"id": "resp_1"}})
    added = _event(
        {"type": "response.output_item.added", "output_index": 0, "item": {"id": "item_1"}}
    )
    out = _run([created, added], vault)
    assert out == [created, added]


def test_non_streaming_body_rehydration(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    adapter = OpenAIResponsesAdapter()
    body = {
        "id": "resp_1",
        "output": [
            {
                "id": "item_1",
                "type": "message",
                "content": [{"type": "output_text", "text": "mail «EMAIL_001»"}],
            },
            {
                "id": "item_2",
                "type": "function_call",
                "arguments": '{"to": "«EMAIL_001»"}',
            },
        ],
        "previous_response_id": "resp_0",
    }
    out = adapter.rehydrate_body(body, Rehydrator(vault))
    assert out["output"][0]["content"][0]["text"] == "mail jane@corp.example"
    assert json.loads(out["output"][1]["arguments"]) == {"to": "jane@corp.example"}
    assert out["previous_response_id"] == "resp_0"
