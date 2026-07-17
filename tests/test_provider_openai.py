import json
from typing import Any

from llm_redact.providers.base import SYSTEM_NOTE, RouteKind
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.rehydrate import RehydratorPool
from llm_redact.sse import SSEEvent
from llm_redact.vault import InMemoryVault


def _chunk(*choices: dict[str, Any]) -> SSEEvent:
    return SSEEvent(data=json.dumps({"object": "chat.completion.chunk", "choices": list(choices)}))


def _content(index: int, text: str, finish: str | None = None) -> dict[str, Any]:
    return {"index": index, "delta": {"content": text}, "finish_reason": finish}


def test_routing() -> None:
    adapter = OpenAIAdapter()
    assert adapter.matches("POST", "/v1/chat/completions") is RouteKind.CHAT
    # Redact-only since 0.6.0: previously passed through unredacted.
    assert adapter.matches("POST", "/v1/embeddings") is RouteKind.REDACT_ONLY


def test_system_note_prepended() -> None:
    adapter = OpenAIAdapter()
    body = adapter.inject_system_note({"messages": [{"role": "user", "content": "hi"}]})
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_NOTE}
    assert len(body["messages"]) == 2


def _run(events: list[SSEEvent], vault: InMemoryVault) -> list[SSEEvent]:
    adapter = OpenAIAdapter()
    pool = RehydratorPool(vault)
    out: list[SSEEvent] = []
    for event in events:
        out.extend(adapter.rehydrate_event(event, pool))
    return out


def _content_of(events: list[SSEEvent], index: int = 0) -> str:
    parts = []
    for e in events:
        if not e.data or e.data == "[DONE]":
            continue
        for choice in json.loads(e.data).get("choices", []):
            if choice.get("index", 0) == index:
                content = choice.get("delta", {}).get("content")
                if isinstance(content, str):
                    parts.append(content)
    return "".join(parts)


def test_content_split_across_chunks(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    events = [
        _chunk(_content(0, "mail «EMA")),
        _chunk(_content(0, "IL_001» done")),
        _chunk(_content(0, "", finish="stop")),
        SSEEvent(data="[DONE]"),
    ]
    out = _run(events, vault)
    assert _content_of(out) == "mail jane@corp.example done"


def test_leftover_flushed_before_finish(vault: InMemoryVault) -> None:
    events = [
        _chunk(_content(0, "dangling «EMAIL_")),
        _chunk(_content(0, "", finish="stop")),
    ]
    out = _run(events, vault)
    assert _content_of(out) == "dangling «EMAIL_"
    # Synthetic flush chunk must precede the finish_reason chunk.
    finish_positions = [
        i
        for i, e in enumerate(out)
        if e.data
        and e.data != "[DONE]"
        and any(c.get("finish_reason") for c in json.loads(e.data)["choices"])
    ]
    leftover_positions = [
        i for i, e in enumerate(out) if e.data and "dangling" not in e.data and "«EMAIL_" in e.data
    ]
    assert leftover_positions and finish_positions
    assert max(leftover_positions) < min(finish_positions)


def test_leftover_flushed_on_done_sentinel(vault: InMemoryVault) -> None:
    events = [
        _chunk(_content(0, "tail «EMAIL_")),
        SSEEvent(data="[DONE]"),
    ]
    out = _run(events, vault)
    assert _content_of(out) == "tail «EMAIL_"


def _field_of(events: list[SSEEvent], field: str, index: int = 0) -> str:
    parts = []
    for e in events:
        if not e.data or e.data == "[DONE]":
            continue
        for choice in json.loads(e.data).get("choices", []):
            if choice.get("index", 0) == index:
                value = choice.get("delta", {}).get(field)
                if isinstance(value, str):
                    parts.append(value)
    return "".join(parts)


def test_reasoning_content_split_at_every_offset(vault: InMemoryVault) -> None:
    # DeepSeek/Groq/xAI stream chain-of-thought in delta.reasoning_content;
    # tokens split across reasoning deltas must reassemble like content.
    vault.placeholder_for("EMAIL", "jane@corp.example")
    full = "thinking about «EMAIL_001» here"
    for split in range(1, len(full)):
        events = [
            _chunk({"index": 0, "delta": {"reasoning_content": full[:split]}}),
            _chunk({"index": 0, "delta": {"reasoning_content": full[split:]}}),
            _chunk({"index": 0, "delta": {}, "finish_reason": "stop"}),
            SSEEvent(data="[DONE]"),
        ]
        out = _run(events, vault)
        assert _field_of(out, "reasoning_content") == "thinking about jane@corp.example here", (
            f"split at {split}"
        )


def test_openrouter_reasoning_field_and_leftover_flush(vault: InMemoryVault) -> None:
    # OpenRouter's unified `reasoning` field is rehydrated on its own
    # channel, and a dangling token flushes before the finish chunk.
    events = [
        _chunk({"index": 0, "delta": {"reasoning": "hmm «EMAIL_"}}),
        _chunk({"index": 0, "delta": {}, "finish_reason": "stop"}),
    ]
    out = _run(events, vault)
    assert _field_of(out, "reasoning") == "hmm «EMAIL_"


def test_reasoning_and_content_are_independent_channels(vault: InMemoryVault) -> None:
    # A token split across reasoning deltas and a separate token split
    # across content deltas must not cross-contaminate.
    vault.placeholder_for("EMAIL", "jane@corp.example")
    vault.placeholder_for("IPV4", "203.0.113.7")
    events = [
        _chunk({"index": 0, "delta": {"reasoning_content": "r «EMA", "content": "c «IPV"}}),
        _chunk({"index": 0, "delta": {"reasoning_content": "IL_001»", "content": "4_001»"}}),
        _chunk({"index": 0, "delta": {}, "finish_reason": "stop"}),
        SSEEvent(data="[DONE]"),
    ]
    out = _run(events, vault)
    assert _field_of(out, "reasoning_content") == "r jane@corp.example"
    assert _content_of(out) == "c 203.0.113.7"
    assert out[-1].data == "[DONE]"


def test_tool_call_arguments_rehydrated(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    events = [
        _chunk(
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"to": "«EM'}}]},
                "finish_reason": None,
            }
        ),
        _chunk(
            {
                "index": 0,
                "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'AIL_001»"}'}}]},
                "finish_reason": None,
            }
        ),
        _chunk({"index": 0, "delta": {}, "finish_reason": "tool_calls"}),
        SSEEvent(data="[DONE]"),
    ]
    out = _run(events, vault)
    argument_parts = []
    for e in out:
        if not e.data or e.data == "[DONE]":
            continue
        for choice in json.loads(e.data).get("choices", []):
            for tc in choice.get("delta", {}).get("tool_calls") or []:
                argument_parts.append(tc["function"]["arguments"])
    assert json.loads("".join(argument_parts)) == {"to": "jane@corp.example"}


def test_multiple_choices_independent(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    events = [
        _chunk(_content(0, "a «EMA"), _content(1, "b «EMA")),
        _chunk(_content(0, "IL_001»"), _content(1, "IL_001»")),
        _chunk(_content(0, "", finish="stop"), _content(1, "", finish="stop")),
    ]
    out = _run(events, vault)
    assert _content_of(out, 0) == "a jane@corp.example"
    assert _content_of(out, 1) == "b jane@corp.example"


def test_non_streaming_arguments_reescaped(vault: InMemoryVault) -> None:
    from llm_redact.rehydrate import Rehydrator

    vault.placeholder_for("SECRET", 'pa"ss\nwor\\d')  # nasty original
    adapter = OpenAIAdapter()
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "done «SECRET_001»",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "save",
                                "arguments": '{"secret": "«SECRET_001»"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    out = adapter.rehydrate_body(body, Rehydrator(vault))
    message = out["choices"][0]["message"]
    # Plain content gets the raw original; arguments stay valid JSON source.
    assert message["content"] == 'done pa"ss\nwor\\d'
    arguments = message["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"secret": 'pa"ss\nwor\\d'}
    assert message["tool_calls"][0]["function"]["name"] == "save"
