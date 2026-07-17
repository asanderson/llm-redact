import json
from typing import Any

from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.base import SYSTEM_NOTE, RouteKind
from llm_redact.rehydrate import RehydratorPool
from llm_redact.sse import SSEEvent
from llm_redact.vault import InMemoryVault


def _event(payload: dict[str, Any], name: str | None = None) -> SSEEvent:
    return SSEEvent(event=name or payload.get("type"), data=json.dumps(payload))


def _delta(index: int, **delta: Any) -> SSEEvent:
    return _event({"type": "content_block_delta", "index": index, "delta": delta})


def test_routing() -> None:
    adapter = AnthropicAdapter()
    assert adapter.matches("POST", "/v1/messages") is RouteKind.CHAT
    assert adapter.matches("POST", "/v1/messages/count_tokens") is RouteKind.REDACT_ONLY
    assert adapter.matches("POST", "/v1/complete") is RouteKind.CHAT
    assert adapter.matches("GET", "/v1/messages") is RouteKind.NONE


def test_system_note_string_and_blocks() -> None:
    adapter = AnthropicAdapter()
    assert adapter.inject_system_note({})["system"] == SYSTEM_NOTE
    merged = adapter.inject_system_note({"system": "be brief"})
    assert merged["system"].startswith("be brief")
    assert SYSTEM_NOTE in merged["system"]
    blocks = adapter.inject_system_note({"system": [{"type": "text", "text": "be brief"}]})
    assert blocks["system"][-1]["text"] == SYSTEM_NOTE


def _run(events: list[SSEEvent], vault: InMemoryVault) -> list[SSEEvent]:
    adapter = AnthropicAdapter()
    pool = RehydratorPool(vault)
    out: list[SSEEvent] = []
    for event in events:
        out.extend(adapter.rehydrate_event(event, pool))
    return out


def _text_of(events: list[SSEEvent]) -> str:
    parts = []
    for e in events:
        if e.data and e.data != "[DONE]":
            payload = json.loads(e.data)
            if payload.get("type") == "content_block_delta":
                delta = payload["delta"]
                parts.append(delta.get("text") or delta.get("partial_json") or "")
    return "".join(parts)


def test_text_delta_split_across_events(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    events = [
        _delta(0, type="text_delta", text="your email is «EMA"),
        _delta(0, type="text_delta", text="IL_0"),
        _delta(0, type="text_delta", text="01» ok"),
        _event({"type": "content_block_stop", "index": 0}),
    ]
    out = _run(events, vault)
    assert _text_of(out) == "your email is jane@corp.example ok"
    assert "«" not in _text_of(out)


def test_leftover_flushed_before_block_stop(vault: InMemoryVault) -> None:
    events = [
        _delta(0, type="text_delta", text="dangling «EMAIL_"),
        _event({"type": "content_block_stop", "index": 0}),
    ]
    out = _run(events, vault)
    # The unknown partial token is emitted verbatim in a synthetic delta
    # placed *before* the stop event.
    types = [json.loads(e.data)["type"] for e in out if e.data]
    assert types == ["content_block_delta", "content_block_delta", "content_block_stop"]
    assert _text_of(out) == "dangling «EMAIL_"


def test_input_json_delta_rehydration(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    events = [
        _delta(1, type="input_json_delta", partial_json='{"to": "«EM'),
        _delta(1, type="input_json_delta", partial_json='AIL_001»"}'),
        _event({"type": "content_block_stop", "index": 1}),
    ]
    out = _run(events, vault)
    assert json.loads(_text_of(out)) == {"to": "jane@corp.example"}


def test_thinking_delta_rehydrated(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    events = [
        _delta(0, type="thinking_delta", thinking="user is «EMAIL_001»"),
        _event({"type": "content_block_stop", "index": 0}),
    ]
    out = _run(events, vault)
    payload = json.loads(out[0].data)
    assert payload["delta"]["thinking"] == "user is jane@corp.example"


def test_passthrough_events_untouched(vault: InMemoryVault) -> None:
    ping = SSEEvent(event="ping", data='{"type": "ping"}')
    message_start = _event({"type": "message_start", "message": {"id": "msg_1"}})
    out = _run([ping, message_start], vault)
    assert out == [ping, message_start]


def test_signature_delta_not_corrupted(vault: InMemoryVault) -> None:
    # signature_delta carries opaque base64; must pass through unmodified.
    event = _delta(0, type="signature_delta", signature="AbC«123»==")
    out = _run([event], vault)
    assert json.loads(out[0].data)["delta"]["signature"] == "AbC«123»=="
