"""Cohere adapter: routing, note injection, non-streaming + streaming rehydrate.

The streaming sweep mirrors the SSE/Gemini convention: a «TOKEN» split across
delta FRAMES (text, tool_plan, tool-call arguments, and the v1 text channel)
must reassemble byte-for-byte and match the non-streaming restoration.
"""

import json
from typing import Any

import pytest

from llm_redact.providers.base import SYSTEM_NOTE, RouteKind
from llm_redact.providers.cohere import CohereAdapter
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent, SSEParser, serialize
from llm_redact.vault import InMemoryVault


def test_routing_matrix() -> None:
    a = CohereAdapter()
    assert a.name == "cohere"
    assert a.matches("POST", "/v2/chat") is RouteKind.CHAT
    assert a.matches("POST", "/v2/embed") is RouteKind.REDACT_ONLY
    assert a.matches("POST", "/v2/rerank") is RouteKind.REDACT_ONLY
    assert a.matches("POST", "/v1/chat") is RouteKind.CHAT
    assert a.matches("POST", "/v1/generate") is RouteKind.CHAT
    assert a.matches("GET", "/v2/chat") is RouteKind.NONE
    assert a.matches("POST", "/v1/chat/completions") is RouteKind.NONE  # OpenAI, not Cohere
    assert a.matches("POST", "/v2/unknown") is RouteKind.NONE


def test_note_injection_variants() -> None:
    a = CohereAdapter()
    # v2 chat: a system message.
    v2 = a.inject_system_note({"messages": [{"role": "user", "content": "hi"}]})
    assert v2["messages"][0] == {"role": "system", "content": SYSTEM_NOTE}
    # v1 chat: preamble is the system slot.
    v1 = a.inject_system_note({"message": "hi", "preamble": "be terse"})
    assert v1["preamble"] == f"be terse\n\n{SYSTEM_NOTE}"
    v1_none = a.inject_system_note({"message": "hi"})
    assert v1_none["preamble"] == SYSTEM_NOTE
    # v1 generate: no system field, untouched.
    gen = a.inject_system_note({"prompt": "write a poem"})
    assert gen == {"prompt": "write a poem"}


def test_non_streaming_rehydrate() -> None:
    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", "jane@corp.example")
    rehydrator = Rehydrator(vault)
    body = {
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": f"mail {token}"}],
            "tool_plan": f"look up {token}",
            "tool_calls": [{"function": {"arguments": json.dumps({"to": token})}}],
        }
    }
    out = CohereAdapter().rehydrate_body(body, rehydrator)
    flat = json.dumps(out, ensure_ascii=False)
    assert token not in flat
    assert out["message"]["content"][0]["text"] == "mail jane@corp.example"
    assert out["message"]["tool_plan"] == "look up jane@corp.example"
    assert json.loads(out["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "to": "jane@corp.example"
    }


def _v2_events(token: str) -> list[SSEEvent]:
    """A canonical v2 stream: text token split across two content-deltas, a
    tool_plan token split, and a tool-call arguments token split, each closed
    by its *-end, then message-end."""
    th, tt = token[:4], token[4:]
    args_full = f'{{"to": "{token}"}}'
    ah, at = args_full[:6], args_full[6:]

    def ev(payload: dict[str, Any]) -> SSEEvent:
        return SSEEvent(data=json.dumps(payload, ensure_ascii=False))

    return [
        ev({"type": "message-start", "delta": {"message": {"role": "assistant"}}}),
        ev({"type": "content-start", "index": 0}),
        ev(
            {
                "type": "content-delta",
                "index": 0,
                "delta": {"message": {"content": {"text": f"mail {th}"}}},
            }
        ),
        ev(
            {
                "type": "content-delta",
                "index": 0,
                "delta": {"message": {"content": {"text": f"{tt} ok"}}},
            }
        ),
        ev({"type": "content-end", "index": 0}),
        ev({"type": "tool-plan-delta", "delta": {"message": {"tool_plan": f"call {th}"}}}),
        ev({"type": "tool-plan-delta", "delta": {"message": {"tool_plan": tt}}}),
        ev({"type": "tool-call-start", "index": 0}),
        ev(
            {
                "type": "tool-call-delta",
                "index": 0,
                "delta": {"message": {"tool_calls": {"function": {"arguments": ah}}}},
            }
        ),
        ev(
            {
                "type": "tool-call-delta",
                "index": 0,
                "delta": {"message": {"tool_calls": {"function": {"arguments": at}}}},
            }
        ),
        ev({"type": "tool-call-end", "index": 0}),
        ev({"type": "message-end", "delta": {"finish_reason": "COMPLETE"}}),
    ]


def _collect_v2(events: list[SSEEvent]) -> dict[str, str]:
    text, plan, args = [], [], []
    for e in events:
        if not e.data:
            continue
        payload = json.loads(e.data)
        msg = (payload.get("delta") or {}).get("message") or {}
        if payload.get("type") == "content-delta":
            text.append(msg["content"]["text"])
        elif payload.get("type") == "tool-plan-delta":
            plan.append(msg["tool_plan"])
        elif payload.get("type") == "tool-call-delta":
            args.append(msg["tool_calls"]["function"]["arguments"])
    return {"text": "".join(text), "plan": "".join(plan), "args": "".join(args)}


@pytest.mark.parametrize("fuzzy", [False, True])
def test_v2_stream_split_at_every_offset(fuzzy: bool) -> None:
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", "jane@corp.example")
    token = "«email-1»" if fuzzy else "«EMAIL_001»"
    raw = b"".join(serialize(e) for e in _v2_events(token))
    expected = {
        "text": "mail jane@corp.example ok",
        "plan": "call jane@corp.example",
        "args": '{"to": "jane@corp.example"}',
    }
    for offset in range(len(raw) + 1):
        adapter = CohereAdapter()
        pool = RehydratorPool(vault, fuzzy=fuzzy)
        parser = SSEParser()
        out: list[SSEEvent] = []
        for chunk in (raw[:offset], raw[offset:]):
            for event in parser.feed(chunk):
                out.extend(adapter.rehydrate_event(event, pool))
        for event in parser.close():
            out.extend(adapter.rehydrate_event(event, pool))
        assert _collect_v2(out) == expected, f"offset {offset}"


def _v1_events(token: str) -> list[SSEEvent]:
    th, tt = token[:4], token[4:]
    return [
        SSEEvent(data=json.dumps({"event_type": "stream-start"})),
        SSEEvent(data=json.dumps({"event_type": "text-generation", "text": f"mail {th}"})),
        SSEEvent(data=json.dumps({"event_type": "text-generation", "text": f"{tt} done"})),
        SSEEvent(data=json.dumps({"event_type": "stream-end", "finish_reason": "COMPLETE"})),
    ]


@pytest.mark.parametrize("fuzzy", [False, True])
def test_v1_stream_split_at_every_offset(fuzzy: bool) -> None:
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", "jane@corp.example")
    token = "«email-1»" if fuzzy else "«EMAIL_001»"
    raw = b"".join(serialize(e) for e in _v1_events(token))
    for offset in range(len(raw) + 1):
        adapter = CohereAdapter()
        pool = RehydratorPool(vault, fuzzy=fuzzy)
        parser = SSEParser()
        out: list[SSEEvent] = []
        for chunk in (raw[:offset], raw[offset:]):
            for event in parser.feed(chunk):
                out.extend(adapter.rehydrate_event(event, pool))
        for event in parser.close():
            out.extend(adapter.rehydrate_event(event, pool))
        text = "".join(
            json.loads(e.data)["text"]
            for e in out
            if e.data and json.loads(e.data).get("event_type") == "text-generation"
        )
        assert text == "mail jane@corp.example done", f"offset {offset}"
