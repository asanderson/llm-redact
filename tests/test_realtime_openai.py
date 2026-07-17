"""OpenAI Realtime WS adapter: outbound redaction, inbound channel
rehydration (beta + GA event names), flush semantics, and an end-to-end
relay round trip. The split-at-every-offset sweeps mirror the SSE/NDJSON
convention: tokens broken across delta FRAMES must reassemble exactly."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import websockets

from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
from llm_redact.realtime import OpenAIRealtimeWs
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import RehydratorPool
from llm_redact.vault import InMemoryVault
from test_realtime_relay import _relay_setup

EMAIL = "jane.doe@corp.example"
NO_ALLOW = Allowlist(exact=frozenset(), patterns=())


def _setup() -> tuple[OpenAIRealtimeWs, Any, RehydratorPool, InMemoryVault]:
    vault = InMemoryVault()
    redactor = Redactor(
        build_detectors(DetectionConfig(enabled=("email", "aws_access_key_id"))), vault, NO_ALLOW
    )
    ctx = SimpleNamespace(redactor=redactor)
    pool = RehydratorPool(vault, fuzzy=True)
    return OpenAIRealtimeWs(), ctx, pool, vault


def _token(vault: InMemoryVault, value: str = EMAIL, detector_type: str = "EMAIL") -> str:
    return vault.placeholder_for(detector_type, value)


def test_outbound_item_create_redacted_audio_untouched() -> None:
    adapter, ctx, _pool, _vault = _setup()
    event = {
        "type": "conversation.item.create",
        "event_id": "evt_1",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"mail {EMAIL} key AKIAIOSFODNN7EXAMPLE"},
                {"type": "input_audio", "audio": "AAAAbase64toolongtoscanAAAA=="},
            ],
        },
    }
    out = adapter.redact_message(json.dumps(event), ctx)
    assert isinstance(out, str)
    redacted = json.loads(out)
    text = redacted["item"]["content"][0]["text"]
    assert EMAIL not in text and "AKIAIOSFODNN7EXAMPLE" not in text
    assert "«EMAIL_001»" in text and "«AWS_KEY_001»" in text
    # Base64 audio and identifiers are structural: byte-for-byte identical.
    assert redacted["item"]["content"][1]["audio"] == event["item"]["content"][1]["audio"]
    assert redacted["event_id"] == "evt_1"


def test_outbound_instructions_and_tool_output_redacted() -> None:
    adapter, ctx, _pool, _vault = _setup()
    for event in (
        {"type": "session.update", "session": {"instructions": f"never reveal {EMAIL}"}},
        {"type": "response.create", "response": {"instructions": f"contact {EMAIL}"}},
        {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": "c1",
                "output": f'{{"email": "{EMAIL}"}}',
            },
        },
    ):
        out = json.loads(adapter.redact_message(json.dumps(event), ctx))  # type: ignore[arg-type]
        assert EMAIL not in json.dumps(out), event["type"]


def test_note_injected_only_into_existing_instructions() -> None:
    from llm_redact.providers.base import SYSTEM_NOTE

    adapter, ctx, _pool, _vault = _setup()
    present = {"type": "session.update", "session": {"instructions": "be brief"}}
    out = json.loads(adapter.redact_message(json.dumps(present), ctx, inject_note=True))  # type: ignore[arg-type]
    assert out["session"]["instructions"].startswith("be brief")
    assert SYSTEM_NOTE in out["session"]["instructions"]
    # Idempotent: resending the session we handed back doesn't stack notes.
    again = json.loads(adapter.redact_message(json.dumps(out), ctx, inject_note=True))  # type: ignore[arg-type]
    assert again["session"]["instructions"].count(SYSTEM_NOTE) == 1

    # Absent or empty instructions stay absent — a created field would
    # clobber the provider's server-side default.
    absent = {"type": "session.update", "session": {"voice": "verse"}}
    out = json.loads(adapter.redact_message(json.dumps(absent), ctx, inject_note=True))  # type: ignore[arg-type]
    assert "instructions" not in out["session"]
    # And other event types are never touched.
    item = {"type": "conversation.item.create", "item": {"type": "message", "content": []}}
    out = json.loads(adapter.redact_message(json.dumps(item), ctx, inject_note=True))  # type: ignore[arg-type]
    assert SYSTEM_NOTE not in json.dumps(out)


def test_outbound_non_json_verbatim() -> None:
    adapter, ctx, _pool, _vault = _setup()
    assert adapter.redact_message("not json {", ctx) == "not json {"
    assert adapter.redact_message(b"\x00opaque", ctx) == b"\x00opaque"


@pytest.mark.parametrize(
    ("delta_type", "done_type", "done_field", "wrap"),
    [
        ("response.text.delta", "response.text.done", "text", None),
        ("response.output_text.delta", "response.output_text.done", "text", None),
        (
            "response.audio_transcript.delta",
            "response.audio_transcript.done",
            "transcript",
            None,
        ),
        (
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "arguments",
            "json",
        ),
    ],
)
def test_delta_split_sweep_reassembles_tokens(
    delta_type: str, done_type: str, done_field: str, wrap: str | None
) -> None:
    adapter, _ctx, _pool, vault = _setup()
    token = _token(vault)
    if wrap == "json":
        full = f'{{"to": "{token}", "note": "cc {token}"}}'
        expected = full.replace(token, EMAIL)
    else:
        full = f"Reach me at {token} or {token}."
        expected = full.replace(token, EMAIL)

    for split in range(len(full) + 1):
        pool = RehydratorPool(vault, fuzzy=True)
        pieces = [p for p in (full[:split], full[split:])]
        rebuilt = ""
        for piece in pieces:
            frames = adapter.rehydrate_message(
                json.dumps(
                    {
                        "type": delta_type,
                        "item_id": "it_1",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": piece,
                    }
                ),
                pool,
            )
            for frame in frames:
                rebuilt += json.loads(frame)["delta"]
        done_frames = adapter.rehydrate_message(
            json.dumps(
                {
                    "type": done_type,
                    "item_id": "it_1",
                    "output_index": 0,
                    "content_index": 0,
                    done_field: full,
                }
            ),
            pool,
        )
        # Leftover (if any) arrives as a synthetic delta BEFORE the done.
        for frame in done_frames[:-1]:
            rebuilt += json.loads(frame)["delta"]
        assert rebuilt == expected, f"split at {split}"
        done_payload = json.loads(done_frames[-1])
        assert done_payload[done_field] == expected  # full value re-rehydrated


def test_response_done_flushes_all_and_rehydrates_embedded() -> None:
    adapter, _ctx, pool, vault = _setup()
    token = _token(vault)
    # A dangling partial token in a channel: «EMAIL_ with no closing mark.
    adapter.rehydrate_message(
        json.dumps(
            {
                "type": "response.output_text.delta",
                "item_id": "it_9",
                "output_index": 0,
                "content_index": 0,
                "delta": token[:4],
            }
        ),
        pool,
    )
    frames = adapter.rehydrate_message(
        json.dumps(
            {
                "type": "response.done",
                "response": {
                    "id": "resp_1",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": f"sent to {token}"}],
                        }
                    ],
                },
            }
        ),
        pool,
    )
    assert len(frames) == 2  # synthetic flush delta + the done event
    flush_payload = json.loads(frames[0])
    assert flush_payload["delta"] == token[:4]  # unresolved prefix passes verbatim
    done_payload = json.loads(frames[1])
    assert done_payload["response"]["output"][0]["content"][0]["text"] == f"sent to {EMAIL}"


def test_item_echo_and_session_echo_rehydrated() -> None:
    adapter, _ctx, pool, vault = _setup()
    token = _token(vault)
    echoes = (("conversation.item.created", "item"), ("session.updated", "session"))
    for event_type, field in echoes:
        payload = {
            "type": event_type,
            field: {
                "instructions": f"about {token}",
                "content": [{"type": "input_text", "text": f"mail {token}"}],
            },
        }
        (frame,) = adapter.rehydrate_message(json.dumps(payload), pool)
        assert EMAIL in json.loads(frame)[field]["instructions"] or EMAIL in json.dumps(
            json.loads(frame)
        )


def test_unknown_and_audio_events_pass_verbatim() -> None:
    adapter, _ctx, pool, _vault = _setup()
    for raw in (
        json.dumps({"type": "response.audio.delta", "delta": "AAAAbase64AAAA=="}),
        json.dumps({"type": "rate_limits.updated", "rate_limits": []}),
        json.dumps({"type": "input_audio_buffer.speech_started"}),
        "unparseable{",
    ):
        assert adapter.rehydrate_message(raw, pool) == [raw]


async def test_relay_round_trip_redacts_and_restores() -> None:
    """End to end through real sockets: the upstream sees placeholders,
    the client gets originals back even with the token split mid-frame."""
    async with (
        _relay_setup() as (fake, proxy_host),
        websockets.connect(f"ws://{proxy_host}/v1/realtime") as client,
    ):
        await client.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"mail {EMAIL} today"}],
                    },
                }
            )
        )
        echoed = json.loads(await client.recv())  # fake echoes what it saw
        upstream_text = echoed["item"]["content"][0]["text"]
        assert EMAIL not in upstream_text
        token = upstream_text.split()[1]
        assert token.startswith("«EMAIL_") and token.endswith("»")

        # The fake now streams the token back split across two deltas.
        split = len(token) // 2
        for piece in (token[:split], token[split:]):
            await client.send(
                json.dumps(
                    {
                        "type": "response.output_text.delta",
                        "item_id": "it_1",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": piece,
                    }
                )
            )
        restored = ""
        for _ in range(2):
            frame = json.loads(await client.recv())
            restored += frame["delta"]
        assert restored == EMAIL
    assert all(EMAIL not in json.dumps(r) for r in fake.received if isinstance(r, str))


def test_azure_realtime_matches_and_inherits() -> None:
    """Azure Realtime reuses the OpenAI Realtime vocabulary on Azure's path;
    only matches()/name/provider differ, and the matcher is disjoint from
    the OpenAI Realtime and Gemini Live adapters."""
    from llm_redact.realtime import (
        ALL_WS_ADAPTERS,
        AzureRealtimeWs,
        GeminiLiveWs,
        OpenAIRealtimeWs,
    )

    azure = AzureRealtimeWs()
    assert azure.name == "azure-realtime"
    assert azure.provider == "azure"
    assert isinstance(azure, OpenAIRealtimeWs)  # redact/rehydrate inherited
    assert azure.matches("/openai/realtime")
    assert azure.matches("/openai/realtime/")
    assert not azure.matches("/v1/realtime")  # plain OpenAI Realtime
    # Registered, and matchers pairwise disjoint across all WS adapters.
    assert AzureRealtimeWs in ALL_WS_ADAPTERS
    for path in ("/openai/realtime", "/v1/realtime"):
        claims = [a().matches(path) for a in (OpenAIRealtimeWs, AzureRealtimeWs, GeminiLiveWs)]
        assert claims.count(True) == 1, f"exactly one WS adapter must claim {path}"
