"""Gemini Live WS adapter: JSON over text OR binary frames, outbound
redaction of setup/clientContent/toolResponse, streamed modelTurn
rehydration with turnComplete flush semantics, and a relay round trip."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import websockets

from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
from llm_redact.realtime import GeminiLiveWs
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import RehydratorPool
from llm_redact.vault import InMemoryVault
from test_realtime_relay import _relay_setup

EMAIL = "jane.doe@corp.example"
NO_ALLOW = Allowlist(exact=frozenset(), patterns=())
LIVE_PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"


def _setup() -> tuple[GeminiLiveWs, Any, RehydratorPool, InMemoryVault]:
    vault = InMemoryVault()
    redactor = Redactor(build_detectors(DetectionConfig(enabled=("email",))), vault, NO_ALLOW)
    ctx = SimpleNamespace(redactor=redactor)
    return GeminiLiveWs(), ctx, RehydratorPool(vault, fuzzy=True), vault


def _server_content(text: str, **flags: Any) -> dict[str, Any]:
    return {"serverContent": {"modelTurn": {"parts": [{"text": text}]}, **flags}}


def test_outbound_setup_and_client_content_redacted_binary_stays_binary() -> None:
    adapter, ctx, _pool, _vault = _setup()
    message = {
        "setup": {
            "model": "models/gemini-live-2.5-flash",
            "systemInstruction": {"parts": [{"text": f"never reveal {EMAIL}"}]},
        }
    }
    out = adapter.redact_message(json.dumps(message).encode(), ctx)
    assert isinstance(out, bytes)  # binary in, binary out
    redacted = json.loads(out)
    assert EMAIL not in json.dumps(redacted)
    assert redacted["setup"]["model"] == "models/gemini-live-2.5-flash"  # structural

    turns = {
        "clientContent": {
            "turns": [{"role": "user", "parts": [{"text": f"mail {EMAIL}"}]}],
            "turnComplete": True,
        }
    }
    out_text = adapter.redact_message(json.dumps(turns), ctx)
    assert isinstance(out_text, str)  # text in, text out
    assert EMAIL not in out_text and "«EMAIL_001»" in out_text


def test_outbound_media_chunks_untouched_tool_response_walked() -> None:
    adapter, ctx, _pool, _vault = _setup()
    media = {
        "realtimeInput": {"mediaChunks": [{"mimeType": "audio/pcm", "data": "AAAAbase64AAAA=="}]}
    }
    out = json.loads(adapter.redact_message(json.dumps(media), ctx))  # type: ignore[arg-type]
    assert out == media  # base64 `data` and mimeType are structural

    tool = {"toolResponse": {"functionResponses": [{"id": "f1", "response": {"contact": EMAIL}}]}}
    out = json.loads(adapter.redact_message(json.dumps(tool), ctx))  # type: ignore[arg-type]
    assert EMAIL not in json.dumps(out)


def test_note_appended_to_existing_system_instruction_only() -> None:
    from llm_redact.providers.base import SYSTEM_NOTE

    adapter, ctx, _pool, _vault = _setup()
    present = {"setup": {"model": "models/x", "systemInstruction": {"parts": [{"text": "hi"}]}}}
    out = json.loads(adapter.redact_message(json.dumps(present), ctx, inject_note=True))  # type: ignore[arg-type]
    assert out["setup"]["systemInstruction"]["parts"][-1]["text"] == SYSTEM_NOTE
    again = json.loads(adapter.redact_message(json.dumps(out), ctx, inject_note=True))  # type: ignore[arg-type]
    parts = again["setup"]["systemInstruction"]["parts"]
    assert sum(1 for p in parts if p["text"] == SYSTEM_NOTE) == 1  # idempotent

    absent = {"setup": {"model": "models/x"}}
    out = json.loads(adapter.redact_message(json.dumps(absent), ctx, inject_note=True))  # type: ignore[arg-type]
    assert "systemInstruction" not in out["setup"]  # never created


@pytest.mark.parametrize("binary", [False, True])
def test_model_turn_split_sweep(binary: bool) -> None:
    adapter, _ctx, _pool, vault = _setup()
    token = vault.placeholder_for("EMAIL", EMAIL)
    full = f"Contact {token} soon."
    expected = full.replace(token, EMAIL)

    for split in range(len(full) + 1):
        pool = RehydratorPool(vault, fuzzy=True)
        rebuilt = ""
        for i, piece in enumerate((full[:split], full[split:])):
            flags = {"turnComplete": True} if i == 1 else {}
            frame = json.dumps(_server_content(piece, **flags))
            frames = adapter.rehydrate_message(frame.encode() if binary else frame, pool)
            for out in frames:
                assert isinstance(out, bytes) == binary  # frame type preserved
                payload = json.loads(out)
                for part in payload["serverContent"]["modelTurn"]["parts"]:
                    rebuilt += part["text"]
        assert rebuilt == expected, f"split at {split}"


def test_turn_complete_without_model_turn_emits_synthetic_frame() -> None:
    adapter, _ctx, pool, vault = _setup()
    token = vault.placeholder_for("EMAIL", EMAIL)
    # Feed a dangling partial token, then a bare turnComplete message.
    adapter.rehydrate_message(json.dumps(_server_content(token[:5])), pool)
    frames = adapter.rehydrate_message(json.dumps({"serverContent": {"turnComplete": True}}), pool)
    assert len(frames) == 2
    synthetic = json.loads(frames[0])
    assert synthetic["serverContent"]["modelTurn"]["parts"][0]["text"] == token[:5]
    assert json.loads(frames[1]) == {"serverContent": {"turnComplete": True}}


def test_thought_and_text_parts_use_separate_channels() -> None:
    adapter, _ctx, pool, vault = _setup()
    token = vault.placeholder_for("EMAIL", EMAIL)
    split = len(token) // 2
    # Interleave: text channel gets the first half, a thought part arrives,
    # then the text channel completes — the token must still reassemble.
    frame1 = {
        "serverContent": {
            "modelTurn": {
                "parts": [{"text": token[:split]}, {"text": "pondering", "thought": True}]
            }
        }
    }
    frame2 = _server_content(token[split:], turnComplete=True)
    out1 = json.loads(adapter.rehydrate_message(json.dumps(frame1), pool)[0])
    out2 = json.loads(adapter.rehydrate_message(json.dumps(frame2), pool)[-1])
    text = "".join(
        p["text"]
        for out in (out1, out2)
        for p in out["serverContent"]["modelTurn"]["parts"]
        if not p.get("thought")
    )
    assert text == EMAIL


def test_tool_call_args_rehydrated_whole() -> None:
    adapter, _ctx, pool, vault = _setup()
    token = vault.placeholder_for("EMAIL", EMAIL)
    frame = {"toolCall": {"functionCalls": [{"id": "f1", "name": "send", "args": {"to": token}}]}}
    (out,) = adapter.rehydrate_message(json.dumps(frame), pool)
    assert json.loads(out)["toolCall"]["functionCalls"][0]["args"]["to"] == EMAIL


def test_bookkeeping_and_unparseable_pass_verbatim() -> None:
    adapter, _ctx, pool, _vault = _setup()
    for raw in (
        json.dumps({"setupComplete": {}}),
        json.dumps({"usageMetadata": {"totalTokenCount": 5}}),
        b"\x00not-json",
    ):
        assert adapter.rehydrate_message(raw, pool) == [raw]


async def test_relay_round_trip_gemini_path() -> None:
    async with (
        _relay_setup(provider="gemini") as (fake, proxy_host),
        websockets.connect(f"ws://{proxy_host}{LIVE_PATH}?key=test-key") as client,
    ):
        await client.send(
            json.dumps(
                {
                    "clientContent": {
                        "turns": [{"role": "user", "parts": [{"text": f"mail {EMAIL}"}]}],
                        "turnComplete": True,
                    }
                }
            )
        )
        echoed = json.loads(await client.recv())
        upstream_text = echoed["clientContent"]["turns"][0]["parts"][0]["text"]
        assert EMAIL not in upstream_text and upstream_text.startswith("mail «EMAIL_")
    # The raw query (with its key) reached the upstream untouched.
    assert fake.paths == [f"{LIVE_PATH}?key=test-key"]
