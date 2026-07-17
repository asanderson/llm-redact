"""Gemini adapter: routing, system note, and the streaming split sweep."""

import json
from typing import Any

import pytest

from llm_redact.providers.base import SYSTEM_NOTE, RouteKind
from llm_redact.providers.gemini import GeminiAdapter
from llm_redact.rehydrate import Rehydrator, RehydratorPool
from llm_redact.sse import SSEEvent, SSEParser, serialize
from llm_redact.vault import InMemoryVault

GEN = "/v1beta/models/gemini-2.5-pro:generateContent"
STREAM = "/v1beta/models/gemini-2.5-pro:streamGenerateContent"


def test_routing_matrix() -> None:
    adapter = GeminiAdapter()
    assert adapter.matches("POST", GEN) is RouteKind.CHAT
    assert adapter.matches("POST", STREAM) is RouteKind.CHAT
    assert adapter.matches("POST", "/v1/models/gemini-2.0-flash:generateContent") is RouteKind.CHAT
    assert adapter.matches("POST", "/v1beta/tunedModels/my-tune:generateContent") is RouteKind.CHAT
    assert (
        adapter.matches("POST", "/v1beta/models/gemini-2.5-pro:countTokens")
        is RouteKind.REDACT_ONLY
    )
    assert adapter.matches("GET", GEN) is RouteKind.NONE
    assert adapter.matches("GET", "/v1beta/models") is RouteKind.NONE
    # Redact-only since 0.6.0: embeddings input is redactable text; the
    # response is vectors with nothing to rehydrate.
    assert adapter.matches("POST", "/v1beta/models/x:embedContent") is RouteKind.REDACT_ONLY
    assert adapter.matches("POST", "/v1/models/x:batchEmbedContents") is RouteKind.REDACT_ONLY
    assert adapter.matches("POST", "/v1/chat/completions") is RouteKind.NONE


def test_imagen_veo_routing_and_redaction() -> None:
    adapter = GeminiAdapter()
    # Imagen/Veo prompts are text; their responses are image bytes / a
    # long-running operation name — nothing to rehydrate.
    assert adapter.matches("POST", "/v1beta/models/imagen-3.0:predict") is RouteKind.REDACT_ONLY
    assert (
        adapter.matches("POST", "/v1beta/models/veo-2.0:predictLongRunning")
        is RouteKind.REDACT_ONLY
    )
    assert adapter.matches("GET", "/v1beta/models/imagen-3.0:predict") is RouteKind.NONE
    # No systemInstruction field in predict bodies: never inject the note.
    assert not adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1beta/models/i:predict")

    # instances[].prompt is redacted by the generic walk.
    from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
    from llm_redact.redactor import Redactor

    redactor = Redactor(
        detectors=build_detectors(DetectionConfig()),
        vault=InMemoryVault(),
        allowlist=Allowlist(exact=frozenset(), patterns=()),
    )
    body = {
        "instances": [{"prompt": "a portrait of jane.doe@corp.example"}],
        "parameters": {"sampleCount": 2, "aspectRatio": "16:9"},
    }
    redacted = redactor.redact_json(body)
    assert redacted["instances"][0]["prompt"] == "a portrait of «EMAIL_001»"
    assert redacted["parameters"] == {"sampleCount": 2, "aspectRatio": "16:9"}


def test_cached_and_batch_routing() -> None:
    adapter = GeminiAdapter()
    # Cache create redacts the cached content; the response is metadata only.
    assert adapter.matches("POST", "/v1beta/cachedContents") is RouteKind.REDACT_ONLY
    assert adapter.matches("POST", "/v1/cachedContents") is RouteKind.REDACT_ONLY
    # Per-cache metadata ops pass through (name/model/expiry, never content).
    assert adapter.matches("GET", "/v1beta/cachedContents/abc") is RouteKind.NONE
    assert adapter.matches("PATCH", "/v1beta/cachedContents/abc") is RouteKind.NONE
    assert adapter.matches("DELETE", "/v1beta/cachedContents/abc") is RouteKind.NONE
    # Batch inlines redactable request content; the response is an operation
    # name (results fetched later), so redact-only.
    assert (
        adapter.matches("POST", "/v1beta/models/gemini-2.5-pro:batchGenerateContent")
        is RouteKind.REDACT_ONLY
    )
    # No system note on either (both redact-only, neither is countTokens): a
    # note would corrupt a batch body and change a cache's behavior.
    assert adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1beta/cachedContents") is False
    assert (
        adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1beta/models/x:batchGenerateContent")
        is False
    )


def test_cached_and_batch_content_is_redacted() -> None:
    """The generic body walk redacts the cached prompt and inlined batch
    request content — the actual leak this closes."""
    vault = InMemoryVault()
    adapter = GeminiAdapter()
    from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
    from llm_redact.redactor import Redactor

    redactor = Redactor(
        build_detectors(DetectionConfig(enabled=("email",))),
        vault,
        Allowlist(exact=frozenset(), patterns=()),
    )
    cache_body = {
        "model": "models/gemini-2.5-pro",
        "contents": [{"role": "user", "parts": [{"text": "mail jane@corp.example"}]}],
    }
    out = adapter.prepare_request(cache_body, redactor, inject_note=False)
    flat = json.dumps(out, ensure_ascii=False)
    assert "jane@corp.example" not in flat
    assert "«EMAIL_001»" in flat


def test_system_note_variants() -> None:
    adapter = GeminiAdapter()
    created = adapter.inject_system_note({"contents": []})
    assert created["systemInstruction"] == {"parts": [{"text": SYSTEM_NOTE}]}

    appended = adapter.inject_system_note({"systemInstruction": {"parts": [{"text": "be brief"}]}})
    assert appended["systemInstruction"]["parts"] == [
        {"text": "be brief"},
        {"text": SYSTEM_NOTE},
    ]

    snake = adapter.inject_system_note({"system_instruction": {"parts": [{"text": "x"}]}})
    assert snake["system_instruction"]["parts"][-1] == {"text": SYSTEM_NOTE}
    assert "systemInstruction" not in snake

    stringy = adapter.inject_system_note({"systemInstruction": "be brief"})
    assert stringy["systemInstruction"]["parts"] == [
        {"text": "be brief"},
        {"text": SYSTEM_NOTE},
    ]


def test_error_body_is_google_shaped() -> None:
    body = GeminiAdapter().error_body("too big")
    assert body["error"]["code"] == 413
    assert body["error"]["status"]


def _chunk(*candidates: dict[str, Any], extra: dict[str, Any] | None = None) -> SSEEvent:
    payload: dict[str, Any] = {"candidates": list(candidates)}
    if extra:
        payload.update(extra)
    return SSEEvent(data=json.dumps(payload, ensure_ascii=False))


def _candidate(index: int, *parts: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    candidate: dict[str, Any] = {"index": index, "content": {"parts": list(parts)}}
    if finish:
        candidate["finishReason"] = finish
    return candidate


def _fixture_events(token: str) -> list[SSEEvent]:
    """Canonical stream: token split across chunks in text AND thought
    channels, a functionCall with a placeholder inside args, a
    metadata-only chunk, and a finish chunk with a trailing partial."""
    head, tail = token[:4], token[4:]
    return [
        _chunk(_candidate(0, {"text": f"mail {head}"})),
        _chunk(
            _candidate(
                0,
                {"text": f"{tail} ok"},
                {"text": f"note {head}", "thought": True},
            )
        ),
        _chunk(
            _candidate(
                0,
                {"text": tail, "thought": True},
                {"functionCall": {"name": "send", "args": {"to": token}}},
            )
        ),
        SSEEvent(data=json.dumps({"usageMetadata": {"totalTokenCount": 5}})),
        _chunk(_candidate(0, {"text": "tail «EMAIL_"}, finish="STOP")),
    ]


def _collect(events: list[SSEEvent]) -> dict[str, Any]:
    text: list[str] = []
    thought: list[str] = []
    args: list[Any] = []
    for event in events:
        if not event.data:
            continue
        payload = json.loads(event.data)
        for candidate in payload.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                if isinstance(part.get("text"), str):
                    (thought if part.get("thought") else text).append(part["text"])
                if "functionCall" in part:
                    args.append(part["functionCall"]["args"])
    return {"text": "".join(text), "thought": "".join(thought), "args": args}


@pytest.mark.parametrize("fuzzy", [False, True])
def test_stream_split_at_every_byte_offset(fuzzy: bool) -> None:
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", "jane@corp.example")
    token = "«email-1»" if fuzzy else "«EMAIL_001»"

    raw = b"".join(serialize(e) for e in _fixture_events(token))
    expected = {
        "text": "mail jane@corp.example oktail «EMAIL_",
        "thought": "note jane@corp.example",
        "args": [{"to": "jane@corp.example"}],
    }

    for offset in range(len(raw) + 1):
        adapter = GeminiAdapter()
        pool = RehydratorPool(vault, fuzzy=fuzzy)
        parser = SSEParser()
        out: list[SSEEvent] = []
        for piece in (raw[:offset], raw[offset:]):
            for event in parser.feed(piece):
                out.extend(adapter.rehydrate_event(event, pool))
        for event in parser.close():
            out.extend(adapter.rehydrate_event(event, pool))
        assert _collect(out) == expected, f"offset {offset}"
        # finishReason flushed every channel: nothing left for stream close.
        assert pool.flush_all() == {}, f"offset {offset}"


def test_list_body_split_across_elements(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    adapter = GeminiAdapter()
    body = [
        {"candidates": [_candidate(0, {"text": "mail «EMA"})]},
        {"candidates": [_candidate(0, {"text": "IL_001» ok"})]},
        {"candidates": [_candidate(0, {"text": " end"}, finish="STOP")]},
    ]
    out = adapter.rehydrate_body(body, Rehydrator(vault))
    text = "".join(
        part["text"]
        for chunk in out
        for candidate in chunk["candidates"]
        for part in candidate["content"]["parts"]
    )
    assert text == "mail jane@corp.example ok end"


def test_list_body_without_finish_still_flushes(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    adapter = GeminiAdapter()
    body = [
        {"candidates": [_candidate(0, {"text": "mail «EMA"})]},
        {"candidates": [_candidate(0, {"text": "IL_001"})]},  # » never arrives... below
    ]
    out = adapter.rehydrate_body(body, Rehydrator(vault))
    text = "".join(
        part["text"]
        for chunk in out
        for candidate in chunk["candidates"]
        for part in candidate["content"]["parts"]
    )
    # The unterminated token is passed through verbatim, never dropped.
    assert text == "mail «EMAIL_001"


def test_dict_body_uses_plain_walk(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    adapter = GeminiAdapter()
    body = {
        "candidates": [
            _candidate(0, {"text": "mail «EMAIL_001»"}, finish="STOP"),
        ]
    }
    out = adapter.rehydrate_body(body, Rehydrator(vault))
    assert out["candidates"][0]["content"]["parts"][0]["text"] == "mail jane@corp.example"


def test_finish_chunk_without_content_gains_leftover_part(vault: InMemoryVault) -> None:
    vault.placeholder_for("EMAIL", "jane@corp.example")
    adapter = GeminiAdapter()
    pool = RehydratorPool(vault)
    events = [
        _chunk(_candidate(0, {"text": "dangling «EMAIL_"})),
        SSEEvent(data=json.dumps({"candidates": [{"index": 0, "finishReason": "STOP"}]})),
    ]
    out: list[SSEEvent] = []
    for event in events:
        out.extend(adapter.rehydrate_event(event, pool))
    assert _collect(out)["text"] == "dangling «EMAIL_"
    assert pool.flush_all() == {}
