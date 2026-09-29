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
    # Model metadata (no :verb): recognized, redact-only (a body-less no-op).
    assert adapter.matches("GET", "/v1beta/models") is RouteKind.REDACT_ONLY
    assert adapter.matches("GET", "/v1beta/models/gemini-2.5-pro") is RouteKind.REDACT_ONLY
    assert adapter.matches("GET", "/v1beta/models/a/b") is RouteKind.NONE
    assert not adapter.wants_system_note(RouteKind.REDACT_ONLY, "/v1beta/models")
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


# Function-call args are caller JSON (an opaque position): keys that are
# structural names elsewhere are the tool's own parameter names.
_ARG_NAMES = ("to", "id", "name", "type", "data", "model", "role", "signature")


def _fixture_payloads(token: str) -> list[dict[str, Any]]:
    """Canonical stream: token split across chunks in text AND thought
    channels, a functionCall whose args hide placeholders under structural
    names, generated code, grounding metadata, a metadata-only chunk, and a
    finish chunk with a trailing partial."""
    head, tail = token[:4], token[4:]
    return [
        {"candidates": [_candidate(0, {"text": f"mail {head}"})]},
        {
            "candidates": [
                _candidate(
                    0,
                    {"text": f"{tail} ok"},
                    {"text": f"note {head}", "thought": True},
                )
            ]
        },
        {
            "candidates": [
                _candidate(
                    0,
                    {"text": tail, "thought": True},
                    {
                        "functionCall": {
                            "id": "call_1",
                            "name": "send",
                            "args": {name: token for name in _ARG_NAMES},
                        }
                    },
                    {"executableCode": {"language": "PYTHON", "code": f"send({token!r})"}},
                )
            ]
        },
        {"usageMetadata": {"totalTokenCount": 5}},
        {
            "candidates": [
                {
                    **_candidate(0, {"text": "tail «EMAIL_"}, finish="STOP"),
                    "groundingMetadata": {"webSearchQueries": [f"who is {token}"]},
                }
            ]
        },
    ]


def _fixture_events(token: str) -> list[SSEEvent]:
    return [
        SSEEvent(data=json.dumps(payload, ensure_ascii=False))
        for payload in _fixture_payloads(token)
    ]


def _collect_payloads(payloads: list[Any]) -> dict[str, Any]:
    text: list[str] = []
    thought: list[str] = []
    args: list[Any] = []
    code: list[str] = []
    queries: list[str] = []
    for payload in payloads:
        for candidate in payload.get("candidates") or []:
            queries.extend((candidate.get("groundingMetadata") or {}).get("webSearchQueries", []))
            for part in (candidate.get("content") or {}).get("parts") or []:
                if isinstance(part.get("text"), str):
                    (thought if part.get("thought") else text).append(part["text"])
                if "functionCall" in part:
                    assert part["functionCall"]["name"] == "send"
                    assert part["functionCall"]["id"] == "call_1"
                    args.append(part["functionCall"]["args"])
                if "executableCode" in part:
                    code.append(part["executableCode"]["code"])
    return {
        "text": "".join(text),
        "thought": "".join(thought),
        "args": args,
        "code": code,
        "queries": queries,
    }


def _collect(events: list[SSEEvent]) -> dict[str, Any]:
    return _collect_payloads([json.loads(event.data) for event in events if event.data])


_EXPECTED = {
    "text": "mail jane@corp.example oktail «EMAIL_",
    "thought": "note jane@corp.example",
    "args": [{name: "jane@corp.example" for name in _ARG_NAMES}],
    "code": ["send('jane@corp.example')"],
    "queries": ["who is jane@corp.example"],
}


@pytest.mark.parametrize("fuzzy", [False, True])
def test_stream_split_at_every_byte_offset(fuzzy: bool) -> None:
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", "jane@corp.example")
    token = "«email-1»" if fuzzy else "«EMAIL_001»"

    raw = b"".join(serialize(e) for e in _fixture_events(token))
    expected = _EXPECTED

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


@pytest.mark.parametrize("fuzzy", [False, True])
def test_array_form_and_buffered_body_match_the_stream(fuzzy: bool) -> None:
    """The three delivery forms of one answer restore the same values: the
    SSE stream (swept above), the non-SSE JSON array of the same chunks, and
    — for everything but the chunk-split text — the buffered response."""
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", "jane@corp.example")
    token = "«email-1»" if fuzzy else "«EMAIL_001»"
    chunks = _fixture_payloads(token)
    array = GeminiAdapter().rehydrate_body(chunks, Rehydrator(vault, fuzzy=fuzzy))
    assert _collect_payloads(array) == _EXPECTED
    # The input list is not left holding anything the walk put in it.
    assert json.loads(json.dumps(chunks)) == _fixture_payloads(token)

    merged = {
        "candidates": [
            {
                **chunks[4]["candidates"][0],
                "content": {
                    "parts": [
                        part
                        for chunk in chunks
                        for candidate in chunk.get("candidates", [])
                        for part in candidate["content"]["parts"]
                        if "text" not in part
                    ]
                },
            }
        ]
    }
    buffered = GeminiAdapter().rehydrate_body(merged, Rehydrator(vault, fuzzy=fuzzy))
    collected = _collect_payloads([buffered])
    for field in ("args", "code", "queries"):
        assert collected[field] == _EXPECTED[field]


def test_non_candidate_chunks_of_the_array_form_are_restored(vault: InMemoryVault) -> None:
    token = vault.placeholder_for("EMAIL", "jane@corp.example")
    body = [{"promptFeedback": {"note": token}}, {"candidates": [_candidate(0, {"text": "x"})]}]
    out = GeminiAdapter().rehydrate_body(body, Rehydrator(vault))
    assert out[0] == {"promptFeedback": {"note": "jane@corp.example"}}


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


# --- end to end: every delivery form restores the same args ------------------------


@pytest.mark.parametrize("form", ["buffered", "sse", "array"])
async def test_every_form_restores_structural_named_args_end_to_end(form: str) -> None:
    import httpx

    from llm_redact.config import Config
    from llm_redact.proxy import create_app

    email = "jane.doe@corp.example"
    token = "«EMAIL_001»"
    args = {name: token for name in _ARG_NAMES}
    call = {"functionCall": {"id": "call_1", "name": "send", "args": args}}
    chunk = {"candidates": [_candidate(0, call, finish="STOP")]}

    def upstream(request: httpx.Request) -> httpx.Response:
        assert email.encode() not in request.content
        if form == "sse":
            body = b"data: " + json.dumps(chunk).encode() + b"\r\n\r\n"
            return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=[chunk] if form == "array" else chunk)

    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    verb = "generateContent" if form == "buffered" else "streamGenerateContent"
    query = "?alt=sse" if form == "sse" else ""
    request = {"contents": [{"role": "user", "parts": [{"text": f"look up {email}"}]}]}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            f"/v1beta/models/gemini-2.5-flash:{verb}{query}",
            json=request,
            headers={"x-goog-api-key": "k"},
        )
    assert response.status_code == 200
    if form == "sse":
        payloads = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
    else:
        body = response.json()
        payloads = body if isinstance(body, list) else [body]
    (restored,) = _collect_payloads(payloads)["args"]
    assert restored == dict.fromkeys(_ARG_NAMES, email)
