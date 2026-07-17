"""Live-API smoke tests. Deselected by default; costs real API credits.

Run explicitly:  LLM_REDACT_LIVE=1 uv run pytest -m live -rA
(or use scripts/live_smoke.py). Each provider's tests skip cleanly when its
API key is absent.

Proof structure (we cannot observe the provider's side directly):
- redaction: the in-process ``state.redactor.counts`` grew — the redactor fed
  the upstream request, so placeholders went out;
- rehydration: the response contains the ORIGINAL values. The model can only
  echo what it received, so originals in the output prove the round trip.
"""

import asyncio
import contextlib
import json
import os

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.eventstream import EventStreamParser
from llm_redact.providers.bedrock import KNOWN_CONVERSE_EVENT_TYPES
from llm_redact.providers.openai_responses import KNOWN_EVENT_TYPES
from llm_redact.proxy import create_app
from llm_redact.sse import SSEParser

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("LLM_REDACT_LIVE") != "1",
        reason="live tests disabled (set LLM_REDACT_LIVE=1)",
    ),
]

needs_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
needs_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
needs_cohere = pytest.mark.skipif(
    not os.environ.get("COHERE_API_KEY"), reason="COHERE_API_KEY not set"
)

EMAIL = "jane.doe@corp-llmredact.example"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
ECHO_PROMPT = (
    f"My email is {EMAIL} and my AWS key id is {AWS_KEY}. "
    "Repeat both values back to me exactly, character for character."
)


def _client_and_state() -> tuple[httpx.AsyncClient, object]:
    app = create_app(Config())
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy", timeout=120.0
    )
    return client, app.state.proxy


async def _collect_sse(response: httpx.Response) -> list[tuple[str | None, str]]:
    parser = SSEParser()
    events: list[tuple[str | None, str]] = []
    async for chunk in response.aiter_bytes():
        for event in parser.feed(chunk):
            events.append((event.event, event.data))
    for event in parser.close():
        events.append((event.event, event.data))
    return events


@needs_anthropic
async def test_anthropic_streaming_echo() -> None:
    client, state = _client_and_state()
    body = {
        "model": "claude-3-5-haiku-latest",
        "max_tokens": 200,
        "stream": True,
        "messages": [{"role": "user", "content": ECHO_PROMPT}],
    }
    async with client.stream(
        "POST",
        "/v1/messages",
        json=body,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
    ) as response:
        assert response.status_code == 200, await response.aread()
        events = await _collect_sse(response)
    assert state.redactor.counts["EMAIL"] >= 1  # type: ignore[attr-defined]
    assert state.redactor.counts["AWS_KEY"] >= 1  # type: ignore[attr-defined]
    text = ""
    for _name, data in events:
        if not data:
            continue
        payload = json.loads(data)
        if payload.get("type") == "content_block_delta":
            text += payload["delta"].get("text", "")
    assert EMAIL in text, f"model did not echo the original email: {text!r}"
    assert AWS_KEY in text
    assert "«" not in text
    await client.aclose()


@needs_anthropic
async def test_anthropic_tool_use_streaming() -> None:
    client, state = _client_and_state()
    body = {
        "model": "claude-3-5-haiku-latest",
        "max_tokens": 300,
        "stream": True,
        "tools": [
            {
                "name": "record_contact",
                "description": "Record a contact email address",
                "input_schema": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                    "required": ["email"],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "record_contact"},
        "messages": [{"role": "user", "content": f"Record the contact {EMAIL}."}],
    }
    async with client.stream(
        "POST",
        "/v1/messages",
        json=body,
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
    ) as response:
        assert response.status_code == 200, await response.aread()
        events = await _collect_sse(response)
    arguments = ""
    for _name, data in events:
        if not data:
            continue
        payload = json.loads(data)
        if payload.get("type") == "content_block_delta":
            arguments += payload["delta"].get("partial_json", "")
    parsed = json.loads(arguments)
    assert parsed["email"] == EMAIL  # json_source path against real chunking
    await client.aclose()


@needs_openai
async def test_openai_chat_streaming_echo() -> None:
    client, state = _client_and_state()
    body = {
        "model": "gpt-4o-mini",
        "max_tokens": 200,
        "stream": True,
        "messages": [{"role": "user", "content": ECHO_PROMPT}],
    }
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json=body,
        headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    ) as response:
        assert response.status_code == 200, await response.aread()
        events = await _collect_sse(response)
    text = ""
    for _name, data in events:
        if not data or data == "[DONE]":
            continue
        for choice in json.loads(data).get("choices", []):
            text += choice.get("delta", {}).get("content") or ""
    assert EMAIL in text
    assert "«" not in text
    await client.aclose()


@needs_cohere
async def test_cohere_v2_chat_streaming_and_drift() -> None:
    from llm_redact.providers.cohere import KNOWN_COHERE_EVENT_TYPES

    client, state = _client_and_state()
    body = {
        "model": "command-r-08-2024",
        "stream": True,
        "messages": [{"role": "user", "content": ECHO_PROMPT}],
    }
    async with client.stream(
        "POST",
        "/v2/chat",
        json=body,
        headers={"authorization": f"Bearer {os.environ['COHERE_API_KEY']}"},
    ) as response:
        assert response.status_code == 200, await response.aread()
        events = await _collect_sse(response)

    observed = set()
    text = ""
    for _name, data in events:
        if not data:
            continue
        payload = json.loads(data)
        event_type = payload.get("type")
        if event_type:
            observed.add(event_type)
        if event_type == "content-delta":
            text += (((payload.get("delta") or {}).get("message") or {}).get("content") or {}).get(
                "text"
            ) or ""
    assert state.redactor.counts["EMAIL"] >= 1  # type: ignore[attr-defined]
    assert EMAIL in text, f"model did not echo the original email: {text!r}"
    assert "«" not in text
    unknown = observed - KNOWN_COHERE_EVENT_TYPES
    assert not unknown, f"unknown Cohere event types (update KNOWN_COHERE_EVENT_TYPES): {unknown}"
    await client.aclose()


@needs_openai
async def test_responses_streaming_and_drift() -> None:
    client, state = _client_and_state()
    body = {
        "model": "gpt-4o-mini",
        "stream": True,
        "input": ECHO_PROMPT,
    }
    async with client.stream(
        "POST",
        "/v1/responses",
        json=body,
        headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    ) as response:
        assert response.status_code == 200, await response.aread()
        events = await _collect_sse(response)

    observed = set()
    text = ""
    for _name, data in events:
        if not data or data == "[DONE]":
            continue
        payload = json.loads(data)
        event_type = payload.get("type")
        if event_type:
            observed.add(event_type)
        if event_type == "response.output_text.delta":
            text += payload["delta"]

    # Drift detector: any event name outside the adapter's known set fails
    # loudly with the exact diff — event-shape drift is the top risk here.
    unknown = observed - KNOWN_EVENT_TYPES
    assert not unknown, f"Responses API emitted unknown event types: {sorted(unknown)}"
    assert EMAIL in text
    assert "«" not in text
    await client.aclose()


needs_bedrock = pytest.mark.skipif(
    not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"), reason="AWS_BEARER_TOKEN_BEDROCK not set"
)


@needs_openai
async def test_openai_realtime_live_echo_and_drift() -> None:
    """Realtime over a REAL proxy socket (WS cannot ride ASGITransport):
    the canary email is redacted outbound, restored in the text output,
    and every observed server event name must be one the adapter knows."""
    import websockets

    from llm_redact.realtime import KNOWN_REALTIME_EVENT_TYPES
    from test_realtime_relay import _proxy

    model = os.environ.get("LLM_REDACT_REALTIME_MODEL", "gpt-realtime")
    with _proxy(Config()) as proxy_host:
        observed: set[str] = set()
        text = ""
        async with websockets.connect(
            f"ws://{proxy_host}/v1/realtime?model={model}",
            additional_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        ) as client:
            await client.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"type": "realtime", "output_modalities": ["text"]},
                    }
                )
            )
            await client.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": ECHO_PROMPT}],
                        },
                    }
                )
            )
            await client.send(json.dumps({"type": "response.create"}))
            while True:
                payload = json.loads(await client.recv())
                event_type = str(payload.get("type"))
                observed.add(event_type)
                if event_type in ("response.output_text.delta", "response.text.delta"):
                    text += payload.get("delta") or ""
                if event_type == "error":
                    raise AssertionError(f"realtime error event: {payload}")
                if event_type == "response.done":
                    break
    unknown = observed - KNOWN_REALTIME_EVENT_TYPES
    assert not unknown, f"Realtime API emitted unknown event types: {sorted(unknown)}"
    assert EMAIL in text, f"model did not echo the original email: {text!r}"
    assert "«" not in text


needs_gemini_live = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)


@needs_gemini_live
async def test_gemini_live_echo_and_drift() -> None:
    import websockets

    from llm_redact.realtime import KNOWN_LIVE_SERVER_KEYS
    from test_realtime_relay import _proxy

    model = os.environ.get("LLM_REDACT_LIVE_MODEL", "models/gemini-2.0-flash-live-001")
    path = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    with _proxy(Config()) as proxy_host:
        observed: set[str] = set()
        text = ""
        done = False
        async with websockets.connect(
            f"ws://{proxy_host}{path}?key={os.environ['GEMINI_API_KEY']}"
        ) as client:
            setup = {"model": model, "generationConfig": {"responseModalities": ["TEXT"]}}
            await client.send(json.dumps({"setup": setup}))
            await client.recv()  # setupComplete
            await client.send(
                json.dumps(
                    {
                        "clientContent": {
                            "turns": [{"role": "user", "parts": [{"text": ECHO_PROMPT}]}],
                            "turnComplete": True,
                        }
                    }
                )
            )
            while not done:
                payload = json.loads(await client.recv())
                observed |= set(payload)
                content = payload.get("serverContent") or {}
                for part in (content.get("modelTurn") or {}).get("parts") or []:
                    if isinstance(part.get("text"), str) and not part.get("thought"):
                        text += part["text"]
                done = bool(content.get("turnComplete"))
    # Key-set drift is the analogue of unknown event types (messages are
    # unnamed) — same as the HTTP Gemini adapter's detector.
    unknown = observed - KNOWN_LIVE_SERVER_KEYS
    assert not unknown, f"Live API sent unknown top-level keys: {sorted(unknown)}"
    assert EMAIL in text, f"model did not echo the original email: {text!r}"
    assert "«" not in text


@needs_bedrock
async def test_bedrock_converse_stream_echo_and_drift() -> None:
    # Bearer-key auth only (SigV4 is a permanent non-goal). The region is
    # baked into the host, so [providers.bedrock] must be set explicitly —
    # same as production configs.
    region = os.environ.get("AWS_REGION", "us-east-1")
    model_id = os.environ.get(
        "LLM_REDACT_BEDROCK_MODEL", "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    )
    config = Config(
        providers={
            **Config().providers,
            "bedrock": ProviderConfig(f"https://bedrock-runtime.{region}.amazonaws.com"),
        }
    )
    app = create_app(config)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy", timeout=120.0
    )
    state = app.state.proxy

    body = {
        "messages": [{"role": "user", "content": [{"text": ECHO_PROMPT}]}],
        "inferenceConfig": {"maxTokens": 300},
    }
    parser = EventStreamParser()
    observed: set[str] = set()
    exceptions: list[str] = []
    text = ""
    async with client.stream(
        "POST",
        f"/model/{model_id}/converse-stream",
        json=body,
        headers={"authorization": f"Bearer {os.environ['AWS_BEARER_TOKEN_BEDROCK']}"},
    ) as response:
        assert response.status_code == 200, await response.aread()
        async for chunk in response.aiter_bytes():
            for frame in parser.feed(chunk):
                if frame.message_type == "exception":
                    exceptions.append(f"{frame.exception_type}: {frame.payload!r}")
                    continue
                if frame.event_type:
                    observed.add(frame.event_type)
                if frame.event_type == "contentBlockDelta":
                    delta = json.loads(frame.payload).get("delta", {})
                    text += delta.get("text", "")
    parser.close()  # a truncated trailing frame raises here

    assert not exceptions, exceptions
    assert state.redactor.counts["EMAIL"] >= 1  # type: ignore[attr-defined]
    assert state.redactor.counts["AWS_KEY"] >= 1  # type: ignore[attr-defined]
    # Drift detector: the adapter forwards unknown frames verbatim, so a new
    # event type is a schema-drift signal, not a crash — fail loudly with
    # the exact diff.
    unknown = observed - KNOWN_CONVERSE_EVENT_TYPES
    assert not unknown, f"ConverseStream emitted unknown event types: {sorted(unknown)}"
    assert EMAIL in text, f"model did not echo the original email: {text!r}"
    assert AWS_KEY in text
    assert "«" not in text
    await client.aclose()


needs_gemini = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)


@needs_gemini
async def test_gemini_streaming_echo_and_drift() -> None:
    from llm_redact.providers.gemini import (
        KNOWN_CANDIDATE_KEYS,
        KNOWN_CHUNK_KEYS,
        KNOWN_PART_KEYS,
    )

    client, state = _client_and_state()
    body = {"contents": [{"role": "user", "parts": [{"text": ECHO_PROMPT}]}]}
    async with client.stream(
        "POST",
        "/v1beta/models/gemini-2.0-flash:streamGenerateContent?alt=sse",
        json=body,
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    ) as response:
        assert response.status_code == 200, await response.aread()
        events = await _collect_sse(response)

    assert state.redactor.counts["EMAIL"] >= 1  # type: ignore[attr-defined]
    chunk_keys: set[str] = set()
    candidate_keys: set[str] = set()
    part_keys: set[str] = set()
    text = ""
    for _name, data in events:
        if not data:
            continue
        payload = json.loads(data)
        chunk_keys |= set(payload)
        for candidate in payload.get("candidates") or []:
            candidate_keys |= set(candidate)
            for part in (candidate.get("content") or {}).get("parts") or []:
                part_keys |= set(part)
                if isinstance(part.get("text"), str) and not part.get("thought"):
                    text += part["text"]

    # Drift detector: Gemini events carry no event names, so key-set drift is
    # the analogue of the Responses adapter's KNOWN_EVENT_TYPES check.
    assert not chunk_keys - KNOWN_CHUNK_KEYS, sorted(chunk_keys - KNOWN_CHUNK_KEYS)
    assert not candidate_keys - KNOWN_CANDIDATE_KEYS, sorted(candidate_keys - KNOWN_CANDIDATE_KEYS)
    assert not part_keys - KNOWN_PART_KEYS, sorted(part_keys - KNOWN_PART_KEYS)
    assert EMAIL in text, f"model did not echo the original email: {text!r}"
    assert "«" not in text
    await client.aclose()


KNOWN_BATCH_RESULT_TYPES = frozenset({"succeeded", "errored", "canceled", "expired"})


@needs_anthropic
async def test_anthropic_batch_round_trip_and_drift() -> None:
    """One-request batch: create redacted, poll briefly, drift-check the
    results line shape. Batches are async — if it does not end within the
    polling budget the test cancels and skips (drift must not flake)."""
    client, _state = _client_and_state()
    headers = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    }
    create = await client.post(
        "/v1/messages/batches",
        headers=headers,
        json={
            "requests": [
                {
                    "custom_id": "drift-1",
                    "params": {
                        "model": "claude-3-5-haiku-latest",
                        "max_tokens": 64,
                        "messages": [{"role": "user", "content": ECHO_PROMPT}],
                    },
                }
            ]
        },
    )
    assert create.status_code == 200, create.text
    batch_id = create.json()["id"]
    ended = False
    try:
        for _ in range(20):
            poll = await client.get(f"/v1/messages/batches/{batch_id}", headers=headers)
            assert poll.status_code == 200, poll.text
            if poll.json().get("processing_status") == "ended":
                ended = True
                break
            await asyncio.sleep(5)
        if not ended:
            pytest.skip("batch did not end within the polling budget")
        results = await client.get(f"/v1/messages/batches/{batch_id}/results", headers=headers)
        assert results.status_code == 200, results.text
        lines = [json.loads(line) for line in results.content.splitlines() if line.strip()]
        assert lines, "empty results stream"
        for line in lines:
            assert {"custom_id", "result"} <= set(line), sorted(line)
            assert line["result"]["type"] in KNOWN_BATCH_RESULT_TYPES
        # The exact tokens the proxy issued are always restorable — their
        # presence in the client-visible results would mean rehydration
        # missed the JSONL path.
        text = json.dumps(lines, ensure_ascii=False)
        assert "\u00abEMAIL_001\u00bb" not in text and "«EMAIL_001»" not in text
    finally:
        if not ended:
            with contextlib.suppress(Exception):
                await client.post(f"/v1/messages/batches/{batch_id}/cancel", headers=headers)
        await client.aclose()


@needs_openai
async def test_openai_files_redaction_round_trip_live() -> None:
    """Upload a batch input file through the proxy, then download it BOTH
    ways: direct from the API (must hold placeholders, never the values)
    and through the proxy (must come back restored)."""
    client, _state = _client_and_state()
    headers = {"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    line = json.dumps(
        {
            "custom_id": "drift-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": ECHO_PROMPT}],
            },
        }
    )
    upload = await client.post(
        "/v1/files",
        headers=headers,
        data={"purpose": "batch"},
        files={"file": ("llm_redact_drift.jsonl", f"{line}\n".encode(), "application/jsonl")},
    )
    assert upload.status_code == 200, upload.text
    file_id = upload.json()["id"]
    try:
        direct = httpx.AsyncClient(base_url="https://api.openai.com", timeout=60.0)
        try:
            raw = await direct.get(f"/v1/files/{file_id}/content", headers=headers)
        finally:
            await direct.aclose()
        if raw.status_code in (400, 403):
            pytest.skip(f"file content download not permitted: {raw.status_code}")
        assert raw.status_code == 200, raw.text
        assert EMAIL not in raw.text and AWS_KEY not in raw.text
        assert "«EMAIL_001»" in raw.text and "«AWS_KEY_001»" in raw.text

        restored = await client.get(f"/v1/files/{file_id}/content", headers=headers)
        assert restored.status_code == 200, restored.text
        got = json.loads(restored.content.splitlines()[0])
        content = got["body"]["messages"][0]["content"]
        assert EMAIL in content and AWS_KEY in content
    finally:
        with contextlib.suppress(Exception):
            await client.delete(f"/v1/files/{file_id}", headers=headers)
        await client.aclose()
