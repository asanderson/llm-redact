"""Structural keys skip SCALARS only; opaque user-JSON positions walk everything.

Regression suite for a leak class: jsonwalk used to skip a structural key's
ENTIRE subtree, so a tool result or document that used `data`, `name`,
`id`, `status`, `format`, … as its own keys went upstream unredacted
(Gemini/Vertex functionResponse.response, Vertex Live
toolResponse.functionResponses[].response, Bedrock Converse
toolResult.content[].json, Cohere v2 documents[].data, realtime events).
"""

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
from llm_redact.jsonwalk import STRUCTURAL_KEYS, transform_strings
from llm_redact.providers import (
    AnthropicAdapter,
    BedrockAdapter,
    CohereAdapter,
    GeminiAdapter,
    OllamaAdapter,
)
from llm_redact.proxy import create_app
from llm_redact.realtime import (
    _GEMINI_LIVE_STRUCTURAL_KEYS,
    _REALTIME_STRUCTURAL_KEYS,
    GeminiLiveWs,
    OpenAIRealtimeWs,
    VertexLiveWs,
)
from llm_redact.redactor import Redactor
from llm_redact.rehydrate import Rehydrator
from llm_redact.vault import InMemoryVault

EMAIL = "jane.doe@corp.example"
NO_ALLOW = Allowlist(exact=frozenset(), patterns=())


def _redactor() -> tuple[Redactor, InMemoryVault]:
    vault = InMemoryVault()
    return Redactor(build_detectors(DetectionConfig()), vault, NO_ALLOW), vault


def _mark(s: str) -> str:
    return "<" + s + ">"


# --- jsonwalk semantics --------------------------------------------------------


def test_structural_scalars_still_skipped() -> None:
    obj = {"model": "m", "type": "image", "id": "x", "data": "aGVsbG8=", "text": "t"}
    assert transform_strings(obj, _mark) == {**obj, "text": "<t>"}


@pytest.mark.parametrize("key", sorted(STRUCTURAL_KEYS | _REALTIME_STRUCTURAL_KEYS))
def test_object_and_array_under_a_structural_name_are_walked(key: str) -> None:
    obj = {key: {"note": "a", "inner": ["b"]}, "k2": {key: ["c", {"z": "d"}]}}
    out = transform_strings(obj, _mark, skip_keys=_GEMINI_LIVE_STRUCTURAL_KEYS)
    assert out == {key: {"note": "<a>", "inner": ["<b>"]}, "k2": {key: ["<c>", {"z": "<d>"}]}}


def test_base64_media_data_strings_stay_skipped() -> None:
    anthropic_image = {"type": "image", "source": {"type": "base64", "data": "QUJD"}}
    gemini_inline = {"inlineData": {"mimeType": "image/png", "data": "QUJD"}}
    live_chunk = {"realtimeInput": {"mediaChunks": [{"mimeType": "audio/pcm", "data": "QUJD"}]}}
    assert transform_strings(anthropic_image, _mark) == anthropic_image
    assert transform_strings(gemini_inline, _mark)["inlineData"]["data"] == "QUJD"
    assert transform_strings(live_chunk, _mark, skip_keys=_GEMINI_LIVE_STRUCTURAL_KEYS) == (
        live_chunk
    )
    append = {"type": "input_audio_buffer.append", "audio": "QUJD", "event_id": "e"}
    assert transform_strings(append, _mark, skip_keys=_REALTIME_STRUCTURAL_KEYS) == append


def test_opaque_positions_walk_structural_scalars_too() -> None:
    body = {
        "contents": [
            {
                "parts": [
                    {"functionCall": {"name": "lookup", "args": {"id": "a", "name": "b"}}},
                    {"functionResponse": {"name": "lookup", "response": {"data": "c"}}},
                ]
            }
        ]
    }
    parts = transform_strings(body, _mark)["contents"][0]["parts"]
    assert parts[0]["functionCall"] == {"name": "lookup", "args": {"id": "<a>", "name": "<b>"}}
    assert parts[1]["functionResponse"] == {"name": "lookup", "response": {"data": "<c>"}}


def test_opaque_position_needs_its_parent() -> None:
    # `args` / `json` / `input` are only opaque at their schema positions.
    assert transform_strings({"args": {"id": "a"}}, _mark) == {"args": {"id": "a"}}
    assert transform_strings({"x": {"json": {"id": "a"}}}, _mark) == {"x": {"json": {"id": "a"}}}
    # `documents` is opaque anywhere (Cohere grounding content).
    assert transform_strings({"documents": [{"id": "a"}, "b"]}, _mark) == {
        "documents": [{"id": "<a>"}, "<b>"]
    }


def test_enum_arrays_skipped_only_at_their_positions() -> None:
    session = {"session": {"modalities": ["text", "audio"], "instructions": "i"}}
    assert transform_strings(session, _mark) == {
        "session": {"modalities": ["text", "audio"], "instructions": "<i>"}
    }
    gen = {"generationConfig": {"responseModalities": ["TEXT"]}}
    assert transform_strings(gen, _mark) == gen
    # Elsewhere, or when the array holds objects, the array is walked.
    assert transform_strings({"modalities": ["text"]}, _mark) == {"modalities": ["<text>"]}
    nested = {"session": {"modalities": [{"k": "v"}]}}
    assert transform_strings(nested, _mark) == {"session": {"modalities": [{"k": "<v>"}]}}


# --- the property: any string inside tool-result / document content ----------

_ANY_KEYS = st.sampled_from(
    sorted(_GEMINI_LIVE_STRUCTURAL_KEYS | {"email", "note", "payload", "value", "x"})
)
# Leaves: the secret, inert scalars, or the secret inside a sentence.
_LEAVES = st.one_of(
    st.just(EMAIL),
    st.just(f"contact {EMAIL} today"),
    st.integers(),
    st.booleans(),
    st.none(),
)
_USER_JSON = st.recursive(
    _LEAVES,
    lambda children: st.one_of(
        st.lists(children, min_size=1, max_size=3),
        st.dictionaries(_ANY_KEYS, children, min_size=1, max_size=4),
    ),
    max_leaves=10,
)
# Tool results / documents are JSON OBJECTS at these positions.
_USER_OBJECT = st.dictionaries(_ANY_KEYS, _USER_JSON, min_size=1, max_size=4)


def _gemini(result: Any) -> dict[str, Any]:
    return {
        "contents": [
            {"role": "model", "parts": [{"functionCall": {"name": "f", "args": result}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": result}}]},
        ]
    }


def _bedrock_converse(result: Any) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t", "input": result}}]},
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "t", "content": [{"json": result}]}}],
            },
        ]
    }


def _anthropic(result: Any) -> dict[str, Any]:
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t", "name": "f", "input": result}],
            }
        ],
    }


def _cohere_v2(result: Any) -> dict[str, Any]:
    return {
        "model": "command-r",
        "documents": [{"id": "d1", "data": result}],
        "messages": [
            {
                "role": "tool",
                "tool_call_id": "t",
                "content": [{"type": "document", "document": {"id": "d", "data": result}}],
            }
        ],
    }


def _cohere_v1(result: Any) -> dict[str, Any]:
    return {
        "message": "hi",
        "documents": [result],
        "tool_results": [{"call": {"name": "f", "parameters": result}, "outputs": [result]}],
    }


def _ollama(result: Any) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "f", "arguments": result}}],
            }
        ]
    }


_HTTP_SHAPES = [
    (GeminiAdapter, _gemini),
    (BedrockAdapter, _bedrock_converse),
    (BedrockAdapter, _anthropic),
    (AnthropicAdapter, _anthropic),
    (CohereAdapter, _cohere_v2),
    (CohereAdapter, _cohere_v1),
    (OllamaAdapter, _ollama),
]


@settings(deadline=None, max_examples=150)
@given(result=_USER_OBJECT, which=st.sampled_from(range(len(_HTTP_SHAPES))))
def test_any_string_in_a_tool_result_or_document_is_redacted(result: Any, which: int) -> None:
    adapter_cls, shape = _HTTP_SHAPES[which]
    redactor, vault = _redactor()
    out = adapter_cls().prepare_request(shape(result), redactor, inject_note=False)
    flat = json.dumps(out, ensure_ascii=False)
    assert EMAIL not in flat
    # And it all comes back: the walk is symmetric.
    assert Rehydrator(vault).rehydrate_json(out) == shape(result)


def _live_ctx(redactor: Redactor) -> Any:
    return SimpleNamespace(redactor=redactor)


@settings(deadline=None, max_examples=100)
@given(result=_USER_OBJECT, snake=st.booleans(), vertex=st.booleans())
def test_any_string_in_a_live_tool_response_is_redacted(
    result: Any, snake: bool, vertex: bool
) -> None:
    adapter = VertexLiveWs() if vertex else GeminiLiveWs()
    redactor, _vault = _redactor()
    if snake:
        message = {"tool_response": {"function_responses": [{"id": "c", "response": result}]}}
    else:
        message = {"toolResponse": {"functionResponses": [{"id": "c", "response": result}]}}
    out = adapter.redact_message(json.dumps(message), _live_ctx(redactor))
    assert EMAIL not in (out if isinstance(out, str) else out.decode())


_PLAIN_KEYS = st.sampled_from(["email", "note", "payload", "value", "x"])
_RT_STRUCTURAL = st.sampled_from(sorted(_REALTIME_STRUCTURAL_KEYS))


def _entries(children: st.SearchStrategy[Any]) -> st.SearchStrategy[dict[str, Any]]:
    # Outside an opaque position a structural name still guards a SCALAR,
    # so here it only ever holds an object/array (the leak this pins).
    containers = st.one_of(
        st.lists(children, min_size=1, max_size=3),
        st.dictionaries(_PLAIN_KEYS, children, min_size=1, max_size=3),
    )
    return st.dictionaries(_PLAIN_KEYS, children, max_size=3).flatmap(
        lambda plain: st.dictionaries(_RT_STRUCTURAL, containers, max_size=3).map(
            lambda structural: {**plain, **structural}
        )
    )


_RT_JSON = st.recursive(
    _LEAVES,
    lambda children: st.one_of(st.lists(children, min_size=1, max_size=3), _entries(children)),
    max_leaves=10,
)


@settings(deadline=None, max_examples=100)
@given(result=_RT_JSON, key=_RT_STRUCTURAL)
def test_any_string_in_a_realtime_structural_subtree_is_redacted(result: Any, key: str) -> None:
    # OpenAI Realtime: objects nested under realtime-structural names (the
    # GA session `audio` object, `format`, `status`, …) are walked.
    redactor, _vault = _redactor()
    message = {"type": "session.update", "session": {key: {"user": result}}}
    out = OpenAIRealtimeWs().redact_message(json.dumps(message), _live_ctx(redactor))
    assert EMAIL not in str(out)


def test_realtime_ga_transcription_prompt_redacted() -> None:
    redactor, _vault = _redactor()
    message = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-4o-transcribe", "prompt": f"ask {EMAIL}"},
                },
                "output": {"voice": "marin"},
            },
        },
    }
    out = json.loads(OpenAIRealtimeWs().redact_message(json.dumps(message), _live_ctx(redactor)))
    audio = out["session"]["audio"]
    assert EMAIL not in audio["input"]["transcription"]["prompt"]
    # Enums and identifiers inside the walked object are untouched.
    assert audio["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio["input"]["transcription"]["model"] == "gpt-4o-transcribe"
    assert audio["output"] == {"voice": "marin"}
    assert out["session"]["output_modalities"] == ["audio"]


# --- end to end through the real app --------------------------------------------


class _Capture:
    def __init__(self) -> None:
        self.bodies: list[bytes] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        return httpx.Response(200, json={"ok": True})


async def _post(config: Config, path: str, body: dict[str, Any]) -> bytes:
    capture = _Capture()
    app = create_app(config, upstream_transport=httpx.MockTransport(capture))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200
    return capture.bodies[0]


def _providers(**extra: ProviderConfig) -> Config:
    return Config(providers={**Config().providers, **extra})


# Each probe hides the secret under a structural-named user key.
_RESULT = {"data": {"owner": EMAIL}, "name": EMAIL, "status": {"note": EMAIL}, "id": [EMAIL]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "path", "body"),
    [
        (
            _providers(),
            "/v1beta/models/gemini-2.0-flash:generateContent",
            _gemini(_RESULT),
        ),
        (
            _providers(vertex=ProviderConfig("https://us-east5-aiplatform.googleapis.com")),
            "/v1/projects/p/locations/us-east5/publishers/google/models/gemini-2.0:generateContent",
            _gemini(_RESULT),
        ),
        (
            _providers(bedrock=ProviderConfig("https://bedrock-runtime.us-east-1.amazonaws.com")),
            "/model/amazon.nova-pro-v1%3A0/converse",
            _bedrock_converse(_RESULT),
        ),
        (_providers(), "/v2/chat", _cohere_v2(_RESULT)),
        (_providers(), "/v1/chat", _cohere_v1(_RESULT)),
    ],
    ids=["gemini", "vertex", "bedrock-converse", "cohere-v2", "cohere-v1"],
)
async def test_tool_results_redacted_end_to_end(
    config: Config, path: str, body: dict[str, Any]
) -> None:
    sent = await _post(config, path, body)
    assert EMAIL.encode() not in sent
    assert b"\xc2\xabEMAIL_" in sent  # «EMAIL_…» placeholders took their place
