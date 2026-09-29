"""Maps whose KEYS the caller chooses are user data, never protocol.

jsonwalk skips a scalar under a structural name (``id``, ``name``, ``type``,
``data``, …) because those names are protocol elsewhere. Inside a map whose
keys the caller picks — OpenAI/Azure/Anthropic ``metadata``, Bedrock
Converse ``requestMetadata``, Responses/Realtime ``prompt.variables``,
Vertex/Gemini ``:predict`` ``instances``/``parameters`` (arbitrary custom
model input) — a key named ``id`` or ``data`` is the caller's own label,
and its value went upstream unredacted (signed with the proxy's identity
under ``auth = "identity"``). Those positions are opaque: every string
below them is walked, in both directions (the echoes restore it).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from llm_redact.config import Config, ProviderConfig
from llm_redact.detection.engine import Allowlist, DetectionConfig, build_detectors
from llm_redact.jsonwalk import STRUCTURAL_KEYS, transform_strings
from llm_redact.proxy import create_app
from llm_redact.realtime import (
    _GEMINI_LIVE_STRUCTURAL_KEYS,
    _REALTIME_STRUCTURAL_KEYS,
    OpenAIRealtimeWs,
)
from llm_redact.redactor import Redactor
from llm_redact.registry import Registry
from llm_redact.rehydrate import RehydratorPool
from llm_redact.vault import InMemoryVault
from test_openai_batches import PREFIXES, BatchStore
from test_upstream_auth import AZURE, VERTEX, _config, _identity, _install

EMAIL = "jane.doe@corp.example"
UPSTREAM = "https://upstream.test"
# Every name a skip set guards, HTTP and realtime alike.
SKIP_NAMES = sorted(STRUCTURAL_KEYS | _REALTIME_STRUCTURAL_KEYS | _GEMINI_LIVE_STRUCTURAL_KEYS)


def _mark(s: str) -> str:
    return "<" + s + ">"


def _labels() -> dict[str, str]:
    """A caller-keyed map using every skip name as its own key."""
    return {name: f"{name} {EMAIL}" for name in SKIP_NAMES}


# --- jsonwalk semantics ------------------------------------------------------------


@pytest.mark.parametrize("key", SKIP_NAMES)
def test_every_skip_name_is_walked_inside_a_caller_keyed_map(key: str) -> None:
    for skip in (STRUCTURAL_KEYS, _GEMINI_LIVE_STRUCTURAL_KEYS):
        body = {
            "metadata": {key: "a"},
            "nested": {"metadata": {key: {"deeper": "b"}}},
            "requestMetadata": {key: "c"},
            "prompt": {"id": "pmpt_1", "variables": {key: "d"}},
            "instances": [{key: "e", "image": {key: "f"}}],
            "parameters": {key: "g"},
        }
        assert transform_strings(body, _mark, skip_keys=skip) == {
            "metadata": {key: "<a>"},
            "nested": {"metadata": {key: {"deeper": "<b>"}}},
            "requestMetadata": {key: "<c>"},
            # `prompt.id` is the stored prompt's id — still protocol.
            "prompt": {"id": "pmpt_1", "variables": {key: "<d>"}},
            "instances": [{key: "<e>", "image": {key: "<f>"}}],
            "parameters": {key: "<g>"},
        }


def test_positional_maps_need_their_position() -> None:
    # `variables` only under `prompt`; `instances`/`parameters` only at the
    # top of a body (the :predict request shape).
    for body in (
        {"variables": {"id": "a"}},
        {"x": {"instances": [{"id": "a"}]}},
        {"x": {"parameters": {"id": "a"}}},
        {"tool": {"prompt": {"id": "a"}}},
    ):
        assert transform_strings(body, _mark) == body


# --- end to end ----------------------------------------------------------------------


class _Echo:
    """Records each request and answers with its own JSON body."""

    def __init__(self) -> None:
        self.bodies: list[bytes] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        return httpx.Response(
            200, content=request.content or b"{}", headers={"content-type": "application/json"}
        )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


@pytest.mark.parametrize(("prefix", "provider"), PREFIXES)
async def test_batch_metadata_under_any_key_is_redacted_and_restored(
    prefix: str, provider: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    store = BatchStore()
    providers = {**Config().providers, provider: ProviderConfig(UPSTREAM)}
    app = create_app(Config(providers=providers), upstream_transport=httpx.MockTransport(store))
    body = {
        "input_file_id": "file-abc123",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": _labels(),
    }
    async with _client(app) as client:
        created = await client.post(f"{prefix}/batches", json=body)
        fetched = await client.get(f"{prefix}/batches/{created.json()['id']}")
    assert all(EMAIL.encode() not in sent for _, _, sent in store.received)
    for response in (created, fetched):
        assert response.status_code == 200
        assert response.json()["metadata"] == _labels()


async def test_azure_batch_metadata_signed_redacted_under_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Echo()
    app = create_app(
        _config(azure=_identity(AZURE)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = {
        "input_file_id": "file-abc123",
        "endpoint": "/chat/completions",
        "completion_window": "24h",
        "metadata": _labels(),
    }
    async with _client(app) as client:
        response = await client.post("/openai/batches?api-version=1", json=body)
    assert response.status_code == 200
    (sent,) = upstream.bodies
    assert EMAIL.encode() not in sent
    assert built[0].calls[0][3] == sent
    assert response.json()["metadata"] == _labels()


async def test_vertex_predict_instances_signed_redacted_under_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Echo()
    app = create_app(
        _config(vertex=_identity(VERTEX)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = {
        "instances": [{"name": EMAIL, "id": EMAIL, "text": f"mail {EMAIL}"}],
        "parameters": {"name": EMAIL},
    }
    async with _client(app) as client:
        response = await client.post(
            "/v1/projects/p/locations/us-east5/endpoints/123:predict", json=body
        )
    assert response.status_code == 200
    (sent,) = upstream.bodies
    assert EMAIL.encode() not in sent
    assert built[0].calls[0][3] == sent


@pytest.mark.parametrize(
    ("config", "path", "body"),
    [
        (
            Config(),
            "/v1/responses",
            {"model": "gpt-4.1", "prompt": {"id": "pmpt_1", "variables": _labels()}},
        ),
        (
            Config(),
            "/v1/chat/completions",
            {
                "model": "gpt-4.1",
                "store": True,
                "metadata": _labels(),
                "messages": [{"role": "user", "content": "hi"}],
            },
        ),
        (
            Config(),
            "/v1/messages",
            {
                "model": "claude-x",
                "max_tokens": 8,
                "metadata": {"user_id": EMAIL},
                "messages": [{"role": "user", "content": "hi"}],
            },
        ),
        (
            Config(
                providers={
                    **Config().providers,
                    "bedrock": ProviderConfig("https://bedrock-runtime.us-east-1.amazonaws.com"),
                }
            ),
            "/model/amazon.nova-pro-v1%3A0/converse",
            {
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "requestMetadata": _labels(),
            },
        ),
        (
            Config(),
            "/v1beta/models/imagen-3.0-generate-002:predict",
            {"instances": [{"prompt": "a cat", "name": EMAIL}], "parameters": {"id": EMAIL}},
        ),
    ],
    ids=[
        "responses-prompt-variables",
        "chat-metadata",
        "anthropic-metadata",
        "converse",
        "gemini-predict",
    ],
)
async def test_caller_keyed_maps_redacted_end_to_end(
    config: Config, path: str, body: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    upstream = _Echo()
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200
    (sent,) = upstream.bodies
    assert EMAIL.encode() not in sent
    assert "«EMAIL_".encode() in sent


# --- realtime -----------------------------------------------------------------------


def _ctx() -> tuple[Any, InMemoryVault]:
    vault = InMemoryVault()
    redactor = Redactor(
        build_detectors(DetectionConfig()), vault, Allowlist(exact=frozenset(), patterns=())
    )
    return SimpleNamespace(redactor=redactor), vault


def test_realtime_metadata_and_prompt_variables_redacted_and_restored() -> None:
    ctx, vault = _ctx()
    adapter = OpenAIRealtimeWs()
    create = {"type": "response.create", "response": {"metadata": _labels()}}
    update = {
        "type": "session.update",
        "session": {"type": "realtime", "prompt": {"id": "pmpt_1", "variables": _labels()}},
    }
    for event in (create, update):
        out = adapter.redact_message(json.dumps(event), ctx)
        assert EMAIL not in str(out)
    redacted = json.loads(adapter.redact_message(json.dumps(create), ctx))
    done = {"type": "response.done", "response": redacted["response"]}
    (frame,) = adapter.rehydrate_message(json.dumps(done), RehydratorPool(vault))
    assert json.loads(frame)["response"]["metadata"] == _labels()
