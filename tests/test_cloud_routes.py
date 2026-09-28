"""The cloud providers' non-chat routes: Vertex AI, Azure OpenAI, Bedrock.

A provider configured ``auth = "identity"`` forwards ONLY the routes an
adapter recognizes (anything else is a recorded local 403), and an
unrecognized route of a key-authorized provider is forwarded UNREDACTED.
These tests pin the endpoints recognized beyond inference — content-bearing
ones redacted (and restored where the response echoes content), metadata
reads recognized as body-less REDACT_ONLY no-ops — and prove:

- every new matcher is claimed by exactly one adapter, of the right
  provider (Vertex vs Gemini vs Claude-on-Vertex; Azure chat vs Azure
  Responses vs OpenAI), and no cloud adapter claims another provider's
  route (custom providers included);
- the system note never touches these bodies;
- under identity auth each is now signed and sent, while a still
  unrecognized path keeps its 403.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llm_redact.providers import ALL_ADAPTERS
from llm_redact.providers.azure_openai import AzureOpenAIAdapter, AzureResponsesAdapter
from llm_redact.providers.base import RouteKind
from llm_redact.providers.bedrock import BedrockAdapter
from llm_redact.providers.custom import build_custom_adapters
from llm_redact.providers.gemini import GeminiAdapter
from llm_redact.providers.vertex import VertexAdapter
from llm_redact.proxy import ProxyState, create_app
from test_api_coverage import MATRIX
from test_upstream_auth import (
    AZURE,
    BEDROCK,
    CLIENT_CREDENTIALS,
    EMAIL,
    TOKEN,
    VERTEX,
    _client,
    _config,
    _identity,
    _install,
    _Upstream,
)

VX = "/v1/projects/p/locations/us-central1"
CHAT, REDACT, NONE = RouteKind.CHAT, RouteKind.REDACT_ONLY, RouteKind.NONE

# (method, concrete path, provider, kind): every route this survey added.
NEW_ROUTES: list[tuple[str, str, str, RouteKind]] = [
    # Vertex
    ("POST", f"{VX}/cachedContents", "vertex", REDACT),
    ("POST", f"/v1beta1{VX[3:]}/cachedContents", "vertex", REDACT),
    ("GET", f"{VX}/cachedContents", "vertex", REDACT),
    ("GET", f"{VX}/cachedContents/c1", "vertex", REDACT),
    ("PATCH", f"{VX}/cachedContents/c1", "vertex", REDACT),
    ("DELETE", f"{VX}/cachedContents/c1", "vertex", REDACT),
    ("POST", f"{VX}/publishers/google/models/gemini-2.5:computeTokens", "vertex", REDACT),
    ("POST", f"{VX}/endpoints/123:computeTokens", "vertex", REDACT),
    ("POST", f"{VX}/publishers/google/models/gemini-embedding:embedContent", "vertex", REDACT),
    ("POST", f"{VX}/publishers/google/models/veo-3:fetchPredictOperation", "vertex", REDACT),
    ("GET", "/v1beta1/publishers/google/models", "vertex", REDACT),
    ("GET", "/v1/publishers/google/models/gemini-2.5-pro", "vertex", REDACT),
    ("GET", f"{VX}/publishers/google/models", "vertex", REDACT),
    ("GET", f"{VX}/publishers/google/models/gemini-2.5-pro", "vertex", REDACT),
    ("GET", f"{VX}/publishers/anthropic/models/claude-x", "vertex", REDACT),
    ("GET", f"{VX}/models", "vertex", REDACT),
    ("GET", f"{VX}/models/tuned-1", "vertex", REDACT),
    # Azure OpenAI
    ("POST", "/openai/deployments/d/completions", "azure", CHAT),
    ("POST", "/openai/v1/completions", "azure", CHAT),
    ("POST", "/openai/deployments/d/images/generations", "azure", REDACT),
    ("POST", "/openai/v1/images/generations", "azure", REDACT),
    ("POST", "/openai/deployments/d/images/edits", "azure", REDACT),
    ("POST", "/openai/deployments/d/audio/speech", "azure", REDACT),
    ("POST", "/openai/v1/files", "azure", REDACT),
    ("GET", "/openai/files", "azure", REDACT),
    ("GET", "/openai/v1/files/file-1", "azure", REDACT),
    ("DELETE", "/openai/files/file-1", "azure", REDACT),
    ("GET", "/openai/v1/files/file-1/content", "azure", CHAT),
    ("POST", "/openai/batches", "azure", CHAT),
    ("POST", "/openai/v1/batches", "azure", CHAT),
    ("GET", "/openai/batches", "azure", REDACT),
    ("GET", "/openai/batches/batch_1", "azure", CHAT),
    ("POST", "/openai/v1/batches/batch_1/cancel", "azure", CHAT),
    ("GET", "/openai/models", "azure", REDACT),
    ("GET", "/openai/models/gpt-4o", "azure", REDACT),
    ("GET", "/openai/v1/models", "azure", REDACT),
    ("GET", "/openai/deployments", "azure", REDACT),
    ("GET", "/openai/deployments/d", "azure", REDACT),
    ("POST", "/openai/v1/conversations", "azure", CHAT),
    ("POST", "/openai/v1/conversations/conv_1/items", "azure", CHAT),
    ("GET", "/openai/v1/conversations/conv_1", "azure", CHAT),
    ("GET", "/openai/v1/conversations/conv_1/items/item_1", "azure", CHAT),
    ("DELETE", "/openai/v1/conversations/conv_1", "azure", REDACT),
    ("DELETE", "/openai/v1/conversations/conv_1/items/item_1", "azure", REDACT),
    ("POST", "/openai/responses/resp_1/cancel", "azure", CHAT),
    ("DELETE", "/openai/v1/responses/resp_1", "azure", REDACT),
    # Bedrock runtime
    ("POST", "/model/anthropic.claude-v2/count-tokens", "bedrock", REDACT),
    ("POST", "/guardrail/gr1abc/version/1/apply", "bedrock", CHAT),
    ("POST", "/guardrail/gr1abc/version/DRAFT/apply", "bedrock", CHAT),
    # Decoded ARN identifiers carry a slash; the ids stay greedy.
    (
        "POST",
        "/guardrail/arn:aws:bedrock:us-east-1:123456789012:guardrail/gr1abc/version/2/apply",
        "bedrock",
        CHAT,
    ),
    ("POST", "/async-invoke", "bedrock", REDACT),
    ("GET", "/async-invoke", "bedrock", REDACT),
    (
        "GET",
        "/async-invoke/arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc",
        "bedrock",
        REDACT,
    ),
]

# Near-miss paths that must stay unrecognized (pass-through, or 403 under
# identity auth).
STILL_UNMATCHED: list[tuple[str, str]] = [
    ("GET", f"{VX}/publishers/google/models/gemini:generateContent"),  # a verb is no GET
    ("GET", f"{VX}/models/m:predict"),
    ("POST", f"{VX}/models"),
    ("PUT", f"{VX}/cachedContents/c1"),
    ("PATCH", f"{VX}/cachedContents"),
    ("GET", "/v1/cachedContents/c1"),  # Gemini API form: pass-through there
    ("POST", f"{VX}/publishers/meta/models/llama:rawPredict"),
    ("POST", f"{VX}/publishers/google/models/g:serverStreamingPredict"),
    ("POST", f"{VX}/batchPredictionJobs"),
    ("GET", "/openai/deployments/d/chat/completions"),
    ("GET", "/openai/v1/deployments"),
    ("POST", "/openai/deployments/d/audio/transcriptions"),
    ("POST", "/openai/v1/threads"),
    ("POST", "/openai/v1/vector_stores/vs_1/search"),
    ("POST", "/openai/v1/files/file-1"),
    ("PUT", "/openai/v1/conversations/conv_1"),
    ("GET", "/guardrail/gr1/version/1"),
    ("GET", "/guardrail/gr1/version/1/apply"),
    ("POST", "/async-invoke/arn"),
    ("DELETE", "/async-invoke"),
    ("POST", "/model/m/count-tokens/extra"),
]


def _claims(method: str, path: str) -> list[str]:
    adapters = [cls() for cls in ALL_ADAPTERS]
    return [type(a).__name__ for a in adapters if a.matches(method, path) is not NONE]


def _first(method: str, path: str) -> tuple[str | None, RouteKind]:
    for adapter in (cls() for cls in ALL_ADAPTERS):
        kind = adapter.matches(method, path)
        if kind is not NONE:
            return adapter.name, kind
    return None, NONE


@pytest.mark.parametrize(("method", "path", "provider", "kind"), NEW_ROUTES)
def test_each_new_route_has_exactly_one_owner(
    method: str, path: str, provider: str, kind: RouteKind
) -> None:
    assert len(_claims(method, path)) == 1, _claims(method, path)
    assert _first(method, path) == (provider, kind)


@pytest.mark.parametrize(("method", "path"), STILL_UNMATCHED)
def test_near_misses_stay_unrecognized(method: str, path: str) -> None:
    assert _claims(method, path) == []


def test_matchers_pairwise_disjoint_across_the_whole_matrix() -> None:
    """No two adapters claim any pinned route — the coverage matrix plus the
    survey's own rows — so first-match order never decides a route."""
    rows = [(m, p.replace("{", "").replace("}", "")) for m, p, _ in MATRIX]
    rows += [(m, p) for m, p, _, _ in NEW_ROUTES]
    for method, path in rows:
        assert len(_claims(method, path)) <= 1, (method, path, _claims(method, path))


def test_cloud_adapters_never_claim_other_providers_routes() -> None:
    """The unrouted behavior of every other provider is unchanged: no cloud
    adapter claims an Anthropic/OpenAI/Gemini route, nor a custom
    provider's prefixed twin of its own routes."""
    cloud = [VertexAdapter(), AzureOpenAIAdapter(), AzureResponsesAdapter(), BedrockAdapter()]
    others = [
        (m, p)
        for m, p, _ in MATRIX
        if p.startswith("/v1/") and "projects" not in p and "publishers" not in p
    ]
    others += [
        ("POST", "/v1beta/models/g:generateContent"),
        ("POST", "/v1beta/cachedContents"),
        ("GET", "/v1beta/models"),
        ("GET", "/v1/models/gpt-4o"),
    ]
    for method, path in others:
        path = path.replace("{", "").replace("}", "")
        assert all(a.matches(method, path) is NONE for a in cloud), (method, path)
    customs = build_custom_adapters(["custom:lm"])
    for method, path, _, _ in NEW_ROUTES:
        prefixed = f"/custom/lm{path}"
        assert all(a.matches(method, prefixed) is NONE for a in cloud), prefixed
        # A custom upstream re-anchors at /v1/ and may claim an OpenAI-shaped
        # tail, but it never claims the bare cloud path itself.
        assert all(a.matches(method, path) is NONE for a in customs), path
    # The Gemini API adapter never claims a Vertex route, and vice versa.
    gemini = GeminiAdapter()
    for method, path, provider, _ in NEW_ROUTES:
        if provider == "vertex":
            assert gemini.matches(method, path) is NONE, path
    assert VertexAdapter().matches("POST", "/v1beta/cachedContents") is NONE


def test_the_system_note_never_touches_the_new_bodies() -> None:
    adapters = {
        "vertex": VertexAdapter(),
        "azure": AzureOpenAIAdapter(),
        "bedrock": BedrockAdapter(),
    }
    responses = AzureResponsesAdapter()
    for method, path, provider, kind in NEW_ROUTES:
        adapter = adapters[provider]
        if provider == "azure" and "/responses" in path:
            continue  # the Responses adapter: cancel carries no body
        if provider == "azure" and path.endswith("/files"):
            # Uploads inject per chat-shaped JSONL line, as on /v1/files.
            assert adapter.wants_system_note(kind, path)
            continue
        assert not adapter.wants_system_note(kind, path), (method, path)
    # The chat routes keep their note.
    assert adapters["azure"].wants_system_note(CHAT, "/openai/v1/chat/completions")
    assert adapters["bedrock"].wants_system_note(CHAT, "/model/m/converse")
    assert adapters["vertex"].wants_system_note(CHAT, f"{VX}/publishers/g/models/m:generateContent")
    assert responses.wants_system_note(CHAT, "/openai/v1/responses")


def test_azure_file_content_on_the_v1_family_is_restored() -> None:
    from llm_redact.rehydrate import Rehydrator
    from llm_redact.vault import InMemoryVault

    vault = InMemoryVault()
    token = vault.placeholder_for("EMAIL", EMAIL)
    line = json.dumps({"response": {"body": {"content": f"hi {token}"}}}).encode()
    out = AzureOpenAIAdapter().rehydrate_raw_body(
        "/openai/v1/files/file-1/content", line, Rehydrator(vault)
    )
    assert out is not None and EMAIL.encode() in out


# --- end to end under identity auth -------------------------------------------------

_TEXT = {"text": f"mail {EMAIL}"}
IDENTITY_CASES: list[tuple[str, str, str, str, Any]] = [
    # (provider, base, method, path, JSON body or None)
    ("vertex", VERTEX, "POST", f"{VX}/cachedContents", {"contents": [{"parts": [_TEXT]}]}),
    ("vertex", VERTEX, "GET", f"{VX}/cachedContents/c1", None),
    ("vertex", VERTEX, "PATCH", f"{VX}/cachedContents/c1", {"ttl": "60s"}),
    ("vertex", VERTEX, "DELETE", f"{VX}/cachedContents/c1", None),
    (
        "vertex",
        VERTEX,
        "POST",
        f"{VX}/publishers/google/models/g:computeTokens",
        {"contents": [{"parts": [_TEXT]}]},
    ),
    (
        "vertex",
        VERTEX,
        "POST",
        f"{VX}/publishers/google/models/e:embedContent",
        {"content": {"parts": [_TEXT]}},
    ),
    ("vertex", VERTEX, "GET", "/v1beta1/publishers/google/models", None),
    ("vertex", VERTEX, "GET", f"{VX}/models/tuned-1", None),
    ("azure", AZURE, "POST", "/openai/deployments/d/completions", {"prompt": f"mail {EMAIL}"}),
    ("azure", AZURE, "POST", "/openai/v1/images/generations", {"prompt": f"draw {EMAIL}"}),
    ("azure", AZURE, "POST", "/openai/v1/conversations", {"items": [{"content": EMAIL}]}),
    ("azure", AZURE, "GET", "/openai/v1/models", None),
    ("azure", AZURE, "GET", "/openai/deployments", None),
    ("azure", AZURE, "GET", "/openai/files", None),
    ("azure", AZURE, "GET", "/openai/batches/batch_1", None),
    ("azure", AZURE, "DELETE", "/openai/responses/resp_1", None),
    (
        "bedrock",
        BEDROCK,
        "POST",
        "/model/anthropic.claude-v2/count-tokens",
        {"input": {"converse": {"messages": [{"role": "user", "content": [_TEXT]}]}}},
    ),
    (
        "bedrock",
        BEDROCK,
        "POST",
        "/guardrail/gr1/version/1/apply",
        {"source": "INPUT", "content": [{"text": _TEXT}]},
    ),
    (
        "bedrock",
        BEDROCK,
        "POST",
        "/async-invoke",
        {"modelId": "amazon.nova-reel-v1:0", "modelInput": {"textToVideoParams": _TEXT}},
    ),
    ("bedrock", BEDROCK, "GET", "/async-invoke", None),
]


@pytest.mark.parametrize(("provider", "base", "method", "path", "body"), IDENTITY_CASES)
async def test_identity_auth_signs_the_newly_recognized_routes(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    base: str,
    method: str,
    path: str,
    body: Any,
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: _identity(base)}), upstream_transport=httpx.MockTransport(upstream)
    )
    content = json.dumps(body).encode() if body is not None else None
    headers = {**CLIENT_CREDENTIALS}
    if content is not None:
        headers["content-type"] = "application/json"
    async with _client(app) as client:
        response = await client.request(method, path, content=content, headers=headers)
    assert response.status_code == 200, response.text
    (auth,) = built
    ((seen_method, seen_url, _, signed),) = auth.calls
    (sent,) = upstream.requests
    assert seen_method == method and str(sent.url) == seen_url == f"{base}{path}"
    assert sent.headers["authorization"] == f"Proxy {provider}"
    assert sent.content == signed  # exactly the authorized bytes were sent
    assert EMAIL.encode() not in sent.content  # content routes redacted
    assert b"character for character" not in sent.content  # never a note here
    if body is not None and EMAIL in json.dumps(body):
        assert TOKEN.encode() in sent.content


@pytest.mark.parametrize(
    ("provider", "base", "method", "path"),
    [
        ("vertex", VERTEX, "POST", f"{VX}/batchPredictionJobs"),
        ("vertex", VERTEX, "GET", f"{VX}/endpoints"),
        ("azure", AZURE, "POST", "/openai/v1/threads"),
        ("azure", AZURE, "POST", "/openai/deployments/d/audio/transcriptions"),
        ("bedrock", BEDROCK, "GET", "/guardrail/gr1/version/1"),
        ("bedrock", BEDROCK, "DELETE", "/async-invoke"),
    ],
)
async def test_an_unrecognized_path_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, provider: str, base: str, method: str, path: str
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: _identity(base)}), upstream_transport=httpx.MockTransport(upstream)
    )
    state: ProxyState = app.state.proxy
    async with _client(app) as client:
        response = await client.request(method, path, headers={"x-api-key": "k"})
    assert response.status_code == 403
    assert 'auth = "identity"' in response.json()["error"]
    assert built[0].calls == [] and upstream.requests == []
    assert state.recent[-1]["status"] == 403


async def test_an_encoded_guardrail_arn_is_matched_and_forwarded_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(bedrock=_identity(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    arn = "arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Aguardrail%2Fgr1"
    raw = f"/guardrail/{arn}/version/3/apply"
    async with _client(app) as client:
        response = await client.post(raw, json={"source": "OUTPUT", "content": [{"text": _TEXT}]})
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert sent.url.raw_path.decode() == raw  # the client's encoding, verbatim
    assert EMAIL.encode() not in sent.content


async def test_apply_guardrail_restores_its_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """ApplyGuardrail is CHAT: the guardrail's rewritten text and quoted
    matches carry the placeholders sent up; the client gets its own text."""
    _install(monkeypatch)
    answer = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": f"contact {TOKEN} redacted by policy"}],
        "assessments": [{"wordPolicy": {"customWords": [{"match": TOKEN, "action": "BLOCKED"}]}}],
    }
    upstream = _Upstream(json.dumps(answer).encode())
    app = create_app(
        _config(bedrock=_identity(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = {"source": "INPUT", "content": [{"text": {"text": f"contact {EMAIL}"}}]}
    async with _client(app) as client:
        response = await client.post("/guardrail/gr1/version/1/apply", json=body)
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert EMAIL.encode() not in sent.content and TOKEN.encode() in sent.content
    out = response.json()
    assert out["outputs"][0]["text"] == f"contact {EMAIL} redacted by policy"
    assert out["assessments"][0]["wordPolicy"]["customWords"][0]["match"] == EMAIL


async def test_azure_batch_metadata_round_trips_and_keyed_setups_forward_body_less_gets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Key-authorized (passthrough) Azure: a recognized metadata GET forwards
    exactly what pass-through did, client credential included; a batch's
    user metadata is redacted out and restored in the echo."""
    from llm_redact.config import ProviderConfig

    _install(monkeypatch)
    upstream = _Upstream()

    def respond(request: httpx.Request) -> httpx.Response:
        upstream.requests.append(request)
        if request.method == "POST":
            echo = json.loads(request.content)
            return httpx.Response(200, json={"id": "batch_1", **echo})
        return httpx.Response(200, json={"data": [], "object": "list"})

    app = create_app(
        _config(azure=ProviderConfig(AZURE)), upstream_transport=httpx.MockTransport(respond)
    )
    async with _client(app) as client:
        listing = await client.get(
            "/openai/models?api-version=2024-10-21", headers={"api-key": "client-azure-key"}
        )
        created = await client.post(
            "/openai/batches?api-version=2024-10-21",
            json={"input_file_id": "file-1", "metadata": {"owner": EMAIL}},
            headers={"api-key": "client-azure-key"},
        )
    assert listing.status_code == 200 and listing.json() == {"data": [], "object": "list"}
    first, second = upstream.requests
    assert first.headers["api-key"] == "client-azure-key" and first.content == b""
    assert str(first.url) == f"{AZURE}/openai/models?api-version=2024-10-21"
    assert EMAIL.encode() not in second.content
    assert b"character for character" not in second.content  # no note in a batch body
    assert created.json()["metadata"] == {"owner": EMAIL}
