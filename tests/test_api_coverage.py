"""Pins the API coverage matrix: routing per endpoint + doc/table sync.

The table below is the executable twin of docs/api-coverage.md. Every row
asserts what ProxyState-style first-match routing yields for that method
and path — sent with the headers its provider's SDKs send — and which
PROVIDER the request is sent to (a pass-through row included: the upstream
it reaches, end to end). The doc-sync test requires the markdown table and
this table to list exactly the same endpoints with the same classification
— so a route drifting to pass-through, a pass-through drifting to another
provider's upstream, or a doc row going stale fails here.
"""

import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.providers import ALL_ADAPTERS
from llm_redact.providers.base import RouteKind
from llm_redact.proxy import create_app

DOC = Path(__file__).resolve().parent.parent / "docs" / "api-coverage.md"

# (method, path, classification) — classification strings match the doc.
CHAT, REDACT_ONLY, PASS = "chat", "redact-only", "pass-through"
_VX = "/v1/projects/{p}/locations/{l}"  # the Vertex project/location prefix
ANTHROPIC_ROWS: list[tuple[str, str, str]] = [
    ("POST", "/v1/messages", CHAT),
    ("POST", "/v1/messages/count_tokens", REDACT_ONLY),
    ("POST", "/v1/messages/batches", REDACT_ONLY),
    ("GET", "/v1/messages/batches", REDACT_ONLY),
    ("GET", "/v1/messages/batches/{id}", REDACT_ONLY),
    ("GET", "/v1/messages/batches/{id}/results", CHAT),
    ("POST", "/v1/messages/batches/{id}/cancel", REDACT_ONLY),
    ("DELETE", "/v1/messages/batches/{id}", REDACT_ONLY),
    ("GET", "/v1/models", REDACT_ONLY),
    ("GET", "/v1/models/{id}", REDACT_ONLY),
    ("POST", "/v1/complete", CHAT),
    ("POST", "/v1/files", PASS),
    ("GET", "/v1/files/{id}/content", PASS),
    ("POST", "/v1/organizations/probe", PASS),
]
OPENAI_ROWS: list[tuple[str, str, str]] = [
    ("POST", "/v1/chat/completions", CHAT),
    ("GET", "/v1/chat/completions/{id}", CHAT),
    ("POST", "/v1/responses", CHAT),
    ("GET", "/v1/responses/{id}", CHAT),
    ("GET", "/v1/responses/{id}/input_items", CHAT),
    ("DELETE", "/v1/responses/{id}", REDACT_ONLY),
    ("POST", "/v1/responses/compact", CHAT),
    ("POST", "/v1/responses/input_tokens", REDACT_ONLY),
    ("POST", "/v1/conversations", CHAT),
    ("POST", "/v1/conversations/{id}/items", CHAT),
    ("GET", "/v1/conversations/{id}", CHAT),
    ("GET", "/v1/conversations/{id}/items", CHAT),
    ("GET", "/v1/conversations/{id}/items/{item_id}", CHAT),
    ("DELETE", "/v1/conversations/{id}", REDACT_ONLY),
    ("DELETE", "/v1/conversations/{id}/items/{item_id}", REDACT_ONLY),
    ("POST", "/v1/embeddings", REDACT_ONLY),
    ("POST", "/v1/files", CHAT),
    ("GET", "/v1/files", CHAT),
    ("GET", "/v1/files/{id}", CHAT),
    ("GET", "/v1/files/{id}/content", CHAT),
    ("DELETE", "/v1/files/{id}", REDACT_ONLY),
    ("POST", "/v1/batches", CHAT),
    ("GET", "/v1/batches", CHAT),
    ("GET", "/v1/batches/{id}", CHAT),
    ("POST", "/v1/batches/{id}/cancel", CHAT),
    ("GET", "/v1/models", REDACT_ONLY),
    ("GET", "/v1/models/{id}", REDACT_ONLY),
    ("POST", "/v1/completions", CHAT),
    ("POST", "/v1/moderations", PASS),
    ("POST", "/v1/audio/transcriptions", PASS),
    ("POST", "/v1/audio/translations", PASS),
    ("POST", "/v1/audio/speech", REDACT_ONLY),
    ("POST", "/v1/images/generations", REDACT_ONLY),
    ("POST", "/v1/images/edits", REDACT_ONLY),
    ("POST", "/v1/images/variations", PASS),
    ("POST", "/v1/videos", CHAT),
    ("GET", "/v1/videos", CHAT),
    ("GET", "/v1/videos/{id}", CHAT),
    ("POST", "/v1/videos/{id}/remix", CHAT),
    ("GET", "/v1/videos/{id}/content", REDACT_ONLY),
    ("DELETE", "/v1/videos/{id}", REDACT_ONLY),
    ("POST", "/v1/fine_tuning/jobs", PASS),
    ("GET", "/v1/fine_tuning/jobs", PASS),
    ("GET", "/v1/fine_tuning/jobs/{id}", PASS),
    ("POST", "/v1/uploads", PASS),
    ("POST", "/v1/uploads/{id}/parts", PASS),
    ("POST", "/v1/vector_stores", PASS),
    ("POST", "/v1/assistants", PASS),
    ("POST", "/v1/threads/{id}/messages", PASS),
    ("GET", "/v1/containers/{id}/files/{file_id}/content", PASS),
    ("GET", "/v1/evals", PASS),
    ("POST", "/v1/realtime/client_secrets", PASS),
    ("GET", "/v1/organization/probe", PASS),
]
_GM = "/v1beta/models/{m}"  # a Gemini API model
GEMINI_ROWS: list[tuple[str, str, str]] = [
    *[
        ("POST", f"{_GM}:{verb}", kind)
        for verb, kind in (
            ("generateContent", CHAT),
            ("streamGenerateContent", CHAT),
            ("countTokens", REDACT_ONLY),
            ("embedContent", REDACT_ONLY),
            ("batchEmbedContents", REDACT_ONLY),
            ("predict", REDACT_ONLY),
            ("predictLongRunning", REDACT_ONLY),
            ("batchGenerateContent", REDACT_ONLY),
            ("asyncBatchEmbedContent", REDACT_ONLY),
        )
    ],
    ("GET", f"{_GM}/operations/{{id}}", PASS),
    ("GET", "/v1beta/models", REDACT_ONLY),
    ("GET", _GM, REDACT_ONLY),
    ("POST", "/v1beta/cachedContents", REDACT_ONLY),
    ("GET", "/v1beta/cachedContents", PASS),
    ("GET", "/v1beta/cachedContents/{id}", PASS),
    ("PATCH", "/v1beta/cachedContents/{id}", PASS),
    ("DELETE", "/v1beta/cachedContents/{id}", PASS),
    ("GET", "/v1beta/batches", CHAT),
    ("GET", "/v1beta/batches/{id}", CHAT),
    ("POST", "/v1beta/batches/{id}:cancel", REDACT_ONLY),
    ("DELETE", "/v1beta/batches/{id}", REDACT_ONLY),
    ("PATCH", "/v1beta/batches/{id}:updateGenerateContentBatch", PASS),
]
VERTEX_ROWS: list[tuple[str, str, str]] = [
    *[
        ("POST", f"{_VX}/publishers/google/models/{{m}}:{verb}", kind)
        for verb, kind in (
            ("generateContent", CHAT),
            ("streamGenerateContent", CHAT),
            ("countTokens", REDACT_ONLY),
            ("computeTokens", REDACT_ONLY),
            ("embedContent", REDACT_ONLY),
            ("predict", REDACT_ONLY),
            ("predictLongRunning", REDACT_ONLY),
            ("fetchPredictOperation", REDACT_ONLY),
        )
    ],
    ("POST", f"{_VX}/endpoints/{{id}}:generateContent", CHAT),
    ("POST", "/v1/publishers/google/models/{m}:generateContent", CHAT),
    ("POST", f"{_VX}/publishers/anthropic/models/{{m}}:rawPredict", CHAT),
    ("POST", f"{_VX}/publishers/anthropic/models/{{m}}:streamRawPredict", CHAT),
    ("POST", f"{_VX}/publishers/meta/models/{{m}}:rawPredict", PASS),
    ("POST", f"{_VX}/cachedContents", REDACT_ONLY),
    ("GET", f"{_VX}/cachedContents", REDACT_ONLY),
    ("GET", f"{_VX}/cachedContents/{{id}}", REDACT_ONLY),
    ("PATCH", f"{_VX}/cachedContents/{{id}}", REDACT_ONLY),
    ("DELETE", f"{_VX}/cachedContents/{{id}}", REDACT_ONLY),
    ("GET", "/v1beta1/publishers/google/models", REDACT_ONLY),
    ("GET", "/v1beta1/projects/{p}/locations/{l}/publishers/google/models/{m}", REDACT_ONLY),
    ("GET", f"{_VX}/models", REDACT_ONLY),
    ("GET", f"{_VX}/models/{{m}}", REDACT_ONLY),
    ("POST", f"{_VX}/batchPredictionJobs", PASS),
]
AZURE_ROWS: list[tuple[str, str, str]] = [
    ("POST", "/openai/deployments/{d}/chat/completions", CHAT),
    ("POST", "/openai/v1/chat/completions", CHAT),
    ("POST", "/openai/deployments/{d}/completions", CHAT),
    ("POST", "/openai/v1/completions", CHAT),
    ("POST", "/openai/deployments/{d}/embeddings", REDACT_ONLY),
    ("POST", "/openai/v1/embeddings", REDACT_ONLY),
    ("POST", "/openai/deployments/{d}/images/generations", REDACT_ONLY),
    ("POST", "/openai/deployments/{d}/images/edits", REDACT_ONLY),
    ("POST", "/openai/deployments/{d}/audio/speech", REDACT_ONLY),
    ("POST", "/openai/deployments/{d}/audio/transcriptions", PASS),
    ("POST", "/openai/responses", CHAT),
    ("POST", "/openai/v1/responses", CHAT),
    ("GET", "/openai/responses/{id}", CHAT),
    ("GET", "/openai/v1/responses/{id}/input_items", CHAT),
    ("POST", "/openai/v1/responses/{id}/cancel", CHAT),
    ("DELETE", "/openai/responses/{id}", REDACT_ONLY),
    ("POST", "/openai/v1/responses/compact", CHAT),
    ("POST", "/openai/responses/compact", CHAT),
    ("POST", "/openai/v1/responses/input_tokens", REDACT_ONLY),
    ("POST", "/openai/v1/conversations", CHAT),
    ("POST", "/openai/v1/conversations/{id}/items", CHAT),
    ("GET", "/openai/v1/conversations/{id}", CHAT),
    ("GET", "/openai/v1/conversations/{id}/items", CHAT),
    ("DELETE", "/openai/v1/conversations/{id}", REDACT_ONLY),
    ("POST", "/openai/files", CHAT),
    ("POST", "/openai/v1/files", CHAT),
    ("GET", "/openai/files", CHAT),
    ("GET", "/openai/files/{id}", CHAT),
    ("GET", "/openai/v1/files", CHAT),
    ("GET", "/openai/v1/files/{id}", CHAT),
    ("DELETE", "/openai/files/{id}", REDACT_ONLY),
    ("GET", "/openai/files/{id}/content", CHAT),
    ("GET", "/openai/v1/files/{id}/content", CHAT),
    ("POST", "/openai/batches", CHAT),
    ("GET", "/openai/batches", CHAT),
    ("GET", "/openai/v1/batches/{id}", CHAT),
    ("POST", "/openai/batches/{id}/cancel", CHAT),
    ("GET", "/openai/models", REDACT_ONLY),
    ("GET", "/openai/v1/models", REDACT_ONLY),
    ("GET", "/openai/v1/models/{id}", REDACT_ONLY),
    ("GET", "/openai/deployments", REDACT_ONLY),
    ("GET", "/openai/deployments/{d}", REDACT_ONLY),
    ("POST", "/openai/v1/fine_tuning/jobs", PASS),
]
BEDROCK_ROWS: list[tuple[str, str, str]] = [
    ("POST", "/model/{m}/invoke", CHAT),
    ("POST", "/model/{m}/invoke-with-response-stream", CHAT),
    ("POST", "/model/{m}/converse", CHAT),
    ("POST", "/model/{m}/converse-stream", CHAT),
    ("POST", "/model/{m}/count-tokens", REDACT_ONLY),
    ("POST", "/guardrail/{id}/version/{v}/apply", CHAT),
    ("POST", "/async-invoke", REDACT_ONLY),
    ("GET", "/async-invoke", REDACT_ONLY),
    ("GET", "/async-invoke/{id}", REDACT_ONLY),
]

# Each doc section's provider, the upstream it is configured with here, and
# the headers its SDKs send (an Anthropic SDK request always carries
# anthropic-version — the marker the shared /v1/files and /v1/models paths
# are told apart by).
SECTIONS: list[tuple[str, str, dict[str, str], list[tuple[str, str, str]]]] = [
    (
        "anthropic",
        "https://api.anthropic.com",
        {"x-api-key": "sk-ant-api03-test", "anthropic-version": "2023-06-01"},
        ANTHROPIC_ROWS,
    ),
    ("openai", "https://api.openai.com", {"authorization": "Bearer sk-proj-test"}, OPENAI_ROWS),
    (
        "gemini",
        "https://generativelanguage.googleapis.com",
        {"x-goog-api-key": "AIza-test"},
        GEMINI_ROWS,
    ),
    (
        "vertex",
        "https://us-central1-aiplatform.googleapis.com",
        {"authorization": "Bearer ya29.test"},
        VERTEX_ROWS,
    ),
    ("azure", "https://res.openai.azure.com", {"api-key": "azure-test"}, AZURE_ROWS),
    (
        "bedrock",
        "https://bedrock-runtime.us-east-1.amazonaws.com",
        {"authorization": "Bearer ABSK-test"},
        BEDROCK_ROWS,
    ),
]
MATRIX: list[tuple[str, str, str]] = [row for *_, rows in SECTIONS for row in rows]
ROUTED: list[tuple[str, dict[str, str], str, str, str]] = [
    (provider, headers, method, path, classification)
    for provider, _, headers, rows in SECTIONS
    for method, path, classification in rows
]

_EXPECTED_KIND = {
    CHAT: RouteKind.CHAT,
    REDACT_ONLY: RouteKind.REDACT_ONLY,
    PASS: RouteKind.NONE,
}


def _concrete(path: str) -> str:
    return re.sub(r"\{[a-z_]+\}", "abc_123", path)


def _route(method: str, path: str, headers: dict[str, str]) -> tuple[str | None, RouteKind]:
    # First-match semantics, identical to ProxyState.route.
    for adapter in (cls() for cls in ALL_ADAPTERS):
        kind = adapter.matches_request(method, path, headers)
        if kind is not RouteKind.NONE:
            return adapter.name, kind
    return None, RouteKind.NONE


@pytest.mark.parametrize(("provider", "headers", "method", "path", "classification"), ROUTED)
def test_route_matches_matrix(
    provider: str, headers: dict[str, str], method: str, path: str, classification: str
) -> None:
    name, kind = _route(method, _concrete(path), headers)
    assert kind is _EXPECTED_KIND[classification], f"{method} {path} expected {classification}"
    assert name in (None, provider), f"{method} {path} matched {name}'s adapter"


@pytest.mark.parametrize(("method", "path", "classification"), OPENAI_ROWS)
def test_openai_routes_match_under_any_base_path(
    method: str, path: str, classification: str
) -> None:
    """The prefix mixins (custom upstreams, the Gemini API's OpenAI surface)
    try only tails of at most MAX_ENDPOINT_SEGMENTS segments after /v1 —
    twice the deepest OpenAI route, pinned here — so every row routes the
    same under a deep base path, with or without its /v1."""
    from llm_redact.providers.custom import (
        MAX_ENDPOINT_SEGMENTS,
        CustomOpenAIAdapter,
        CustomResponsesAdapter,
    )

    concrete = _concrete(path)
    expected = _EXPECTED_KIND[classification]
    if expected is not RouteKind.NONE:
        assert len(concrete.split("/")) - 2 <= MAX_ENDPOINT_SEGMENTS // 2, path
    adapters = (CustomOpenAIAdapter("lm"), CustomResponsesAdapter("lm"))
    for inner in (concrete, concrete.removeprefix("/v1")):
        target = "/custom/lm" + "/base" * 20 + inner
        kinds = {adapter.matches(method, target) for adapter in adapters} - {RouteKind.NONE}
        assert kinds == ({expected} - {RouteKind.NONE}), f"{method} {target}"


def _config() -> Config:
    providers = dict(Config().providers)
    for provider, base, _, _ in SECTIONS:
        providers[provider] = ProviderConfig(base)
    return Config(providers=providers)


@pytest.mark.parametrize(("provider", "headers", "method", "path", "classification"), ROUTED)
async def test_every_row_reaches_its_own_providers_upstream(
    provider: str, headers: dict[str, str], method: str, path: str, classification: str
) -> None:
    """End to end: each row — pass-through included — is forwarded to its
    section's upstream with the client's own credential, and to no other
    (an unrecognized route once went to the anthropic default)."""
    sent: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    body: dict[str, Any] | None = {} if method in ("POST", "PATCH") else None
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.request(method, _concrete(path), headers=headers, json=body)
    assert response.status_code == 200, response.text
    (request,) = sent
    expected_host = httpx.URL(dict((p, b) for p, b, _, _ in SECTIONS)[provider]).host
    assert request.url.host == expected_host, f"{method} {path} reached {request.url.host}"


# Doc rows deliberately not route-pinned (prose paths, no concrete route).
EXTRA_DOC_ROWS = {("GET", "/v1/organizations/...", PASS), ("GET", "/v1/organization/...", PASS)}
# Table rows probed for routing but expressed as prose in the doc.
TABLE_ONLY_ROWS = {
    ("POST", "/v1/organizations/probe", PASS),
    ("GET", "/v1/organization/probe", PASS),
}


def test_doc_and_matrix_agree() -> None:
    """docs/api-coverage.md rows == this table, both directions."""
    text = DOC.read_text(encoding="utf-8")
    doc_rows: set[tuple[str, str, str]] = set()
    for match in re.finditer(
        r"^\| `(GET|POST|PATCH|DELETE) ([^`]+)`[^|]* \| (chat|redact-only|pass-through) \|",
        text,
        flags=re.M,
    ):
        doc_rows.add((match.group(1), match.group(2).strip(), match.group(3)))
    table_rows = set(MATRIX) - TABLE_ONLY_ROWS
    missing_in_doc = table_rows - doc_rows
    assert not missing_in_doc, f"rows missing from docs/api-coverage.md: {sorted(missing_in_doc)}"
    stale_in_doc = doc_rows - table_rows - EXTRA_DOC_ROWS
    assert not stale_in_doc, f"doc rows not pinned by this table: {sorted(stale_in_doc)}"


# --- WebSocket realtime routes ----------------------------------------------------------

# (path, WS adapter name) — the executable twin of the doc's "WebSocket `…`"
# rows. `{version}` is probed as v1beta.
WS_MATRIX: list[tuple[str, str]] = [
    ("/v1/realtime", "openai-realtime"),
    ("/openai/realtime", "azure-realtime"),
    ("/openai/v1/realtime", "azure-realtime"),
    (
        "/ws/google.ai.generativelanguage.{version}.GenerativeService.BidiGenerateContent",
        "gemini-live",
    ),
    ("/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent", "vertex-live"),
    ("/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent", "vertex-live"),
]


@pytest.mark.parametrize(("path", "adapter_name"), WS_MATRIX)
def test_ws_route_matches_matrix(path: str, adapter_name: str) -> None:
    from llm_redact.realtime import ALL_WS_ADAPTERS, ws_adapter_for

    adapter = ws_adapter_for(
        path.replace("{version}", "v1beta"), [cls() for cls in ALL_WS_ADAPTERS]
    )
    assert adapter is not None and adapter.name == adapter_name


def test_ws_doc_and_matrix_agree() -> None:
    """The doc's WebSocket rows == WS_MATRIX, both directions, and every WS
    adapter has at least one row (a new adapter needs its doc row)."""
    from llm_redact.realtime import ALL_WS_ADAPTERS

    text = DOC.read_text(encoding="utf-8")
    doc_paths = set(re.findall(r"^\| WebSocket `([^`]+)` \| websocket \|", text, flags=re.M))
    assert doc_paths == {path for path, _ in WS_MATRIX}
    assert {cls.name for cls in ALL_WS_ADAPTERS} == {name for _, name in WS_MATRIX}
