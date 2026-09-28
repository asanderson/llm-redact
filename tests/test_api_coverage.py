"""Pins the API coverage matrix: routing per endpoint + doc/table sync.

The table below is the executable twin of docs/api-coverage.md. Every row
asserts what ProxyState-style first-match routing yields for that method
and path, and the doc-sync test requires the markdown table and this table
to list exactly the same endpoints with the same classification — so a
route drifting to pass-through (or a doc row going stale) fails here.
"""

import re
from pathlib import Path

import pytest

from llm_redact.providers import ALL_ADAPTERS
from llm_redact.providers.base import RouteKind

DOC = Path(__file__).resolve().parent.parent / "docs" / "api-coverage.md"

# (method, path, classification) — classification strings match the doc.
CHAT, REDACT_ONLY, PASS = "chat", "redact-only", "pass-through"
_VX = "/v1/projects/{p}/locations/{l}"  # the Vertex project/location prefix
MATRIX: list[tuple[str, str, str]] = [
    # Anthropic
    ("POST", "/v1/messages", CHAT),
    ("POST", "/v1/messages/count_tokens", REDACT_ONLY),
    ("POST", "/v1/messages/batches", REDACT_ONLY),
    ("GET", "/v1/messages/batches", PASS),
    ("GET", "/v1/messages/batches/{id}", PASS),
    ("GET", "/v1/messages/batches/{id}/results", CHAT),
    ("POST", "/v1/messages/batches/{id}/cancel", PASS),
    ("DELETE", "/v1/messages/batches/{id}", PASS),
    ("GET", "/v1/models", PASS),
    ("GET", "/v1/models/{id}", PASS),
    ("POST", "/v1/complete", CHAT),
    ("POST", "/v1/organizations/probe", PASS),
    # OpenAI
    ("POST", "/v1/chat/completions", CHAT),
    ("GET", "/v1/chat/completions/{id}", CHAT),
    ("POST", "/v1/responses", CHAT),
    ("GET", "/v1/responses/{id}", CHAT),
    ("GET", "/v1/responses/{id}/input_items", CHAT),
    ("DELETE", "/v1/responses/{id}", PASS),
    ("POST", "/v1/conversations", CHAT),
    ("POST", "/v1/conversations/{id}/items", CHAT),
    ("GET", "/v1/conversations/{id}", CHAT),
    ("GET", "/v1/conversations/{id}/items", CHAT),
    ("GET", "/v1/conversations/{id}/items/{item_id}", CHAT),
    ("DELETE", "/v1/conversations/{id}", PASS),
    ("POST", "/v1/embeddings", REDACT_ONLY),
    ("POST", "/v1/files", REDACT_ONLY),
    ("GET", "/v1/files", PASS),
    ("GET", "/v1/files/{id}", PASS),
    ("GET", "/v1/files/{id}/content", CHAT),
    ("DELETE", "/v1/files/{id}", PASS),
    ("POST", "/v1/batches", PASS),
    ("GET", "/v1/batches", PASS),
    ("GET", "/v1/batches/{id}", PASS),
    ("POST", "/v1/batches/{id}/cancel", PASS),
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
    ("GET", "/v1/videos/{id}/content", PASS),
    ("DELETE", "/v1/videos/{id}", PASS),
    ("POST", "/v1/fine_tuning/jobs", PASS),
    ("GET", "/v1/fine_tuning/jobs", PASS),
    # Google Vertex AI
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
    # Azure OpenAI
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
    ("POST", "/openai/v1/conversations", CHAT),
    ("POST", "/openai/v1/conversations/{id}/items", CHAT),
    ("GET", "/openai/v1/conversations/{id}", CHAT),
    ("GET", "/openai/v1/conversations/{id}/items", CHAT),
    ("DELETE", "/openai/v1/conversations/{id}", REDACT_ONLY),
    ("POST", "/openai/files", REDACT_ONLY),
    ("POST", "/openai/v1/files", REDACT_ONLY),
    ("GET", "/openai/files", REDACT_ONLY),
    ("GET", "/openai/files/{id}", REDACT_ONLY),
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
    # AWS Bedrock (runtime)
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

_EXPECTED_KIND = {
    CHAT: RouteKind.CHAT,
    REDACT_ONLY: RouteKind.REDACT_ONLY,
    PASS: RouteKind.NONE,
}


def _route(method: str, path: str) -> RouteKind:
    # First-match semantics, identical to ProxyState.route.
    for adapter in (cls() for cls in ALL_ADAPTERS):
        kind = adapter.matches(method, path)
        if kind is not RouteKind.NONE:
            return kind
    return RouteKind.NONE


@pytest.mark.parametrize(("method", "path", "classification"), MATRIX)
def test_route_matches_matrix(method: str, path: str, classification: str) -> None:
    concrete = re.sub(r"\{[a-z]+\}", "abc_123", path)
    assert _route(method, concrete) is _EXPECTED_KIND[classification], (
        f"{method} {path} expected {classification}"
    )


# Doc rows deliberately not route-pinned (prose paths, no concrete route).
EXTRA_DOC_ROWS = {("GET", "/v1/organizations/...", PASS)}
# Table rows probed for routing but expressed as prose in the doc.
TABLE_ONLY_ROWS = {("POST", "/v1/organizations/probe", PASS)}


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
