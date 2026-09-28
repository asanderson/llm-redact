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
    concrete = path.replace("{id}", "abc_123")
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
        r"^\| `(GET|POST|DELETE) ([^`]+)`[^|]* \| (chat|redact-only|pass-through) \|",
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
