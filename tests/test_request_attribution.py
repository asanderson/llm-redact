"""Where a request goes — and when it goes nowhere (C-R2-01, C-R2-06, C-R2-07).

Every test runs the real app against one recording fake upstream
(``httpx.MockTransport``) and asserts the HOST each request reached, with
which credential, and whether its body was redacted — or that nothing
reached any upstream at all:

- a request no adapter recognizes goes only to a provider the proxy can
  POSITIVELY attribute it to (a path family, or one provider's markers);
  anything else is a recorded local 404, never the old anthropic guess;
- a spelling that is not the API's own (an empty segment, a trailing slash,
  another case) of a route llm-redact recognizes is refused, never
  forwarded unredacted;
- the documented tool setups keep reaching the right provider, redacted.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.providers.attribution import attribute, provider_markers, unattributed_reason
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
OPENAI_KEY = "sk-proj-FAKEFAKEFAKE"
ANTHROPIC_KEY = "sk-ant-api03-FAKEFAKE"
GOOGLE_KEY = "AIzaFAKEFAKEFAKE"
PROXY = "http://127.0.0.1:8787"

ANTHROPIC = {"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"}
OPENAI = {"authorization": f"Bearer {OPENAI_KEY}"}
GOOGLE = {"x-goog-api-key": GOOGLE_KEY, "x-goog-api-client": "google-genai-sdk/1.0"}

HOSTS = {
    "anthropic": "api.anthropic.com",
    "openai": "api.openai.com",
    "gemini": "generativelanguage.googleapis.com",
    "ollama": "127.0.0.1",
    "cohere": "api.cohere.com",
    "azure": "res.openai.azure.com",
    "bedrock": "bedrock-runtime.us-east-1.amazonaws.com",
    "custom:lm": "lm.local",
}


class Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True})


def _config() -> Config:
    providers = dict(Config().providers)
    providers["azure"] = ProviderConfig("https://res.openai.azure.com")
    providers["bedrock"] = ProviderConfig("https://bedrock-runtime.us-east-1.amazonaws.com")
    providers["custom:lm"] = ProviderConfig("http://lm.local")
    return Config(providers=providers)


def _app(config: Config | None = None) -> tuple[Any, Recorder]:
    recorder = Recorder()
    app = create_app(config or _config(), upstream_transport=httpx.MockTransport(recorder))
    return app, recorder


async def _send(
    app: Any,
    method: str,
    target: str,
    *,
    headers: dict[str, str] | None = None,
    body: Any = None,
) -> httpx.Response:
    """``target`` is sent as the raw request target (full URL form keeps a
    leading ``//`` a path, not a scheme-relative host)."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        content = None if body is None else json.dumps(body).encode()
        request_headers = dict(headers or {})
        if content is not None:
            request_headers["content-type"] = "application/json"
        return await client.request(
            method, PROXY + target, headers=request_headers, content=content
        )


def _chat(text: str = f"mail {EMAIL}") -> dict[str, Any]:
    return {"model": "m", "messages": [{"role": "user", "content": text}]}


# --- the documented tool setups: each reaches its provider, redacted --------------------

# (tool, method, target, headers, body, provider, redacted?)
TOOL_SHAPES: list[tuple[str, str, str, dict[str, str], Any, str, bool]] = [
    # Claude Code / the Anthropic SDKs (ANTHROPIC_BASE_URL=http://127.0.0.1:8787).
    ("claude", "POST", "/v1/messages?beta=true", ANTHROPIC, _chat(), "anthropic", True),
    (
        "claude",
        "POST",
        "/v1/messages/count_tokens?beta=true",
        ANTHROPIC,
        _chat(),
        "anthropic",
        True,
    ),
    ("claude", "GET", "/v1/models?limit=20", ANTHROPIC, None, "anthropic", False),
    ("claude", "GET", "/v1/models/claude-x", ANTHROPIC, None, "anthropic", False),
    (
        "claude-oauth",
        "GET",
        "/v1/models",
        {"authorization": "Bearer sk-ant-oat01-FAKE", "anthropic-version": "2023-06-01"},
        None,
        "anthropic",
        False,
    ),
    (
        "anthropic-sdk",
        "GET",
        "/v1/messages/batches/msgbatch_1",
        ANTHROPIC,
        None,
        "anthropic",
        False,
    ),
    ("anthropic-sdk", "DELETE", "/v1/files/file_011", ANTHROPIC, None, "anthropic", False),
    ("anthropic-sdk", "GET", "/v1/skills", ANTHROPIC, None, "anthropic", False),
    # Codex CLI / OpenCode / the OpenAI SDKs (OPENAI_BASE_URL=http://127.0.0.1:8787/v1).
    ("codex", "POST", "/v1/responses", OPENAI, {"model": "m", "input": EMAIL}, "openai", True),
    ("openai-sdk", "POST", "/v1/chat/completions", OPENAI, _chat(), "openai", True),
    ("openai-sdk", "GET", "/v1/models", OPENAI, None, "openai", False),
    ("openai-sdk", "DELETE", "/v1/conversations/conv_123", OPENAI, None, "openai", False),
    ("openai-sdk", "DELETE", "/v1/conversations/conv_1/items/msg_1", OPENAI, None, "openai", False),
    (
        "openai-sdk",
        "GET",
        "/v1/containers/cntr_1/files/cf_1/content",
        OPENAI,
        None,
        "openai",
        False,
    ),
    ("openai-sdk", "GET", "/v1/organization/usage/completions", OPENAI, None, "openai", False),
    ("openai-sdk", "GET", "/v1/evals", OPENAI, None, "openai", False),
    ("openai-sdk", "GET", "/v1/fine_tuning/jobs", OPENAI, None, "openai", False),
    ("openai-sdk", "POST", "/v1/uploads", OPENAI, {"filename": "a.jsonl"}, "openai", False),
    ("openai-key-unknown-path", "GET", "/v1/some_new_resource", OPENAI, None, "openai", False),
    (
        "openai-beta-header",
        "GET",
        "/v1/some_new_resource",
        {"openai-beta": "assistants=v2"},
        None,
        "openai",
        False,
    ),
    # Gemini CLI / google-genai (GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:8787).
    (
        "gemini",
        "POST",
        "/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse",
        GOOGLE,
        {"contents": [{"role": "user", "parts": [{"text": f"mail {EMAIL}"}]}]},
        "gemini",
        True,
    ),
    ("gemini", "GET", "/v1beta/models", GOOGLE, None, "gemini", False),
    ("gemini", "GET", f"/v1/models?key={GOOGLE_KEY}", {}, None, "gemini", False),
    ("gemini", "GET", "/v1/models", {"x-goog-api-key": GOOGLE_KEY}, None, "gemini", False),
    ("gemini", "POST", "/upload/v1beta/files", GOOGLE, {"file": {}}, "gemini", False),
    # Gemini's OpenAI-compatible surface (base .../v1beta/openai/).
    (
        "gemini-openai",
        "POST",
        "/v1beta/openai/chat/completions",
        {"authorization": f"Bearer {GOOGLE_KEY}"},
        _chat(),
        "gemini",
        True,
    ),
    (
        "gemini-openai",
        "POST",
        "/v1beta/openai/embeddings",
        {"authorization": f"Bearer {GOOGLE_KEY}"},
        {"model": "m", "input": EMAIL},
        "gemini",
        True,
    ),
    # The ollama CLI and native clients (OLLAMA_HOST=http://127.0.0.1:8787).
    ("ollama", "POST", "/api/chat", {}, _chat(), "ollama", True),
    ("ollama", "GET", "/api/tags", {}, None, "ollama", False),
    ("ollama", "POST", "/api/show", {}, {"model": "llama3"}, "ollama", False),
    # Cohere (base_url=http://127.0.0.1:8787).
    ("cohere", "POST", "/v2/chat", {"authorization": "Bearer co-key"}, _chat(), "cohere", True),
    ("cohere", "GET", "/v2/models", {"authorization": "Bearer co-key"}, None, "cohere", False),
    (
        "cohere-sdk",
        "GET",
        "/v1/models",
        {"authorization": "Bearer co-key", "x-fern-sdk-name": "cohere"},
        None,
        "cohere",
        False,
    ),
    # Azure / Bedrock / a named custom upstream.
    (
        "azure",
        "POST",
        "/openai/deployments/d/chat/completions?api-version=2024-10-21",
        {"api-key": "az"},
        _chat(),
        "azure",
        True,
    ),
    (
        "bedrock",
        "POST",
        "/model/m/converse",
        {"authorization": "Bearer ABSK"},
        {"messages": [{"role": "user", "content": [{"text": f"mail {EMAIL}"}]}]},
        "bedrock",
        True,
    ),
    ("custom", "POST", "/custom/lm/v1/chat/completions", OPENAI, _chat(), "custom:lm", True),
]


@pytest.mark.parametrize(
    ("tool", "method", "target", "headers", "body", "provider", "redacted"),
    TOOL_SHAPES,
    ids=[f"{row[0]}:{row[1]} {row[2]}" for row in TOOL_SHAPES],
)
async def test_documented_tool_requests_reach_their_provider(
    tool: str,
    method: str,
    target: str,
    headers: dict[str, str],
    body: Any,
    provider: str,
    redacted: bool,
) -> None:
    app, upstream = _app()
    response = await _send(app, method, target, headers=headers, body=body)
    assert response.status_code == 200, response.text
    (sent,) = upstream.requests
    assert sent.url.host == HOSTS[provider]
    if redacted:
        assert EMAIL not in sent.content.decode("utf-8") and "«EMAIL_001»" in sent.content.decode()
    # The client's own credential went to the provider it belongs to only.
    if headers is ANTHROPIC:
        assert sent.headers["x-api-key"] == ANTHROPIC_KEY and sent.url.host == HOSTS["anthropic"]
    if headers is OPENAI:
        assert sent.url.host in (HOSTS["openai"], HOSTS["custom:lm"])


async def test_the_proxy_root_is_answered_locally() -> None:
    """The ollama CLI's heartbeat (HEAD /) — no upstream owns the root."""
    app, upstream = _app()
    for method in ("HEAD", "GET"):
        response = await _send(app, method, "/")
        assert response.status_code == 200
    assert upstream.requests == []
    assert not app.state.proxy.recent


# --- unattributable: a recorded local 404, nothing forwarded -----------------------------


@pytest.mark.parametrize(
    ("method", "target", "headers"),
    [
        # Codex/OpenAI SDKs with the old base URL (no /v1): /responses was
        # sent to api.anthropic.com with the OpenAI key and the prompt.
        ("POST", "/responses", OPENAI),
        ("POST", "/chat/completions", OPENAI),
        ("POST", "/embeddings", OPENAI),
        ("GET", "/models", OPENAI),
        ("GET", "/fine_tuning/jobs", OPENAI),
    ],
)
async def test_an_openai_path_without_v1_is_refused_with_the_base_url_fix(
    method: str, target: str, headers: dict[str, str]
) -> None:
    app, upstream = _app()
    body = {"model": "m", "input": EMAIL} if method == "POST" else None
    response = await _send(app, method, target, headers=headers, body=body)
    assert response.status_code == 404
    assert "/v1" in response.json()["error"]["message"]
    assert upstream.requests == []
    (row,) = app.state.proxy.recent
    assert row["status"] == 404 and row["path"] == target.split("?")[0]
    assert EMAIL not in response.text


@pytest.mark.parametrize(
    ("method", "target", "headers", "why"),
    [
        ("POST", "/v1/anything/at/all", {}, "no path family"),
        ("GET", "/v1/foo", {"x-api-key": ANTHROPIC_KEY}, "no path family"),
        ("GET", "/somewhere", {}, "no path family"),
        ("POST", "/", {}, "no path family"),
        (
            "GET",
            "/v1/models",
            {"anthropic-version": "2023-06-01", "x-goog-api-key": GOOGLE_KEY},
            "more than one provider (anthropic, gemini)",
        ),
    ],
)
async def test_an_unattributable_request_is_a_recorded_local_404(
    method: str, target: str, headers: dict[str, str], why: str
) -> None:
    app, upstream = _app()
    body = _chat() if method == "POST" else None
    response = await _send(app, method, target, headers=headers, body=body)
    assert response.status_code == 404
    assert why in response.json()["error"]["message"]
    assert upstream.requests == []
    (row,) = app.state.proxy.recent
    assert row["status"] == 404 and row["provider"] is None


async def test_a_path_under_an_extra_prefix_is_refused() -> None:
    """ANTHROPIC_BASE_URL=http://127.0.0.1:8787/v1 sends /v1/v1/messages —
    once forwarded (anthropic-version) unredacted to Anthropic, which 404s."""
    app, upstream = _app()
    for target, headers in (
        ("/v1/v1/messages", ANTHROPIC),
        ("/v1/v1/chat/completions", OPENAI),
        ("/v1beta/v1beta/models/m:generateContent", GOOGLE),
        ("/api/api/chat", {}),
    ):
        response = await _send(app, "POST", target, headers=headers, body=_chat())
        assert response.status_code == 404, target
        assert "prefix" in response.text
    assert upstream.requests == []


# --- spellings of a recognized route (C-R2-07) ---------------------------------------------


@pytest.mark.parametrize(
    ("target", "headers"),
    [
        ("/v1//chat/completions", OPENAI),
        ("//v1/chat/completions", OPENAI),
        ("/v1//responses", OPENAI),
        ("//api/chat", {}),
        (f"//v1beta/models/m:generateContent?key={GOOGLE_KEY}", {}),
        ("/custom/lm/v1//chat/completions", OPENAI),
        ("/custom/lm//v1/chat/completions", OPENAI),
        ("/v1/%2F/chat/completions", OPENAI),
    ],
)
async def test_an_empty_path_segment_is_refused_before_anything(
    target: str, headers: dict[str, str]
) -> None:
    app, upstream = _app()
    response = await _send(app, "POST", target, headers=headers, body=_chat())
    assert response.status_code == 400
    assert "empty segment" in response.json()["error"]
    assert upstream.requests == []
    assert not app.state.proxy.recent  # the path may still carry an identity key


@pytest.mark.parametrize(
    ("target", "headers"),
    [
        ("/v1/chat/completions/", OPENAI),
        ("/v1/responses/", OPENAI),
        ("/v1/messages/", ANTHROPIC),
        ("/V1/chat/completions", OPENAI),
        ("/v1/Chat/Completions", OPENAI),
        ("/V1/messages", ANTHROPIC),
        ("/api/chat/", {}),
        ("/openai/deployments/d/chat/completions/?api-version=1", {"api-key": "az"}),
        ("/openai/deployments/d/Chat/Completions?api-version=1", {"api-key": "az"}),
        ("/model/m/converse/", {"authorization": "Bearer ABSK"}),
        ("/model/m/Converse", {"authorization": "Bearer ABSK"}),
        ("/custom/lm/v1/chat/completions/", OPENAI),
        ("/custom/lm/v1/Chat/Completions", OPENAI),
        ("/custom/lm/chat/completions/", OPENAI),
    ],
)
async def test_another_spelling_of_a_recognized_route_is_refused(
    target: str, headers: dict[str, str]
) -> None:
    app, upstream = _app()
    response = await _send(app, "POST", target, headers=headers, body=_chat())
    assert response.status_code == 400, response.text
    assert "spelled exactly" in response.text
    assert EMAIL not in response.text
    assert upstream.requests == []
    (row,) = app.state.proxy.recent
    assert row["status"] == 400


# --- OpenAI-compatible bases without a /v1 segment (C-R2-06) -------------------------------


@pytest.mark.parametrize(
    "inner",
    [
        "/v1beta/openai/chat/completions",  # Gemini's OpenAI-compatible base
        "/inference/chat/completions",  # GitHub Models
        "/models/chat/completions",  # Azure AI model inference
        "/v1/acct/gw/openai/chat/completions",  # Cloudflare AI Gateway
        "/api/paas/v4/chat/completions",  # a /v4 base
        "/openai/v1/chat/completions",  # Groq (already re-anchored)
        "/chat/completions",  # /v1 baked into upstream_base_url
    ],
)
async def test_custom_upstream_chat_under_any_base_path_is_redacted(inner: str) -> None:
    app, upstream = _app()
    response = await _send(app, "POST", "/custom/lm" + inner, headers=OPENAI, body=_chat())
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert sent.url.host == "lm.local"
    assert sent.url.raw_path == inner.encode()  # the stripped path, byte-for-byte
    body = sent.content.decode("utf-8")
    assert EMAIL not in body and "«EMAIL_001»" in body


async def test_custom_upstream_tail_matching_keeps_unknown_tails_pass_through() -> None:
    app, upstream = _app()
    response = await _send(
        app, "POST", "/custom/lm/v1/some/unknown/tail", headers=OPENAI, body=_chat()
    )
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert sent.url.host == "lm.local"
    assert EMAIL in sent.content.decode("utf-8")  # pass-through, as documented


async def test_gemini_openai_compat_streams_are_rehydrated() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        token = body["messages"][-1]["content"].split()[-1]
        chunk = {"choices": [{"index": 0, "delta": {"content": f"hi {token}"}}]}
        done = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        payload = (
            f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(done)}\n\ndata: [DONE]\n\n"
        ).encode()
        return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})

    app = create_app(_config(), upstream_transport=httpx.MockTransport(upstream))
    response = await _send(
        app,
        "POST",
        "/v1beta/openai/chat/completions",
        headers={"authorization": f"Bearer {GOOGLE_KEY}"},
        body={**_chat(), "stream": True},
    )
    assert response.status_code == 200
    assert f"hi {EMAIL}" in response.text


# --- the attribution rules themselves --------------------------------------------------------


def test_markers_name_one_provider_each() -> None:
    assert provider_markers({"Anthropic-Version": "x"}) == {"anthropic"}
    assert provider_markers({"X-Goog-User-Project": "p"}) == {"gemini"}
    assert provider_markers({}, "alt=sse&%24key=AIza") == {"gemini"}
    assert provider_markers({"OpenAI-Organization": "org"}) == {"openai"}
    assert provider_markers({"x-fern-sdk-name": "cohere-ai"}) == {"cohere"}
    assert provider_markers({"x-fern-sdk-name": "other"}) == frozenset()
    assert provider_markers(None) == frozenset()


def test_attribution_prefers_families_then_markers_then_openai() -> None:
    # An explicit family wins over any marker (express-mode Vertex keys).
    assert attribute("/v1/projects/p/x", {"x-goog-api-key": "k"}) == "vertex"
    assert attribute("/v1beta/files", {"anthropic-version": "1"}) == "gemini"
    assert attribute("/custom/lm/x", {}) == "custom:lm"
    # Segment-aware: Anthropic's Admin API is not OpenAI's.
    assert attribute("/v1/organizations/users", {}) == "anthropic"
    assert attribute("/v1/organization/users", {}) == "openai"
    assert attribute("/v1/embed", {}) is None  # not /v1/embeddings
    # A bearer key: OpenAI's, never Anthropic's OAuth token.
    assert attribute("/x", {"Authorization": "Bearer sk-abc"}) == "openai"
    assert attribute("/x", {"authorization": "Bearer sk-ant-oat01-x"}) is None
    assert attribute("/x", {"authorization": "Basic sk-abc"}) is None
    assert attribute("/x", {"authorization": "Bearer"}) is None
    # Two providers' markers: nothing.
    assert attribute("/v1/files", {"anthropic-version": "1", "openai-beta": "x"}) is None
    assert "more than one provider (anthropic, openai)" in unattributed_reason(
        {"anthropic-version": "1", "openai-beta": "x"}
    )
    assert unattributed_reason({}) == "no path family or provider marker names its provider"


def test_custom_tails_never_cross_into_a_nested_resource() -> None:
    """A tail below an OpenAI resource is that resource's sub-resource, not
    an endpoint under a base path: a container's file download restored as
    a Files API download would read a code interpreter's output in the
    wrong session."""
    from llm_redact.providers.base import RouteKind
    from llm_redact.providers.custom import CustomOpenAIAdapter

    adapter = CustomOpenAIAdapter("lm")
    for method, path in (
        ("GET", "/custom/lm/v1/containers/cntr_1/files/cf_1/content"),
        ("GET", "/custom/lm/v1/vector_stores/vs_1/files"),
        ("GET", "/custom/lm/v1/chat/completions/c_1/messages"),
        ("POST", "/custom/lm/v1/threads/th_1/messages"),
    ):
        assert adapter.matches(method, path) is RouteKind.NONE, path
    for method, path, kind in (
        ("GET", "/custom/lm/v1/files/f_1/content", RouteKind.CHAT),
        ("POST", "/custom/lm/openai/deployments/d/chat/completions", RouteKind.CHAT),
        ("POST", "/custom/lm/models/embeddings", RouteKind.REDACT_ONLY),
        ("GET", "/custom/lm/inference/models", RouteKind.REDACT_ONLY),
    ):
        assert adapter.matches(method, path) is kind, path
    assert adapter.matches("POST", "/elsewhere/chat/completions") is RouteKind.NONE
    # The tail is chosen for the request's method (and, from the note hook,
    # for the matched kind): POST /models/completions is legacy completions
    # — no note, its body has no messages — not a GET of a model.
    assert adapter.matches("POST", "/custom/lm/models/completions") is RouteKind.CHAT
    assert adapter.matches("GET", "/custom/lm/models/completions") is RouteKind.REDACT_ONLY
    assert not adapter.wants_system_note(RouteKind.CHAT, "/custom/lm/models/completions")
    assert adapter.wants_system_note(RouteKind.CHAT, "/custom/lm/models/chat/completions")
    assert not adapter.wants_system_note(RouteKind.REDACT_ONLY, "/custom/lm/models/embeddings")


async def test_gemini_openai_compat_responses_restores_only_its_own_answer() -> None:
    """A Responses create on Gemini's OpenAI-compatible prefix is redacted and
    its answer restored in the request's session; a stored response read by
    id is left as the provider sends it (no session is known to hold its
    tokens), and still reaches the Gemini upstream."""
    from llm_redact.providers.base import RouteKind
    from llm_redact.providers.custom import GeminiOpenAIAdapter, GeminiOpenAIResponsesAdapter

    responses = GeminiOpenAIResponsesAdapter()
    assert responses.matches("POST", "/v1beta/openai/responses") is RouteKind.CHAT
    assert responses.matches("GET", "/v1beta/openai/responses/resp_1") is RouteKind.NONE
    assert GeminiOpenAIAdapter().matches("GET", "/v1beta/openai/responses/resp_1") is RouteKind.NONE
    app, upstream = _app()
    headers = {"authorization": f"Bearer {GOOGLE_KEY}"}
    created = await _send(
        app, "POST", "/v1beta/openai/responses", headers=headers, body={"input": EMAIL}
    )
    read = await _send(app, "GET", "/v1beta/openai/responses/resp_1", headers=headers)
    assert created.status_code == read.status_code == 200
    first, second = upstream.requests
    assert first.url.host == second.url.host == HOSTS["gemini"]
    assert EMAIL not in first.content.decode("utf-8")
