"""Two optional ``SessionRouter`` seams for stored-object ownership:

- ``object_access_refusal`` lets a router refuse a request that reaches
  another namespace's stored object (a recorded, provider-shaped 403 before
  the audit START row, redaction, any upstream credential and any upstream
  contact — routed or not);
- ``listing_item_session`` names, per listed stored object, the session its
  placeholders are restored in (the owner's own listing), through the
  adapter hooks ``lists_objects`` / ``listing_items``.

Keyless: scripted fakes on a bare Registry stand in for llm-redact-pro.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from fake_router import ROUTE_HEADER, FakeRouter, Hop, install, routed_config
from license_fixtures import resolved
from llm_redact.audit import AuditRecord
from llm_redact.config import AuditConfig, Config, ProviderConfig, VaultConfig
from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.azure_openai import AzureOpenAIAdapter
from llm_redact.providers.base import ProviderAdapter
from llm_redact.providers.custom import CustomOpenAIAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import create_app
from llm_redact.registry import Registry
from llm_redact.vault import InMemoryVaultManager

UPSTREAM = "https://upstream.test"
AZURE = "https://res.openai.azure.com"
REFUSAL = "llm-redact: that stored object belongs to another user"


class ScriptedRouter:
    """A session router with both optional ownership members, scripted."""

    def __init__(
        self,
        *,
        mode: str = "per-user",
        session: str = "user:n1:main",
        verdict: Any = None,
        error: Exception | None = None,
        owners: dict[str, str] | None = None,
        listing_error: Exception | None = None,
    ) -> None:
        self.mode = mode
        self.session = session
        self.verdict = verdict
        self.error = error
        self.owners = dict(owners or {})
        self.listing_error = listing_error
        self.checks: list[tuple[str | None, str, str, Any, bool]] = []
        self.listed: list[str] = []
        self.resolved = 0

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        self.resolved += 1
        return self.session

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def object_access_refusal(
        self, adapter_name: str | None, method: str, path: str, body: Any, *, identity: bool
    ) -> Any:
        self.checks.append((adapter_name, method, path, body, identity))
        if self.error is not None:
            raise self.error
        return self.verdict

    def listing_item_session(self, object_id: str) -> str | None:
        self.listed.append(object_id)
        if self.listing_error is not None:
            raise self.listing_error
        return self.owners.get(object_id)


class OlderRouter:
    """A router from before either member (today's behavior must hold)."""

    mode = "per-user"

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return "user:n1:main"

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None


class FakeAuth:
    def __init__(self) -> None:
        self.calls = 0

    async def authorize(
        self, method: str, url: str, headers: list[tuple[str, str]], body: bytes
    ) -> list[tuple[str, str]]:
        self.calls += 1
        return [*headers, ("authorization", "Proxy azure")]

    def close(self) -> None:
        pass


class FakeAudit:
    def __init__(self) -> None:
        self.begun: list[AuditRecord] = []
        self.recorded: list[AuditRecord] = []

    def record(self, entry: AuditRecord) -> None:
        self.recorded.append(entry)

    def begin(self, entry: AuditRecord) -> object:
        self.begun.append(entry)
        return len(self.begun)

    def finalize(self, token: object, entry: AuditRecord) -> None:
        pass

    def recent(self, limit: int) -> list[dict[str, object]]:
        return []

    def count(self) -> int:
        return 0

    def close(self) -> None:
        pass


class Upstream:
    def __init__(self, body: Any = None, *, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.body = {"id": "file-1", "object": "file"} if body is None else body
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)


def _registry(
    monkeypatch: pytest.MonkeyPatch, router: Any, registry: Registry | None = None
) -> Registry:
    reg = registry or Registry()
    reg.build_session_router = lambda config, **kw: router
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg


def _app(
    monkeypatch: pytest.MonkeyPatch,
    router: Any,
    upstream: Upstream,
    *,
    providers: dict[str, ProviderConfig] | None = None,
    registry: Registry | None = None,
    **config: Any,
) -> Any:
    _registry(monkeypatch, router, registry)
    merged = {**Config().providers, "openai": ProviderConfig(UPSTREAM), **(providers or {})}
    return create_app(
        Config(providers=merged, **config), upstream_transport=httpx.MockTransport(upstream)
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


# --- object_access_refusal -------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "adapter_name", "body"),
    [
        ("DELETE", "/v1/files/file-1", None, None),  # pass-through: no adapter, no body
        ("GET", "/v1/files/file-1/content", "openai", None),
        ("POST", "/v1/batches/batch_1/cancel", "openai", None),
        ("DELETE", "/v1/videos/video_1", None, None),
        ("POST", "/v1/videos/video_1/remix", "openai", {"prompt": "a dog"}),
    ],
)
async def test_a_refusal_is_a_recorded_403_and_nothing_is_forwarded(
    method: str,
    path: str,
    adapter_name: str | None,
    body: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ScriptedRouter(verdict=REFUSAL)
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream)
    async with _client(app) as client:
        response = await client.request(method, path, json=body)
    assert response.status_code == 403
    expected = (
        OpenAIAdapter().error_body(REFUSAL, status=403)
        if adapter_name is not None
        else {"error": REFUSAL}
    )
    assert response.json() == expected
    assert upstream.requests == []
    assert router.checks == [(adapter_name, method, path, body, False)]
    assert router.resolved == 0  # refused before the session is even resolved
    (row,) = app.state.proxy.recent
    assert row["status"] == 403 and row["path"] == path and row["provider"] == "openai"


async def test_none_forwards_and_the_check_runs_in_every_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mode in ("static", "per-user"):
        router = ScriptedRouter(mode=mode)
        upstream = Upstream()
        app = _app(monkeypatch, router, upstream)
        async with _client(app) as client:
            response = await client.get("/v1/files/file-1/content")
        assert response.status_code == 200 and len(upstream.requests) == 1
        assert router.checks == [("openai", "GET", "/v1/files/file-1/content", None, False)]


@pytest.mark.parametrize("verdict", [True, "", 403])
async def test_a_non_reason_answer_refuses_with_the_fixed_text(
    verdict: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = Upstream()
    app = _app(monkeypatch, ScriptedRouter(verdict=verdict), upstream)
    async with _client(app) as client:
        response = await client.delete("/v1/files/file-1")
    assert response.status_code == 403
    assert "ownership check failed" in response.json()["error"]
    assert upstream.requests == []


async def test_a_failing_router_refuses_and_logs_the_type_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = Upstream()
    router = ScriptedRouter(error=LookupError("file-secret-id"))
    app = _app(monkeypatch, router, upstream)
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        async with _client(app) as client:
            response = await client.get("/v1/files/file-1/content")
    assert response.status_code == 403 and upstream.requests == []
    assert "ownership check failed" in response.json()["error"]["message"]
    assert "LookupError" in caplog.text and "file-secret-id" not in caplog.text


async def test_identity_is_reported_and_a_refusal_never_reaches_the_authorizer_or_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = FakeAuth()
    audit = FakeAudit()
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("pro")
    reg.build_access_gate = lambda cfg, lic: None
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    reg.build_audit = lambda cfg: audit if cfg.enabled else None
    router = ScriptedRouter(verdict=REFUSAL)
    upstream = Upstream()
    app = _app(
        monkeypatch,
        router,
        upstream,
        providers={"azure": ProviderConfig(AZURE, auth="identity")},
        registry=reg,
        audit=AuditConfig(enabled=True, required=True),
    )
    path = "/openai/v1/files/file-1"
    async with _client(app) as client:
        response = await client.delete(path)
    assert response.status_code == 403
    assert response.json() == AzureOpenAIAdapter().error_body(REFUSAL, status=403)
    assert router.checks == [("azure", "DELETE", path, None, True)]
    assert upstream.requests == [] and auth.calls == 0
    assert audit.begun == []  # no write-ahead START row for a refused request
    # The same request, allowed, is authorized, audited and forwarded.
    router.verdict = None
    async with _client(app) as client:
        assert (await client.delete(path)).status_code == 200
    assert auth.calls == 1 and len(audit.begun) == 1 and len(upstream.requests) == 1


async def test_a_routed_request_is_checked_before_the_router_plans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRouter({"r": [Hop("x", "http://x.example/v1/messages")]})
    reg, _ = install(monkeypatch, fake)
    router = ScriptedRouter(verdict=REFUSAL)
    _registry(monkeypatch, router, reg)
    app = create_app(routed_config())
    body = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
    async with _client(app) as client:
        response = await client.post("/v1/messages", json=body, headers={ROUTE_HEADER: "r"})
    assert response.status_code == 403
    assert response.json() == AnthropicAdapter().error_body(REFUSAL, status=403)
    assert fake.inbounds == []  # never planned, so no hop was ever issued
    assert router.checks == [("anthropic", "POST", "/v1/messages", body, False)]


async def test_an_older_router_is_never_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = Upstream()
    app = _app(monkeypatch, OlderRouter(), upstream)
    async with _client(app) as client:
        assert (await client.delete("/v1/files/file-1")).status_code == 200
    assert len(upstream.requests) == 1


# --- lists_objects / listing_items -----------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "path", "listed"),
    [
        (OpenAIAdapter(), "/v1/files", True),
        (OpenAIAdapter(), "/v1/batches/", True),
        (OpenAIAdapter(), "/v1/videos", True),
        (OpenAIAdapter(), "/v1/chat/completions", True),
        (OpenAIAdapter(), "/v1/files/file-1", False),
        (OpenAIAdapter(), "/v1/chat/completions/chatcmpl-1/messages", False),
        (OpenAIAdapter(), "/v1/conversations/conv_1/items", False),
        (OpenAIAdapter(), "/v1/models", False),
        (AzureOpenAIAdapter(), "/openai/files", True),
        (AzureOpenAIAdapter(), "/openai/v1/batches", True),
        (AzureOpenAIAdapter(), "/openai/v1/chat/completions", True),
        (CustomOpenAIAdapter("lm"), "/custom/lm/v1/files", True),
        (CustomOpenAIAdapter("lm"), "/custom/lm/api/v1/videos", True),
        (AnthropicAdapter(), "/v1/messages/batches", False),
    ],
)
def test_listings_of_stored_objects_are_recognized(
    adapter: ProviderAdapter, path: str, listed: bool
) -> None:
    assert adapter.lists_objects("GET", path) is listed
    assert adapter.lists_objects("POST", path) is False


def test_listing_items_is_the_openai_list_envelope_only() -> None:
    adapter = OpenAIAdapter()
    items = [{"id": "file-1"}]
    body = {"object": "list", "data": items}
    assert adapter.listing_items(body) is items
    assert adapter.listing_items({"object": "file", "data": items}) is None
    assert adapter.listing_items({"object": "list", "data": {"id": "x"}}) is None
    assert adapter.listing_items([items]) is None
    assert AnthropicAdapter().listing_items(body) is None  # the base default


def test_the_memory_manager_answers_without_creating_a_session() -> None:
    manager = InMemoryVaultManager()
    assert manager.has_session("absent") is False
    assert manager.session_count() == 0
    manager.get("empty")
    assert manager.has_session("empty") is False
    manager.get("full").placeholder_for("EMAIL", "ada@corp.example")
    assert manager.has_session("full") is True


# --- listing_item_session ---------------------------------------------------------------

ADA = "ada@corp.example"
BOB = "bob@corp.example"
TOKEN = "«EMAIL_001»"


def _listing() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "file-own", "object": "file", "filename": f"notes {TOKEN}.jsonl"},
            {"id": "file-other", "object": "file", "filename": f"notes {TOKEN}.jsonl"},
            {"id": "file-gone", "object": "file", "filename": f"notes {TOKEN}.jsonl"},
            "not-an-object",
            {"object": "file", "filename": TOKEN},  # no id
        ],
        "first_id": TOKEN,
        "has_more": False,
    }


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_only_the_items_the_router_names_are_restored(
    backend: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = ScriptedRouter(
        session="read-empty",
        owners={"file-own": "user:n1:main", "file-gone": "user:n1:never-created"},
    )
    upstream = Upstream(_listing())
    vault = VaultConfig(backend=backend, path=str(tmp_path / "v.db"))
    app = _app(monkeypatch, router, upstream, vault=vault)
    state = app.state.proxy
    state.vault_manager.get("user:n1:main").placeholder_for("EMAIL", ADA)
    async with _client(app) as client:
        response = await client.get("/v1/files?limit=5")  # pass-through: no adapter
    assert response.status_code == 200
    body = response.json()
    expected = _listing()
    expected["data"][0]["filename"] = f"notes {ADA}.jsonl"
    assert body == expected  # the rest (and every top-level field) untouched
    assert router.listed == ["file-own", "file-other", "file-gone"]
    # The named-but-absent session was never created; nothing was recorded.
    listed_sessions = [row["session"] for row in state.vault_manager.sessions_summary()]
    assert "user:n1:never-created" not in listed_sessions
    assert state.vault_manager.lookup_response_session("file-own") is None
    (row,) = state.recent
    assert row["rehydrations"] == {"EMAIL": 1}


async def test_a_named_item_is_restored_from_the_provider_bytes_not_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # GET /v1/videos is a CHAT route: the request's own session restores the
    # whole listing first; a named item is then restored from the PROVIDER's
    # bytes in its owner's session — never from the already-restored text.
    # An item named to an EMPTY session (another namespace's, in a listing
    # read in a populated session) keeps the provider's placeholder instead
    # of the request session's value.
    router = ScriptedRouter(
        session="shared", owners={"video_a": "user:n1:main", "video_c": "never-written"}
    )
    listing = {
        "object": "list",
        "data": [
            {"id": "video_a", "object": "video", "prompt": f"film {TOKEN}"},
            {"id": "video_b", "object": "video", "prompt": f"film {TOKEN}"},
            {"id": "video_c", "object": "video", "prompt": f"film {TOKEN}"},
        ],
    }
    app = _app(monkeypatch, router, Upstream(listing))
    manager = app.state.proxy.vault_manager
    manager.get("shared").placeholder_for("EMAIL", BOB)
    manager.get("user:n1:main").placeholder_for("EMAIL", ADA)
    async with _client(app) as client:
        response = await client.get("/v1/videos")
    data = response.json()["data"]
    assert data[0]["prompt"] == f"film {ADA}"  # the owner's value
    assert data[1]["prompt"] == f"film {BOB}"  # the request's session, as today
    assert data[2]["prompt"] == f"film {TOKEN}"  # an empty session restores nothing
    assert manager.has_session("never-written") is False  # and was never created


@pytest.mark.parametrize(
    ("provider", "path"),
    [
        ("azure", "/openai/v1/chat/completions"),
        ("azure", "/openai/files"),
        ("custom:lm", "/custom/lm/v1/batches"),
    ],
)
async def test_azure_and_custom_listings_are_restored_too(
    provider: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = ScriptedRouter(session="read-empty", owners={"obj_1": "user:n1:main"})
    listing = {"object": "list", "data": [{"id": "obj_1", "metadata": {"who": TOKEN}}]}
    upstream = Upstream(listing)
    app = _app(monkeypatch, router, upstream, providers={provider: ProviderConfig(UPSTREAM)})
    app.state.proxy.vault_manager.get("user:n1:main").placeholder_for("EMAIL", ADA)
    async with _client(app) as client:
        response = await client.get(path)
    assert response.json()["data"][0]["metadata"]["who"] == ADA


async def test_listings_are_left_alone_when_nothing_applies(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    raw_listing = json.dumps(_listing(), indent=1).encode()
    tokenless = json.dumps({"object": "list", "data": [{"id": "file-own"}]}, indent=1).encode()
    named = {"file-own": "owner"}
    cases: list[tuple[Any, int, bytes, str]] = [
        (ScriptedRouter(), 200, raw_listing, "application/json"),  # names nothing
        (ScriptedRouter(listing_error=RuntimeError("x")), 200, raw_listing, "application/json"),
        (ScriptedRouter(owners=named), 500, raw_listing, "application/json"),  # an error
        (ScriptedRouter(owners=named), 200, b'{"object": "file"}', "application/json"),
        (ScriptedRouter(owners=named), 200, b"not json", "application/json"),
        (ScriptedRouter(owners=named), 200, raw_listing, "text/plain"),
        (ScriptedRouter(owners=named), 200, tokenless, "application/json"),  # nothing to restore
        (OlderRouter(), 200, raw_listing, "application/json"),  # no listing pass at all
    ]
    for router, status, content, content_type in cases:

        def handler(
            request: httpx.Request,
            status: int = status,
            content: bytes = content,
            content_type: str = content_type,
        ) -> httpx.Response:
            return httpx.Response(status, content=content, headers={"content-type": content_type})

        _registry(monkeypatch, router)
        app = create_app(
            Config(providers={**Config().providers, "openai": ProviderConfig(UPSTREAM)}),
            upstream_transport=httpx.MockTransport(handler),
        )
        app.state.proxy.vault_manager.get("owner").placeholder_for("EMAIL", ADA)
        with caplog.at_level(logging.WARNING, logger="llm_redact"):
            async with _client(app) as client:
                response = await client.get("/v1/files")
        assert response.content == content  # byte-identical
    assert "listing_item_session failed (RuntimeError)" in caplog.text
