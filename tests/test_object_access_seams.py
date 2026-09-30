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
from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from license_fixtures import resolved
from llm_redact.audit import AuditRecord
from llm_redact.config import AuditConfig, Config, ProviderConfig, VaultConfig
from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.azure_openai import AzureOpenAIAdapter
from llm_redact.providers.base import ProviderAdapter
from llm_redact.providers.custom import CustomOpenAIAdapter
from llm_redact.providers.ollama import OllamaAdapter
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
        ("DELETE", "/v1/vector_stores/vs_1", None, None),  # pass-through: no adapter, no body
        ("DELETE", "/v1/files/file-1", "openai", None),  # recognized: an id-only route
        ("GET", "/v1/files/file-1/content", "openai", None),
        ("POST", "/v1/batches/batch_1/cancel", "openai", None),
        ("DELETE", "/v1/videos/video_1", "openai", None),
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
        response = await client.delete("/v1/vector_stores/vs_1")  # pass-through
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


@pytest.mark.parametrize(
    ("plan_kwargs", "identity"),
    [
        pytest.param({}, True, id="plan-without-the-member"),  # fail closed
        pytest.param({"proxy_credential": True}, True, id="proxy-credential"),
        pytest.param({"proxy_credential": False}, False, id="client-credential"),
    ],
)
async def test_a_routed_request_is_planned_then_checked_before_any_hop(
    plan_kwargs: dict[str, Any], identity: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The router plans first — the check must know whose credential the
    request spends — and the check refuses before begin(): no hop, no
    redaction, no session resolved. A plan that lends the proxy's own
    credential (or does not say) is checked like identity auth."""
    fake = FakeRouter(
        {"r": [Hop("x", "http://x.example/v1/messages"), Stop()]}, plan_kwargs={"r": plan_kwargs}
    )
    reg, _ = install(monkeypatch, fake)
    router = ScriptedRouter(verdict=REFUSAL)
    _registry(monkeypatch, router, reg)
    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(Upstream()))
    body = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
    async with _client(app) as client:
        response = await client.post("/v1/messages", json=body, headers={ROUTE_HEADER: "r"})
    assert response.status_code == 403
    assert response.json() == AnthropicAdapter().error_body(REFUSAL, status=403)
    (plan,) = fake.plans
    assert plan.begun == [] and plan.hops_issued == []  # planned, never begun
    assert router.checks == [("anthropic", "POST", "/v1/messages", body, identity)]
    assert router.resolved == 0
    # Allowed, the same plan is begun and its hop issued.
    router.verdict = None
    async with _client(app) as client:
        await client.post("/v1/messages", json=body, headers={ROUTE_HEADER: "r"})
    assert len(fake.plans) == 2 and len(fake.plans[1].begun) == 1


async def test_an_unrouted_request_is_checked_with_the_clients_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRouter()
    reg, _ = install(monkeypatch, fake)
    router = ScriptedRouter()
    _registry(monkeypatch, router, reg)
    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(Upstream()))
    body = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
    async with _client(app) as client:
        response = await client.post("/v1/messages", json=body)  # plan() -> None
    assert response.status_code == 200
    assert router.checks == [("anthropic", "POST", "/v1/messages", body, False)]


# --- routed pass-through bodies under the proxy's credential ---------------------------


def _routed_pass_through(
    monkeypatch: pytest.MonkeyPatch, router: Any, *, proxy_credential: bool | None = None
) -> tuple[Any, FakeRouter]:
    hop = Hop("x", "http://x.example/v1/vector_stores")
    kwargs = {} if proxy_credential is None else {"proxy_credential": proxy_credential}
    fake = FakeRouter({"r": [hop, Stop()]}, plan_kwargs={"r": kwargs})
    reg, _ = install(monkeypatch, fake)
    _registry(monkeypatch, router, reg)
    upstream = Upstream({"id": "vs_1"})
    return create_app(routed_config(), upstream_transport=httpx.MockTransport(upstream)), fake


@pytest.mark.parametrize("lends", [None, True], ids=["member-absent", "true"])
async def test_a_routed_pass_through_under_the_proxys_credential_is_refused_unread(
    lends: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrecognized route is never lent a credential the proxy holds
    (C-R2-04): refused before its body is read or the ownership check is
    asked — it once went out unredacted with the operator's key."""
    router = ScriptedRouter()
    app, fake = _routed_pass_through(monkeypatch, router, proxy_credential=lends)
    raw = b'{"name": "mine",   "file_ids": ["file-a"]}'
    async with _client(app) as client:
        response = await client.post("/v1/vector_stores", content=raw, headers={ROUTE_HEADER: "r"})
    assert response.status_code == 403
    assert "credential the proxy holds" in response.json()["error"]
    assert router.checks == []
    (plan,) = fake.plans
    assert plan.begun == []  # never begun: no audit row, no hop


@pytest.mark.parametrize(
    ("proxy_credential", "routed", "checks_ownership"),
    [
        pytest.param(False, True, True, id="client-credential"),
        pytest.param(None, False, True, id="unrouted"),
        pytest.param(False, True, False, id="no-ownership-check"),
    ],
)
async def test_other_pass_through_bodies_are_never_parsed(
    proxy_credential: bool | None,
    routed: bool,
    checks_ownership: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router: Any = ScriptedRouter() if checks_ownership else OlderRouter()
    app, fake = _routed_pass_through(monkeypatch, router, proxy_credential=proxy_credential)
    headers = {ROUTE_HEADER: "r"} if routed else {}
    async with _client(app) as client:
        response = await client.post(
            "/v1/vector_stores",
            content=b'{"file_ids": ["file-a"], "file_ids": ["file-b"]}',
            headers={**headers, "content-encoding": "gzip"},  # never inspected here
        )
    assert response.status_code == 200
    if checks_ownership:
        assert router.checks == [(None, "POST", "/v1/vector_stores", None, False)]


@pytest.mark.parametrize(
    ("raw", "headers"),
    [
        pytest.param(b'{"a": 1}', {"content-encoding": "gzip"}, id="gzip"),
        pytest.param(
            b'{"a": 1}',
            [("content-encoding", "identity"), ("content-encoding", "identity, br")],
            id="second-encoding-header",
        ),
        pytest.param(b'{"file_ids": ["file-a"], "file_ids": ["file-b"]}', {}, id="repeated-key"),
        pytest.param(
            b" \n\xef\xbb\xbf" + b'{"file_ids": []' + b" " * 64 + b"}", {}, id="oversized-json"
        ),
        pytest.param(b"\x00[" + b"\x00 " * 40 + b"\x00]", {}, id="oversized-utf16"),
        pytest.param(b"\xff\xfb" + b"\x00" * 200, {}, id="oversized-audio"),
    ],
)
async def test_no_pass_through_body_is_read_under_the_proxys_credential(
    raw: bytes,
    headers: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever the body — one the check could not read, or one it need not
    — an unrecognized route under the proxy's credential is refused before
    the (unbounded) body read: the same recorded 403 for every body."""
    router = ScriptedRouter()
    hop = Hop("x", "http://x.example/v1/vector_stores")
    fake = FakeRouter({"r": [hop, Stop()]})
    reg, _ = install(monkeypatch, fake)
    _registry(monkeypatch, router, reg)
    upstream = Upstream()
    app = create_app(
        routed_config(max_body_bytes=64), upstream_transport=httpx.MockTransport(upstream)
    )
    request_headers = httpx.Headers(headers)
    request_headers[ROUTE_HEADER] = "r"
    async with _client(app) as client:
        response = await client.post("/v1/vector_stores", content=raw, headers=request_headers)
    assert response.status_code == 403
    assert "credential the proxy holds" in response.json()["error"]
    assert router.checks == [] and upstream.requests == []
    assert fake.plans[0].begun == []
    (row,) = app.state.proxy.recent
    assert row["status"] == 403 and row["provider"] == "openai"


async def test_with_the_clients_own_credential_a_pass_through_body_is_forwarded_unread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ScriptedRouter()
    hop = Hop("x", "http://x.example/v1/audio/transcriptions")
    fake = FakeRouter({"r": [hop, Stop()]}, plan_kwargs={"r": {"proxy_credential": False}})
    reg, _ = install(monkeypatch, fake)
    _registry(monkeypatch, router, reg)
    app = create_app(
        routed_config(max_body_bytes=64), upstream_transport=httpx.MockTransport(Upstream())
    )
    raw = b"\xff\xfb" + b"\x00" * 200  # an MP3 frame, not JSON
    async with _client(app) as client:
        response = await client.post(
            "/v1/audio/transcriptions", content=raw, headers={ROUTE_HEADER: "r"}
        )
    assert response.status_code == 200
    assert router.checks == [(None, "POST", "/v1/audio/transcriptions", None, False)]
    assert fake.plans[0].begun[0][0] == raw  # forwarded byte-for-byte


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
    assert OllamaAdapter().listing_items(body) is None  # the base default
    assert AnthropicAdapter().listing_items(body) is items  # its Files API list


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
    # What the core mirrored when each object was created (a durable map
    # only — the memory manager keeps none).
    state.vault_manager.record_response_session("file-own", "user:n1:main")
    state.vault_manager.record_response_session("file-gone", "user:n1:never-created")
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
    assert state.vault_manager.lookup_response_session("file-other") is None
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


# --- a listing is never restored from a session younger than the object ---------------


def _stored_completion_listing() -> dict[str, Any]:
    item = {"id": "chatcmpl-1", "object": "chat.completion", "note": f"mail {TOKEN}"}
    return {"object": "list", "data": [item]}


async def test_a_pruned_and_recreated_session_never_restores_an_older_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A router that remembers owners forever keeps naming the conversation
    session a stored completion was created in. Once that session is pruned
    (its mappings AND the core's durable ownership row go together) and the
    same opener recreates it, «EMAIL_001» means a NEW value there: the item
    keeps its placeholder (never-wrong-value)."""
    router = ScriptedRouter(session="read-empty", owners={"chatcmpl-1": "user:n1:conv-x"})
    app = _app(
        monkeypatch,
        router,
        Upstream(_stored_completion_listing()),
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    manager = app.state.proxy.vault_manager
    manager.get("user:n1:conv-x").placeholder_for("EMAIL", ADA)
    manager.record_response_session("chatcmpl-1", "user:n1:conv-x")  # the create's record
    async with _client(app) as client:
        before = (await client.get("/v1/chat/completions")).json()["data"][0]
        assert before["note"] == f"mail {ADA}"  # the creator's own session, as recorded
        assert manager.forget_sessions(["user:n1:conv-x"]) == 1  # the whole-session prune
        manager.get("user:n1:conv-x").placeholder_for("EMAIL", BOB)  # recreated
        after = (await client.get("/v1/chat/completions")).json()["data"][0]
    assert after["note"] == f"mail {TOKEN}"
    assert router.listed == ["chatcmpl-1", "chatcmpl-1"]


async def test_a_failing_ownership_check_leaves_the_listed_items_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    router = ScriptedRouter(session="read-empty", owners={"chatcmpl-1": "user:n1:main"})
    app = _app(
        monkeypatch,
        router,
        Upstream(_stored_completion_listing()),
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    manager = app.state.proxy.vault_manager
    manager.get("user:n1:main").placeholder_for("EMAIL", ADA)
    manager.record_response_session("chatcmpl-1", "user:n1:main")

    def broken(ids: Any) -> dict[str, str]:
        raise OSError("disk says no to chatcmpl-1")

    monkeypatch.setattr(manager, "lookup_response_sessions", broken)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with _client(app) as client:
            item = (await client.get("/v1/chat/completions")).json()["data"][0]
    assert item["note"] == f"mail {TOKEN}"
    assert "listing ownership check failed (OSError)" in caplog.text
    assert "chatcmpl-1" not in caplog.text


async def test_a_manager_without_the_batched_lookup_is_asked_per_item(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = ScriptedRouter(
        session="read-empty", owners={"chatcmpl-1": "user:n1:main", "chatcmpl-2": "user:n1:gone"}
    )
    listing = _stored_completion_listing()
    listing["data"].append({"id": "chatcmpl-2", "note": f"mail {TOKEN}"})
    app = _app(
        monkeypatch,
        router,
        Upstream(listing),
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    manager = app.state.proxy.vault_manager
    manager.get("user:n1:main").placeholder_for("EMAIL", ADA)
    manager.get("user:n1:gone").placeholder_for("EMAIL", BOB)
    manager.record_response_session("chatcmpl-1", "user:n1:main")  # chatcmpl-2: no row
    monkeypatch.delattr(type(manager), "lookup_response_sessions")
    async with _client(app) as client:
        data = (await client.get("/v1/chat/completions")).json()["data"]
    assert [item["note"] for item in data] == [f"mail {ADA}", f"mail {TOKEN}"]


class BatchedRouter(ScriptedRouter):
    """A router answering a whole listing at once (``listing_item_sessions``)."""

    def __init__(self, *, answer: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.answer = answer
        self.batches: list[list[str]] = []

    def listing_item_sessions(self, object_ids: list[str]) -> Any:
        self.batches.append(list(object_ids))
        if self.answer is not None:
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer
        return [self.owners.get(object_id) for object_id in object_ids]


async def test_a_listing_is_answered_by_one_batched_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = BatchedRouter(session="read-empty", owners={"a": "user:n1:main", "c": "user:n1:main"})
    listing = {
        "object": "list",
        "data": [{"id": object_id, "note": TOKEN} for object_id in ("a", "b", "c", "a")],
    }
    app = _app(monkeypatch, router, Upstream(listing))
    app.state.proxy.vault_manager.get("user:n1:main").placeholder_for("EMAIL", ADA)
    async with _client(app) as client:
        data = (await client.get("/v1/files")).json()["data"]
    assert [item["note"] for item in data] == [ADA, TOKEN, ADA, ADA]
    assert router.batches == [["a", "b", "c"]] and router.listed == []  # never per item


@pytest.mark.parametrize(
    ("answer", "logged"),
    [
        (RuntimeError("secret-id"), "listing_item_sessions failed (RuntimeError)"),
        (["user:n1:main"], "listing_item_sessions miscounted"),
    ],
)
async def test_a_failing_batched_answer_leaves_the_listing_alone(
    answer: Any, logged: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    router = BatchedRouter(session="read-empty", answer=answer)
    listing = {"object": "list", "data": [{"id": "a", "note": TOKEN}, {"id": "b", "note": TOKEN}]}
    app = _app(monkeypatch, router, Upstream(listing))
    app.state.proxy.vault_manager.get("user:n1:main").placeholder_for("EMAIL", ADA)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with _client(app) as client:
            response = await client.get("/v1/files")
    assert [item["note"] for item in response.json()["data"]] == [TOKEN, TOKEN]
    assert logged in caplog.text and "secret-id" not in caplog.text


@pytest.mark.parametrize(
    ("answer", "logged"),
    [
        (RuntimeError("secret-id"), "listing_item_sessions failed (RuntimeError)"),
        (["user:n1:main"], "listing_item_sessions miscounted"),
    ],
)
async def test_a_failing_answer_delivers_every_item_as_the_provider_sent_it(
    answer: Any, logged: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The listing is read in a POPULATED session (an unattributed reader on
    the static path holds its own «EMAIL_001»): a router that cannot answer
    vouches for no item, so none is restored with that session's values —
    each goes out exactly as the provider sent it."""
    router = BatchedRouter(session="shared", answer=answer)
    listing = {"object": "list", "data": [{"id": "a", "note": TOKEN}, {"id": "b", "note": TOKEN}]}
    app = _app(monkeypatch, router, Upstream(listing))
    app.state.proxy.vault_manager.get("shared").placeholder_for("EMAIL", BOB)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with _client(app) as client:
            response = await client.get("/v1/files")
    assert [item["note"] for item in response.json()["data"]] == [TOKEN, TOKEN]
    assert BOB not in response.text
    assert logged in caplog.text and "secret-id" not in caplog.text
    # One listing, one bookkeeping fault (/status bookkeeping_errors_total).
    assert app.state.proxy.bookkeeping_errors == {"listing": 1}


async def test_an_item_whose_answer_fails_is_delivered_as_the_provider_sent_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class OneFails(ScriptedRouter):
        def listing_item_session(self, object_id: str) -> str | None:
            self.listed.append(object_id)
            if object_id == "b":
                raise LookupError("secret-b")
            return None  # the router names no session: the listing's own session

    router = OneFails(session="shared")
    listing = {"object": "list", "data": [{"id": "a", "note": TOKEN}, {"id": "b", "note": TOKEN}]}
    app = _app(monkeypatch, router, Upstream(listing))
    app.state.proxy.vault_manager.get("shared").placeholder_for("EMAIL", BOB)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with _client(app) as client:
            data = (await client.get("/v1/files")).json()["data"]
    assert [item["note"] for item in data] == [BOB, TOKEN]
    assert "listing_item_session failed (LookupError)" in caplog.text
    assert "secret-b" not in caplog.text
    assert app.state.proxy.bookkeeping_errors == {"listing": 1}


class BatchedOnlyRouter:
    """A router offering only the batched listing member."""

    mode = "per-user"

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return "read-empty"

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def listing_item_sessions(self, object_ids: list[str]) -> list[str | None]:
        self.batches.append(list(object_ids))
        return ["user:n1:main" for _ in object_ids]


async def test_the_batched_member_alone_enables_listing_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = BatchedOnlyRouter()
    listing = {"object": "list", "data": [{"id": "a", "note": TOKEN}]}
    app = _app(monkeypatch, router, Upstream(listing))
    app.state.proxy.vault_manager.get("user:n1:main").placeholder_for("EMAIL", ADA)
    async with _client(app) as client:
        data = (await client.get("/v1/files")).json()["data"]
    assert data == [{"id": "a", "note": ADA}] and router.batches == [["a"]]


# --- the vault managers' batched response-map lookup ------------------------------------


def test_the_sqlite_manager_looks_up_many_ids_per_query(tmp_path: Path) -> None:
    from llm_redact.vault import LOOKUP_CHUNK, SqliteVaultManager

    manager = SqliteVaultManager(tmp_path / "v.db")
    try:
        ids = [f"obj-{n}" for n in range(LOOKUP_CHUNK + 1)]
        for object_id in ids[::2]:
            manager.record_response_session(object_id, f"s-{object_id}")
        queries: list[str] = []
        manager._conn.set_trace_callback(queries.append)
        found = manager.lookup_response_sessions([*ids, ids[0]])  # a repeat is asked once
        manager._conn.set_trace_callback(None)
        assert found == {object_id: f"s-{object_id}" for object_id in ids[::2]}
        assert len([q for q in queries if q.startswith("SELECT")]) == 2  # one per chunk
        assert manager.lookup_response_sessions([]) == {}
    finally:
        manager.close()


def test_the_memory_manager_has_no_durable_answers() -> None:
    assert InMemoryVaultManager().lookup_response_sessions(["a", "b"]) == {}


# --- an uploaded file's lines are requests too ------------------------------------------


class LineRouter(ScriptedRouter):
    """Refuses an upload (a list body: what it cites) that names ``file-a``
    anywhere — in an uploaded line or a form field."""

    def object_access_refusal(
        self, adapter_name: str | None, method: str, path: str, body: Any, *, identity: bool
    ) -> Any:
        self.checks.append((adapter_name, method, path, body, identity))
        return REFUSAL if isinstance(body, list) and "file-a" in json.dumps(body) else None


def _batch_line(file_id: str) -> dict[str, Any]:
    content = [{"type": "input_file", "file_id": file_id}]
    return {
        "custom_id": "1",
        "method": "POST",
        "url": "/v1/responses",
        "body": {"model": "m", "input": [{"role": "user", "content": content}]},
    }


async def test_an_uploaded_batch_files_lines_are_checked_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = LineRouter()
    upstream = Upstream()
    app = _app(monkeypatch, router, upstream)
    lines = [_batch_line("file-a"), {"custom_id": "2", "body": {"input": f"hi {ADA}"}}]
    upload = b"".join(json.dumps(line).encode() + b"\n" for line in lines) + b"not json\n"
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            files={"file": ("in.jsonl", upload, "application/jsonl")},
            data={"purpose": "batch"},
        )
    assert response.status_code == 403 and REFUSAL in response.text
    assert upstream.requests == []
    # ONE check, of everything the upload cites (its form field, then the
    # file's JSON lines), asked BEFORE redaction: nothing was written.
    assert router.checks == [("openai", "POST", "/v1/files", [{"purpose": "batch"}, *lines], False)]
    assert app.state.proxy.vault_manager.total_entries() == 0
    (row,) = app.state.proxy.recent
    assert row["status"] == 403


@pytest.mark.parametrize("detection", [True, False], ids=["detection-on", "detection-off"])
async def test_an_upload_is_checked_once_with_what_it_cites(
    detection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The check reads the upload whether or not the route redacts; a CSV
    # file has no JSON line to cite. Where redaction applies, its lines —
    # not JSONL — are then refused by the scanned-body rule; detection =
    # false forwards the upload as sent.
    router = LineRouter()
    upstream = Upstream()
    app = _app(
        monkeypatch,
        router,
        upstream,
        providers={"openai": ProviderConfig(UPSTREAM, detection=detection)},
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            files={"file": ("contacts.csv", b"name,email\nada,x\n", "text/csv")},
            data={"purpose": "assistants"},
        )
    assert response.status_code == (400 if detection else 200)
    assert len(upstream.requests) == (0 if detection else 1)
    assert router.checks == [("openai", "POST", "/v1/files", [{"purpose": "assistants"}], False)]


@pytest.mark.parametrize(
    ("proxy_credential", "detection", "status"),
    [(None, True, 400), (None, False, 400), (False, True, 400), (False, False, 200)],
)
async def test_an_upload_the_check_cannot_read_is_never_sent_with_the_proxys_credential(
    proxy_credential: bool | None, detection: bool, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multipart upload outside the canonical grammar would be forwarded
    unread (its lines never reach the check): refused when the routed plan
    would send it with the proxy's own credential, and — the scanned-body
    rule — wherever redaction applies; forwarded as sent only with the
    client's own credential and detection = false."""
    router = LineRouter()
    kwargs = {} if proxy_credential is None else {"proxy_credential": proxy_credential}
    fake = FakeRouter(
        {"r": [Hop("x", "http://x.example/v1/files"), Stop()]}, plan_kwargs={"r": kwargs}
    )
    reg, _ = install(monkeypatch, fake)
    _registry(monkeypatch, router, reg)
    upstream = Upstream()
    openai = ProviderConfig(UPSTREAM, detection=detection)
    app = create_app(
        routed_config(providers={**Config().providers, "openai": openai}),
        upstream_transport=httpx.MockTransport(upstream),
    )
    line = json.dumps(_batch_line("file-a")).encode()
    body = (  # LF line endings: outside the canonical CRLF grammar
        b'--b\nContent-Disposition: form-data; name="file"; filename="in.jsonl"\n\n'
        + line
        + b"\n--b--\n"
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            content=body,
            headers={"content-type": "multipart/form-data; boundary=b", ROUTE_HEADER: "r"},
        )
    assert response.status_code == status
    if status == 400:
        assert "outside the canonical form" in response.text
        assert fake.plans[0].begun == []
    else:
        assert fake.plans[0].begun[0][0] == body  # forwarded byte-for-byte
