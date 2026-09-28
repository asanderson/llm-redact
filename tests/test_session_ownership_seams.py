"""Two optional session seams an access-control plugin builds on:

- ``AccessGate.bind_sessions`` hands the gate the live vault's sessions
  (``plugin_api.SessionStore``) so it can drop a deleted user's sessions
  through the running proxy's own vault manager;
- ``SessionRouter.record_object_id`` reports the ids of objects the provider
  stores for later reads (files, batches, stored conversations) with the
  session that created them.

Keyless: scripted fakes on a bare Registry stand in for llm-redact-pro.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.plugin_api import Admission, SessionStore
from llm_redact.providers.anthropic import AnthropicAdapter
from llm_redact.providers.gemini import GeminiAdapter
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.vertex import VertexAdapter
from llm_redact.proxy import create_app
from llm_redact.registry import Registry
from llm_redact.vault import InMemoryVaultManager, SqliteVaultManager

UPSTREAM = "https://upstream.test"


class BindingGate:
    def __init__(self) -> None:
        self.store: SessionStore | None = None

    def admit(self, conn: Any, surface: str) -> Admission:
        return Admission()

    def status(self) -> dict[str, Any]:
        return {}

    async def handle(self, request: Any, host: Any) -> Any:
        raise AssertionError("not reached")

    def close(self) -> None:
        pass

    def bind_sessions(self, store: SessionStore) -> None:
        self.store = store


def _registry(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Registry:
    reg = Registry()
    if "build_access_gate" in overrides:  # a paid tier is where a gate belongs
        reg.resolve_license = lambda *args, **kwargs: resolved("team")
    for name, value in overrides.items():
        setattr(reg, name, value)
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg


# --- bind_sessions / SessionStore ------------------------------------------------------------


def test_the_gate_gets_the_live_sessions_and_can_forget_whole_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = BindingGate()
    _registry(monkeypatch, build_access_gate=lambda config, license: gate)
    app = create_app(
        Config(vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"), session="main"))
    )
    state = app.state.proxy
    for session in ("main", "user:n1:main", "user:n1:conv-a", "user:n2:main"):
        state.vault_manager.get(session).placeholder_for("EMAIL", f"{session}@corp.example")
    state.vault_manager.record_response_session("resp_1", "user:n1:conv-a")
    store = gate.store
    assert store is not None
    assert sorted(store.session_ids()) == ["main", "user:n1:conv-a", "user:n1:main", "user:n2:main"]
    # The configured static session is never forgotten, whatever is asked.
    assert store.forget(["user:n1:main", "user:n1:conv-a", "main", "missing"]) == 2
    assert sorted(store.session_ids()) == ["main", "user:n2:main"]
    assert state.vault_manager.lookup_response_session("resp_1") is None
    # A forgotten session's cached view is gone: it comes back empty.
    assert len(state.vault_manager.get("user:n1:main")) == 0
    assert store.forget([]) == 0


def test_a_gate_without_the_member_is_not_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    class Plain(BindingGate):
        bind_sessions = None  # type: ignore[assignment]  # an older gate

    gate = Plain()
    _registry(monkeypatch, build_access_gate=lambda config, license: gate)
    create_app(Config())  # no AttributeError, nothing bound
    assert gate.store is None


def test_every_vault_manager_forgets_whole_sessions(tmp_path: Path) -> None:
    for manager in (InMemoryVaultManager(), SqliteVaultManager(tmp_path / "m.db")):
        manager.get("a").placeholder_for("EMAIL", "a@corp.example")
        manager.get("b").placeholder_for("EMAIL", "b@corp.example")
        assert manager.forget_sessions(["a", "a", "nope"]) == 1
        assert [row["session"] for row in manager.sessions_summary()] == ["b"]
        assert manager.forget_sessions(["b"]) == 1  # counts the ones that existed
        assert manager.forget_sessions(["b"]) == 0
        assert manager.forget_sessions([]) == 0
        manager.close()


# --- record_object_id ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "tracked"),
    [
        ("POST", "/v1/files", True),
        ("POST", "/v1/batches", True),
        ("POST", "/v1/conversations", True),
        ("GET", "/v1/batches/batch_1", True),
        ("POST", "/v1/batches/batch_1/cancel", True),
        ("POST", "/openai/files", True),  # Azure's prefix
        ("POST", "/custom/lm/api/v1/batches", True),  # a custom provider's prefix
        ("POST", "/v1/conversations/conv_1/items", False),
        ("GET", "/v1/files", False),
        ("GET", "/v1/files/file-1/content", False),
        ("POST", "/v1/chat/completions", False),
    ],
)
def test_openai_tracks_stored_objects(method: str, path: str, tracked: bool) -> None:
    assert OpenAIAdapter().tracks_object_ids(method, path) is tracked


def test_stored_object_ids_are_read_from_the_body() -> None:
    openai = OpenAIAdapter()
    assert openai.object_ids_from_body("POST", "/v1/files", {"id": "file-1"}) == ("file-1",)
    batch = {"id": "batch_1", "output_file_id": "file-out", "error_file_id": None}
    assert openai.object_ids_from_body("POST", "/v1/batches", batch) == ("batch_1", "file-out")
    # A batch's status names its output files, never the batch id again.
    status = {"id": "batch_1", "output_file_id": "file-out", "error_file_id": "file-err"}
    assert openai.object_ids_from_body("GET", "/v1/batches/batch_1", status) == (
        "file-out",
        "file-err",
    )
    assert openai.object_ids_from_body("POST", "/v1/files", ["not", "a", "dict"]) == ()
    anthropic = AnthropicAdapter()
    assert anthropic.tracks_object_ids("POST", "/v1/messages/batches")
    assert not anthropic.tracks_object_ids("GET", "/v1/messages/batches/msgbatch_1/results")
    assert anthropic.object_ids_from_body("POST", "/v1/messages/batches", {"id": "msgbatch_1"}) == (
        "msgbatch_1",
    )
    gemini = GeminiAdapter()
    assert gemini.tracks_object_ids("POST", "/v1beta/cachedContents")
    assert gemini.tracks_object_ids("POST", "/v1/cachedContents")
    assert not gemini.tracks_object_ids("GET", "/v1beta/cachedContents/abc")
    assert not gemini.tracks_object_ids("POST", "/v1beta/models/m:generateContent")
    assert not gemini.tracks_object_ids("POST", "/v1beta/models/m:batchGenerateContent")
    created = {"name": "cachedContents/abc123", "model": "models/gemini-2.5-pro"}
    assert gemini.object_ids_from_body("POST", "/v1beta/cachedContents", created) == (
        "cachedContents/abc123",
    )
    assert gemini.object_ids_from_body("POST", "/v1beta/cachedContents", {"name": ""}) == ()
    assert gemini.object_ids_from_body("POST", "/v1beta/cachedContents", ["x"]) == ()
    # Vertex answers on its own paths, which the Gemini API matcher never
    # claims; its cache create is tracked too, reported in the Gemini form
    # (`cachedContents/<id>`) that later generateContent bodies are read in.
    vertex_create = "/v1/projects/p/locations/us-central1/cachedContents"
    assert not gemini.tracks_object_ids("POST", vertex_create)
    vertex = VertexAdapter()
    assert vertex.tracks_object_ids("POST", vertex_create)
    assert not vertex.tracks_object_ids("GET", vertex_create + "/abc")
    full = {"name": "projects/p/locations/us-central1/cachedContents/abc123"}
    assert vertex.object_ids_from_body("POST", vertex_create, full) == ("cachedContents/abc123",)
    assert vertex.object_ids_from_body("POST", vertex_create, {"name": "odd"}) == ("odd",)
    assert vertex.object_ids_from_body("POST", vertex_create, {}) == ()


class OwnershipRouter:
    def __init__(self, mode: str = "per-user", answer: bool | None = True) -> None:
        self.mode = mode
        self.answer = answer
        self.objects: list[tuple[str, str]] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return "user:n1:main"

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def record_object_id(self, object_id: str, session_id: str) -> bool | None:
        self.objects.append((object_id, session_id))
        return self.answer


def _files_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, router: OwnershipRouter, status: int = 200
) -> Any:
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"id": "batch_9", "output_file_id": None})

    config = Config(
        providers={"openai": ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    return create_app(config, upstream_transport=httpx.MockTransport(upstream))


async def _post_batch(app: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.post("/v1/batches", json={"input_file_id": "file-in"})


@pytest.mark.parametrize(("answer", "mirrored"), [(True, True), (None, True), (False, False)])
async def test_created_objects_are_reported_with_their_session(
    answer: bool | None, mirrored: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter(answer=answer)
    app = _files_app(monkeypatch, tmp_path, router)
    response = await _post_batch(app)
    assert response.status_code == 200 and response.json()["id"] == "batch_9"  # untouched
    assert router.objects == [("batch_9", "user:n1:main")]
    durable = app.state.proxy.vault_manager.lookup_response_session("batch_9")
    assert (durable == "user:n1:main") is mirrored


async def test_a_created_gemini_cache_is_reported_with_its_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter()
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": "cachedContents/c1", "model": "models/g"})

    config = Config(
        providers={"gemini": ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        body = {"model": "models/g", "contents": [{"parts": [{"text": "ada@corp.example"}]}]}
        response = await client.post("/v1beta/cachedContents", json=body)
    assert response.status_code == 200
    assert router.objects == [("cachedContents/c1", "user:n1:main")]
    durable = app.state.proxy.vault_manager.lookup_response_session("cachedContents/c1")
    assert durable == "user:n1:main"


async def test_a_created_vertex_cache_is_reported_in_the_gemini_form(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Vertex names the cache by its full resource name; the router hears the
    # `cachedContents/<id>` form a later generateContent body cites.
    router = OwnershipRouter()
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)
    seen: list[bytes] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        name = "projects/p/locations/us-central1/cachedContents/c2"
        return httpx.Response(200, json={"name": name, "model": "m"})

    config = Config(
        providers={"vertex": ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        body = {"model": "m", "contents": [{"parts": [{"text": "ada@corp.example"}]}]}
        response = await client.post(
            "/v1/projects/p/locations/us-central1/cachedContents", json=body
        )
    assert response.status_code == 200
    assert b"ada@corp.example" not in seen[0]  # the cached content was redacted
    assert router.objects == [("cachedContents/c2", "user:n1:main")]


async def test_nothing_is_reported_for_errors_or_static_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter()
    await _post_batch(_files_app(monkeypatch, tmp_path, router, status=400))
    assert router.objects == []
    static = OwnershipRouter(mode="static")
    await _post_batch(_files_app(monkeypatch, tmp_path, static))
    assert static.objects == []


async def test_a_router_without_the_member_is_never_called(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Older(OwnershipRouter):
        record_object_id = None  # type: ignore[assignment]

    app = _files_app(monkeypatch, tmp_path, Older())
    assert (await _post_batch(app)).status_code == 200
    assert app.state.proxy.vault_manager.lookup_response_session("batch_9") is None
