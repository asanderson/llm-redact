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
    # A batch job is read back by its operation name (batches/<id>).
    assert gemini.tracks_object_ids("POST", "/v1beta/models/m:batchGenerateContent")
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


async def test_nothing_is_reported_for_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter()
    await _post_batch(_files_app(monkeypatch, tmp_path, router, status=400))
    assert router.objects == []


async def test_static_mode_reports_with_the_static_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A router in static mode may still separate namespaces (llm-redact-pro
    # serves unattributed traffic on the static path next to named users):
    # what the SHARED session created is reported under its name, so a
    # named user's later reference to it can be told apart from their own.
    static = OwnershipRouter(mode="static")
    app = _files_app(monkeypatch, tmp_path, static)
    assert (await _post_batch(app)).status_code == 200
    assert static.objects == [("batch_9", "default")]
    assert app.state.proxy.vault_manager.lookup_response_session("batch_9") == "default"


async def test_a_router_without_the_member_is_never_called(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Older(OwnershipRouter):
        record_object_id = None  # type: ignore[assignment]

    app = _files_app(monkeypatch, tmp_path, Older())
    assert (await _post_batch(app)).status_code == 200
    assert app.state.proxy.vault_manager.lookup_response_session("batch_9") is None


@pytest.mark.parametrize(
    ("upload_purpose", "file_purpose", "reported"),
    [
        ("user_data", "user_data", True),
        ("fine-tune", None, True),
        (None, "assistants", True),
        ("batch", "batch", False),
        ("user_data", "batch", False),  # either one saying batch: unsure
        ("Batch", None, False),
        (None, None, False),  # no purpose stated: unsure
        (7, None, False),
    ],
)
def test_a_completed_upload_of_batch_requests_is_never_reported(
    upload_purpose: Any, file_purpose: Any, reported: bool
) -> None:
    """The Uploads API forwards its parts unread (an opaque byte range can
    split a line), so the requests a batch input file assembled from them
    holds were never checked for the stored objects they cite: the file is
    not reported as its uploader's — it stays an object nobody is recorded
    creating (the session router's unknown-object case)."""
    body: dict[str, Any] = {"id": "upload_1", "object": "upload"}
    body["file"] = {"id": "file-big", "object": "file"}
    if upload_purpose is not None:
        body["purpose"] = upload_purpose
    if file_purpose is not None:
        body["file"]["purpose"] = file_purpose
    ids = OpenAIAdapter().object_ids_from_body("POST", "/v1/uploads/upload_1/complete", body)
    assert ids == (("file-big",) if reported else ())


@pytest.mark.parametrize(("purpose", "reported"), [("batch", False), ("user_data", True)])
async def test_a_completed_batch_upload_reaches_no_router(
    purpose: str, reported: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter()
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)
    completed = {
        "id": "upload_1",
        "object": "upload",
        "status": "completed",
        "purpose": purpose,
        "file": {"id": "file-parts", "object": "file", "purpose": purpose},
    }
    config = Config(
        providers={**Config().providers, "openai": ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    app = create_app(
        config,
        upstream_transport=httpx.MockTransport(lambda r: httpx.Response(200, json=completed)),
    )
    response = await _post(app, "/v1/uploads/upload_1/complete", {"part_ids": ["part_1"]})
    assert response.status_code == 200
    assert router.objects == ([("file-parts", "user:n1:main")] if reported else [])
    recorded = app.state.proxy.vault_manager.lookup_response_session("file-parts")
    assert recorded == ("user:n1:main" if reported else None)


# --- video jobs and stored chat completions -------------------------------------------


@pytest.mark.parametrize(
    ("path", "body", "tracked"),
    [
        ("/v1/videos", None, True),
        ("/v1/videos/video_1/remix", None, True),  # a remix creates a NEW video
        ("/openai/v1/videos", None, True),  # Azure's prefix
        ("/custom/lm/v1/videos", None, True),  # a custom provider's prefix
        ("/v1/videos/video_1", None, False),
        ("/v1/videos/video_1/content", None, False),
        ("/v1/chat/completions", {"store": True}, True),
        ("/openai/v1/chat/completions", {"store": True}, True),
        ("/openai/deployments/gpt/chat/completions", {"store": True}, True),
        ("/custom/lm/v1/chat/completions", {"store": True}, True),
        ("/v1/chat/completions", {"store": False}, False),
        ("/v1/chat/completions", {"store": "true"}, False),  # a real boolean only
        ("/v1/chat/completions", {}, False),
        ("/v1/chat/completions", None, False),
        ("/v1/chat/completions/chatcmpl-1", {"store": True}, False),  # metadata update
    ],
)
def test_openai_tracks_videos_and_stored_completions(path: str, body: Any, tracked: bool) -> None:
    assert OpenAIAdapter().tracks_object_ids("POST", path, body=body) is tracked
    assert not OpenAIAdapter().tracks_object_ids("GET", path, body=body)


def _openai_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    router: OwnershipRouter,
    respond: Any,
    provider: str = "openai",
) -> Any:
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)
    config = Config(
        providers={**Config().providers, provider: ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    return create_app(config, upstream_transport=httpx.MockTransport(respond))


async def _post(app: Any, path: str, body: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.post(path, json=body)


def _chat(store: bool | None) -> dict[str, Any]:
    body: dict[str, Any] = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    if store is not None:
        body["store"] = store
    return body


@pytest.mark.parametrize(
    ("provider", "path"),
    [
        ("openai", "/v1/chat/completions"),
        ("azure", "/openai/deployments/gpt/chat/completions"),
        ("azure", "/openai/v1/chat/completions"),
    ],
)
@pytest.mark.parametrize(("store", "reported"), [(True, True), (False, False), (None, False)])
async def test_a_stored_chat_completion_is_reported_with_its_creator(
    provider: str,
    path: str,
    store: bool | None,
    reported: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    router = OwnershipRouter()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "chatcmpl-9", "object": "chat.completion"})

    app = _openai_app(monkeypatch, tmp_path, router, respond, provider)
    assert (await _post(app, path, _chat(store))).status_code == 200
    assert router.objects == ([("chatcmpl-9", "user:n1:main")] if reported else [])


async def test_a_streamed_stored_chat_completion_is_reported_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter()
    chunk = '{"id": "chatcmpl-7", "object": "chat.completion.chunk", "choices": []}'
    stream = f": keep-alive\n\ndata: {chunk}\n\ndata: {chunk}\n\ndata: [DONE]\n\n"

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=stream.encode(), headers={"content-type": "text/event-stream"}
        )

    app = _openai_app(monkeypatch, tmp_path, router, respond)
    response = await _post(app, "/v1/chat/completions", {**_chat(True), "stream": True})
    assert response.status_code == 200 and "chatcmpl-7" in response.text
    assert router.objects == [("chatcmpl-7", "user:n1:main")]
    # Unstored, or a failed stream: nothing.
    router.objects.clear()
    await _post(app, "/v1/chat/completions", {**_chat(False), "stream": True})
    assert router.objects == []


async def test_a_streamed_error_reports_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter()

    def respond(request: httpx.Request) -> httpx.Response:
        data = b'data: {"id": "chatcmpl-x"}\n\n'
        return httpx.Response(500, content=data, headers={"content-type": "text/event-stream"})

    app = _openai_app(monkeypatch, tmp_path, router, respond)
    await _post(app, "/v1/chat/completions", {**_chat(True), "stream": True})
    assert router.objects == []


@pytest.mark.parametrize(
    ("provider", "path"),
    [
        ("openai", "/v1/videos"),
        ("openai", "/v1/videos/video_0/remix"),
        ("azure", "/openai/v1/videos"),
    ],
)
async def test_a_created_video_is_reported_with_its_creator(
    provider: str, path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "video_5", "object": "video", "status": "queued"})

    app = _openai_app(monkeypatch, tmp_path, router, respond, provider)
    assert (await _post(app, path, {"model": "sora-2", "prompt": "a cat"})).status_code == 200
    assert router.objects == [("video_5", "user:n1:main")]


async def test_a_custom_provider_reports_its_stored_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "chatcmpl-c", "object": "chat.completion"})

    _registry(monkeypatch, build_session_router=lambda config, **kw: router)
    config = Config(
        providers={**Config().providers, "custom:lm": ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(respond))
    assert (await _post(app, "/custom/lm/v1/chat/completions", _chat(True))).status_code == 200
    assert router.objects == [("chatcmpl-c", "user:n1:main")]


# --- more stored objects: Anthropic files, OpenAI uploads, long-running jobs -----------


def test_every_provider_reports_the_jobs_and_files_read_back_by_id() -> None:
    from llm_redact.providers.bedrock import BedrockAdapter

    anthropic = AnthropicAdapter()
    assert anthropic.tracks_object_ids("POST", "/v1/files")
    assert not anthropic.tracks_object_ids("GET", "/v1/files/file_1")
    assert not anthropic.tracks_object_ids("POST", "/v1/files/file_1")
    assert anthropic.object_ids_from_body("POST", "/v1/files", {"id": "file_011"}) == ("file_011",)
    assert anthropic.object_ids_from_body("POST", "/v1/files", {"id": ""}) == ()
    openai = OpenAIAdapter()
    complete = "/v1/uploads/upload_1/complete"
    assert openai.tracks_object_ids("POST", complete)
    assert openai.tracks_object_ids("POST", "/openai/v1/uploads/upload_1/complete/")
    assert not openai.tracks_object_ids("POST", "/v1/uploads/upload_1/parts")
    assert not openai.tracks_object_ids("POST", "/v1/uploads")
    done = {
        "id": "upload_1",
        "object": "upload",
        "purpose": "user_data",
        "file": {"id": "file-big", "object": "file", "purpose": "user_data"},
    }
    assert openai.object_ids_from_body("POST", complete, done) == ("file-big",)
    assert openai.object_ids_from_body("POST", complete, {"id": "upload_1", "file": None}) == ()
    assert openai.object_ids_from_body("POST", complete, ["x"]) == ()
    gemini = GeminiAdapter()
    for verb in ("batchGenerateContent", "predictLongRunning"):
        assert gemini.tracks_object_ids("POST", f"/v1beta/models/m:{verb}")
    assert not gemini.tracks_object_ids("POST", "/v1beta/models/m:predict")
    assert not gemini.tracks_object_ids("GET", "/v1beta/models/m:predictLongRunning")
    operation = {"name": "models/veo-3.0-generate-001/operations/op1"}
    assert gemini.object_ids_from_body(
        "POST", "/v1beta/models/m:predictLongRunning", operation
    ) == ("models/veo-3.0-generate-001/operations/op1",)
    vertex = VertexAdapter()
    veo = "/v1/projects/p/locations/l/publishers/google/models/veo-3.0-generate-001"
    assert vertex.tracks_object_ids("POST", veo + ":predictLongRunning")
    assert not vertex.tracks_object_ids("POST", veo + ":fetchPredictOperation")
    assert not vertex.tracks_object_ids("POST", veo + ":generateContent")
    assert not vertex.tracks_object_ids("POST", "/v1/projects/p/locations/l/unknown")
    full = {"name": veo[4:] + "/operations/op2"}
    assert vertex.object_ids_from_body("POST", veo + ":predictLongRunning", full) == (full["name"],)
    bedrock = BedrockAdapter()
    arn = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"
    assert bedrock.tracks_object_ids("POST", "/async-invoke")
    assert not bedrock.tracks_object_ids("GET", "/async-invoke")
    assert not bedrock.tracks_object_ids("GET", f"/async-invoke/{arn}")
    assert bedrock.object_ids_from_body("POST", "/async-invoke", {"invocationArn": arn}) == (arn,)
    assert bedrock.object_ids_from_body("POST", "/async-invoke", {"invocationArn": 7}) == ()
    assert bedrock.object_ids_from_body("POST", "/async-invoke", None) == ()


def test_the_gemini_files_api_is_tracked() -> None:
    """The Gemini API's Files API: a file is created by the media upload
    (the multipart protocol, or a resumable upload's finalizing chunk), the
    metadata-only create, or ``files:register``; a batch's status names its
    output file once it finished. Each is reported as ``files/<id>``."""
    gemini = GeminiAdapter()
    for path in ("/upload/v1beta/files", "/v1beta/files", "/v1beta/files:register"):
        assert gemini.tracks_object_ids("POST", path), path
    assert gemini.tracks_object_ids("GET", "/v1beta/batches/b1")
    for method, path in [
        ("GET", "/v1beta/files/abc"),
        ("DELETE", "/v1beta/files/abc"),
        ("GET", "/v1beta/files"),
        ("GET", "/download/v1beta/files/abc:download"),
        ("POST", "/upload/v1beta/files/abc"),
        ("GET", "/v1beta/batches"),
        ("POST", "/v1beta/batches/b1:cancel"),
        ("DELETE", "/v1beta/batches/b1"),
        ("POST", "/v1/files"),  # OpenAI's collection, never the Gemini API's
    ]:
        assert not gemini.tracks_object_ids(method, path), (method, path)
    uri = "https://generativelanguage.googleapis.com/v1beta/files/abc-123"
    uploaded = {"file": {"name": "files/abc-123", "uri": uri, "state": "ACTIVE"}}
    for path in ("/upload/v1beta/files", "/v1beta/files"):
        assert gemini.object_ids_from_body("POST", path, uploaded) == ("files/abc-123",)
    assert gemini.object_ids_from_body("POST", "/upload/v1beta/files", {"file": {}}) == ()
    assert gemini.object_ids_from_body("POST", "/upload/v1beta/files", {"file": "x"}) == ()
    assert gemini.object_ids_from_body("POST", "/upload/v1beta/files", ["x"]) == ()
    registered = {"files": [{"name": "files/a"}, {"name": "files/b"}, {"name": 7}, "x"]}
    assert gemini.object_ids_from_body("POST", "/v1beta/files:register", registered) == (
        "files/a",
        "files/b",
    )
    assert gemini.object_ids_from_body("POST", "/v1beta/files:register", {"files": "x"}) == ()
    # The `files` array is register's alone: a create answering with one is a
    # LISTING (an upstream that ran another method), never created files —
    # and register answers with no single `file`.
    for path in ("/upload/v1beta/files", "/v1beta/files"):
        assert gemini.object_ids_from_body("POST", path, registered) == ()
        both = {**uploaded, **registered}
        assert gemini.object_ids_from_body("POST", path, both) == ("files/abc-123",)
    assert gemini.object_ids_from_body("POST", "/v1beta/files:register", uploaded) == ()
    status = {
        "name": "batches/b1",
        "done": True,
        "metadata": {"state": "BATCH_STATE_SUCCEEDED", "output": {"responsesFile": "files/o1"}},
        "response": {"@type": "type.googleapis.com/x", "responsesFile": "files/o1"},
    }
    assert gemini.object_ids_from_body("GET", "/v1beta/batches/b1", status) == ("files/o1",)
    running = {"name": "batches/b1", "done": False, "metadata": {"state": "BATCH_STATE_RUNNING"}}
    assert gemini.object_ids_from_body("GET", "/v1beta/batches/b1", running) == ()
    inline = {"name": "batches/b1", "response": {"inlinedResponses": {"inlinedResponses": []}}}
    assert gemini.object_ids_from_body("GET", "/v1beta/batches/b1", inline) == ()
    assert gemini.object_ids_from_body("GET", "/v1beta/batches/b1", {"response": "x"}) == ()
    # Only a Files API name counts (never an operation or a stray string).
    odd = {"file": {"name": "batches/b1"}}
    assert gemini.object_ids_from_body("POST", "/upload/v1beta/files", odd) == ()


async def test_a_gemini_file_upload_and_a_batch_output_are_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gemini Files routes: the upload's answer names the file, and a
    finished batch's status its output file — both reported with the
    session that created (or first read) them."""
    router = OwnershipRouter()
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)
    uri = "https://generativelanguage.googleapis.com/v1beta/files/abc-123"

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"file": {"name": "files/abc-123", "uri": uri}})
        return httpx.Response(
            200, json={"name": "batches/b1", "response": {"responsesFile": "files/o1"}}
        )

    config = Config(
        providers={**Config().providers, "gemini": ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        upload = await client.post(
            "/upload/v1beta/files",
            content=b"--b\r\n\r\n{}\r\n--b--",
            headers={
                "x-goog-upload-protocol": "multipart",
                "content-type": "multipart/related; boundary=b",
            },
        )
        status = await client.get("/v1beta/batches/b1")
    assert upload.status_code == 200 and status.status_code == 200
    assert router.objects == [("files/abc-123", "user:n1:main"), ("files/o1", "user:n1:main")]
    manager = app.state.proxy.vault_manager
    assert manager.lookup_response_session("files/abc-123") == "user:n1:main"


def test_gemini_file_downloads_reach_the_gemini_upstream() -> None:
    state = create_app(Config()).state.proxy
    for path in ("/download/v1beta/files/abc:download", "/upload/v1beta/files"):
        assert state.provider_for(None, path) == "gemini", path


async def test_an_anthropic_files_upload_is_reported_with_its_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Anthropic's Files API upload (a text document is redacted, a binary
    # one forwarded as sent with the client's own key); the
    # anthropic-version header names the provider whose adapter tracks it.
    router = OwnershipRouter()
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "file_011", "type": "file"})

    config = Config(
        providers={**Config().providers, "anthropic": ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        response = await client.post(
            "/v1/files",
            files={"file": ("doc.pdf", b"%PDF-1.7", "application/pdf")},
            headers={"anthropic-version": "2023-06-01"},
        )
    assert response.status_code == 200
    assert router.objects == [("file_011", "user:n1:main")]


async def test_a_bedrock_async_invocation_is_reported_with_its_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arn = "arn:aws:bedrock:us-east-1:123456789012:async-invoke/abc123"
    router = OwnershipRouter()
    _registry(monkeypatch, build_session_router=lambda config, **kw: router)

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"invocationArn": arn})

    config = Config(
        providers={**Config().providers, "bedrock": ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    body = {"modelId": "amazon.nova-reel-v1:0", "modelInput": {"text": "a dog"}}
    assert (await _post(app, "/async-invoke", body)).status_code == 200
    assert router.objects == [(arn, "user:n1:main")]
