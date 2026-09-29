"""The optional ``SessionRouter.sealed`` seam: a session the router says must
stay EMPTY is read for rehydration but never written. A request that would
redact a value into it is refused (a recorded, provider-shaped 403) before
any upstream contact, and the session is left untouched; a request with
nothing to redact is served, its answer restored from the (empty) session,
so the provider's placeholders pass through.

Keyless: a scripted router on a bare Registry stands in for llm-redact-pro.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.proxy import _SEALED_REFUSAL, create_app
from llm_redact.registry import Registry

UPSTREAM = "https://upstream.test"
EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"
SEALED = "user:n1:conv-sealed"


class SealingRouter:
    def __init__(self, *, session: str = SEALED, seal: Any = True) -> None:
        self.mode = "per-user"
        self.session = session
        self.seal = seal
        self.asked: list[str] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return self.session

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def sealed(self, session_id: str) -> Any:
        self.asked.append(session_id)
        if isinstance(self.seal, Exception):
            raise self.seal
        return self.seal


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        content = f"you said {TOKEN}"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, router: Any, upstream: Upstream) -> Any:
    reg = Registry()
    reg.build_session_router = lambda config, **kw: router
    monkeypatch.setattr(registry_mod, "_registry", reg)
    config = Config(
        providers={**Config().providers, "openai": ProviderConfig(UPSTREAM)},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db")),
    )
    return create_app(config, upstream_transport=httpx.MockTransport(upstream))


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _chat(text: str) -> dict[str, Any]:
    return {"model": "gpt-4o", "messages": [{"role": "user", "content": text}]}


async def test_a_value_to_redact_into_a_sealed_session_is_refused_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = SealingRouter()
    upstream = Upstream()
    app = _app(monkeypatch, tmp_path, router, upstream)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 403
    assert response.json() == OpenAIAdapter().error_body(_SEALED_REFUSAL, status=403)
    assert EMAIL not in response.text
    assert upstream.requests == []
    assert router.asked == [SEALED]
    state = app.state.proxy
    assert len(state.vault_manager.get(SEALED)) == 0  # nothing was written
    (row,) = state.recent
    assert row["status"] == 403 and row["session"] == SEALED and row["detections"] == {}


async def test_nothing_to_redact_is_served_and_restores_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = SealingRouter()
    upstream = Upstream()
    app = _app(monkeypatch, tmp_path, router, upstream)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat("go on"))
    assert response.status_code == 200 and len(upstream.requests) == 1
    # The provider's placeholder passes through: the session is empty.
    assert response.json()["choices"][0]["message"]["content"] == f"you said {TOKEN}"


async def test_a_populated_sealed_session_still_restores_but_never_grows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Sealing is about writes: a sealed session's existing mappings are read.
    router = SealingRouter()
    upstream = Upstream()
    app = _app(monkeypatch, tmp_path, router, upstream)
    app.state.proxy.vault_manager.get(SEALED).placeholder_for("EMAIL", EMAIL)
    async with _client(app) as client:
        served = await client.post("/v1/chat/completions", json=_chat("go on"))
        refused = await client.post("/v1/chat/completions", json=_chat("and bob@corp.example too"))
    assert served.json()["choices"][0]["message"]["content"] == f"you said {EMAIL}"
    assert refused.status_code == 403
    assert len(app.state.proxy.vault_manager.get(SEALED)) == 1


async def test_a_multipart_upload_into_a_sealed_session_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream = Upstream()
    app = _app(monkeypatch, tmp_path, SealingRouter(), upstream)
    line = json.dumps({"custom_id": "a", "body": {"messages": [{"content": EMAIL}]}})
    async with _client(app) as client:
        response = await client.post(
            "/v1/files",
            data={"purpose": "batch"},
            files={"file": ("in.jsonl", f"{line}\n".encode(), "application/jsonl")},
        )
    assert response.status_code == 403 and upstream.requests == []


@pytest.mark.parametrize("seal", [False, 0, None, ""])
async def test_an_unsealed_session_redacts_as_usual(
    seal: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream = Upstream()
    app = _app(monkeypatch, tmp_path, SealingRouter(seal=seal), upstream)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 200
    assert EMAIL.encode() not in upstream.requests[0].content
    assert response.json()["choices"][0]["message"]["content"] == f"you said {EMAIL}"


async def test_the_routers_own_reason_is_the_refusal_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reason = "llm-redact: this conversation's earlier turns are no longer known"
    upstream = Upstream()
    app = _app(monkeypatch, tmp_path, SealingRouter(seal=reason), upstream)
    async with _client(app) as client:
        refused = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
        served = await client.post("/v1/chat/completions", json=_chat("go on"))
    assert refused.status_code == 403
    assert refused.json() == OpenAIAdapter().error_body(reason, status=403)
    assert served.status_code == 200 and len(upstream.requests) == 1


async def test_a_failing_router_seals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = Upstream()
    router = SealingRouter(seal=LookupError("conv-secret"))
    app = _app(monkeypatch, tmp_path, router, upstream)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        async with _client(app) as client:
            response = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 403 and upstream.requests == []
    assert "LookupError" in caplog.text and "conv-secret" not in caplog.text


async def test_the_static_session_can_be_sealed_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A router may resolve to the configured static session and seal it:
    # the prebuilt static context is not reused then.
    upstream = Upstream()
    app = _app(monkeypatch, tmp_path, SealingRouter(session="default"), upstream)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 403 and upstream.requests == []


async def test_a_router_without_the_member_never_seals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Older(SealingRouter):
        sealed = None  # type: ignore[assignment]

    upstream = Upstream()
    app = _app(monkeypatch, tmp_path, Older(), upstream)
    async with _client(app) as client:
        response = await client.post("/v1/chat/completions", json=_chat(f"mail {EMAIL}"))
    assert response.status_code == 200 and len(upstream.requests) == 1
