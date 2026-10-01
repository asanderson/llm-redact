"""Refusal overrides never apply under a credential the PROXY holds
(docs/overrides.md, "What can never be overridden"): its own cloud identity
(``auth = "identity"``) or a routed plan's operator key
(``RoutePlan.proxy_credential``). Every refusal under one — a block-mode
value, a verbatim field, values in an inspected binary upload, a realtime
frame — carries no code, mints no pending record, and an every-time rule
the requester approved elsewhere (a value rule applies on any route) does
not pass it. Pinned end to end through the real app; nothing reaches the
upstream or the authorizer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets

import llm_redact.registry as registry_mod
from lent_routes import client as lent_client
from lent_routes import lent_app
from llm_redact.config import Config, OverridesConfig, ProviderConfig
from llm_redact.detection.engine import DetectionConfig
from llm_redact.overrides import OverrideStore
from llm_redact.proxy import create_app
from llm_redact.registry import Registry
from test_realtime_identity import FakeAuth, _closed
from test_realtime_relay import FakeUpstream, _proxy
from test_upload_inspection import FakeInspector, reads

EMAIL = "jane.doe@corp.example"
AZURE = "https://res.openai.azure.com"
AZURE_CHAT = "/openai/deployments/gpt/chat/completions?api-version=2024-10-21"
AZURE_FILES = "/openai/files?api-version=2024-10-21"
PDF = b"%PDF-1.7\n1 0 obj << >> endobj\n%%EOF\n"
FORM = {"content-type": "multipart/form-data; boundary=b"}


def _form(content: bytes) -> bytes:
    return (
        b'--b\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nassistants\r\n'
        b'--b\r\nContent-Disposition: form-data; name="file"; filename="r.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n" + content + b"\r\n--b--\r\n"
    )


def _store(tmp_path: Path) -> OverrideStore:
    return OverrideStore(tmp_path / "overrides.db")


def _approved_always(tmp_path: Path, kind: str = "block") -> None:
    """An every-time rule for EMAIL the local operator approved (a value rule
    covers every route)."""
    store = _store(tmp_path)
    code = store.record_pending(
        kind, "", "openai", "POST", "/v1/chat/completions", [("EMAIL", EMAIL)]
    )
    store.approve("always", approver=None, code=code)


def _no_new_pending(tmp_path: Path) -> None:
    assert [e for e in _store(tmp_path).entries() if e.state == "pending"] == []


class _Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"id": "x", "object": "file", "choices": []})


def _identity_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    upstream: _Upstream,
    auth: FakeAuth,
    *,
    inspector: Any = None,
    detection: DetectionConfig | None = None,
) -> Any:
    reg = Registry()
    reg.build_upstream_auth = lambda name, provider: (  # type: ignore[method-assign]
        auth if provider.auth == "identity" else None
    )
    if inspector is not None:
        reg.build_upload_inspector = lambda config, tier: inspector  # type: ignore[method-assign]
    monkeypatch.setattr(registry_mod, "_registry", reg)
    config = Config(
        providers={**Config().providers, "azure": ProviderConfig(AZURE, auth="identity")},
        detection=detection or DetectionConfig(modes=(("email", "block"),)),
        overrides=OverridesConfig(enabled=True, path=str(tmp_path / "overrides.db")),
    )
    return create_app(config, upstream_transport=httpx.MockTransport(upstream))


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


@pytest.mark.parametrize("approved", [False, True])
async def test_a_block_value_under_identity_is_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approved: bool
) -> None:
    if approved:
        _approved_always(tmp_path)
    upstream, auth = _Upstream(), FakeAuth()
    app = _identity_app(tmp_path, monkeypatch, upstream, auth)
    body = {"messages": [{"role": "user", "content": f"mail {EMAIL}"}]}
    async with _client(app) as client:
        for _ in range(2):
            refused = await client.post(AZURE_CHAT, json=body)
            assert refused.status_code == 400
            assert "llm-redact override" not in refused.text
            assert "dashboard" not in refused.text
    assert upstream.requests == [] and auth.calls == []
    _no_new_pending(tmp_path)


@pytest.mark.parametrize("vouches", [False, True])
@pytest.mark.parametrize("approved", [False, True])
async def test_values_in_a_binary_upload_under_identity_carry_no_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approved: bool, vouches: bool
) -> None:
    # Whether or not the inspector vouches for the proxy's credential
    # (``Inspection.proxy_credential``; the core extraction's default does
    # not): no refusal names a code (UP-4), and an every-time value rule
    # approved elsewhere never clears a part a reading vouched for — the
    # file would leave with its value under the proxy's identity.
    if approved:
        _approved_always(tmp_path, "binary_values")
    upstream, auth = _Upstream(), FakeAuth()
    inspector = FakeInspector(reads(f"contact {EMAIL} for details", proxy_credential=vouches))
    app = _identity_app(
        tmp_path, monkeypatch, upstream, auth, inspector=inspector, detection=DetectionConfig()
    )
    async with _client(app) as client:
        refused = await client.post(AZURE_FILES, content=_form(PDF), headers=FORM)
    assert refused.status_code == 400, refused.text
    assert "llm-redact override" not in refused.text
    assert upstream.requests == [] and auth.calls == []
    _no_new_pending(tmp_path)


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("kind", ["block", "verbatim_field"])
async def test_refusals_under_a_routed_operator_key_are_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approved: bool, kind: str
) -> None:
    if approved:
        _approved_always(tmp_path, kind)
    upstream = _Upstream()
    block = kind == "block"
    app, _ = lent_app(
        monkeypatch,
        "openai",
        "http://upstream",
        upstream,
        detection=DetectionConfig(modes=(("email", "block"),) if block else ()),
        overrides=OverridesConfig(enabled=True, path=str(tmp_path / "overrides.db")),
    )
    path, body = (
        ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": EMAIL}]})
        if block
        else ("/v1/fine_tuning/jobs", {"model": "m", "training_file": "file-a", "suffix": EMAIL})
    )
    async with lent_client(app) as client:
        refused = await client.post(path, json=body)
    assert refused.status_code == 400, refused.text
    assert "llm-redact override" not in refused.text
    assert ("blocked" if block else "`suffix`") in refused.text
    assert upstream.requests == []
    _no_new_pending(tmp_path)


async def test_the_client_key_through_the_router_still_gets_a_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The control: a routed plan spending the CLIENT's own key is no
    # credential of the proxy's — its refusal carries the code as before.
    upstream = _Upstream()
    app, _ = lent_app(
        monkeypatch,
        "openai",
        "http://upstream",
        upstream,
        proxy_credential=False,
        detection=DetectionConfig(modes=(("email", "block"),)),
        overrides=OverridesConfig(enabled=True, path=str(tmp_path / "overrides.db")),
    )
    chat = {"model": "gpt-4o", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]}
    async with lent_client(app) as client:
        refused = await client.post(
            "/v1/chat/completions", json=chat, headers={"authorization": "Bearer sk-client"}
        )
    assert refused.status_code == 400 and "llm-redact override" in refused.text


@pytest.mark.parametrize("approved", [False, True])
async def test_a_realtime_frame_under_identity_is_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approved: bool
) -> None:
    if approved:
        _approved_always(tmp_path)
    auth = FakeAuth()
    reg = Registry()
    reg.build_upstream_auth = lambda name, provider: (  # type: ignore[method-assign]
        auth if provider.auth == "identity" else None
    )
    monkeypatch.setattr(registry_mod, "_registry", reg)
    frame = json.dumps(
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "content": [{"type": "input_text", "text": EMAIL}]},
        }
    )
    async with FakeUpstream() as fake:
        config = Config(
            providers={
                **Config().providers,
                "azure": ProviderConfig(f"http://127.0.0.1:{fake.port}", auth="identity"),
            },
            detection=DetectionConfig(modes=(("email", "block"),)),
            overrides=OverridesConfig(enabled=True, path=str(tmp_path / "overrides.db")),
        )
        with _proxy(config) as host:
            async with websockets.connect(f"ws://{host}/openai/v1/realtime?model=m") as ws:
                await ws.send(frame)
                closed = await _closed(ws)
    assert closed.code == 1008
    assert "llm-redact override" not in closed.reason
    assert fake.received == []
    _no_new_pending(tmp_path)
