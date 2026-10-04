"""The access gate's authorization seams on a realtime upgrade.

``authorize_request`` is asked once per upgrade — after admission and adapter
resolution, before the session is opened, the ``[audit] required`` START row,
the upstream authorizer and any dial — with the connection's facts (the
``model`` query parameter included). A refusal is an accept-then-close 1008
with the gate's reason, recorded 403 (kind ``authorization``); nothing is
dialled. ``detection_overlay`` is asked once per connection and holds for
its every frame. A reload that lands while an awaited check runs refuses the
connection like a reload before the dial (1012).

Real sockets end to end (the test_realtime_reload harness: uvicorn on port 0
in a thread, a fake ``websockets`` upstream in the test's loop).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
import websockets

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact.authorization import AUTHORIZATION_FAULT, AUTHORIZATION_STAGE, OVERLAY_FAULT
from llm_redact.plugin_api import AuthorizationRequest, DetectionOverlay
from llm_redact.registry import Registry
from local_refusals import refused_once
from test_authorization_seam import AuthorizingGate, BothGate, OverlayGate
from test_realtime_identity import EMAIL, FakeAuth, _recent
from test_realtime_reload import (
    PATHS,
    Upstream,
    _cfg,
    _closed,
    _connect,
    _frame,
    _reload_close,
    _serve,
    _until,
)

pytestmark = pytest.mark.asyncio

REASON = "llm-redact: your role does not grant realtime"
DENIED = "Project Zebra"
IDENTITY = ("azure", "vertex")
ADAPTERS = {
    "openai": "openai-realtime",
    "azure": "azure-realtime",
    "gemini": "gemini-live",
    "vertex": "vertex-live",
}


def _install(monkeypatch: pytest.MonkeyPatch, gate: Any) -> FakeAuth:
    auth = FakeAuth()
    reg = Registry()
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda config, license: gate
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return auth


def _config(provider: str, base: str, rules: dict[str, Any] | None = None) -> Any:
    auth = "identity" if provider in IDENTITY else "passthrough"
    return _cfg({provider: {"upstream_base_url": base, "auth": auth}}, rules)


@pytest.mark.parametrize("provider", sorted(ADAPTERS))
async def test_an_upgrade_is_asked_with_its_facts_and_a_refusal_is_never_dialled(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    gate = AuthorizingGate(lambda request: REASON)
    auth = _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}{PATHS[provider]}")
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", provider)
            stored = len(proxy.state.vault)
    assert closed is not None and (closed.code, closed.reason) == (1008, REASON)
    assert (row["status"], row["user"]) == (403, "ada")
    (request,) = gate.requests
    assert request == AuthorizationRequest(
        surface="websocket",
        provider=provider,
        adapter=ADAPTERS[provider],
        kind="chat",
        method="GET",
        path=PATHS[provider].split("?")[0],
        model="d" if provider == "azure" else None,  # azure's path carries ?model=d
        identity=provider in IDENTITY,
    )
    assert gate.users == ["ada"]
    # Never dialled, never authorized upstream, nothing numbered.
    assert fake.paths == [] and auth.calls == [] and stored == 0


async def test_an_allowed_upgrade_relays_and_reports_its_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def decide(request: AuthorizationRequest) -> None:
        await asyncio.sleep(0)

    gate = AuthorizingGate(decide)
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            url = f"ws://{proxy.host}/v1/realtime?model=gpt-realtime"
            async with websockets.connect(url) as client:
                await client.send(_frame("openai", f"mail {EMAIL}"))
                await client.recv()
    assert [r.model for r in gate.requests] == ["gpt-realtime"]
    assert EMAIL not in fake.texts()[0] and "«EMAIL_001»" in fake.texts()[0]


@pytest.mark.parametrize(
    "decide",
    [lambda request: 1 / 0, lambda request: False],
    ids=["raises", "nonsense"],
)
async def test_a_failing_check_closes_with_the_core_text(
    monkeypatch: pytest.MonkeyPatch, decide: Any
) -> None:
    _install(monkeypatch, AuthorizingGate(decide))
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}/v1/realtime")
            closed = await _closed(client)
            await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "openai")
            faults = proxy.state.bookkeeping_errors[AUTHORIZATION_STAGE]
    assert (closed.code, closed.reason) == (1008, AUTHORIZATION_FAULT)
    assert faults == 1 and fake.paths == []


@pytest.mark.parametrize("provider", ["openai", "gemini"])
async def test_the_overlay_holds_for_every_frame(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    gate = OverlayGate(DetectionOverlay(deny=(DENIED,), modes=(("email", "block"),)))
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_frame(provider, f"about {DENIED}"))
                await client.recv()
                await client.send(_frame(provider, f"mail {EMAIL}"))  # blocked
                closed = await _closed(client)
    assert gate.asked == 1  # once per connection
    [first] = fake.texts()
    assert DENIED not in first and "«DENY_001»" in first
    assert closed.code == 1008 and "EMAIL" in closed.reason


@pytest.mark.parametrize("provider", ["openai", "gemini"])
async def test_an_extra_deny_string_inside_a_redacted_value_keeps_its_token(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    # The overlay only tightens: the email keeps its own token over the
    # union with the deny string inside it, never a placeholder cut into it.
    _install(monkeypatch, OverlayGate(DetectionOverlay(deny=("acme",))))
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_frame(provider, "mail bob.private@acme-corp.example"))
                await client.recv()
    [sent] = fake.texts()
    assert "bob.private" not in sent and "acme" not in sent and "«EMAIL_001»" in sent


async def test_an_overlay_the_core_cannot_apply_refuses_the_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, OverlayGate(DetectionOverlay(modes=(("email", "redact"),))))
    async with Upstream() as fake:
        rules = {"modes": {"email": "block"}}
        with _serve(_config("openai", fake.url(), rules)) as proxy:
            client = await _connect(f"ws://{proxy.host}/v1/realtime")
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "openai")
    assert (closed.code, closed.reason) == (1008, OVERLAY_FAULT[:123])
    assert row["status"] == 403 and fake.paths == []


async def test_a_reload_during_an_awaited_check_is_never_dialled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    async def decide(request: AuthorizationRequest) -> None:
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)

    gate = BothGate(decide, DetectionOverlay(deny=(DENIED,)))
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            connecting = asyncio.ensure_future(_connect(f"ws://{proxy.host}/v1/realtime"))
            await _until(entered.is_set)
            await proxy.apply(
                _cfg(
                    {"openai": {"upstream_base_url": fake.url(), "enabled": True}},
                    {"deny": ["x-new"]},
                )
            )
            release.set()
            client = await connecting
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "reload", "openai")
    _reload_close(closed, "[detection] policy")
    assert row["status"] == 503 and fake.paths == []
    assert gate.asked == 0  # the overlay is never built against stale objects


async def test_a_refused_upgrade_writes_no_audit_start_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_audit_required import FakeAudit

    audit = FakeAudit()
    _install(monkeypatch, AuthorizingGate(lambda request: REASON))
    registry_mod._registry.build_audit = lambda cfg: audit if cfg.enabled else None
    async with Upstream() as fake:
        config = _cfg(
            {"openai": {"upstream_base_url": fake.url()}},
            audit={"enabled": True, "required": True},
        )
        with _serve(config) as proxy:
            client = await _connect(f"ws://{proxy.host}/v1/realtime")
            closed = await _closed(client)
            await _recent(proxy.host, lambda r: r["method"] == "WS")
    assert closed.code == 1008
    assert audit.begun == [] and fake.paths == []
    assert [entry.status for entry in audit.recorded] == [403]
