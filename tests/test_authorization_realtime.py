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
import dataclasses
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import websockets

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact import realtime
from llm_redact.authorization import AUTHORIZATION_FAULT, AUTHORIZATION_STAGE, OVERLAY_FAULT
from llm_redact.plugin_api import AuthorizationRequest, DetectionOverlay
from llm_redact.registry import Registry
from local_refusals import refused_once
from test_authorization_seam import AuthorizingGate, BothGate, OverlayGate
from test_realtime_identity import AZURE_PREVIEW, EMAIL, GEMINI_LIVE, FakeAuth, _recent
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
        # Gemini and Vertex Live: the setup frame names the model.
        model_in_frame=provider in ("gemini", "vertex"),
    )
    assert gate.users == ["ada"]
    # Never dialled, never authorized upstream, nothing numbered.
    assert fake.paths == [] and auth.calls == [] and stored == 0


@pytest.mark.parametrize(
    ("provider", "path", "model"),
    [
        # Azure's preview runs the deployment the query names, whatever
        # `model` says.
        ("azure", f"{AZURE_PREVIEW}?api-version=v&deployment=dep&model=other", "dep"),
        # Gemini Live: the setup frame names the model, after the check.
        ("gemini", f"{GEMINI_LIVE}?model=models/gemini-2.5-flash", None),
        # A transcription session's model is set by its frames.
        ("openai", "/v1/realtime?intent=transcription&model=gpt-realtime", None),
    ],
    ids=["azure-preview", "gemini-live", "openai-intent"],
)
async def test_an_upgrade_reports_the_model_its_upstream_runs(
    monkeypatch: pytest.MonkeyPatch, provider: str, path: str, model: str | None
) -> None:
    gate = AuthorizingGate(lambda request: REASON)
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}{path}")
            closed = await _closed(client)
            await _recent(proxy.host, lambda r: r["method"] == "WS")
    assert closed is not None and closed.code == 1008
    (request,) = gate.requests
    assert request.model == model


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
    _install(monkeypatch, OverlayGate(DetectionOverlay(modes=(("no_such_rule", "block"),))))
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


async def test_a_block_the_overlay_added_closes_with_no_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The configured policy redacts the email; the overlay blocks it — a
    # refusal no approval passes: the close reason carries no code.
    _install(monkeypatch, OverlayGate(DetectionOverlay(modes=(("email", "block"),))))
    store = tmp_path / "overrides.db"
    async with Upstream() as fake:
        config = _cfg(
            {"openai": {"upstream_base_url": fake.url()}},
            overrides={"enabled": True, "path": str(store)},
        )
        with _serve(config) as proxy:
            async with websockets.connect(f"ws://{proxy.host}/v1/realtime") as client:
                await client.send(_frame("openai", f"mail {EMAIL}"))
                closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
    assert closed is not None and closed.code == 1008 and "EMAIL" in closed.reason
    assert "llm-redact override" not in closed.reason and "dashboard" not in closed.reason
    assert fake.texts() == [] and row["override"] is None


async def test_an_access_revocation_during_an_awaited_check_is_never_dialled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The relay is held — revocable — from admission on, the awaited check
    # included: the gate ending the user's access while it decides refuses
    # the connection right after (1008 with the gate's reason, row 403),
    # never dialled.
    entered = threading.Event()
    release = threading.Event()

    async def decide(request: AuthorizationRequest) -> None:
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)

    _install(monkeypatch, AuthorizingGate(decide))
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            connecting = asyncio.ensure_future(_connect(f"ws://{proxy.host}/v1/realtime"))
            await _until(entered.is_set)
            closed_now = proxy.state.connections.close(subject="ada", reason="access revoked")
            release.set()
            client = await connecting
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "access_gate", "openai")
            await _until(lambda: not proxy.state.realtime_relays)
            tracked = proxy.state.connections.open_counts()
    assert closed_now == 1
    assert closed is not None and (closed.code, closed.reason) == (1008, "access revoked")
    assert row["status"] == 403 and fake.paths == []
    assert tracked == {}


async def test_a_refused_upgrade_leaves_nothing_tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, AuthorizingGate(lambda request: REASON))
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}/v1/realtime")
            closed = await _closed(client)
            await _recent(proxy.host, lambda r: r["method"] == "WS")
            await _until(lambda: not proxy.state.realtime_relays)
            tracked = proxy.state.connections.open_counts()
    assert closed is not None and closed.code == 1008
    assert tracked == {} and fake.paths == []


@pytest.mark.parametrize(
    ("member", "text"),
    [("authorize_request", AUTHORIZATION_FAULT), ("detection_overlay", OVERLAY_FAULT)],
    ids=["authorize_request", "detection_overlay"],
)
async def test_a_member_that_cannot_be_called_refuses_every_upgrade(
    monkeypatch: pytest.MonkeyPatch, member: str, text: str
) -> None:
    from test_access_seam import FakeGate

    gate = FakeGate()
    setattr(gate, member, "not callable")
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}/v1/realtime")
            closed = await _closed(client)
            await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "openai")
    assert closed is not None and (closed.code, closed.reason) == (1008, text[:123])
    assert fake.paths == []


# --- the model a Gemini/Vertex Live setup frame names ----------------------------------

LIVE = ("gemini", "vertex")
ALLOWED = "gemini-live-2.5-flash"
SETUP_NAMES = {
    "gemini": f"models/{ALLOWED}",
    "vertex": f"projects/p/locations/l/publishers/google/models/{ALLOWED}",
}


def _setup(model: Any = None, **fields: Any) -> bytes:
    """A Live setup message (JSON in a binary frame, as the SDKs send it)."""
    setup = dict(fields)
    if model is not None:
        setup["model"] = model
    return json.dumps({"setup": setup}).encode()


def _models_only(allowed: str = ALLOWED) -> Callable[[AuthorizationRequest], str | None]:
    """A model-restricting policy: an upgrade whose model a frame names is
    admitted on its other facts; any other model must be ``allowed``."""

    def decide(request: AuthorizationRequest) -> str | None:
        if request.model_in_frame:
            return None
        return None if request.model == allowed else REASON

    return decide


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "awaitable"])
@pytest.mark.parametrize("provider", LIVE)
async def test_an_admitted_setup_model_is_relayed(
    monkeypatch: pytest.MonkeyPatch, provider: str, asynchronous: bool
) -> None:
    policy = _models_only()

    async def later(request: AuthorizationRequest) -> str | None:
        await asyncio.sleep(0)
        return policy(request)

    gate = AuthorizingGate(later if asynchronous else policy)
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_setup(SETUP_NAMES[provider]))
                await client.recv()
                await client.send(_frame(provider, f"mail {EMAIL}"))
                await client.recv()
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
    upgrade, setup = gate.requests
    assert (upgrade.model, upgrade.model_in_frame) == (None, True)
    # Asked again with the same facts and the frame's model, as HTTP reads it.
    assert setup == dataclasses.replace(upgrade, model=ALLOWED, model_in_frame=False)
    assert gate.users == ["ada", "ada"]  # in the connection's own context
    first, second = fake.texts()
    assert json.loads(first) == {"setup": {"model": SETUP_NAMES[provider]}}
    assert EMAIL not in second and "«EMAIL_001»" in second
    assert row["status"] == 101 and len(gate.requests) == 2  # content frames: never asked


@pytest.mark.parametrize("provider", LIVE)
async def test_a_refused_setup_model_is_never_forwarded(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    gate = AuthorizingGate(_models_only("another-model"))
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}{PATHS[provider]}")
            await client.send(_setup(SETUP_NAMES[provider], systemInstruction=EMAIL))
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", provider)
            stored = len(proxy.state.vault)
    assert closed is not None and (closed.code, closed.reason) == (1008, REASON)
    assert (row["status"], row["user"]) == (403, "ada")
    assert [r.model for r in gate.requests] == [None, ALLOWED]
    # Dialled at the upgrade, but nothing of the frame sent or numbered.
    assert len(fake.paths) == 1 and fake.texts() == [] and stored == 0


@pytest.mark.parametrize(
    "first",
    [
        json.dumps({"clientContent": {"turns": [{"role": "user", "parts": [{"text": "hi"}]}]}}),
        _setup(generationConfig={"responseModalities": ["TEXT"]}),
        _setup(""),
        _setup(7),
        json.dumps({"setup": f"models/{ALLOWED}"}),
        json.dumps([{"setup": {"model": f"models/{ALLOWED}"}}]),
        "not json at all",
        b"\xff\xfe binary that is not UTF-8",
    ],
    ids=[
        "content",
        "no-model",
        "empty-model",
        "model-not-a-string",
        "setup-not-an-object",
        "array",
        "not-json",
        "binary",
    ],
)
async def test_the_first_frame_must_be_a_setup_naming_a_model(
    monkeypatch: pytest.MonkeyPatch, first: str | bytes
) -> None:
    gate = AuthorizingGate(_models_only())
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config("gemini", fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}{GEMINI_LIVE}")
            await client.send(first)
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "gemini")
    assert closed is not None and (closed.code, closed.reason) == (1008, realtime.SETUP_FIRST)
    assert row["status"] == 403 and fake.texts() == []
    assert len(gate.requests) == 1  # the upgrade only: nothing to ask about


@pytest.mark.parametrize(
    ("later", "reason"),
    [
        (_setup("models/another-model"), REASON),
        (_setup(generationConfig={}), realtime.SETUP_MODEL),
        ("not json at all", realtime.FRAME_NOT_JSON),
    ],
    ids=["refused-model", "no-model", "not-json"],
)
async def test_a_later_setup_is_checked_alike(
    monkeypatch: pytest.MonkeyPatch, later: str | bytes, reason: str
) -> None:
    gate = AuthorizingGate(_models_only())
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config("gemini", fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}{GEMINI_LIVE}")
            await client.send(_setup(SETUP_NAMES["gemini"]))
            await client.recv()
            await client.send(_frame("gemini", "plain words"))
            await client.recv()
            await client.send(later)
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "gemini")
    assert closed is not None and (closed.code, closed.reason) == (1008, reason)
    assert row["status"] == 403 and len(fake.texts()) == 2  # the later frame never sent
    asked = 3 if reason == REASON else 2
    assert [r.model for r in gate.requests][:asked] == [None, ALLOWED, "another-model"][:asked]


@pytest.mark.parametrize(
    "decide",
    [lambda request: 1 / 0, lambda request: False, "times-out"],
    ids=["raises", "nonsense", "times-out"],
)
async def test_a_failing_setup_check_closes_with_the_core_text(
    monkeypatch: pytest.MonkeyPatch, decide: Any
) -> None:
    async def never(request: AuthorizationRequest) -> None:
        await asyncio.sleep(3600)

    def frame_check(request: AuthorizationRequest) -> Any:
        if request.model_in_frame:
            return None
        return never(request) if decide == "times-out" else decide(request)

    _install(monkeypatch, AuthorizingGate(frame_check))
    async with Upstream() as fake:
        with _serve(_config("gemini", fake.url())) as proxy:
            proxy.state.authorization.timeout = 0.05
            client = await _connect(f"ws://{proxy.host}{GEMINI_LIVE}")
            await client.send(_setup(SETUP_NAMES["gemini"]))
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "gemini")
            faults = proxy.state.bookkeeping_errors[AUTHORIZATION_STAGE]
    assert closed is not None and (closed.code, closed.reason) == (1008, AUTHORIZATION_FAULT)
    assert row["status"] == 403 and faults == 1 and fake.texts() == []


@pytest.mark.parametrize("revocation", ["reload", "access"])
async def test_a_revocation_during_an_awaited_setup_check_forwards_nothing(
    monkeypatch: pytest.MonkeyPatch, revocation: str
) -> None:
    entered = threading.Event()
    release = threading.Event()

    async def decide(request: AuthorizationRequest) -> None:
        if request.model_in_frame:
            return
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)

    gate = AuthorizingGate(decide)
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config("gemini", fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}{GEMINI_LIVE}")
            await client.send(_setup(SETUP_NAMES["gemini"]))
            await _until(entered.is_set)
            if revocation == "reload":
                await proxy.apply(
                    _cfg({"gemini": {"upstream_base_url": fake.url()}}, {"deny": ["x-new"]})
                )
            else:
                proxy.state.connections.close(subject="ada", reason="access revoked")
            release.set()
            closed = await _closed(client)
            await _upstream_closed_once(fake)
    if revocation == "reload":
        _reload_close(closed, "[detection] policy")
    else:
        assert closed is not None and (closed.code, closed.reason) == (1008, "access revoked")
    # The gate allowed the model, but the frame was read under the admission
    # the revocation ended: never checked further, redacted or sent.
    assert fake.texts() == []


async def _upstream_closed_once(fake: Upstream) -> None:
    await _until(lambda: len(fake.close_codes) >= 1)


async def test_the_model_check_runs_before_the_session_routers_frame_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_realtime_frame_check import FrameRouter

    router = FrameRouter()
    gate = AuthorizingGate(_models_only("another-model"))
    _install(monkeypatch, gate)
    registry_mod._registry.build_session_router = lambda config, **kw: router
    async with Upstream() as fake:
        with _serve(_config("gemini", fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}{GEMINI_LIVE}")
            await client.send(_setup(SETUP_NAMES["gemini"]))
            closed = await _closed(client)
            await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "gemini")
    assert closed is not None and (closed.code, closed.reason) == (1008, REASON)
    assert router.calls == [] and fake.texts() == []


async def test_an_admitted_setup_reaches_the_session_routers_frame_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_realtime_frame_check import FrameRouter

    router = FrameRouter()
    _install(monkeypatch, AuthorizingGate(_models_only()))
    registry_mod._registry.build_session_router = lambda config, **kw: router
    async with Upstream() as fake:
        with _serve(_config("gemini", fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{GEMINI_LIVE}") as client:
                await client.send(_setup(SETUP_NAMES["gemini"]))
                await client.recv()
    assert [call["frame"] for call in router.calls] == [{"setup": {"model": SETUP_NAMES["gemini"]}}]
    assert len(fake.texts()) == 1


@pytest.mark.parametrize("gate", ["none", "overlay-only"])
async def test_without_authorize_request_no_setup_is_required(
    monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    from test_access_seam import FakeGate

    _install(monkeypatch, FakeGate() if gate == "none" else OverlayGate(None))
    async with Upstream() as fake:
        with _serve(_config("gemini", fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{GEMINI_LIVE}") as client:
                await client.send(_frame("gemini", "plain words"))
                await client.recv()
                await client.send("not json at all")
                await client.recv()
    assert fake.texts() == [_frame("gemini", "plain words").decode(), "not json at all"]


async def test_detection_off_sends_the_setup_the_gate_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A repeated `model` key: the check reads the last occurrence, and the
    # frame is re-serialized from that reading, so an upstream reading the
    # first could never run another model.
    gate = AuthorizingGate(_models_only())
    _install(monkeypatch, gate)
    frame = b'{"setup": {"model": "models/other", "model": "models/' + ALLOWED.encode() + b'"}}'
    async with Upstream() as fake:
        config = _cfg({"gemini": {"upstream_base_url": fake.url(), "detection": False}})
        with _serve(config) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{GEMINI_LIVE}") as client:
                await client.send(frame)
                await client.recv()
    assert [r.model for r in gate.requests] == [None, ALLOWED]
    assert fake.texts() == [json.dumps({"setup": {"model": f"models/{ALLOWED}"}})]


@pytest.mark.parametrize(
    ("adapter", "name", "model"),
    [
        (realtime.GeminiLiveWs(), "models/gemini-live-2.5-flash", "gemini-live-2.5-flash"),
        (realtime.GeminiLiveWs(), "tunedModels/mine", "tunedModels/mine"),
        (realtime.GeminiLiveWs(), "gemini-live-2.5-flash", "gemini-live-2.5-flash"),
        (
            realtime.VertexLiveWs(),
            "projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash",
            "gemini-2.0-flash",
        ),
        (realtime.VertexLiveWs(), "publishers/google/models/m", "m"),
        (realtime.VertexLiveWs(), "projects/p/locations/l/endpoints/123", "endpoints/123"),
        (realtime.VertexLiveWs(), "gemini-2.0-flash", "gemini-2.0-flash"),
    ],
)
async def test_a_setup_model_is_read_as_the_http_adapters_report_it(
    adapter: realtime.WsAdapter, name: str, model: str
) -> None:
    assert adapter.model_in_frame
    assert adapter.sets_model({"setup": {"model": name}})
    assert adapter.frame_model({"setup": {"model": name}}) == model


@pytest.mark.parametrize(
    "payload",
    [{"setup": {}}, {"setup": None}, {"setup": {"model": ""}}, {"setup": {"model": 1}}, []],
)
async def test_a_setup_naming_no_model_reads_as_none(payload: Any) -> None:
    for adapter in (realtime.GeminiLiveWs(), realtime.VertexLiveWs()):
        assert adapter.frame_model(payload) is None


@pytest.mark.parametrize("cls", [realtime.OpenAIRealtimeWs, realtime.AzureRealtimeWs])
async def test_openai_vocabulary_adapters_read_session_update_models(cls: Any) -> None:
    adapter = cls()
    assert not adapter.model_in_frame and adapter.model_in_update
    update = {"type": "session.update", "session": {"model": "x"}}
    assert adapter.sets_model(update) and adapter.frame_model(update) == "x"
    for frame in (
        {"type": "session.update", "session": {"instructions": "hi"}},
        {"type": "session.update", "session": None},
        {"type": "response.create", "response": {"model": "x"}},
        {"setup": {"model": "y"}},
        [update],
    ):
        assert not adapter.sets_model(frame) and adapter.frame_model(frame) is None
    for model in (None, "", 7, ["x"], {"id": "x"}):
        unnamed = {"type": "session.update", "session": {"model": model}}
        assert adapter.sets_model(unnamed) and adapter.frame_model(unnamed) is None


# --- an OpenAI/Azure session.update naming session.model ----------------------------

UPGRADE_MODELS = {"openai": "gpt-realtime", "azure": "d"}
OPENAI_PATHS = {"openai": "/v1/realtime?model=gpt-realtime", "azure": PATHS["azure"]}


def _update(**session: Any) -> str:
    return json.dumps({"type": "session.update", "session": session})


def _upgrade_models_only(allowed: str) -> Callable[[AuthorizationRequest], str | None]:
    """Admits the upgrade's own model; a frame's model must be ``allowed``."""

    def decide(request: AuthorizationRequest) -> str | None:
        if request.model in UPGRADE_MODELS.values() or request.model == allowed:
            return None
        return REASON

    return decide


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "awaitable"])
@pytest.mark.parametrize("provider", ["openai", "azure"])
async def test_an_admitted_session_update_model_is_relayed(
    monkeypatch: pytest.MonkeyPatch, provider: str, asynchronous: bool
) -> None:
    policy = _upgrade_models_only("gpt-realtime-mini")

    async def later(request: AuthorizationRequest) -> str | None:
        await asyncio.sleep(0)
        return policy(request)

    gate = AuthorizingGate(later if asynchronous else policy)
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{OPENAI_PATHS[provider]}") as client:
                await client.send(_update(model="gpt-realtime-mini", instructions="be brief"))
                await client.recv()
                await client.send(_frame(provider, f"mail {EMAIL}"))
                await client.recv()
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
    upgrade, update = gate.requests
    assert (upgrade.model, upgrade.model_in_frame) == (UPGRADE_MODELS[provider], False)
    assert update == dataclasses.replace(upgrade, model="gpt-realtime-mini")
    first, second = fake.texts()
    assert json.loads(first)["session"]["model"] == "gpt-realtime-mini"
    assert EMAIL not in second and row["status"] == 101


@pytest.mark.parametrize("provider", ["openai", "azure"])
async def test_a_refused_session_update_model_is_never_forwarded(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    gate = AuthorizingGate(_upgrade_models_only("gpt-realtime-mini"))
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config(provider, fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}{OPENAI_PATHS[provider]}")
            await client.send(_frame(provider, "plain words"))
            await client.recv()
            await client.send(_update(model="gpt-4o-realtime", instructions=f"mail {EMAIL}"))
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", provider)
            stored = len(proxy.state.vault)
    assert closed is not None and (closed.code, closed.reason) == (1008, REASON)
    assert (row["status"], row["user"]) == (403, "ada")
    assert [r.model for r in gate.requests] == [UPGRADE_MODELS[provider], "gpt-4o-realtime"]
    assert len(fake.texts()) == 1 and stored == 0  # the update never sent or numbered


@pytest.mark.parametrize("model", [None, "", 7, ["gpt-realtime"]], ids=repr)
async def test_a_session_update_model_that_is_no_name_closes(
    monkeypatch: pytest.MonkeyPatch, model: Any
) -> None:
    gate = AuthorizingGate(lambda request: None)
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}/v1/realtime?model=gpt-realtime")
            await client.send(_update(model=model))
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "openai")
    reason = realtime.OpenAIRealtimeWs.no_model_reason
    assert len(reason.encode()) <= 123
    assert closed is not None and (closed.code, closed.reason) == (1008, reason)
    assert row["status"] == 403 and fake.texts() == [] and len(gate.requests) == 1


async def test_a_failing_session_update_check_closes_with_the_core_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def decide(request: AuthorizationRequest) -> Any:
        return None if request.model == "gpt-realtime" else 1 / 0

    _install(monkeypatch, AuthorizingGate(decide))
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            client = await _connect(f"ws://{proxy.host}/v1/realtime?model=gpt-realtime")
            await client.send(_update(model="other"))
            closed = await _closed(client)
            await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "openai")
            faults = proxy.state.bookkeeping_errors[AUTHORIZATION_STAGE]
    assert closed is not None and (closed.code, closed.reason) == (1008, AUTHORIZATION_FAULT)
    assert faults == 1 and fake.texts() == []


async def test_a_revocation_during_an_awaited_session_update_check_forwards_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxies: list[Any] = []

    async def decide(request: AuthorizationRequest) -> None:
        if request.model == "other":
            proxies[0].state.connections.close(subject="ada", reason="access revoked")

    _install(monkeypatch, AuthorizingGate(decide))
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            proxies.append(proxy)
            client = await _connect(f"ws://{proxy.host}/v1/realtime?model=gpt-realtime")
            await client.send(_update(model="other"))
            closed = await _closed(client)
            await _upstream_closed_once(fake)
    assert closed is not None and (closed.code, closed.reason) == (1008, "access revoked")
    assert fake.texts() == []


@pytest.mark.parametrize("detection", [True, False])
async def test_frames_without_a_session_model_go_as_before(
    monkeypatch: pytest.MonkeyPatch, detection: bool
) -> None:
    # A session.update without session.model, any other event and a frame
    # that is not JSON: never asked about, never refused for it.
    gate = AuthorizingGate(lambda request: None)
    _install(monkeypatch, gate)
    frames = [
        _update(instructions="be brief"),
        json.dumps({"type": "response.create", "response": {"model": "x"}}),
        "not json at all",
    ]
    async with Upstream() as fake:
        config = _cfg({"openai": {"upstream_base_url": fake.url(), "detection": detection}})
        with _serve(config) as proxy:
            url = f"ws://{proxy.host}/v1/realtime?model=gpt-realtime"
            async with websockets.connect(url) as client:
                for frame in frames:
                    await client.send(frame)
                    await client.recv()
    assert len(gate.requests) == 1
    if detection:
        assert len(fake.texts()) == 3 and fake.texts()[2] == "not json at all"
    else:
        assert fake.texts() == frames  # byte-identical, as without the member


async def test_with_detection_off_a_checked_update_is_sent_as_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A repeated `model` key: the check reads the last occurrence, and the
    # frame is re-serialized from that reading.
    gate = AuthorizingGate(_upgrade_models_only("gpt-realtime-mini"))
    _install(monkeypatch, gate)
    frame = '{"type": "session.update", "session": {"model": "x", "model": "gpt-realtime-mini"}}'
    async with Upstream() as fake:
        config = _cfg({"openai": {"upstream_base_url": fake.url(), "detection": False}})
        with _serve(config) as proxy:
            url = f"ws://{proxy.host}/v1/realtime?model=gpt-realtime"
            async with websockets.connect(url) as client:
                await client.send(frame)
                await client.recv()
    assert [r.model for r in gate.requests] == ["gpt-realtime", "gpt-realtime-mini"]
    assert fake.texts() == [_update(model="gpt-realtime-mini")]


@pytest.mark.parametrize("gate", ["none", "overlay-only"])
async def test_without_authorize_request_a_session_update_model_is_not_checked(
    monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    from test_access_seam import FakeGate

    _install(monkeypatch, FakeGate() if gate == "none" else OverlayGate(None))
    async with Upstream() as fake:
        with _serve(_config("openai", fake.url())) as proxy:
            async with websockets.connect(f"ws://{proxy.host}/v1/realtime") as client:
                for model in ("anything", 7):
                    await client.send(_update(model=model))
                    await client.recv()
    assert [json.loads(t)["session"]["model"] for t in fake.texts()] == ["anything", 7]


async def test_a_revocation_landing_with_the_awaited_answer_forwards_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The gate allows the model while the user's access is being ended: the
    # relay re-checks its revocation right after the awaited answer, so the
    # frame read under the ended admission is never checked further or sent.
    proxies: list[Any] = []

    async def decide(request: AuthorizationRequest) -> None:
        if not request.model_in_frame:
            proxies[0].state.connections.close(subject="ada", reason="access revoked")

    _install(monkeypatch, AuthorizingGate(decide))
    async with Upstream() as fake:
        with _serve(_config("gemini", fake.url())) as proxy:
            proxies.append(proxy)
            client = await _connect(f"ws://{proxy.host}{GEMINI_LIVE}")
            await client.send(_setup(SETUP_NAMES["gemini"]))
            closed = await _closed(client)
            await _upstream_closed_once(fake)
    assert closed is not None and (closed.code, closed.reason) == (1008, "access revoked")
    assert fake.texts() == []
