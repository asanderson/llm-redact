"""The access gate's optional ``authorize_content`` on realtime client frames.

Asked for every client frame AFTER its redaction and BEFORE its send, with
the frame's OWN facts (the per-connection tee counter's growth across that
frame) and the upgrade's ``AuthorizationRequest`` — on a connection whose
setup frames name the model, the latest request the model check asked with.
A refusal closes 1008 with the gate's reason, nothing of the frame sent, row
403 (kind ``authorization``); a connection revoked while an awaited answer
runs forwards nothing more. Upstream frames are never asked about.

Real sockets end to end (the test_realtime_reload harness).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import websockets

from llm_redact.authorization import AUTHORIZATION_FAULT, AUTHORIZATION_STAGE, GateAuthorization
from llm_redact.plugin_api import AuthorizationRequest, ContentFacts
from local_refusals import refused_once
from test_authorization_content import BothGate, ContentGate
from test_authorization_realtime import (
    ALLOWED,
    SETUP_NAMES,
    _config,
    _install,
    _setup,
    _upstream_closed_once,
)
from test_realtime_identity import EMAIL, _recent
from test_realtime_reload import PATHS, Upstream, _cfg, _closed, _connect, _frame

pytestmark = pytest.mark.asyncio

REASON = "llm-redact: your role may not send credentials"
OTHER = "john.roe@corp.example"
MODEL_URL = "/v1/realtime?model=gpt-realtime"


@pytest.mark.parametrize("provider", ["openai", "gemini"])
async def test_each_frame_is_asked_with_its_own_counts(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    gate = ContentGate()
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve_config(provider, fake) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_frame(provider, f"mail {EMAIL}"))
                await client.recv()
                await client.send(_frame(provider, f"mail {EMAIL} and {OTHER}"))
                await client.recv()
                await client.send(_frame(provider, "nothing here"))
                await client.recv()
    assert [content for _request, content in gate.asked] == [
        ContentFacts(scanned=True, detected=(("EMAIL", 1),)),
        # The frame's own: EMAIL_001 again plus one new value.
        ContentFacts(scanned=True, detected=(("EMAIL", 2),)),
        ContentFacts(scanned=True),
    ]
    request = gate.asked[0][0]
    assert (request.surface, request.provider, request.kind) == ("websocket", provider, "chat")
    assert all(asked is request for asked, _content in gate.asked)  # the upgrade's, each time
    assert len(fake.texts()) == 3 and EMAIL not in fake.texts()[0]


def _serve_config(provider: str, fake: Upstream, **provider_settings: Any) -> Any:
    from test_realtime_reload import _serve

    if provider_settings:
        return _serve(_cfg({provider: {"upstream_base_url": fake.url(), **provider_settings}}))
    return _serve(_config(provider, fake.url()))


async def test_the_upgrade_request_carries_its_model(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = ContentGate()
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve_config("openai", fake) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{MODEL_URL}") as client:
                await client.send(_frame("openai", "hi"))
                await client.recv()
    ((request, _content),) = gate.asked
    assert request == AuthorizationRequest(
        surface="websocket",
        provider="openai",
        adapter="openai-realtime",
        kind="chat",
        method="GET",
        path="/v1/realtime",
        model="gpt-realtime",
        identity=False,
    )


async def test_a_refused_frame_closes_1008_and_is_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ContentGate(lambda request, content: REASON if content.detected else None)
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve_config("openai", fake) as proxy:
            client = await _connect(f"ws://{proxy.host}{MODEL_URL}")
            await client.send(_frame("openai", "hello"))
            await client.recv()
            await client.send(_frame("openai", f"mail {EMAIL}"))
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "openai")
    assert closed is not None and (closed.code, closed.reason) == (1008, REASON)
    assert (row["status"], row["user"]) == (403, "ada")
    # Only the clean frame reached the upstream.
    assert len(fake.texts()) == 1 and "hello" in fake.texts()[0]


async def test_a_long_reason_is_cut_to_a_close_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    long_reason = "llm-redact: " + "no " * 80
    _install(monkeypatch, ContentGate(lambda request, content: long_reason))
    async with Upstream() as fake:
        with _serve_config("openai", fake) as proxy:
            client = await _connect(f"ws://{proxy.host}{MODEL_URL}")
            await client.send(_frame("openai", "hello"))
            closed = await _closed(client)
            await _recent(proxy.host, lambda r: r["method"] == "WS")
    assert closed is not None and closed.code == 1008
    assert len(closed.reason.encode()) <= 123 and long_reason.startswith(closed.reason[:-1])
    assert fake.texts() == []


@pytest.mark.parametrize(
    "decide",
    [lambda request, content: 1 / 0, lambda request, content: 7],
    ids=["raises", "nonsense"],
)
async def test_a_failing_check_closes_with_the_core_text(
    monkeypatch: pytest.MonkeyPatch, decide: Any
) -> None:
    _install(monkeypatch, ContentGate(decide))
    async with Upstream() as fake:
        with _serve_config("openai", fake) as proxy:
            client = await _connect(f"ws://{proxy.host}{MODEL_URL}")
            await client.send(_frame("openai", "hello"))
            closed = await _closed(client)
            await _recent(proxy.host, lambda r: r["method"] == "WS")
            refused_once(proxy.state, "authorization", "openai")
            faults = proxy.state.bookkeeping_errors[AUTHORIZATION_STAGE]
    assert closed is not None and (closed.code, closed.reason) == (1008, AUTHORIZATION_FAULT)
    assert faults == 1 and fake.texts() == []


async def test_a_revocation_during_an_awaited_check_forwards_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxies: list[Any] = []

    async def decide(request: AuthorizationRequest, content: ContentFacts) -> None:
        proxies[0].state.connections.close(subject="ada", reason="access revoked")
        await asyncio.sleep(0)

    _install(monkeypatch, ContentGate(decide))
    async with Upstream() as fake:
        with _serve_config("openai", fake) as proxy:
            proxies.append(proxy)
            client = await _connect(f"ws://{proxy.host}{MODEL_URL}")
            await client.send(_frame("openai", f"mail {EMAIL}"))
            closed = await _closed(client)
            await _upstream_closed_once(fake)
    assert closed is not None and (closed.code, closed.reason) == (1008, "access revoked")
    assert fake.texts() == []


async def test_an_awaited_allowance_sends_the_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    async def decide(request: AuthorizationRequest, content: ContentFacts) -> None:
        await asyncio.sleep(0)

    _install(monkeypatch, ContentGate(decide))
    async with Upstream() as fake:
        with _serve_config("openai", fake) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{MODEL_URL}") as client:
                await client.send(_frame("openai", f"mail {EMAIL}"))
                await client.recv()
    assert len(fake.texts()) == 1 and "«EMAIL_001»" in fake.texts()[0]


@pytest.mark.parametrize("provider", ["gemini", "vertex"])
async def test_a_live_frame_is_asked_with_the_setups_model(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    gate = BothGate()
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve_config(provider, fake) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_setup(SETUP_NAMES[provider]))
                await client.recv()
                await client.send(_frame(provider, f"mail {EMAIL}"))
                await client.recv()
    upgrade, setup = gate.requests
    assert (upgrade.model, setup.model) == (None, ALLOWED)
    assert [(request is setup, content) for request, content in gate.asked] == [
        (True, ContentFacts(scanned=True)),
        (True, ContentFacts(scanned=True, detected=(("EMAIL", 1),))),
    ]


async def test_detection_off_and_non_json_frames_are_unscanned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = ContentGate()
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve_config("openai", fake, detection=False) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{MODEL_URL}") as client:
                await client.send(_frame("openai", f"mail {EMAIL}"))
                await client.recv()
    assert [content for _request, content in gate.asked] == [ContentFacts(scanned=False)]
    assert EMAIL in fake.texts()[0]

    gate = ContentGate()
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve_config("openai", fake) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{MODEL_URL}") as client:
                await client.send("not json at all")
                await client.recv()
                await client.send(_frame("openai", f"mail {EMAIL}"))
                await client.recv()
    assert [content for _request, content in gate.asked] == [
        ContentFacts(scanned=False),
        ContentFacts(scanned=True, detected=(("EMAIL", 1),)),
    ]
    assert fake.texts()[0] == "not json at all"


async def test_server_frames_are_never_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = ContentGate()
    _install(monkeypatch, gate)
    async with Upstream() as fake:
        with _serve_config("openai", fake) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{MODEL_URL}") as client:
                await client.send(_frame("openai", "hello"))
                echoed = await client.recv()
    assert json.loads(echoed)  # the upstream's echo reached the client
    assert len(gate.asked) == 1  # the client frame only


async def test_without_the_member_frames_go_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_access_seam import FakeGate

    monkeypatch.setattr(GateAuthorization, "content_refusal", _never)
    _install(monkeypatch, FakeGate())
    async with Upstream() as fake:
        with _serve_config("openai", fake) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{MODEL_URL}") as client:
                await client.send(_frame("openai", f"mail {EMAIL}"))
                await client.recv()
                await client.send("not json at all")
                await client.recv()
    assert "«EMAIL_001»" in fake.texts()[0] and fake.texts()[1] == "not json at all"


async def test_a_frame_reports_the_values_an_override_let_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The realtime twin of the HTTP fact: a block-mode value an approved
    # override lets through goes upstream RAW, so the frame's facts carry
    # its TYPE (distinct values), each frame its own.
    from llm_redact.config import Config, OverridesConfig, ProviderConfig
    from llm_redact.detection.engine import DetectionConfig
    from llm_redact.overrides import OverrideStore
    from test_overrides_realtime import _frame as _blocked_frame
    from test_overrides_realtime import _refused
    from test_realtime_relay import FakeUpstream, _proxy

    gate = ContentGate(
        lambda request, content: REASON if refuse[0] and content.overridden_types else None,
        subject=None,
    )
    refuse = [True]
    _install(monkeypatch, gate)
    store_path = tmp_path / "overrides.db"
    async with FakeUpstream() as fake:
        config = Config(
            providers={
                **Config().providers,
                "openai": ProviderConfig(f"http://127.0.0.1:{fake.port}"),
            },
            detection=DetectionConfig(modes=(("email", "block"),)),
            overrides=OverridesConfig(enabled=True, path=str(store_path)),
        )
        with _proxy(config) as host:
            url = f"ws://{host}{MODEL_URL}"
            code = await _refused(url)
            OverrideStore(store_path).approve("always", approver=None, code=code)
            async with websockets.connect(url) as client:
                await client.send(_blocked_frame())
                with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                    await client.recv()
            assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008
            assert fake.received == []
            refuse[0] = False
            async with websockets.connect(url) as client:
                await client.send(_blocked_frame())
                await client.recv()
                await client.send(_frame("openai", "nothing here"))
                await client.recv()
    assert sum(EMAIL in str(frame) for frame in fake.received) == 1
    approved = ContentFacts(scanned=True, overridden=True, overridden_types=(("EMAIL", 1),))
    # The blocked frame before the approval is refused by the redaction
    # itself (never asked); then the gate's refusal, the allowed frame and a
    # frame of its own with nothing in it.
    assert [content for _request, content in gate.asked] == [
        approved,
        approved,
        ContentFacts(scanned=True),
    ]


def _never(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("asked without the member")
