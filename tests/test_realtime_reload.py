"""A hot reload reaches OPEN realtime relays.

An established relay used to keep the admission it was opened with for its
whole life. A reload that withdrew the proxy's own identity (``auth`` back to
``passthrough``, the provider disabled, its upstream moved) kept spending it
on the live upstream session, and a reload that tightened redaction (detection
re-enabled, a deny string added, an allowlist narrowed, a rule moved to
block) never reached new frames — while /status already reported the new
policy. ``apply_config`` now revokes, in the same synchronous step as the
swap, every open relay whose admission it changed: the relay closes both
sides with 1012 (Service Restart: reconnect) and never redacts or forwards a
frame it reads after the swap. A reload that changes nothing a relay depends
on leaves it open.

Real sockets end to end (uvicorn on port 0 in a thread, a fake ``websockets``
upstream in the test's loop — the test_realtime_relay harness). The reload
runs ON the server's event loop, as SIGHUP (``loop.add_signal_handler``) and
the config editor (an HTTP handler) run it in production.

A client still WRITING frames while the proxy closes may see a reset instead
of the close frame: uvicorn closes the socket right after writing its close
frame (it does not wait for the client's), so the client's in-flight frames
meet a closed socket and the client library aborts on the failed write. That
holds for every proxy-initiated close (a block's 1008 too). Tests that write
after the reload therefore accept either the 1012 frame or that reset — the
proxy's own log line pins the 1012 — while the upstream-side assertions (what
the provider received) stay exact.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
import websockets

import llm_redact.registry as registry_mod
from llm_redact import realtime
from llm_redact.config import Config, load_config, parse_config
from llm_redact.detection.engine import build_allowlist, build_detectors, build_modes
from llm_redact.plugin_api import UpstreamAuthError
from llm_redact.proxy import ProxyState, create_app
from llm_redact.realtime import RealtimeRelay
from llm_redact.registry import Registry
from test_realtime_identity import (
    AZURE_GA,
    EMAIL,
    GEMINI_LIVE,
    PROXY_TOKEN,
    VERTEX_V1,
    FakeAuth,
    _item_created,
    _recent,
)
from test_realtime_relay import FakeUpstream

DENY = "Project Nightingale"
PATHS = {
    "openai": "/v1/realtime",
    "azure": f"{AZURE_GA}?model=d",
    "gemini": GEMINI_LIVE,
    "vertex": VERTEX_V1,
}


class Upstream(FakeUpstream):
    """The echoing fake upstream, also recording how each connection ended
    and able to hold the opening handshake (``hold``) so a reload can land
    while the proxy is dialling it."""

    def __init__(self) -> None:
        super().__init__()
        self.close_codes: list[int | None] = []
        self.dialled = asyncio.Event()
        self.hold: asyncio.Event | None = None

    async def __aenter__(self) -> Upstream:
        self.server = await websockets.serve(
            self._handler,
            "127.0.0.1",
            0,
            select_subprotocol=self._select,
            process_request=self._process_request,
        )
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def _process_request(self, connection: Any, request: Any) -> None:
        self.dialled.set()
        if self.hold is not None:
            await self.hold.wait()

    async def _handler(self, connection: Any) -> None:
        try:
            await super()._handler(connection)
        finally:
            self.close_codes.append(connection.close_code)

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def texts(self) -> list[str]:
        return [f.decode() if isinstance(f, bytes) else f for f in self.received]


class HeldAuth(FakeAuth):
    """Authorizes an upgrade only once released, so a reload can land while
    the proxy is obtaining its credential (the authorizer runs on the
    server's loop; the gates are thread-safe)."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    async def authorize(
        self, method: str, url: str, headers: list[tuple[str, str]], body: bytes
    ) -> list[tuple[str, str]]:
        self.entered.set()
        while not self.release.is_set():
            await asyncio.sleep(0.005)
        return await super().authorize(method, url, headers, body)


class _LoopCapture:
    """ASGI wrapper recording the event loop uvicorn serves the app on."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.loop: asyncio.AbstractEventLoop | None = None

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        self.loop = asyncio.get_running_loop()
        await self.app(scope, receive, send)


class Proxy:
    def __init__(self, host: str, state: ProxyState, capture: _LoopCapture) -> None:
        self.host = host
        self.state = state
        self._capture = capture

    async def on_loop(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the server's event loop (where SIGHUP runs reload)."""
        assert self._capture.loop is not None
        done: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def run() -> None:
            try:
                done.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 — re-raised in the test
                done.set_exception(exc)

        self._capture.loop.call_soon_threadsafe(run)
        return await asyncio.wrap_future(done)

    async def apply(self, fresh: Config) -> Any:
        return await self.on_loop(lambda: self.state.apply_config(fresh))


@contextlib.contextmanager
def _serve(config: Config, config_path: Path | None = None) -> Iterator[Proxy]:
    app = create_app(config, config_path=config_path)
    capture = _LoopCapture(app)
    server = uvicorn.Server(uvicorn.Config(capture, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield Proxy(
            f"127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}",
            app.state.proxy,
            capture,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _cfg(
    providers: dict[str, dict[str, Any]], rules: dict[str, Any] | None = None, **top: Any
) -> Config:
    raw: dict[str, Any] = {"providers": providers, **top}
    if rules is not None:
        raw["detection"] = rules
    return parse_config(raw, "test")


def _install(monkeypatch: pytest.MonkeyPatch, auth: FakeAuth) -> None:
    reg = Registry()
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    monkeypatch.setattr(registry_mod, "_registry", reg)


def _frame(provider: str, text: str) -> str | bytes:
    if provider in ("gemini", "vertex"):
        # Gemini Live JSON rides binary frames.
        turns = [{"role": "user", "parts": [{"text": text}]}]
        return json.dumps({"clientContent": {"turns": turns}}).encode()
    return _item_created(text)


async def _connect(url: str) -> Any:
    return await websockets.connect(url)


async def _send(client: Any, frame: str | bytes) -> None:
    # The proxy may already have closed the connection (1012) by the time
    # the frame is written; either way it must never reach the upstream.
    with contextlib.suppress(websockets.exceptions.ConnectionClosed):
        await client.send(frame)


async def _closed(client: Any) -> Any:
    """The proxy's close frame — None when the connection was reset under a
    frame the client was still writing (module docstring) — draining any
    echo still in flight. A relay the reload left open fails here, listing
    what the upstream echoed back after the reload (what it received)."""
    relayed: list[str | bytes] = []
    try:
        while True:
            relayed.append(await asyncio.wait_for(client.recv(), 5))
    except websockets.exceptions.ConnectionClosed as closed:
        return closed.rcvd
    except TimeoutError:
        raise AssertionError(
            f"the relay stayed open after the reload; it relayed {relayed!r}"
        ) from None


def _reload_close(closed: Any, changed: str) -> None:
    assert closed is not None, "the connection was reset, not closed"
    assert closed.code == 1012
    assert closed.reason == (
        f"llm-redact config reload changed this connection's {changed}; reconnect"
    )


def _reload_close_while_writing(closed: Any, caplog: pytest.LogCaptureFixture) -> None:
    # The 1012 frame or the reset of a client still writing — never any
    # other close (a 1008 block, a 1000) — and the proxy says it sent 1012.
    assert closed is None or closed.code == 1012
    assert "closed 1012 (a config reload changed its" in caplog.text


async def _until(predicate: Callable[[], bool]) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


async def _upstream_closed(fake: Upstream, connections: int = 1) -> None:
    await _until(lambda: len(fake.close_codes) >= connections)


# --- identity withdrawn ---------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"auth": "passthrough"},
        {"auth": "identity", "enabled": False},
        {"auth": "identity", "upstream_base_url": "http://127.0.0.1:9"},
    ],
    ids=["identity-to-passthrough", "disabled", "upstream-moved"],
)
async def test_a_withdrawn_identity_closes_the_live_relay(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, change: dict[str, Any]
) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with Upstream() as fake:
        opened = {"azure": {"upstream_base_url": fake.url(), "auth": "identity"}}
        with _serve(_cfg(opened)) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS['azure']}") as client:
                await client.send(_frame("azure", f"hello {EMAIL}"))
                await client.recv()  # the relay is live, on the proxy's identity
                await proxy.apply(_cfg({"azure": {"upstream_base_url": fake.url(), **change}}))
                closed = await _closed(client)
            await _upstream_closed(fake)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            await _until(lambda: not proxy.state.realtime_relays)
    _reload_close(closed, "[providers.azure] settings")
    # The upstream session the withdrawn identity opened carried nothing
    # after the reload, and the proxy ended it.
    assert fake.headers[0]["authorization"] == PROXY_TOKEN
    assert len(fake.received) == 1 and "«EMAIL_" in fake.texts()[0]
    assert fake.close_codes == [1000]
    assert len(auth.calls) == 1
    # Recorded like any relay (its one row at close), value-free.
    assert (row["status"], row["provider"], row["path"]) == (101, "azure", AZURE_GA)
    assert "closed 1012 (a config reload changed its [providers.azure] settings)" in caplog.text
    assert EMAIL not in caplog.text


async def test_frames_written_after_a_withdrawn_identity_never_reach_its_session(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The finding's trigger: the client keeps sending conversation items.
    auth = FakeAuth()
    _install(monkeypatch, auth)
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with Upstream() as fake:
        opened = {"azure": {"upstream_base_url": fake.url(), "auth": "identity"}}
        with _serve(_cfg(opened)) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS['azure']}") as client:
                await client.send(_frame("azure", "hello"))
                await client.recv()
                await proxy.apply(_cfg({"azure": {"upstream_base_url": fake.url()}}))
                for text in (f"mail {EMAIL}", "and more", "and more still"):
                    await _send(client, _frame("azure", text))
                closed = await _closed(client)
            await _upstream_closed(fake)
    _reload_close_while_writing(closed, caplog)
    assert len(fake.received) == 1 and "hello" in fake.texts()[0]
    assert fake.close_codes == [1000]


# --- tightened redaction --------------------------------------------------------------


# (provider, provider section, detection before, detection after, the value)
TIGHTENED = {
    "detection-reenabled": ("openai", {"detection": False}, None, None, EMAIL),
    "deny-added-vertex-identity": ("vertex", {"auth": "identity"}, None, {"deny": [DENY]}, DENY),
    "allowlist-narrowed-azure-identity": (
        "azure",
        {"auth": "identity"},
        {"allowlist": [EMAIL]},
        None,
        EMAIL,
    ),
    "block-mode-gemini": (
        "gemini",
        {},
        {"modes": {"email": "warn"}},
        {"modes": {"email": "block"}},
        EMAIL,
    ),
}


@pytest.mark.parametrize("case", list(TIGHTENED), ids=list(TIGHTENED))
async def test_tightened_redaction_closes_the_live_relay(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, case: str
) -> None:
    provider, section, before, after, value = TIGHTENED[case]
    _install(monkeypatch, FakeAuth())
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with Upstream() as fake:

        def config(rules: dict[str, Any] | None, **overrides: Any) -> Config:
            merged = {"upstream_base_url": fake.url(), **section, **overrides}
            return _cfg({provider: merged}, rules)

        loosened = config(before)
        reenabled = case == "detection-reenabled"
        tightened = config(after, detection=True) if reenabled else config(after)
        with _serve(loosened) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                # Before the reload the value leaves as the old policy says
                # (detection off / not yet denied / allowlisted / warn mode).
                await client.send(_frame(provider, f"first {value}"))
                await client.recv()
                await proxy.apply(tightened)
                await _send(client, _frame(provider, f"second {value}"))
                closed = await _closed(client)
            await _upstream_closed(fake)
    # Never a 1008 block either: the frame read after the reload is neither
    # redacted under the old policy nor judged under the new one — the
    # client reconnects under the new one.
    _reload_close_while_writing(closed, caplog)
    changed = f"[providers.{provider}] settings" if case == "detection-reenabled" else "[detection]"
    assert f"a config reload changed its {changed}" in caplog.text
    [first] = fake.texts()
    assert f"first {value}" in first
    assert value not in caplog.text


# --- the SIGHUP path ------------------------------------------------------------------


async def test_a_sighup_reload_closes_the_relay(tmp_path: Path) -> None:
    async with Upstream() as fake:
        path = tmp_path / "config.toml"
        section = f'[providers.openai]\nupstream_base_url = "{fake.url()}"\n'
        path.write_text(section + "detection = false\n")
        with _serve(load_config(path), config_path=path) as proxy:
            async with websockets.connect(f"ws://{proxy.host}/v1/realtime") as client:
                await client.send(_frame("openai", f"first {EMAIL}"))
                await client.recv()
                path.write_text(section)
                await proxy.on_loop(proxy.state.reload)  # what the SIGHUP handler runs
                assert proxy.state.config.providers["openai"].detection is True
                closed = await _closed(client)
            await _upstream_closed(fake)
    _reload_close(closed, "[providers.openai] settings")
    assert len(fake.received) == 1


# --- exactly which frame can still leave ----------------------------------------------


async def test_the_frame_in_hand_when_the_reload_lands_is_the_last_one_sent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The reload lands while the relay is handling a frame it had already
    # read. (Simulated from inside the frame's processing: in production a
    # reload can interleave only with the frame's send, once its bytes are
    # queued on the upstream socket.) That frame leaves under the admission
    # it was read under; the next frame the relay reads never leaves.
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with Upstream() as fake:
        opened = {"openai": {"upstream_base_url": fake.url(), "detection": False}}
        with _serve(_cfg(opened)) as proxy:
            real_floors = realtime.frame_floors

            def floors_then_reload(data: str | bytes) -> dict[str, int]:
                if "in hand" in (data if isinstance(data, str) else data.decode()):
                    proxy.state.apply_config(_cfg({"openai": {"upstream_base_url": fake.url()}}))
                return real_floors(data)

            monkeypatch.setattr(realtime, "frame_floors", floors_then_reload)
            async with websockets.connect(f"ws://{proxy.host}/v1/realtime") as client:
                await client.send(_frame("openai", f"in hand {EMAIL}"))
                await _send(client, _frame("openai", f"read after {EMAIL}"))
                closed = await _closed(client)
            await _upstream_closed(fake)
    _reload_close_while_writing(closed, caplog)
    # Relayed as detection = false said when it was read — verbatim.
    assert fake.texts() == [_frame("openai", f"in hand {EMAIL}")]


# --- a reload during the upgrade -----------------------------------------------------


async def test_a_reload_while_the_upgrade_is_authorized_is_never_dialled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = HeldAuth()
    _install(monkeypatch, auth)
    async with Upstream() as fake:
        opened = {"azure": {"upstream_base_url": fake.url(), "auth": "identity"}}
        with _serve(_cfg(opened)) as proxy:
            connecting = asyncio.ensure_future(_connect(f"ws://{proxy.host}{PATHS['azure']}"))
            await _until(auth.entered.is_set)
            await proxy.apply(_cfg({"azure": {"upstream_base_url": fake.url()}}))
            auth.release.set()
            client = await connecting
            closed = await _closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
            await _until(lambda: not proxy.state.realtime_relays)
    _reload_close(closed, "[providers.azure] settings")
    # The credential minted under the withdrawn identity is never used.
    assert fake.paths == []
    assert (row["status"], row["provider"]) == (503, "azure")


async def test_a_reload_while_the_upstream_is_dialled_relays_no_frame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with Upstream() as fake:
        fake.hold = asyncio.Event()
        # The upstream greets at once (session.created): never relayed.
        fake.greeting = '{"type": "session.created"}'
        opened = {"openai": {"upstream_base_url": fake.url(), "detection": False}}
        with _serve(_cfg(opened)) as proxy:
            connecting = asyncio.ensure_future(_connect(f"ws://{proxy.host}/v1/realtime"))
            await asyncio.wait_for(fake.dialled.wait(), 10)
            await proxy.apply(_cfg({"openai": {"upstream_base_url": fake.url()}}))
            fake.hold.set()
            client = await connecting
            await _send(client, _frame("openai", f"mail {EMAIL}"))
            relayed: list[Any] = []
            try:
                while True:
                    relayed.append(await asyncio.wait_for(client.recv(), 5))
            except websockets.exceptions.ConnectionClosed as ended:
                closed = ended.rcvd
            await _upstream_closed(fake)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
    # The upgrade already in flight completes (the HTTP in-flight rule) and
    # closes; the client is never served on it: no frame either way, the
    # upstream's greeting included, recorded as the reload's 503.
    assert relayed == []
    assert closed is None or closed.code == 1012
    assert "refused after dialling (a config reload changed its" in caplog.text
    assert fake.paths == ["/v1/realtime"]
    assert fake.received == []
    assert fake.close_codes == [1000]
    assert row["status"] == 503


# --- after the reload ------------------------------------------------------------------


async def test_the_client_reconnects_under_the_new_policy_and_the_same_session() -> None:
    async with Upstream() as fake:
        opened = {"openai": {"upstream_base_url": fake.url()}}
        with _serve(_cfg(opened)) as proxy:
            url = f"ws://{proxy.host}/v1/realtime"
            async with websockets.connect(url) as client:
                await client.send(_frame("openai", f"mail {EMAIL}"))
                await client.recv()
                await proxy.apply(_cfg(opened, {"deny": [DENY]}))
                closed = await _closed(client)
            async with websockets.connect(url) as client:
                await client.send(_frame("openai", f"{DENY}: mail {EMAIL}"))
                echoed = json.loads(await client.recv())
    _reload_close(closed, "[detection] policy")
    # The new connection redacts with the new deny string, and restores the
    # token the old connection issued: one vault session across the reload.
    first, second = (json.loads(text)["item"]["content"][0]["text"] for text in fake.texts())
    token = first.removeprefix("mail ")
    assert token.startswith("«EMAIL_") and DENY not in second and token in second
    assert echoed["item"]["content"][0]["text"] == f"{DENY}: mail {EMAIL}"


# --- a reload that changes nothing relevant ------------------------------------------


UNRELATED = {
    "inject-note": {"inject_system_note": False},
    "body-limits": {"max_body_bytes": 1 << 20, "max_body_strings": 50_000},
    "fuzzy": {"rehydration": {"fuzzy": False}},
    "other-provider": {"providers_extra": {"anthropic": {"upstream_base_url": "http://a.example"}}},
    "other-provider-disabled": {"providers_extra": {"cohere": {"enabled": False}}},
}


@pytest.mark.parametrize("case", list(UNRELATED), ids=list(UNRELATED))
@pytest.mark.parametrize("provider", ["openai", "azure"])
async def test_a_reload_that_changes_nothing_relevant_leaves_the_relay_open(
    monkeypatch: pytest.MonkeyPatch, case: str, provider: str
) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    async with Upstream() as fake:
        section = {"upstream_base_url": fake.url()}
        if provider == "azure":
            section["auth"] = "identity"
        top = dict(UNRELATED[case])
        extra = top.pop("providers_extra", {})
        with _serve(_cfg({provider: section})) as proxy:
            async with websockets.connect(f"ws://{proxy.host}{PATHS[provider]}") as client:
                await client.send(_frame(provider, "hello"))
                await client.recv()
                [relay] = proxy.state.realtime_relays
                await proxy.apply(_cfg({provider: section, **extra}, **top))
                # The relay carries on, redacting and restoring as before.
                await client.send(_frame(provider, f"mail {EMAIL}"))
                echoed = json.loads(await client.recv())
                assert relay.revoked is None
                assert proxy.state.realtime_relays == {relay}
    assert echoed["item"]["content"][0]["text"] == f"mail {EMAIL}"
    assert EMAIL not in fake.texts()[1] and "«EMAIL_" in fake.texts()[1]
    assert len(auth.calls) == (1 if provider == "azure" else 0)


# --- the registry holds open relays only ----------------------------------------------


async def test_a_relay_leaves_the_registry_when_its_connection_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Upstream() as fake:
        with _serve(_cfg({"openai": {"upstream_base_url": fake.url()}})) as proxy:
            async with websockets.connect(f"ws://{proxy.host}/v1/realtime") as client:
                await client.send(_frame("openai", "hello"))
                await client.recv()
                assert len(proxy.state.realtime_relays) == 1
            await _until(lambda: not proxy.state.realtime_relays)
    # Refused after admission: the authorizer failed, or the dial did.
    auth = FakeAuth(error=UpstreamAuthError("Microsoft Entra ID credentials"))
    _install(monkeypatch, auth)
    async with Upstream() as fake:
        refused = {"azure": {"upstream_base_url": fake.url(), "auth": "identity"}}
        dead = {"openai": {"upstream_base_url": "http://127.0.0.1:9"}}
        for opened, path in ((refused, PATHS["azure"]), (dead, "/v1/realtime")):
            with _serve(_cfg(opened)) as proxy:
                async with websockets.connect(f"ws://{proxy.host}{path}") as client:
                    closed = await _closed(client)
                assert closed is not None and closed.code == 1011
                await _until(lambda: not proxy.state.realtime_relays)


# --- the admission a reload compares -------------------------------------------------


async def test_stale_names_what_the_reload_changed() -> None:
    state: ProxyState = create_app(Config()).state.proxy
    azure = state.config.providers["azure"]
    relay = RealtimeRelay("azure", azure, None, state.detectors, state.allowlist, state.modes)
    assert relay.stale(state) is None

    def with_azure(**change: Any) -> None:
        state.config = dataclasses.replace(
            state.config,
            providers={**state.config.providers, "azure": dataclasses.replace(azure, **change)},
        )

    for change in (
        {"upstream_base_url": "https://res.openai.azure.com"},
        {"enabled": False},
        {"detection": False},
        {"auth": "identity"},
        {"region": "eu-west-1"},
    ):
        with_azure(**change)
        assert relay.stale(state) == "[providers.azure] settings", change
    with_azure()  # the same settings again (an equal, rebuilt ProviderConfig)
    state.config = dataclasses.replace(
        state.config,
        providers={
            **state.config.providers,
            "vertex": dataclasses.replace(azure, upstream_base_url="https://x.example"),
        },
    )
    assert relay.stale(state) is None  # another provider's settings
    state.upstream_auth = {"azure": FakeAuth()}
    assert relay.stale(state) == "upstream authorizer"
    state.upstream_auth = {}
    # Rebuilt detection objects — even equal ones: apply_config rebuilds
    # them exactly when [detection] changed.
    for name, rebuilt in (
        ("detectors", build_detectors(state.config.detection)),
        ("allowlist", build_allowlist(state.config.detection)),
        ("modes", build_modes(state.config.detection)),
    ):
        live = getattr(state, name)
        setattr(state, name, rebuilt)
        assert relay.stale(state) == "[detection] policy", name
        setattr(state, name, live)
    assert relay.stale(state) is None


async def test_revoke_is_first_wins_and_never_raises() -> None:
    state: ProxyState = create_app(Config()).state.proxy
    relay = RealtimeRelay(
        "openai",
        state.config.providers["openai"],
        None,
        state.detectors,
        state.allowlist,
        state.modes,
    )
    relay.revoke("[detection] policy")
    await asyncio.wait_for(relay.wait_revoked(), 5)
    relay.revoke("[providers.openai] settings")
    assert relay.revoked == "[detection] policy"
    # A relay whose loop is gone (its connection with it) still revokes.
    orphan = RealtimeRelay(
        "openai",
        state.config.providers["openai"],
        None,
        state.detectors,
        state.allowlist,
        state.modes,
    )
    gone = asyncio.new_event_loop()
    gone.close()
    orphan._loop = gone
    orphan.revoke("[detection] policy")
    assert orphan.revoked == "[detection] policy"


async def test_apply_config_revokes_only_the_relays_it_changed() -> None:
    state: ProxyState = create_app(Config()).state.proxy

    def relay_for(provider: str) -> RealtimeRelay:
        return RealtimeRelay(
            provider,
            state.config.providers[provider],
            None,
            state.detectors,
            state.allowlist,
            state.modes,
        )

    openai, gemini = relay_for("openai"), relay_for("gemini")
    state.realtime_relays |= {openai, gemini}
    state.apply_config(_cfg({"gemini": {"enabled": False}}))
    assert (openai.revoked, gemini.revoked) == (None, "[providers.gemini] settings")
    # A reload that fails to build swaps nothing and revokes nothing.
    broken = _cfg({}, {"allowlist_patterns": ["("]})
    with pytest.raises(ValueError, match="allowlist_patterns"):
        state.apply_config(broken)
    assert openai.revoked is None
    state.apply_config(_cfg({"gemini": {"enabled": False}}, {"deny": [DENY]}))
    assert openai.revoked == "[detection] policy"


async def test_rebuilt_authorizers_revoke_the_identity_relays_they_displace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = Registry()
    reg.build_upstream_auth = lambda name, provider: (
        FakeAuth() if provider.auth == "identity" else None
    )
    monkeypatch.setattr(registry_mod, "_registry", reg)
    azure = {"upstream_base_url": "https://res.openai.azure.com", "auth": "identity"}
    vertex = {"upstream_base_url": "https://us-east5-aiplatform.googleapis.com", "auth": "identity"}
    state: ProxyState = create_app(_cfg({"azure": azure, "vertex": vertex})).state.proxy
    relay = RealtimeRelay(
        "azure",
        state.config.providers["azure"],
        state.upstream_auth["azure"],
        state.detectors,
        state.allowlist,
        state.modes,
    )
    state.realtime_relays.add(relay)
    # A change that rebuilds no authorizer (a passthrough provider's) keeps it.
    passthrough_moved = {"openai": {"upstream_base_url": "http://o.example"}}
    state.apply_config(_cfg({"azure": azure, "vertex": vertex, **passthrough_moved}))
    assert relay.revoked is None
    # Vertex's upstream moves: every authorizer is rebuilt, Azure's included,
    # and the one that opened the relay is closed — so is the relay.
    moved = {**vertex, "upstream_base_url": "https://europe-west4-aiplatform.googleapis.com"}
    state.apply_config(_cfg({"azure": azure, "vertex": moved}))
    assert relay.revoked == "upstream authorizer"
