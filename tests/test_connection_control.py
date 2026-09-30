"""Open long-lived connections end when their admission ends
(plugin_api.ConnectionControl, Admission.grant/recheck).

An access gate admits a connection once, when it opens. A realtime relay or
the dashboard's live-events stream admitted under a user who is then
revoked, a key that is revoked, or a session that signs out used to stay
open until its client closed it. Now:

- the gate gets a handle at startup (``bind_connections``) and closes a
  subject's or a grant's open connections at once (``close``), from the
  event loop or any other thread;
- the core re-asks every open connection's ``recheck`` each
  ``recheck_interval`` seconds (the backstop for what the gate cannot see
  in this process), and closes on False, a reason string, an exception, a
  timeout or a nonsense answer (fail closed).

A realtime client is closed 1008 with the gate's reason and the upstream
1000; the connection's row stays the relay's 101. An events stream ends.
Real sockets for realtime (uvicorn on port 0, a fake ``websockets``
upstream — the test_realtime_reload harness); raw ASGI for the infinite
events stream (never through ASGITransport).
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import threading
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import websockets
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse, Response

import llm_redact.registry as registry_mod
from llm_redact import connections as connections_mod
from llm_redact.config import Config, ConfigError, ProviderConfig
from llm_redact.connections import (
    DEFAULT_REVOKED_REASON,
    RECHECK_FAILED_REASON,
    EventStream,
    LiveConnections,
    recheck_interval,
)
from llm_redact.plugin_api import Admission, ConnectionControl, DashboardHost
from llm_redact.proxy import ProxyState, create_app
from llm_redact.registry import Registry
from test_realtime_identity import _recent
from test_realtime_reload import Upstream, _cfg, _serve, _until, _upstream_closed

SECRET = "lrk_the-actual-key-never-logged"
GATE_REASON = "gate-reason-for-the-client-only"


class ConnGate:
    """An access gate with the connection members: admits every connection
    as ``subject`` under ``grant`` (both switchable per test), with a
    scripted recheck."""

    guards_dashboard = True

    def __init__(self, *, verdict: Any = True) -> None:
        self.control: ConnectionControl | None = None
        self.subject = "ada"
        self.grant = "grant-1"
        self.verdict = verdict
        self.rechecks = 0

    def admit(self, conn: HTTPConnection, surface: str) -> Admission:
        return Admission(subject=self.subject, grant=self.grant, recheck=self.recheck)

    def recheck(self) -> Any:
        self.rechecks += 1
        verdict = self.verdict
        if isinstance(verdict, BaseException):
            raise verdict
        if callable(verdict):
            return verdict()
        return verdict

    def bind_connections(self, control: ConnectionControl) -> None:
        self.control = control

    def status(self) -> dict[str, Any]:
        return {}

    async def handle(self, request: Request, host: DashboardHost) -> Response:
        return JSONResponse({})

    def close(self) -> None:
        pass


def _install(monkeypatch: pytest.MonkeyPatch, gate: object) -> None:
    reg = Registry()
    reg.build_access_gate = lambda config, license: gate
    monkeypatch.setattr(registry_mod, "_registry", reg)


# --- LiveConnections, unit ---------------------------------------------------------


class FakeConn:
    kind = "fake"

    def __init__(self, subject: str | None = None, grant: str | None = None, recheck: Any = None):
        self.subject = subject
        self.grant = grant
        self.recheck = recheck
        self.reasons: list[str] = []

    @property
    def closing(self) -> bool:
        return bool(self.reasons)

    def close_for_access(self, reason: str) -> bool:
        if self.reasons:
            return False
        self.reasons.append(reason)
        return True


def test_close_selects_by_subject_grant_or_both() -> None:
    live = LiveConnections(Counter())
    ada_1 = FakeConn("ada", "k1")
    ada_2 = FakeConn("ada", "k2")
    bob = FakeConn("bob", "k1")
    anonymous = FakeConn()
    for conn in (ada_1, ada_2, bob, anonymous):
        live.track(conn)
    with pytest.raises(ValueError, match="subject or a grant"):
        live.close(reason="x")
    # Every selector given must match.
    assert live.close(subject="ada", grant="k2", reason="key revoked") == 1
    assert ada_2.reasons == ["key revoked"]
    # Already closing: not closed (or counted) twice.
    assert live.close(subject="ada", reason="user revoked") == 1
    assert ada_1.reasons == ["user revoked"] and ada_2.reasons == ["key revoked"]
    # An empty reason falls back to the default.
    assert live.close(grant="k1", reason="") == 1
    assert bob.reasons == [DEFAULT_REVOKED_REASON]
    assert live.close(subject="nobody", reason="x") == 0
    assert anonymous.reasons == []
    assert live.closed == Counter({"revoked": 3})
    live.untrack(anonymous)
    # Closed connections still tracked (their handlers have not ended) are
    # no longer counted open.
    assert live.open_counts() == {}


async def _slow() -> bool:
    await asyncio.sleep(5)
    return True


async def _refuse_later() -> str:
    await asyncio.sleep(0)
    return "signed out"


@pytest.mark.parametrize(
    ("verdict", "reasons", "cause"),
    [
        (lambda: True, [], None),
        (lambda: None, [], None),
        (lambda: False, [DEFAULT_REVOKED_REASON], "recheck"),
        (lambda: "user deactivated", ["user deactivated"], "recheck"),
        (lambda: "", [DEFAULT_REVOKED_REASON], "recheck"),
        (_refuse_later, ["signed out"], "recheck"),
        (lambda: 1, [RECHECK_FAILED_REASON], "recheck_error"),
        (_slow, [RECHECK_FAILED_REASON], "recheck_error"),
    ],
    ids=["true", "none", "false", "reason", "empty", "awaitable", "nonsense", "timeout"],
)
async def test_recheck_verdicts(
    verdict: Callable[[], Any], reasons: list[str], cause: str | None
) -> None:
    errors: Counter[str] = Counter()
    live = LiveConnections(errors, interval=5)
    live.timeout = 0.05
    conn = FakeConn(recheck=verdict)
    live.track(conn)
    live.track(FakeConn())  # no recheck: never asked, never closed
    await live.recheck_all()
    assert conn.reasons == reasons
    assert live.closed == (Counter({cause: 1}) if cause else Counter())
    assert errors == (Counter({"recheck": 1}) if cause == "recheck_error" else Counter())


async def test_a_recheck_that_swallows_its_cancellation_cannot_hold_the_pass() -> None:
    # asyncio.wait_for waits for a timed-out awaitable to finish cancelling;
    # a check that swallows the cancellation held the pass — and every later
    # re-check — open indefinitely (fail open). The pass now ends at the
    # timeout and closes the connection.
    live = LiveConnections(Counter())
    live.timeout = 0.05
    release = asyncio.Event()

    async def stubborn() -> bool:
        while not release.is_set():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue  # swallowed
        return True

    conn = FakeConn(recheck=stubborn)
    live.track(conn)
    await asyncio.wait_for(live.recheck_all(), 2)
    assert conn.reasons == [RECHECK_FAILED_REASON]
    assert live.closed == Counter({"recheck_error": 1})
    release.set()
    await asyncio.sleep(0.01)


async def test_a_cancelled_pass_cancels_the_check_it_was_waiting_on() -> None:
    live = LiveConnections(Counter())
    live.timeout = 30
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def held() -> bool:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return True

    live.track(FakeConn(recheck=held))
    pass_ = asyncio.ensure_future(live.recheck_all())
    await started.wait()
    pass_.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pass_
    await asyncio.wait_for(cancelled.wait(), 1)


async def test_an_abandoned_check_that_later_fails_is_not_reported() -> None:
    live = LiveConnections(Counter())
    live.timeout = 0.01
    release = asyncio.Event()

    async def late_failure() -> bool:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await release.wait()
            raise RuntimeError(SECRET) from None
        return True

    conn = FakeConn(recheck=late_failure)
    live.track(conn)
    await live.recheck_all()
    assert conn.reasons == [RECHECK_FAILED_REASON]
    release.set()
    for _ in range(5):
        await asyncio.sleep(0)


async def test_stopping_mid_check_never_reports_the_checks_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # stop() cancels the pass while it waits on a check; a check that fails
    # once cancelled (a gate cleanup whose message could carry a URL or a
    # token) must never surface as an unretrieved task exception with its
    # repr at shutdown.
    caplog.set_level(logging.ERROR, logger="asyncio")
    live = LiveConnections(Counter(), interval=5.0)
    live.interval, live.timeout = 0.01, 5.0
    started = asyncio.Event()

    async def fails_when_cancelled() -> bool:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise RuntimeError(SECRET) from None
        return True

    live.start()
    live.track(FakeConn(recheck=fails_when_cancelled))
    await asyncio.wait_for(started.wait(), 2)
    await live.stop()
    for _ in range(5):
        await asyncio.sleep(0)
    gc.collect()
    assert "never retrieved" not in caplog.text
    assert SECRET not in caplog.text


async def test_a_failing_recheck_closes_and_logs_its_type_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    errors: Counter[str] = Counter()
    live = LiveConnections(errors)

    def boom() -> bool:
        raise RuntimeError(SECRET)

    conn = FakeConn(recheck=boom)
    live.track(conn)
    caplog.set_level(logging.INFO, logger="llm_redact")
    await live.recheck_all()
    assert conn.reasons == [RECHECK_FAILED_REASON]
    assert errors == Counter({"recheck": 1})
    assert "re-check failed (RuntimeError)" in caplog.text
    assert SECRET not in caplog.text
    # The connection is still tracked until its handler ends, but it is
    # closing: a second pass never asks again, counts no second error and
    # logs nothing more.
    caplog.clear()
    await live.recheck_all()
    assert live.closed == Counter({"recheck_error": 1})
    assert errors == Counter({"recheck": 1})
    assert caplog.text == ""


async def test_a_closing_connection_is_never_rechecked_again() -> None:
    # A connection closed for access stays tracked until its handler ends —
    # indefinitely, for an events client that stopped reading. It was asked
    # again every interval, counting a failure (and logging it) each time.
    errors: Counter[str] = Counter()
    live = LiveConnections(errors)
    calls = 0

    def refusing() -> bool:
        nonlocal calls
        calls += 1
        raise RuntimeError("down")

    conn = FakeConn("ada", recheck=refusing)
    other = FakeConn("bob", recheck=lambda: True)
    live.track(conn)
    live.track(other)
    assert live.close(subject="ada", reason="gone") == 1
    for _ in range(3):
        await live.recheck_all()
    assert calls == 0 and errors == Counter()
    assert live.open_counts() == {"fake": 1}


async def test_a_connection_with_a_check_still_running_is_not_asked_again() -> None:
    # A check that swallows its cancellation keeps running after its pass
    # abandoned it. The connection it asked about is never asked again while
    # it runs (a failed check: closed), so abandoned checks cannot pile up
    # per connection, pass after pass.
    errors: Counter[str] = Counter()
    live = LiveConnections(errors)
    live.timeout = 0.01
    release = asyncio.Event()
    started = 0

    async def stubborn() -> bool:
        nonlocal started
        started += 1
        while not release.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await release.wait()
        return True

    class Lingering(FakeConn):
        # A connection whose close never takes (a third-party connection
        # type without a closing state): the running check alone stops the
        # next one.
        @property
        def closing(self) -> bool:
            return False

    conn = Lingering(recheck=stubborn)
    live.track(conn)
    try:
        for _ in range(4):
            await live.recheck_all()
        assert started == 1
        assert len(live._abandoned) == 1
        assert errors == Counter({"recheck": 4})
    finally:
        release.set()  # the stubborn checks end, pass or fail
    for _ in range(5):
        await asyncio.sleep(0)
    assert live._abandoned == {}  # forgotten once it ended
    await live.recheck_all()
    assert started == 2


async def test_abandoned_checks_are_capped_across_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Past MAX_ABANDONED_CHECKS still running, a new awaitable recheck is a
    # failed check and never started (its coroutine closed, never awaited).
    monkeypatch.setattr(connections_mod, "MAX_ABANDONED_CHECKS", 2)
    errors: Counter[str] = Counter()
    live = LiveConnections(errors)
    live.timeout = 0.01
    release = asyncio.Event()
    started = 0

    async def stubborn() -> bool:
        nonlocal started
        started += 1
        while not release.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await release.wait()
        return True

    first = [FakeConn(recheck=stubborn) for _ in range(2)]
    for conn in first:
        live.track(conn)
    try:
        await live.recheck_all()
        assert started == 2 and len(live._abandoned) == 2
        late = FakeConn(recheck=stubborn)
        live.track(late)
        with warnings_as_errors():
            await live.recheck_all()
            gc.collect()
        assert started == 2  # never started
        assert late.reasons == [RECHECK_FAILED_REASON]
        assert errors == Counter({"recheck": 3})
    finally:
        release.set()  # the stubborn checks end, pass or fail
    for _ in range(5):
        await asyncio.sleep(0)
    assert live._abandoned == {}


@contextlib.contextmanager
def warnings_as_errors() -> Any:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


async def test_recheck_passes_run_concurrently() -> None:
    live = LiveConnections(Counter())
    live.timeout = 1
    started = asyncio.Event()
    release = asyncio.Event()

    async def held() -> bool:
        started.set()
        await release.wait()
        return True

    async def quick() -> str:
        await started.wait()
        release.set()
        return "gone"

    slow_conn, quick_conn = FakeConn(recheck=held), FakeConn(recheck=quick)
    live.track(slow_conn)
    live.track(quick_conn)
    # One slow check does not delay the others (it would deadlock here).
    await asyncio.wait_for(live.recheck_all(), 2)
    assert quick_conn.reasons == ["gone"] and slow_conn.reasons == []


@pytest.mark.parametrize("value", [True, "30", float("nan"), 4.9, 3601, -1])
def test_a_malformed_recheck_interval_is_a_config_error(value: Any) -> None:
    class Gate:
        recheck_interval = value

    with pytest.raises(ConfigError, match="recheck_interval"):
        recheck_interval(Gate())


def test_recheck_interval_default_and_bounds() -> None:
    class Gate:
        recheck_interval = 5

    assert recheck_interval(object()) == 30.0
    assert recheck_interval(None) == 30.0
    assert recheck_interval(Gate()) == 5.0
    Gate.recheck_interval = 3600.0
    assert recheck_interval(Gate()) == 3600.0


def test_startup_refuses_a_gate_with_a_malformed_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = ConnGate()
    gate.recheck_interval = 1  # type: ignore[attr-defined]
    _install(monkeypatch, gate)
    with pytest.raises(ConfigError, match="recheck_interval"):
        create_app(Config())


async def test_the_backstop_runs_until_stopped_and_survives_a_failed_pass(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    live = LiveConnections(Counter(), interval=0.01)
    passes = 0
    real = live.recheck_all

    async def counted() -> None:
        nonlocal passes
        passes += 1
        if passes == 1:
            raise RuntimeError(SECRET)
        await real()

    monkeypatch.setattr(live, "recheck_all", counted)
    caplog.set_level(logging.INFO, logger="llm_redact")
    # Not serving yet: a connection with a recheck does not start it.
    conn = FakeConn(recheck=lambda: False)
    live.track(conn)
    assert live._task is None
    live.start()  # serving, and a pending recheck: started at once
    await _until(lambda: conn.reasons != [])
    assert passes >= 2
    assert "re-check pass failed (RuntimeError)" in caplog.text
    assert SECRET not in caplog.text
    await live.stop()
    assert live._task is None
    await live.stop()  # idempotent


async def test_the_backstop_starts_lazily_or_eagerly() -> None:
    lazy = LiveConnections(Counter())
    lazy.start()
    assert lazy._task is None
    lazy.track(FakeConn())  # no recheck: still not needed
    assert lazy._task is None
    lazy.track(FakeConn(recheck=lambda: True))
    assert lazy._task is not None
    await lazy.stop()
    eager = LiveConnections(Counter(), eager=True)
    eager.start()
    assert eager._task is not None
    await eager.stop()


async def test_a_recheck_cancelled_from_outside_closes_and_the_backstop_lives_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A recheck awaiting a shared lookup that another path cancels ends in
    # CancelledError: a failed check (the connection closes, cause
    # recheck_error) — never the end of the backstop, whose later passes
    # still see a revocation only the recheck can.
    errors: Counter[str] = Counter()
    live = LiveConnections(errors, interval=0.01, eager=True)
    shared: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    asked = asyncio.Event()

    async def flaky() -> bool:
        asked.set()
        return await shared

    revoked = False
    ada = FakeConn("ada", recheck=flaky)
    bob = FakeConn("bob", recheck=lambda: "revoked" if revoked else True)
    caplog.set_level(logging.INFO, logger="llm_redact")
    live.start()
    live.track(ada)
    live.track(bob)
    await asyncio.wait_for(asked.wait(), 5)
    shared.cancel()
    await _until(lambda: ada.reasons != [])
    assert ada.reasons == [RECHECK_FAILED_REASON]
    assert errors == Counter({"recheck": 1})
    assert "re-check failed (CancelledError)" in caplog.text
    assert live._task is not None and not live._task.done()
    revoked = True
    await _until(lambda: bob.reasons != [])
    assert bob.reasons == ["revoked"]
    assert live.closed == Counter({"recheck_error": 1, "recheck": 1})
    await live.stop()
    assert live._task is None


async def test_stopping_the_backstop_mid_pass_still_cancels_it() -> None:
    # The backstop's OWN cancellation (stop) still ends it, even while a
    # recheck is awaited, and closes nothing.
    live = LiveConnections(Counter(), interval=0.01, eager=True)
    asked = asyncio.Event()

    async def hangs() -> bool:
        asked.set()
        await asyncio.Event().wait()
        return True

    conn = FakeConn(recheck=hangs)
    live.timeout = 30
    live.start()
    live.track(conn)
    await asyncio.wait_for(asked.wait(), 5)
    task = live._task
    assert task is not None
    await asyncio.wait_for(live.stop(), 5)
    assert task.cancelled() and conn.reasons == [] and live.closed == Counter()


async def test_a_pass_ended_by_a_cancellation_is_logged_and_the_loop_goes_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    live = LiveConnections(Counter(), interval=0.01)
    passes = 0

    async def cancelled_pass() -> None:
        nonlocal passes
        passes += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(live, "recheck_all", cancelled_pass)
    caplog.set_level(logging.INFO, logger="llm_redact")
    live.start()
    live.track(FakeConn(recheck=lambda: True))
    await _until(lambda: passes >= 2)
    assert "re-check pass failed (CancelledError)" in caplog.text
    await live.stop()


async def test_a_backstop_task_that_ended_is_restarted() -> None:
    live = LiveConnections(Counter(), interval=0.01)
    live.start()
    live.track(FakeConn(recheck=lambda: True))
    first = live._task
    assert first is not None
    first.cancel()  # ended by something other than stop()
    with contextlib.suppress(asyncio.CancelledError):
        await first
    conn = FakeConn(recheck=lambda: False)
    live.track(conn)
    assert live._task is not first
    await _until(lambda: conn.reasons != [])
    await live.stop()


# --- realtime relays, real sockets ---------------------------------------------------


def _openai(fake: Upstream) -> Config:
    return _cfg({"openai": {"upstream_base_url": fake.url()}})


async def _access_closed(client: Any) -> Any:
    try:
        while True:
            await asyncio.wait_for(client.recv(), 5)
    except websockets.exceptions.ConnectionClosed as closed:
        return closed.rcvd
    except TimeoutError:
        raise AssertionError("the relay stayed open") from None


async def _open_relay(proxy: Any) -> Any:
    client = await websockets.connect(f"ws://{proxy.host}/v1/realtime")
    await client.send('{"type": "noop"}')
    await client.recv()  # relayed both ways: live
    return client


async def _connect(url: str) -> Any:
    return await websockets.connect(url)


def _in_thread(fn: Callable[[], int]) -> int:
    """Run ``fn`` on a thread of its own (neither the test's loop nor the
    server's): close must be safe from anywhere."""
    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(fn()))
    thread.start()
    thread.join(5)
    return result[0]


@pytest.mark.parametrize("selector", ["subject", "grant"])
async def test_the_gate_closes_a_live_relay_from_another_thread(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, selector: str
) -> None:
    gate = ConnGate()
    _install(monkeypatch, gate)
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with Upstream() as fake:
        with _serve(_openai(fake)) as proxy:
            assert gate.control is proxy.state.connections
            client = await _open_relay(proxy)
            other = await _open_relay(proxy)
            assert proxy.state.connections.open_counts() == {"realtime": 2}
            # A different user's / credential's connection stays open.
            gate.subject, gate.grant = "bob", "grant-2"
            bob = await _open_relay(proxy)
            control = gate.control
            assert control is not None
            selectors = {"subject": "ada"} if selector == "subject" else {"grant": "grant-1"}
            closed_count = _in_thread(lambda: control.close(**selectors, reason=GATE_REASON))
            assert closed_count == 2
            for conn in (client, other):
                closed = await _access_closed(conn)
                assert closed is not None and (closed.code, closed.reason) == (1008, GATE_REASON)
            await _upstream_closed(fake, 2)
            await bob.send('{"type": "still"}')
            assert await bob.recv() == '{"type": "still"}'
            await bob.close()
            row = await _recent(proxy.host, lambda r: r["method"] == "WS" and r["user"] == "ada")
            await _until(lambda: not proxy.state.realtime_relays)
            status = httpx.get(f"http://{proxy.host}/__llm-redact/status").json()
            metrics = httpx.get(f"http://{proxy.host}/__llm-redact/metrics").text
    assert fake.close_codes[:2] == [1000, 1000]
    # The relay's own row, as after a reload revocation.
    assert (row["status"], row["path"]) == (101, "/v1/realtime")
    assert "closed 1008 (its access was revoked)" in caplog.text
    # The gate's reason reaches the client only, never the log.
    assert GATE_REASON not in caplog.text
    assert status["connections"]["closed_total"] == {"revoked": 2, "recheck": 0, "recheck_error": 0}
    assert status["connections"]["open"] == {}
    assert status["connections"]["recheck_interval_seconds"] == 30.0
    assert 'llm_redact_connections_closed_total{cause="revoked"} 2' in metrics
    # Never the grant or the subject in /status.
    assert "grant-1" not in str(status) and "ada" not in str(status["connections"])


@pytest.mark.parametrize(
    ("verdict", "code", "reason", "cause"),
    [
        (False, 1008, DEFAULT_REVOKED_REASON, "recheck"),
        ("user deactivated", 1008, "user deactivated", "recheck"),
        (RuntimeError(SECRET), 1008, RECHECK_FAILED_REASON, "recheck_error"),
        (_slow, 1008, RECHECK_FAILED_REASON, "recheck_error"),
    ],
    ids=["false", "reason", "exception", "timeout"],
)
async def test_the_backstop_closes_a_relay_its_recheck_refuses(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    verdict: Any,
    code: int,
    reason: str,
    cause: str,
) -> None:
    gate = ConnGate()
    _install(monkeypatch, gate)
    caplog.set_level(logging.INFO, logger="llm_redact")
    async with Upstream() as fake:
        with _serve(_openai(fake)) as proxy:
            live = proxy.state.connections
            live.interval, live.timeout = 0.05, 0.2
            client = await _open_relay(proxy)
            await _until(lambda: gate.rechecks >= 1)  # the backstop runs, admits
            gate.verdict = verdict
            closed = await _access_closed(client)
            await _upstream_closed(fake)
            await _until(lambda: not proxy.state.realtime_relays)
            status = httpx.get(f"http://{proxy.host}/__llm-redact/status").json()
    assert closed is not None and (closed.code, closed.reason) == (code, reason)
    assert fake.close_codes == [1000]
    assert status["connections"]["closed_total"][cause] == 1
    if cause == "recheck_error":
        assert status["bookkeeping_errors_total"] == {"recheck": 1}
    assert SECRET not in caplog.text


async def test_a_relay_revoked_while_authorizing_is_never_dialled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The dial's only await before it: the proxy's own credential. A
    # revocation landing there refuses the connection (recorded 403).
    from test_realtime_identity import AZURE_GA
    from test_realtime_reload import HeldAuth

    gate = ConnGate()
    auth = HeldAuth()
    reg = Registry()
    reg.build_access_gate = lambda config, license: gate
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    monkeypatch.setattr(registry_mod, "_registry", reg)
    async with Upstream() as fake:
        providers = {"azure": {"upstream_base_url": fake.url(), "auth": "identity"}}
        with _serve(_cfg(providers)) as proxy:
            # The handshake completes only once the proxy has decided.
            dialling = asyncio.ensure_future(_connect(f"ws://{proxy.host}{AZURE_GA}?model=d"))
            await asyncio.get_running_loop().run_in_executor(None, auth.entered.wait, 5)
            assert gate.control is not None
            assert gate.control.close(subject="ada", reason="revoked meanwhile") == 1
            auth.release.set()
            client = await dialling
            closed = await _access_closed(client)
            row = await _recent(proxy.host, lambda r: r["method"] == "WS")
    assert closed is not None and (closed.code, closed.reason) == (1008, "revoked meanwhile")
    assert row["status"] == 403
    assert fake.paths == []


async def test_a_gate_without_the_members_keeps_working(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_access_seam import FakeGate

    old = FakeGate(strict=True)
    _install(monkeypatch, old)
    async with Upstream() as fake:
        with _serve(_openai(fake)) as proxy:
            live = proxy.state.connections
            assert live.interval == 30.0
            async with websockets.connect(
                f"ws://{proxy.host}/v1/realtime", additional_headers={"x-test-key": "good"}
            ) as client:
                await client.send('{"type": "noop"}')
                assert await client.recv() == '{"type": "noop"}'
                # Tracked (attributed), but nothing to re-check: no backstop.
                [relay] = proxy.state.realtime_relays
                assert (relay.subject, relay.grant, relay.recheck) == (
                    "ada via websocket",
                    None,
                    None,
                )
                assert live._task is None
                # The core's own handle still closes by subject.
                assert live.close(subject="ada via websocket", reason="bye") == 1
                closed = await _access_closed(client)
    assert closed is not None and closed.code == 1008


# --- the live-events stream (raw ASGI) ---------------------------------------------------


def _events_scope() -> dict[str, Any]:
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/__llm-redact/events",
        "raw_path": b"/__llm-redact/events",
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:8787")],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8787),
    }


async def _open_events(app: Any) -> tuple[asyncio.Task[None], asyncio.Queue[dict[str, Any]]]:
    frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def receive() -> dict[str, Any]:
        await asyncio.Event().wait()  # the client never disconnects
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        await frames.put(message)

    task = asyncio.create_task(app(_events_scope(), receive, send))
    assert (await asyncio.wait_for(frames.get(), 5))["status"] == 200
    assert (await asyncio.wait_for(frames.get(), 5))["body"] == b": connected\n\n"
    return task, frames


async def _stream_ended(task: asyncio.Task[None], frames: asyncio.Queue[dict[str, Any]]) -> None:
    await asyncio.wait_for(task, 5)
    last: dict[str, Any] = {}
    while not frames.empty():
        last = frames.get_nowait()
    assert last.get("more_body") is False


def _app() -> Any:
    return create_app(Config(providers={"anthropic": ProviderConfig("http://upstream")}))


@pytest.mark.parametrize("how", ["loop", "thread", "recheck", "recheck_error"])
async def test_an_events_stream_ends_when_its_admission_ends(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    gate = ConnGate()
    _install(monkeypatch, gate)
    app = _app()
    state: ProxyState = app.state.proxy
    task, frames = await _open_events(app)
    try:
        [stream] = [c for c in state.connections._connections if isinstance(c, EventStream)]
        assert (stream.subject, stream.grant) == ("ada", "grant-1")
        assert state.connections.open_counts() == {"events": 1}
        control = gate.control
        assert control is not None
        if how == "loop":
            assert control.close(grant="grant-1", reason="signed out") == 1
        elif how == "thread":
            assert _in_thread(lambda: control.close(subject="ada", reason="revoked")) == 1
        elif how == "recheck":
            gate.verdict = "signed out"
            await state.connections.recheck_all()
        else:
            gate.verdict = RuntimeError(SECRET)
            await state.connections.recheck_all()
        await _stream_ended(task, frames)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert state.event_subscribers == set()
    assert state.connections.open_counts() == {}
    cause = {"loop": "revoked", "thread": "revoked"}.get(how, how)
    assert state.connections.closed == Counter({cause: 1})


async def test_a_full_events_queue_still_ends_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    # A slow reader's full queue gives up a row so the wake-up fits.
    gate = ConnGate()
    _install(monkeypatch, gate)
    app = _app()
    state: ProxyState = app.state.proxy
    frames_task, frames = await _open_events(app)
    try:
        [stream] = [c for c in state.connections._connections if isinstance(c, EventStream)]
        # Fill the queue synchronously (the stream cannot drain it meanwhile)
        # and close in the same step.
        while not stream.queue.full():
            stream.queue.put_nowait({"row": True})
        assert stream.close_for_access("bye") is True
        assert stream.close_for_access("again") is False
        await _stream_ended(frames_task, frames)
    finally:
        frames_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await frames_task


async def test_events_without_a_gate_are_tracked_unattributed() -> None:
    app = _app()
    state: ProxyState = app.state.proxy
    task, _frames = await _open_events(app)
    try:
        [stream] = [c for c in state.connections._connections if isinstance(c, EventStream)]
        assert (stream.subject, stream.grant, stream.recheck) == (None, None, None)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert state.connections.open_counts() == {}


async def test_the_lifespan_starts_and_stops_the_backstop(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = ConnGate()
    gate.recheck_interval = 5  # type: ignore[attr-defined]
    _install(monkeypatch, gate)
    app = _app()
    live: LiveConnections = app.state.proxy.connections
    assert live.interval == 5.0 and live.timeout == 5.0
    async with app.router.lifespan_context(app):
        assert live._task is not None  # declared an interval: eager
        task = live._task
    assert live._task is None and task.cancelled()


def test_close_before_a_loop_exists_never_raises() -> None:
    # A connection whose loop has closed has nothing left to wake.
    loop = asyncio.new_event_loop()

    async def make() -> EventStream:
        return EventStream(asyncio.Queue(maxsize=1), subject="s", grant=None, recheck=None)

    stream = loop.run_until_complete(make())
    loop.close()
    assert stream.close_for_access("gone") is True


def test_module_constants_are_value_free() -> None:
    assert connections_mod.CAUSES == ("revoked", "recheck", "recheck_error")
    assert connections_mod.ACCESS_CLOSE_CODE == 1008


def test_an_admission_compares_and_prints_by_its_verdict_only() -> None:
    # The connection fields are bookkeeping: a recheck is a fresh closure per
    # admission, and a grant is never shown.
    def recheck() -> bool:
        return True

    with_fields = Admission(subject="ada", grant="grant-secretish", recheck=recheck)
    assert with_fields == Admission(subject="ada")
    assert with_fields.grant == "grant-secretish" and with_fields.recheck is recheck
    assert "grant-secretish" not in repr(with_fields)
    assert hash(with_fields) == hash(Admission(subject="ada"))


async def test_an_events_stream_closed_for_access_is_closing_and_not_rechecked() -> None:
    errors: Counter[str] = Counter()
    live = LiveConnections(errors)
    asked = 0

    def recheck() -> bool:
        nonlocal asked
        asked += 1
        return False

    stream = EventStream(asyncio.Queue(1), subject="ada", grant=None, recheck=recheck)
    live.track(stream)
    assert not stream.closing and live.open_counts() == {"events": 1}
    await live.recheck_all()  # refused: closed
    assert asked == 1 and stream.closing
    # Its generator never ran its finally (a client that stopped reading):
    # still tracked, yet neither counted open nor asked again.
    await live.recheck_all()
    assert asked == 1 and live.open_counts() == {}
    assert live.closed == Counter({"recheck": 1})
