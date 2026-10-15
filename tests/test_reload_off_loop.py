"""SIGHUP reloads off the event loop (``ProxyState.reload_off_loop``,
``reloads.ReloadScheduler``).

A reload that changes ``[detection]`` loads every NER model — seconds — and
SIGHUP once ran it on the event loop, holding every request meanwhile. The
signal now starts a reload whose file read, parse and detection build run
on a worker thread; the apply alone runs on the loop, adopting that build
(``apply_config(prepared=)``) exactly as the config editor's save does.
These tests pin: the build's thread and a loop that keeps answering; one
reload at a time, with the signals arriving meanwhile coalesced into ONE
more that re-reads the file; failures that keep the running configuration
and are logged as before (an unexpected one by type only); an apply landing
during the build (an editor save) making the reload start over, never put
back the file it read before that save — bounded, then built on the loop;
a shutdown that cancels a reload still building and closes the build its
worker finishes later; the signal handler the lifespan registers; and, since
the swap now follows the signal by the build's length, where reloads stand
(/status ``config_reload``: a caller waits for its reload before reading the
configuration back), the restart-required lines a failed build still logs
first, and the WARNING for a reload that keeps running.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.proxy as proxy_mod
from llm_redact.config import apply_env_overrides, load_config
from llm_redact.proxy import ProxyState, create_app
from llm_redact.reloads import SLOW_RELOAD_SECONDS, ReloadScheduler, _Handoff, build_off_loop

PROXY = "http://127.0.0.1:8787"
EMAIL_ONLY = '[detection]\nenabled = ["email"]\n'
THREE_RULES = '[detection]\nenabled = ["email", "ipv4", "jwt"]\n'
TWO_RULES = '[detection]\nenabled = ["email", "ipv4"]\n'
HOLD_SECONDS = 10


class GatedBuilds:
    """``build_detectors`` (what every detection build calls) recording the
    thread each build runs on; a build OFF the test's event-loop thread
    waits for ``release`` — the slow NER model load — while one on the
    loop (an apply's own build) runs at once."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.loop_thread = threading.get_ident()
        self.threads: list[int] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.most_active = 0
        self._lock = threading.Lock()
        real = proxy_mod.build_detectors

        def build(detection: Any, **kwargs: Any) -> Any:
            with self._lock:
                self.threads.append(threading.get_ident())
                self.active += 1
                self.most_active = max(self.most_active, self.active)
            try:
                if threading.get_ident() != self.loop_thread:
                    self.entered.set()
                    assert self.release.wait(HOLD_SECONDS)
                return real(detection, **kwargs)
            finally:
                with self._lock:
                    self.active -= 1

        monkeypatch.setattr(proxy_mod, "build_detectors", build)

    @property
    def on_loop(self) -> int:
        return self.threads.count(self.loop_thread)

    @property
    def off_loop(self) -> int:
        return len(self.threads) - self.on_loop

    async def wait_entered(self) -> None:
        assert await asyncio.to_thread(self.entered.wait, HOLD_SECONDS)


def _app(tmp_path: Path, body: str = EMAIL_ONLY) -> tuple[Any, ProxyState, Path]:
    path = tmp_path / "config.toml"
    path.write_text(body)
    app = create_app(load_config(path), config_path=path)
    return app, app.state.proxy, path


def _enabled(state: ProxyState) -> tuple[str, ...]:
    return tuple(state.config.detection.enabled)


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=PROXY)


def _closed_builds(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every prepared build closed (``_PreparedConfig.close``), in order."""
    closed: list[Any] = []
    real = proxy_mod._PreparedConfig.close

    def close(self: Any) -> None:
        closed.append(self)
        real(self)

    monkeypatch.setattr(proxy_mod._PreparedConfig, "close", close)
    return closed


# --- off the loop -----------------------------------------------------------------------


async def test_a_sighup_reload_builds_off_the_loop_while_the_proxy_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state, path = _app(tmp_path)
    builds = GatedBuilds(monkeypatch)
    before = state.detection_snapshot
    path.write_text(THREE_RULES)
    async with app.router.lifespan_context(app), _client(app) as client:
        state.reloads.request()  # what SIGHUP's handler runs
        await builds.wait_entered()
        # The build waits on its worker; the loop keeps answering, and the
        # running configuration is the old one until the apply.
        began = time.perf_counter()
        health = await client.get("/__llm-redact/healthz")
        assert health.status_code == 200
        assert time.perf_counter() - began < HOLD_SECONDS / 2
        assert state.reloads.running
        assert _enabled(state) == ("email",) and state.detection_snapshot is before
        builds.release.set()
        await state.reloads.wait()
        assert _enabled(state) == ("email", "ipv4", "jwt")
        assert len(state.detectors) == 3
        assert state._static_context.redactor is state.redactor
        # Built once, on the worker: the apply adopted it.
        assert builds.off_loop == 1 and builds.on_loop == 0
        assert not state.reloads.running


async def test_an_unchanged_detection_builds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state, path = _app(tmp_path)
    builds = GatedBuilds(monkeypatch)
    builds.release.set()
    detectors = state.detectors
    path.write_text("inject_system_note = false\n\n" + EMAIL_ONLY)
    state.reloads.request()
    await state.reloads.wait()
    assert state.config.inject_system_note is False
    assert state.detectors is detectors and builds.threads == []


async def test_the_reload_keeps_reporting_restart_only_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    app, state, path = _app(tmp_path, '[vault]\nsession = "one"\n')
    path.write_text('[vault]\nsession = "two"\n')
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        state.reloads.request()
        await state.reloads.wait()
    messages = [record.getMessage() for record in caplog.records]
    assert "config reload: [vault] changes require restart; keeping current" in messages
    assert f"config reloaded ({len(state.detectors)} detection rules)" in messages
    assert state.config.vault.session == "one"


# --- one at a time, coalesced --------------------------------------------------------------


async def test_signals_during_a_reload_coalesce_into_one_more_that_rereads_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state, path = _app(tmp_path)
    builds = GatedBuilds(monkeypatch)
    reads: list[str] = []
    real_load = proxy_mod.load_config

    def load(config_path: Any = None) -> Any:
        reads.append(config_path.read_text())
        return real_load(config_path)

    monkeypatch.setattr(proxy_mod, "load_config", load)
    path.write_text(THREE_RULES)
    state.reloads.request()
    await builds.wait_entered()
    first = state.reloads._task
    for _ in range(3):
        state.reloads.request()  # three more signals while it builds
    assert state.reloads._task is first  # never a second reload at once
    path.write_text(TWO_RULES)  # what the last signal found in the file
    builds.release.set()
    await state.reloads.wait()
    # Exactly one more reload ran, after the first, reading the file again.
    assert reads == [THREE_RULES, TWO_RULES]
    assert builds.most_active == 1 and builds.off_loop == 2 and builds.on_loop == 0
    assert _enabled(state) == ("email", "ipv4")
    # Idle again: the next signal starts a new reload.
    state.reloads.request()
    await state.reloads.wait()
    assert len(reads) == 3


# --- failures keep the running configuration ---------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "this is not toml [ [",  # the parse fails (on the worker)
        "not_a_real_key = true\n",
        '[detection]\nenabled = ["no_such_rule"]\n',  # the build fails
    ],
)
async def test_a_failed_reload_keeps_the_running_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, body: str
) -> None:
    app, state, path = _app(tmp_path)
    before, snapshot = state.config, state.detection_snapshot
    path.write_text(body)
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        state.reloads.request()
        await state.reloads.wait()
    assert state.config is before and state.detection_snapshot is snapshot
    assert [record.getMessage().split(":")[0] for record in caplog.records] == [
        "config reload failed; keeping current config"
    ]
    # The next signal reloads as usual.
    path.write_text(THREE_RULES)
    state.reloads.request()
    await state.reloads.wait()
    assert _enabled(state) == ("email", "ipv4", "jwt")


async def test_a_missing_file_keeps_the_running_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app, state, path = _app(tmp_path)
    before = state.config
    path.unlink()
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        state.reloads.request()
        await state.reloads.wait()
    assert state.config is before
    assert "config reload failed; keeping current config" in caplog.text


async def test_a_failed_apply_keeps_the_running_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The build off the loop succeeds; the apply on the loop refuses (a
    # router whose reconfigure raises, say): logged as before, nothing
    # swapped, the prepared build closed.
    app, state, path = _app(tmp_path)
    closed = _closed_builds(monkeypatch)
    before = state.config
    path.write_text(THREE_RULES)

    def refuse(fresh: Any, *, prepared: Any = None) -> list[str]:
        raise proxy_mod.ConfigError("the apply refused")

    monkeypatch.setattr(state, "apply_config", refuse)
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        state.reloads.request()
        await state.reloads.wait()
    assert state.config is before
    assert "config reload failed; keeping current config: the apply refused" in caplog.text
    assert len(closed) == 1 and closed[0].take() is None


async def test_an_unexpected_failure_is_logged_by_type_and_never_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    app, state, path = _app(tmp_path)
    before = state.config
    real = proxy_mod._build_detection

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("a message that could quote the file: SECRET")

    monkeypatch.setattr(proxy_mod, "_build_detection", broken)
    path.write_text(THREE_RULES)
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        state.reloads.request()
        await state.reloads.wait()
    assert state.config is before
    assert "config reload failed unexpectedly (TypeError)" in caplog.text
    assert "SECRET" not in caplog.text
    assert state.reloads.failed == 1 and state.reloads.last is not None
    assert state.reloads.last.outcome == "failed"
    # The scheduler survived it: the next signal reloads.
    monkeypatch.setattr(proxy_mod, "_build_detection", real)
    state.reloads.request()
    await state.reloads.wait()
    assert _enabled(state) == ("email", "ipv4", "jwt")


# --- the lines a failed build logs ---------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # The file parses; its [detection] build fails (on the worker): the
        # restart-required line apply_config logs before its build comes
        # first, as the synchronous reload logs it.
        (
            '[vault]\nsession = "two"\n[detection]\nenabled = ["no_such_rule"]\n',
            [
                "config reload: [vault] changes require restart; keeping current",
                "config reload failed; keeping current config",
            ],
        ),
        # The file does not parse: nothing is compared, the failure alone.
        (
            '[vault]\nsession = "two"\nnot toml [ [\n',
            ["config reload failed; keeping current config"],
        ),
    ],
)
async def test_a_failed_reload_logs_what_the_synchronous_reload_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, body: str, expected: list[str]
) -> None:
    app, state, path = _app(tmp_path, '[vault]\nsession = "one"\n')
    path.write_text(body)

    def logged() -> list[str]:
        failed = "config reload failed; keeping current config"
        # A failure's cause cut off (a path, the built-in rule list).
        lines = [record.getMessage() for record in caplog.records]
        caplog.clear()
        return [failed if line.startswith(failed) else line for line in lines]

    with caplog.at_level(logging.INFO, logger="llm_redact"):
        assert state.reload() is False
        synchronous = logged()
        state.reloads.request()  # what SIGHUP runs
        await state.reloads.wait()
        off_loop = logged()
    assert off_loop == synchronous == expected
    assert state.config.vault.session == "one"
    assert state.reloads.failed == 1


# --- where reloads stand (/status config_reload) ------------------------------------------


async def test_status_says_a_reload_is_still_building_and_how_it_ended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The swap follows the signal by the build's length, so /status answers
    # from the configuration before it meanwhile — the config-edit command's
    # read-back would report an opt-out the file adds as absent. The
    # config_reload block says a reload runs, and how the one asked for
    # ended: the caller waits for it before reading the posture back.
    app, state, path = _app(tmp_path)
    builds = GatedBuilds(monkeypatch)
    path.write_text(EMAIL_ONLY + '[detection.mcp]\nexempt_servers = ["github"]\n')
    async with app.router.lifespan_context(app), _client(app) as client:

        async def status() -> dict[str, Any]:
            reply = await client.get("/__llm-redact/status")
            assert reply.status_code == 200
            body: dict[str, Any] = reply.json()
            return body

        idle = (await status())["config_reload"]
        assert idle == {
            "running": False,
            "running_seconds": None,
            "requested_total": 0,
            "applied_total": 0,
            "failed_total": 0,
            "last": None,
        }
        state.reloads.request()  # SIGHUP
        await builds.wait_entered()
        state.reloads.request()  # another while it builds: coalesced
        building = await status()
        assert building["mcp_exempt_servers"] == 0  # still the configuration before it
        block = building["config_reload"]
        assert block["running"] is True and block["requested_total"] == 2
        assert isinstance(block["running_seconds"], float) and block["running_seconds"] >= 0
        assert block["applied_total"] == 0 and block["last"] is None
        builds.release.set()
        await state.reloads.wait()
        applied = await status()
        assert applied["mcp_exempt_servers"] == 1
        block = applied["config_reload"]
        assert block["running"] is False and block["running_seconds"] is None
        # Two reloads ran for the two signals (the second re-read the file).
        assert block["requested_total"] == 2 and block["applied_total"] == 2
        assert block["failed_total"] == 0
        assert block["last"]["outcome"] == "applied"
        assert block["last"]["at"].endswith("+00:00")
        # A reload that fails keeps the running configuration and says so.
        path.write_text('[detection]\nenabled = ["no_such_rule"]\n')
        state.reloads.request()
        await state.reloads.wait()
        failed = await status()
        block = failed["config_reload"]
        assert block["requested_total"] == 3 and block["failed_total"] == 1
        assert block["last"]["outcome"] == "failed"
        assert failed["mcp_exempt_servers"] == 1  # the configuration kept


async def test_a_reload_still_running_past_the_bound_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A build that hangs (a stalled model volume, a plugin that never
    # answers): /healthz stays green and every later SIGHUP waits behind it,
    # so a WARNING names it — once per reload, never for one that ends in
    # time.
    assert SLOW_RELOAD_SECONDS == 120
    release = asyncio.Event()
    runs: list[str] = []

    async def reload() -> bool:
        runs.append("reload")
        if len(runs) == 1:
            await release.wait()
        return True

    scheduler = ReloadScheduler(reload, slow_seconds=0.05)
    message = (
        "config reload still running after 0 s; the configuration before it stays live"
        " and SIGHUPs meanwhile wait for it"
    )
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        scheduler.request()
        for _ in range(500):
            if caplog.records:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        assert [record.getMessage() for record in caplog.records] == [message]
        assert scheduler.status()["running"] is True
        release.set()
        await scheduler.wait()
        scheduler.request()  # one that ends in time
        await scheduler.wait()
        await asyncio.sleep(0.1)
    assert [record.getMessage() for record in caplog.records] == [message]
    assert runs == ["reload", "reload"] and scheduler.applied == 2


async def test_running_stays_true_until_a_cancelled_reload_has_ended() -> None:
    unwinding, finish = asyncio.Event(), asyncio.Event()

    async def reload() -> bool:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            unwinding.set()
            await finish.wait()  # still unwinding (a build closing, say)
            raise
        return True

    scheduler = ReloadScheduler(reload)
    scheduler.request()
    await asyncio.sleep(0)
    closing = asyncio.create_task(scheduler.close())
    await unwinding.wait()
    assert scheduler.running and not closing.done()
    finish.set()
    await closing
    assert not scheduler.running
    assert scheduler.status()["running"] is False
    assert scheduler.applied == scheduler.failed == 0  # cancelled: no outcome


# --- an apply landing during the build ----------------------------------------------------


async def test_an_editor_save_during_the_build_is_never_undone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reload read THREE_RULES and builds it; meanwhile the config editor
    # writes TWO_RULES and applies it (one synchronous step on the loop).
    # Applying what the reload read would put back the file as it was
    # before the save: the reload starts over and reads the file again.
    app, state, path = _app(tmp_path)
    builds = GatedBuilds(monkeypatch)
    closed = _closed_builds(monkeypatch)
    path.write_text(THREE_RULES)
    state.reloads.request()
    await builds.wait_entered()
    path.write_text(TWO_RULES)
    state.apply_config(apply_env_overrides(load_config(path)))  # the save
    assert _enabled(state) == ("email", "ipv4")
    builds.release.set()
    await state.reloads.wait()
    assert _enabled(state) == ("email", "ipv4")  # the saved file, not the one read before
    # The stale build was closed, never adopted; the second read found the
    # running [detection] and built nothing.
    assert builds.off_loop == 1 and builds.on_loop == 1
    assert closed and all(prepared.take() is None for prepared in closed)


async def test_an_apply_during_the_build_makes_the_reload_build_again_off_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An apply that changes the running [detection] without touching the
    # file: the first build was made against a generation no longer
    # running, so the reload builds again — off the loop — and that build
    # is the one adopted.
    app, state, path = _app(tmp_path)
    builds = GatedBuilds(monkeypatch)
    path.write_text(THREE_RULES)
    state.reloads.request()
    await builds.wait_entered()
    other = dataclasses.replace(state.config.detection, enabled=("email", "ipv4", "jwt", "ipv6"))
    state.apply_config(dataclasses.replace(state.config, detection=other))
    builds.release.set()
    await state.reloads.wait()
    assert _enabled(state) == ("email", "ipv4", "jwt")
    assert builds.on_loop == 1  # the other apply's own build, nothing else
    assert builds.off_loop == 2  # the stale build and the adopted one


async def test_builds_overlapped_every_time_fall_back_to_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Bounded: after RELOAD_BUILD_ATTEMPTS overlapped builds the reload
    # reads and builds on the loop, as every reload once did.
    monkeypatch.setattr(proxy_mod, "RELOAD_BUILD_ATTEMPTS", 2)
    app, state, path = _app(tmp_path)
    builds = GatedBuilds(monkeypatch)
    builds.release.set()
    loop = asyncio.get_running_loop()
    real = state._prepare_reload
    overlaps: list[int] = []

    def overlapped(live: Any) -> Any:
        built = real(live)
        applied = threading.Event()

        def apply() -> None:
            note = not state.config.inject_system_note
            state.apply_config(dataclasses.replace(state.config, inject_system_note=note))
            overlaps.append(1)
            applied.set()

        loop.call_soon_threadsafe(apply)
        assert applied.wait(HOLD_SECONDS)
        return built

    monkeypatch.setattr(state, "_prepare_reload", overlapped)
    path.write_text(THREE_RULES)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        state.reloads.request()
        await state.reloads.wait()
    assert len(overlaps) == 2
    assert (
        "config reload: the configuration changed during 2 builds off the event loop;"
        " building on it"
    ) in caplog.text
    assert _enabled(state) == ("email", "ipv4", "jwt")
    assert builds.off_loop == 2 and builds.on_loop == 1  # the fallback built on the loop


@pytest.mark.parametrize("stage", ["parse", "build"])
async def test_a_failure_overlapped_by_an_apply_is_judged_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, stage: str
) -> None:
    # A file read, parse or build that failed against a configuration no
    # longer running is not reported (nor its restart-required lines): the
    # reload reads and builds again, against the running one.
    app, state, path = _app(tmp_path)
    loop = asyncio.get_running_loop()
    real = state._prepare_reload
    calls: list[int] = []

    def first_fails(live: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            applied = threading.Event()

            def apply() -> None:
                state.apply_config(dataclasses.replace(state.config))
                applied.set()

            loop.call_soon_threadsafe(apply)
            assert applied.wait(HOLD_SECONDS)
            failure = proxy_mod.ConfigError("judged against the old configuration")
            if stage == "build":
                fresh = dataclasses.replace(
                    state.config, vault=dataclasses.replace(state.config.vault, session="two")
                )
                raise proxy_mod._DetectionBuildFailed(fresh, failure)
            raise failure
        return real(live)

    monkeypatch.setattr(state, "_prepare_reload", first_fails)
    path.write_text(THREE_RULES)
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        state.reloads.request()
        await state.reloads.wait()
    assert len(calls) == 2
    assert "config reload failed" not in caplog.text
    assert "require restart" not in caplog.text
    assert _enabled(state) == ("email", "ipv4", "jwt")


# --- shutdown ------------------------------------------------------------------------------


async def test_a_reload_building_at_shutdown_is_cancelled_and_its_build_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state, path = _app(tmp_path)
    builds = GatedBuilds(monkeypatch)
    closed = _closed_builds(monkeypatch)
    applied: list[Any] = []
    real_apply = state.apply_config
    monkeypatch.setattr(
        state, "apply_config", lambda *a, **k: (applied.append(a), real_apply(*a, **k))[1]
    )
    before = state.config
    path.write_text(THREE_RULES)
    async with app.router.lifespan_context(app):
        state.reloads.request()
        await builds.wait_entered()
    # The shutdown cancelled the reload before anything closed; the build is
    # still running on its worker.
    assert not state.reloads.running and closed == []
    builds.release.set()
    for _ in range(500):
        if closed:
            break
        await asyncio.sleep(0.01)
    # Finished after the cancellation: closed (its models freed), never applied.
    assert len(closed) == 1 and closed[0].take() is None
    assert applied == [] and state.config is before
    # No reload is begun once the shutdown has begun.
    state.reloads.request()
    assert not state.reloads.running


async def test_build_off_loop_closes_an_answer_whose_caller_is_cancelled_as_it_is_made() -> None:
    # The cancellation lands as the worker hands its answer over (queued on
    # the loop before the answer is): the caller never sees the answer, and
    # it is closed — by the cancellation or by the worker, whichever decides
    # second (both orders: test_the_handoff_closes_a_build_...).
    loop = asyncio.get_running_loop()
    closed: list[str] = []
    task: asyncio.Task[str]

    def make() -> str:
        loop.call_soon_threadsafe(task.cancel)
        return "build"

    async def caller() -> str:
        return await build_off_loop(make, closed.append)

    task = asyncio.create_task(caller())
    [outcome] = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(outcome, asyncio.CancelledError)
    for _ in range(500):
        if closed:
            break
        await asyncio.sleep(0.01)
    assert closed == ["build"]


def test_the_handoff_closes_a_build_whichever_side_decides_first() -> None:
    closed: list[str] = []
    # Made first, then abandoned: abandon closes it.
    handoff: _Handoff[str] = _Handoff(lambda: "first", closed.append)
    assert handoff.make() == "first" and closed == []
    handoff.abandon()
    assert closed == ["first"]
    handoff.abandon()  # once only
    assert closed == ["first"]
    # Abandoned first, then made: the worker closes it.
    late: _Handoff[str] = _Handoff(lambda: "late", closed.append)
    late.abandon()
    assert late.make() == "late" and closed == ["first", "late"]


async def test_build_off_loop_raises_what_the_build_raises() -> None:
    closed: list[Any] = []

    def make() -> Any:
        raise ValueError("the file does not parse")

    with pytest.raises(ValueError, match="does not parse"):
        await build_off_loop(make, closed.append)
    assert closed == []


async def test_closing_an_idle_scheduler_refuses_later_requests() -> None:
    ran: list[int] = []

    async def reload() -> None:
        ran.append(1)

    scheduler = ReloadScheduler(reload)
    await scheduler.close()
    scheduler.request()
    assert not scheduler.running and ran == []


async def test_a_scheduler_closed_mid_reload_runs_no_coalesced_one() -> None:
    entered = asyncio.Event()
    ran: list[str] = []

    async def reload() -> None:
        ran.append("start")
        entered.set()
        await asyncio.Event().wait()

    scheduler = ReloadScheduler(reload)
    scheduler.request()
    await entered.wait()
    scheduler.request()  # coalesced
    await scheduler.close()
    assert ran == ["start"] and not scheduler.running


async def test_a_coalesced_reload_is_dropped_when_the_scheduler_closes_meanwhile() -> None:
    # A reload ending after close() refused further requests (the shutdown
    # began while it applied) starts no coalesced one.
    ran: list[str] = []
    scheduler: ReloadScheduler

    async def reload() -> None:
        ran.append("reload")
        scheduler.request()  # a signal while it runs
        scheduler._closed = True  # the shutdown began

    scheduler = ReloadScheduler(reload)
    scheduler.request()
    await scheduler.wait()
    assert ran == ["reload"]


# --- the signal handler -----------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
async def test_the_lifespan_registers_the_off_loop_reload_for_sighup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, state, path = _app(tmp_path)
    loop = asyncio.get_running_loop()
    added: list[tuple[int, Callable[[], None]]] = []
    removed: list[int] = []
    monkeypatch.setattr(loop, "add_signal_handler", lambda sig, cb: added.append((sig, cb)))
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: removed.append(sig))
    async with app.router.lifespan_context(app):
        assert added == [(signal.SIGHUP, state.reloads.request)]
        path.write_text(THREE_RULES)
        added[0][1]()  # the signal arrives
        await state.reloads.wait()
        assert _enabled(state) == ("email", "ipv4", "jwt")
    assert removed == [signal.SIGHUP]


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
@pytest.mark.parametrize("error", [NotImplementedError, RuntimeError, ValueError])
async def test_no_signal_handler_where_the_loop_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    # Windows event loops, a loop off the main thread: reload unavailable,
    # the proxy serves and shuts down as before.
    app, state, path = _app(tmp_path)
    loop = asyncio.get_running_loop()
    removed: list[int] = []

    def refuse(sig: int, cb: Callable[[], None]) -> None:
        raise error

    monkeypatch.setattr(loop, "add_signal_handler", refuse)
    monkeypatch.setattr(loop, "remove_signal_handler", lambda sig: removed.append(sig))
    async with app.router.lifespan_context(app), _client(app) as client:
        assert (await client.get("/__llm-redact/healthz")).status_code == 200
    assert removed == []
