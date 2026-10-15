"""SIGHUP reloads, one at a time, their builds off the event loop.

A configuration reload reads the config file and builds what changed — for
a changed ``[detection]`` every NER model, which takes seconds (the ``llm``
backend's model-list check waits up to 10 s). Run on the event loop, as the
signal handler once ran it, that build held every request for its whole
length. ``ProxyState.reload_off_loop`` reads, parses and builds on a worker
thread (``build_off_loop``) and only then applies on the loop, in the one
synchronous step ``apply_config`` always was.

``ReloadScheduler`` is what SIGHUP calls (``request``): it runs those
reloads one at a time, coalesces the signals arriving while one runs into
exactly one more reload after it (which reads the file again, so the last
signal's file is the one applied), and is closed first thing at shutdown —
a reload still building then is cancelled, applies nothing, and the build
its worker thread finishes later is closed rather than applied.

Since the swap now follows the signal by the build's length, the scheduler
also says where its reloads stand (``status``: the ``/status``
``config_reload`` block) — one running, how many were asked for, applied or
failed, the last one's outcome — so whoever sent the signal (the
``config-edit`` agent command, an operator) can wait for it to apply before
reading the configuration back, and a reload that keeps running is logged.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Generic, Literal, NamedTuple, TypeVar

logger = logging.getLogger("llm_redact")

_T = TypeVar("_T")

# How long one reload may run before a WARNING says it still does (once per
# reload): far beyond a slow NER model load, so it names a build that hangs
# — a stalled model volume, a plugin's model-load policy or config section
# that never answers — which /healthz cannot show (the proxy keeps
# answering) and behind which every later SIGHUP waits.
SLOW_RELOAD_SECONDS = 120.0

ReloadOutcome = Literal["applied", "failed"]


class _Finished(NamedTuple):
    """The last reload's outcome and when it ended (``time.time()``)."""

    outcome: ReloadOutcome
    at: float


class _Handoff(Generic[_T]):
    """One build made on a worker thread for a caller awaiting it on the
    event loop. The build is handed to that caller — or closed when the
    caller is gone (cancelled: the shutdown began), whether it went before
    or after the build was made. Both sides decide under one lock, so a
    build is either handed over or closed: never left holding its NER
    models for the garbage collector, never applied by anyone."""

    def __init__(self, make: Callable[[], _T], close: Callable[[_T], None]) -> None:
        self._make = make
        self._close = close
        self._lock = threading.Lock()
        self._made: _T | None = None
        self._abandoned = False

    def make(self) -> _T:
        """On the worker: the build — closed already when its caller
        abandoned it while it was being made (nobody reads that answer)."""
        made = self._make()
        with self._lock:
            if not self._abandoned:
                self._made = made
                return made
        self._close(made)
        return made

    def abandon(self) -> None:
        """The caller is gone: close the build already made for it, and have
        one still being made closed when it is done."""
        with self._lock:
            self._abandoned = True
            made, self._made = self._made, None
        if made is not None:
            self._close(made)


async def build_off_loop(make: Callable[[], _T], close: Callable[[_T], None]) -> _T:
    """``make()`` on a worker thread (``asyncio.to_thread``): its answer, or
    its exception raised here, as the call made inline would raise it.
    Cancelled while it runs — or after it answered but before this
    coroutine took the answer — the answer is closed with ``close`` (now, or
    on the worker when ``make`` finishes): the caller never sees it, nobody
    applies it and nothing it holds outlives the cancellation. A ``make``
    still queued when the cancellation lands never runs at all."""
    handoff = _Handoff(make, close)
    try:
        return await asyncio.to_thread(handoff.make)
    except BaseException:
        handoff.abandon()
        raise


class ReloadScheduler:
    """Runs reloads (``reload``: ``ProxyState.reload_off_loop``, answering
    whether it applied) one at a time on the event loop: ``request`` is the
    SIGHUP handler.

    - A request while no reload runs starts one, as a task this scheduler
      holds (a strong reference: an unreferenced task may be collected
      mid-run).
    - Requests while one runs schedule exactly ONE more reload after it,
      however many arrive: it reads the file again, so what the last signal
      found in the file is what is applied. Two reloads never run at once.
    - A reload that raises is logged by exception TYPE only (never its
      message, which could quote the file), counted as failed and is never
      fatal: the next request starts a reload as usual.
    - A reload still running after ``slow_seconds`` is logged (WARNING,
      once): the requests arriving meanwhile wait for it.
    - ``status`` is the ``/status`` ``config_reload`` block.
    - ``close`` (first thing at shutdown) refuses every later request and
      cancels a reload still running: it applies nothing, and its build is
      closed by ``build_off_loop``."""

    def __init__(
        self,
        reload: Callable[[], Awaitable[bool]],
        *,
        slow_seconds: float = SLOW_RELOAD_SECONDS,
    ) -> None:
        self._reload = reload
        self._slow_seconds = slow_seconds
        self._task: asyncio.Task[None] | None = None
        self._again = False
        self._closed = False
        # When the reload running now began (time.monotonic()), else None.
        self._began: float | None = None
        self.requested = 0
        self.applied = 0
        self.failed = 0
        self.last: _Finished | None = None

    @property
    def running(self) -> bool:
        """Whether a reload is running (or pending after the one running) —
        until it has ENDED: one ``close`` cancelled still runs while it
        unwinds."""
        return self._task is not None and not self._task.done()

    def request(self) -> None:
        """Ask for a reload (SIGHUP's handler, on the event loop). Ignored
        once ``close`` has begun the shutdown."""
        if self._closed:
            return
        self.requested += 1
        if self.running:
            self._again = True
            return
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name="llm-redact config reload"
        )

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            # Cleared as each reload begins: a request that arrived before
            # it (it reads the file after them) is served by it; one arriving
            # while it runs asks for the next.
            self._again = False
            self._began = time.monotonic()
            slow = loop.call_later(self._slow_seconds, self._still_running)
            try:
                applied = await self._reload()
            except Exception as exc:  # noqa: BLE001 — a reload is never fatal
                logger.error("config reload failed unexpectedly (%s)", type(exc).__name__)
                applied = False
            finally:
                slow.cancel()
                self._began = None
            self._ended("applied" if applied else "failed")
            if not self._again or self._closed:
                return

    def _still_running(self) -> None:
        logger.warning(
            "config reload still running after %d s; the configuration before it stays live"
            " and SIGHUPs meanwhile wait for it",
            self._slow_seconds,
        )

    def _ended(self, outcome: ReloadOutcome) -> None:
        if outcome == "applied":
            self.applied += 1
        else:
            self.failed += 1
        self.last = _Finished(outcome, time.time())

    def status(self) -> dict[str, object]:
        """The ``/status`` ``config_reload`` block — counts, a flag and
        times, never a file name or a value: whether a reload is running
        (or due after the one running) and for how long the current one has
        run; the reloads asked for (SIGHUPs, coalesced ones included),
        applied and failed (the running configuration kept) since startup;
        and the last one's outcome and end. A caller that noted
        ``requested_total`` before its signal knows the reload it asked for
        has ended once ``requested_total`` is above that and ``running`` is
        false: ``last`` is then its outcome or a later one's."""
        began = self._began
        seconds = None
        if self.running:
            seconds = 0.0 if began is None else round(time.monotonic() - began, 1)
        last = None
        if self.last is not None:
            ended = datetime.fromtimestamp(self.last.at, tz=UTC)
            last = {"outcome": self.last.outcome, "at": ended.isoformat(timespec="seconds")}
        return {
            "running": self.running,
            "running_seconds": seconds,
            "requested_total": self.requested,
            "applied_total": self.applied,
            "failed_total": self.failed,
            "last": last,
        }

    async def wait(self) -> None:
        """Until no reload runs or is pending — every reload requested so
        far has applied, failed or been cancelled. Never raises what a
        reload raised (the scheduler logged it); cancelling the waiter
        leaves the reload running."""
        while self._task is not None and not self._task.done():
            await asyncio.wait({self._task})

    async def close(self) -> None:
        """Refuse every later request and cancel the reload still running,
        then wait for it to end: once this returns, no reload is begun or
        applied again in this process (and ``running`` reads False only
        then). A build its worker thread finishes later is closed, never
        applied (``build_off_loop``)."""
        self._closed = True
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
