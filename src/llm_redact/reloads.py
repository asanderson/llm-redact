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
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

logger = logging.getLogger("llm_redact")

_T = TypeVar("_T")


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
    """Runs reloads (``reload``: ``ProxyState.reload_off_loop``) one at a
    time on the event loop: ``request`` is the SIGHUP handler.

    - A request while no reload runs starts one, as a task this scheduler
      holds (a strong reference: an unreferenced task may be collected
      mid-run).
    - Requests while one runs schedule exactly ONE more reload after it,
      however many arrive: it reads the file again, so what the last signal
      found in the file is what is applied. Two reloads never run at once.
    - A reload that raises is logged by exception TYPE only (never its
      message, which could quote the file) and is never fatal: the next
      request starts a reload as usual.
    - ``close`` (first thing at shutdown) refuses every later request and
      cancels a reload still running: it applies nothing, and its build is
      closed by ``build_off_loop``."""

    def __init__(self, reload: Callable[[], Awaitable[None]]) -> None:
        self._reload = reload
        self._task: asyncio.Task[None] | None = None
        self._again = False
        self._closed = False

    @property
    def running(self) -> bool:
        """Whether a reload is running (or pending after the one running)."""
        return self._task is not None and not self._task.done()

    def request(self) -> None:
        """Ask for a reload (SIGHUP's handler, on the event loop). Ignored
        once ``close`` has begun the shutdown."""
        if self._closed:
            return
        if self.running:
            self._again = True
            return
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name="llm-redact config reload"
        )

    async def _run(self) -> None:
        while True:
            # Cleared as each reload begins: a request that arrived before
            # it (it reads the file after them) is served by it; one arriving
            # while it runs asks for the next.
            self._again = False
            try:
                await self._reload()
            except Exception as exc:  # noqa: BLE001 — a reload is never fatal
                logger.error("config reload failed unexpectedly (%s)", type(exc).__name__)
            if not self._again or self._closed:
                return

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
        applied again in this process. A build its worker thread finishes
        later is closed, never applied (``build_off_loop``)."""
        self._closed = True
        task, self._task = self._task, None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
