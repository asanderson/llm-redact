"""Open long-lived connections and their admission (plugin_api.ConnectionControl).

An access gate (llm-redact-pro) admits a connection once, when it opens. A
request lives for one response, but a realtime WebSocket relay and the
dashboard's ``/__llm-redact/events`` stream stay open for as long as their
client keeps them, so a revocation, a sign-out or an expiring credential
must reach them too. Each such connection records its admission's
``subject``, ``grant`` and ``recheck`` (``plugin_api.Admission``) in the one
synchronous stretch after admission, and ``LiveConnections`` holds every
open one:

- ``close(subject=..., grant=...)`` is the handle the gate receives through
  its optional ``bind_connections`` member: it closes, at once, every open
  connection matching the selectors (a revoked user, a revoked key, a
  signed-out session).
- The backstop (``run``) asks every open connection's ``recheck`` again each
  ``recheck_interval`` seconds, so a change the gate never sees in this
  process — another process revoking in a shared registry, an identity
  provider deactivating a user, a token expiring — closes it within one
  interval. A recheck that fails, times out or answers nonsense CLOSES the
  connection (fail closed).

The core holds no credential logic: it never interprets a grant, and never
logs, records or reports a grant, a subject or a refusal reason — only
connection kinds and close causes are counted.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import math
import threading
from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol

from .config import ConfigError
from .plugin_api import ConnectionRecheck as Recheck

logger = logging.getLogger("llm_redact.connections")

# What the core does when an open connection loses its admission: a
# WebSocket closes 1008 (policy violation) — its client should not simply
# reconnect with the same credential — and an events stream ends.
ACCESS_CLOSE_CODE = 1008
# The close reason when the gate supplied none (a recheck answering False).
DEFAULT_REVOKED_REASON = "llm-redact access for this connection was revoked"
# The close reason when a recheck failed (an exception, a timeout, a verdict
# that is neither a bool, None nor a string): fail closed.
RECHECK_FAILED_REASON = "llm-redact could not re-check this connection's access"
# Close causes, as counted (/status, llm_redact_connections_closed_total).
CAUSES = ("revoked", "recheck", "recheck_error")
# recheck_interval: default, and the bounds a gate's value must lie within.
DEFAULT_RECHECK_INTERVAL = 30.0
MIN_RECHECK_INTERVAL = 5.0
MAX_RECHECK_INTERVAL = 3600.0
# An awaitable recheck gets at most this long (and never longer than the
# interval): a check that cannot answer in time is a failed check.
MAX_RECHECK_TIMEOUT = 10.0
# Checks abandoned still running (timed out, or their pass cancelled; one that
# swallows its cancellation runs on) that may be alive at once: past it, a
# new awaitable recheck is a failed check and is never started. A
# connection with one of its own still running is never asked again either.
MAX_ABANDONED_CHECKS = 64


def recheck_interval(gate: object) -> float:
    """The gate's optional ``recheck_interval`` in seconds (read once at
    startup): the default without one; a bool, a non-number, a non-finite
    value or one outside [5, 3600] is a ConfigError."""
    raw = getattr(gate, "recheck_interval", None)
    if raw is None:
        return DEFAULT_RECHECK_INTERVAL
    if (
        isinstance(raw, bool)
        or not isinstance(raw, int | float)
        or not math.isfinite(raw)
        or not MIN_RECHECK_INTERVAL <= raw <= MAX_RECHECK_INTERVAL
    ):
        raise ConfigError(
            "the access gate's recheck_interval must be a number of seconds from"
            f" {MIN_RECHECK_INTERVAL:g} to {MAX_RECHECK_INTERVAL:g}"
        )
    return float(raw)


class TrackedConnection(Protocol):
    """One open long-lived connection, as ``LiveConnections`` sees it."""

    kind: str
    subject: str | None
    grant: str | None
    recheck: Recheck | None

    @property
    def closing(self) -> bool:
        """Whether the connection is already closing (its admission ended):
        never re-checked again, never counted open, though it stays tracked
        until its handler ends (a client that stopped reading can hold that
        off indefinitely)."""
        ...

    def close_for_access(self, reason: str) -> bool:
        """Close the connection because its admission ended: at once,
        thread-safe, never raising. True when this call closed it (False:
        it was already closing)."""
        ...


class EventStream:
    """One open ``/__llm-redact/events`` stream: its subscriber queue and
    the admission it was opened under. ``close_for_access`` ends the stream
    (the dashboard reconnects, and its gate then decides again)."""

    kind = "events"
    __slots__ = ("_lock", "_loop", "at_shutdown", "closed", "grant", "queue", "recheck", "subject")

    # Put on the queue to wake the stream so it sees ``closed`` (compared
    # by identity: never a row).
    WAKE: dict[str, Any] = {}

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        *,
        subject: str | None,
        grant: str | None,
        recheck: Recheck | None,
    ) -> None:
        self.queue = queue
        self.subject = subject
        self.grant = grant
        self.recheck = recheck
        self.closed = False
        # Ended because the server is shutting down (``end_for_shutdown``),
        # not because its admission ended.
        self.at_shutdown = False
        self._lock = threading.Lock()
        self._loop = asyncio.get_running_loop()

    @property
    def closing(self) -> bool:
        return self.closed

    def close_for_access(self, reason: str) -> bool:
        return self._end(shutdown=False)

    def end_for_shutdown(self) -> bool:
        """End the stream because the server is shutting down: the server
        waits for every open response before the app's shutdown runs, and
        this one would otherwise never end."""
        return self._end(shutdown=True)

    def _end(self, *, shutdown: bool) -> bool:
        with self._lock:
            if self.closed:
                return False
            self.closed = True
            self.at_shutdown = shutdown
        # Thread-safe, and fine from the loop's own thread. A closed loop
        # has no stream left to end.
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._wake)
        return True

    def _wake(self) -> None:
        # A full queue (a slow reader) gives up its oldest row to make room:
        # the stream is ending anyway.
        if self.queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(self.WAKE)


def _being_cancelled() -> bool:
    """Whether the running task itself is being cancelled — as opposed to a
    CancelledError raised by something it awaited that another path
    cancelled."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


async def _bounded(
    awaitable: Any, timeout: float, abandon: Callable[[asyncio.Future[Any]], None]
) -> Any:
    """``awaitable``'s result within ``timeout`` seconds, else TimeoutError.
    Unlike ``asyncio.wait_for`` this never waits for the cancelled check to
    finish: one that swallows its cancellation cannot hold the pass (and so
    every later re-check) open. A check left running — timed out, or its
    pass cancelled — is handed to ``abandon``, which cancels it and
    discards its outcome whenever it ends (never an unretrieved exception,
    whose repr a gate's message could fill)."""
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except BaseException:
        abandon(task)  # this pass is being cancelled: take the check with it
        raise
    if not done:
        abandon(task)
        raise TimeoutError("the access re-check did not answer in time")
    return task.result()


def _discard(task: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned check's outcome so it is never reported as an
    unretrieved exception (whose repr a gate's message could fill)."""
    if not task.cancelled():
        task.exception()


def _never_started(awaitable: Any) -> None:
    """Close an awaitable recheck that will not be awaited (a coroutine
    that never ran: nothing reported as never awaited)."""
    if inspect.iscoroutine(awaitable):
        awaitable.close()


class LiveConnections:
    """Every open realtime relay and events stream, with the admission each
    was opened under; the core's ``plugin_api.ConnectionControl``.

    ``close`` is synchronous and safe from any thread (the set is guarded by
    a lock, and each connection wakes its own event loop thread-safely).
    ``run`` is the periodic backstop, started by the app lifespan (``start``)
    once a connection with a recheck is open or the gate declares a
    ``recheck_interval``, and cancelled at shutdown (``stop``)."""

    def __init__(
        self,
        bookkeeping_errors: Counter[str],
        *,
        interval: float = DEFAULT_RECHECK_INTERVAL,
        eager: bool = False,
    ) -> None:
        self.interval = interval
        # An awaitable recheck's time limit.
        self.timeout = min(interval, MAX_RECHECK_TIMEOUT)
        self.closed: Counter[str] = Counter()
        self._bookkeeping_errors = bookkeeping_errors
        self._connections: set[TrackedConnection] = set()
        # Checks abandoned still running, each with the connection it asks
        # about (``MAX_ABANDONED_CHECKS``).
        self._abandoned: dict[asyncio.Future[Any], TrackedConnection] = {}
        self._lock = threading.Lock()
        # Start the backstop with the lifespan even before any connection
        # carries a recheck (the gate declared a recheck_interval).
        self._eager = eager
        self._serving = False
        self._task: asyncio.Task[None] | None = None

    # -- tracking (the event loop's thread) --

    def track(self, connection: TrackedConnection) -> None:
        with self._lock:
            self._connections.add(connection)
        if connection.recheck is not None:
            self._ensure_running()

    def untrack(self, connection: TrackedConnection) -> None:
        with self._lock:
            self._connections.discard(connection)

    def end_event_streams(self) -> int:
        """Server shutdown starts (``serving.ProxyServer``): end every open
        ``/__llm-redact/events`` stream, which would otherwise keep the
        server's drain — and so the app's shutdown — waiting for as long as
        its client stays connected. Realtime relays are closed by the server
        itself (1012). Not a close cause: nothing was revoked."""
        with self._lock:
            streams = [c for c in self._connections if isinstance(c, EventStream)]
        ended = sum(1 for stream in streams if stream.end_for_shutdown())
        if ended:
            logger.info("ended %d events stream(s): the server is shutting down", ended)
        return ended

    def open_counts(self) -> dict[str, int]:
        """Open connections by kind (``/status``): counts only — one already
        closing is no longer open, whenever its handler ends."""
        with self._lock:
            return dict(Counter(c.kind for c in self._connections if not c.closing))

    # -- plugin_api.ConnectionControl --

    def close(self, *, subject: str | None = None, grant: str | None = None, reason: str) -> int:
        """Close every open connection admitted for ``subject`` and/or under
        ``grant`` (every selector given must match; at least one is
        required) with ``reason``; returns how many this call closed."""
        if subject is None and grant is None:
            raise ValueError("ConnectionControl.close needs a subject or a grant")
        with self._lock:
            matching = [
                c
                for c in self._connections
                if (subject is None or c.subject == subject) and (grant is None or c.grant == grant)
            ]
        closed = sum(1 for c in matching if c.close_for_access(reason or DEFAULT_REVOKED_REASON))
        if closed:
            with self._lock:
                self.closed["revoked"] += closed
            logger.info("closed %d open connection(s): their access was revoked", closed)
        return closed

    # -- the backstop --

    async def recheck_all(self) -> None:
        """Ask every open connection's recheck once, concurrently; close each
        one no longer admitted."""
        with self._lock:
            # One already closing is never asked again: its admission ended.
            due = [c for c in self._connections if c.recheck is not None and not c.closing]
        if due:
            await asyncio.gather(*(self._recheck_one(c) for c in due))

    async def _recheck_one(self, connection: TrackedConnection) -> None:
        recheck = connection.recheck
        assert recheck is not None
        try:
            if connection in self._abandoned.values():
                raise TimeoutError("an earlier access re-check is still running")
            verdict = recheck()
            if inspect.isawaitable(verdict):
                if len(self._abandoned) >= MAX_ABANDONED_CHECKS:
                    _never_started(verdict)
                    raise TimeoutError("too many earlier access re-checks are still running")
                verdict = await _bounded(
                    verdict, self.timeout, lambda task: self._abandon(task, connection)
                )
            if not (verdict is None or isinstance(verdict, bool | str)):
                raise TypeError("a recheck answered neither a bool, None nor a string")
        except asyncio.CancelledError as problem:
            if _being_cancelled():
                raise  # the backstop itself is stopping
            # The recheck's own awaitable was cancelled (a shared lookup
            # another path cancelled): a failed check, which closes — it
            # must never end the backstop.
            self._check_failed(connection, problem)
            return
        except Exception as problem:  # noqa: BLE001 — a failed check closes (fail closed)
            self._check_failed(connection, problem)
            return
        if verdict is None or verdict is True:
            return
        reason = verdict if isinstance(verdict, str) and verdict else DEFAULT_REVOKED_REASON
        logger.info("%s connection closed: its access re-check refused it", connection.kind)
        self._close_one(connection, reason, "recheck")

    def _abandon(self, task: asyncio.Future[Any], connection: TrackedConnection) -> None:
        """Cancel a check left running; it counts as abandoned (bounded) until
        it ends, and its outcome is discarded then."""
        task.cancel()
        self._abandoned[task] = connection
        task.add_done_callback(self._abandoned_ended)

    def _abandoned_ended(self, task: asyncio.Future[Any]) -> None:
        self._abandoned.pop(task, None)
        _discard(task)

    def _check_failed(self, connection: TrackedConnection, problem: BaseException) -> None:
        self._bookkeeping_errors["recheck"] += 1
        logger.warning(
            "%s connection closed: its access re-check failed (%s)",
            connection.kind,
            type(problem).__name__,
        )
        self._close_one(connection, RECHECK_FAILED_REASON, "recheck_error")

    def _close_one(self, connection: TrackedConnection, reason: str, cause: str) -> None:
        if connection.close_for_access(reason):
            with self._lock:
                self.closed[cause] += 1

    async def run(self) -> None:
        """Recheck every ``interval`` seconds until cancelled; one pass at a
        time (a pass is awaited before the next sleep starts)."""
        while True:
            await asyncio.sleep(self.interval)
            try:
                await self.recheck_all()
            except asyncio.CancelledError as problem:
                if _being_cancelled():
                    raise  # stop(): the only way the backstop ends
                logger.error("connection re-check pass failed (%s)", type(problem).__name__)
            except Exception as problem:  # noqa: BLE001 — the backstop never dies
                logger.error("connection re-check pass failed (%s)", type(problem).__name__)

    def start(self) -> None:
        """The app lifespan started: run the backstop from now on, at once
        when the gate declared an interval, else with the first connection
        carrying a recheck."""
        self._serving = True
        if self._eager:
            self._ensure_running()
        else:
            with self._lock:
                pending = any(c.recheck is not None for c in self._connections)
            if pending:
                self._ensure_running()

    def _ensure_running(self) -> None:
        # A backstop task that ended anyway (whatever ended it) is replaced:
        # a done task would otherwise leave every later revocation unseen.
        if self._serving and (self._task is None or self._task.done()):
            self._task = asyncio.get_running_loop().create_task(self.run())

    async def stop(self) -> None:
        self._serving = False
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
