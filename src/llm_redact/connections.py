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
    __slots__ = ("_lock", "_loop", "closed", "grant", "queue", "recheck", "subject")

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
        self._lock = threading.Lock()
        self._loop = asyncio.get_running_loop()

    def close_for_access(self, reason: str) -> bool:
        with self._lock:
            if self.closed:
                return False
            self.closed = True
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

    def open_counts(self) -> dict[str, int]:
        """Open connections by kind (``/status``): counts only."""
        with self._lock:
            return dict(Counter(c.kind for c in self._connections))

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
            due = [c for c in self._connections if c.recheck is not None]
        if due:
            await asyncio.gather(*(self._recheck_one(c) for c in due))

    async def _recheck_one(self, connection: TrackedConnection) -> None:
        recheck = connection.recheck
        assert recheck is not None
        try:
            verdict = recheck()
            if inspect.isawaitable(verdict):
                verdict = await asyncio.wait_for(verdict, self.timeout)
            if not (verdict is None or isinstance(verdict, bool | str)):
                raise TypeError("a recheck answered neither a bool, None nor a string")
        except Exception as problem:  # noqa: BLE001 — a failed check closes (fail closed)
            self._bookkeeping_errors["recheck"] += 1
            logger.warning(
                "%s connection closed: its access re-check failed (%s)",
                connection.kind,
                type(problem).__name__,
            )
            self._close_one(connection, RECHECK_FAILED_REASON, "recheck_error")
            return
        if verdict is None or verdict is True:
            return
        reason = verdict if isinstance(verdict, str) and verdict else DEFAULT_REVOKED_REASON
        logger.info("%s connection closed: its access re-check refused it", connection.kind)
        self._close_one(connection, reason, "recheck")

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
        if self._serving and self._task is None:
            self._task = asyncio.get_running_loop().create_task(self.run())

    async def stop(self) -> None:
        self._serving = False
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
