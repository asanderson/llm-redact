"""The durable maps' background writer: one thread per vault manager.

After the provider answered, the proxy records which session a Responses
chain, a stored object and a Live resumption handle belong to — the vault
manager's ``record_response_session`` / ``record_object_session`` /
``record_handle_session``. Each is a synchronous database write (an fsync
on sqlite, a round trip to a remote RDBMS); run on the event loop, a slow
disk or database stalled every other request behind it. A manager the
proxy switches to background writes (its ``write_maps_in_background``)
hands them to a ``MapWriter`` instead:

- ONE writer thread per manager, with its OWN database connection (opened
  in that thread — a sqlite connection is never shared across threads), so
  the writes keep their order and never land inside a request's open
  ``run_batched`` batch on the proxy's connection;
- READ-YOUR-WRITES: until a write lands, the manager's lookups answer from
  this writer's OVERLAY (``verdict``) — the map as it will read once every
  queued write is applied — so a client that sends ``previous_response_id``
  or resumes a Live handle right after the answer is served exactly as when
  the write was synchronous. A write that lands leaves the overlay (the
  database answers); one that FAILS leaves it too: the record reads as
  unknown, as a failed synchronous write did (refused / sealed, never a
  wrong value);
- BOUNDED: at most ``MAX_PENDING_WRITES`` wait. Past that a write is not
  queued: it is counted under its stage, kept in the overlay only (at most
  ``MAX_UNWRITTEN`` such records, the oldest forgotten first), so THIS
  process still answers for it — after a restart (or on another replica)
  it reads as unknown: refused or sealed, never a wrong value;
- SESSION DELETES stay exact and NEVER wait for the writer: a
  whole-session delete (prune, forget — on the event loop) runs inside
  ``deleting``, which only marks it active. Once it committed, every
  queued write of a deleted session becomes an ERASE of its key, the write
  IN FLIGHT is erased again right after it ran, and the overlay answers
  "absent" meanwhile; the writer settles a write only while no delete is
  active, so a delete that committed before (or while) the write ran is
  always seen. The map then reads exactly as if those writes had landed
  before the delete removed them (a handle or chain into a
  pruned-and-recreated session stays unknown) — and a hung writer
  connection never holds a delete, nor the loop it runs on;
- FAULTS are contained and counted in the proxy's ``bookkeeping_errors``
  under the write's own stage (``response_id``, ``object_ids``,
  ``handle_map``), logged once per outage by exception TYPE only: the
  writer posts each outcome to the event loop that submitted it, so the
  counters and the outage state are only ever touched there;
- SHUTDOWN drains: the proxy's lifespan waits (``drain``, bounded by
  ``SHUTDOWN_DRAIN_SECONDS``) before the vault closes; ``close`` drops what
  is still queued — and gives up on a write still in flight once its
  thread does not finish in ``STOP_JOIN_SECONDS`` — counting each under its
  stage and logging their NUMBER only (a write left unwritten reads as
  unknown after the restart).

The writer thread exits after ``IDLE_SECONDS`` without work (closing its
connection) and starts again with the next write.

WAITING FOR A WRITE (``[vault] map_writes = "before_answer"``): the overlay
answers for THIS process only — another replica sharing the vault reads a
record once its write landed. A caller that must not hand a client the id a
write records before every replica can read it collects the writes it
submits inside ``awaited_writes()``: each queued write then carries a
``concurrent.futures.Future`` (``MapWrite.landed``) set once the write left
the writer — landed, failed (the record reads as unknown, as it did
written synchronously), dropped at ``close`` or lost with a dying writer
thread — so the waiter is always released, and the proxy awaits those
futures, bounded, before it sends the answer on. A write that is not
queued (the queue is full, the writer closed) carries none: nothing would
ever land to wait for. Outside ``awaited_writes()`` nothing is created.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, InvalidStateError
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

if TYPE_CHECKING:
    from collections import Counter

logger = logging.getLogger("llm_redact")

# Writes waiting for the writer thread, at most (each a few ids: bounded
# memory). Past it a write is kept in the overlay only (``MAX_UNWRITTEN``).
MAX_PENDING_WRITES = 10_000
# Records this process answers for although their write was never queued
# (the queue was full): the newest are kept, the oldest read as unknown.
MAX_UNWRITTEN = 10_000
# The writer thread exits after this long without work (it starts again
# with the next write): an idle proxy holds no extra connection.
IDLE_SECONDS = 5.0
# How long the proxy's shutdown waits for queued writes to land before the
# vault closes (``drain``); what is still queued then is counted and dropped.
SHUTDOWN_DRAIN_SECONDS = 5.0
# How long ``close`` waits for the writer thread to finish the write it is
# in (a daemon thread: a write stuck past it never holds the exit).
STOP_JOIN_SECONDS = 1.0


# The futures of the writes submitted inside ``awaited_writes()`` (None:
# nobody waits, no future is created).
_AWAITED: ContextVar[list[Future[None]] | None] = ContextVar(
    "llm_redact_awaited_map_writes", default=None
)


@contextmanager
def awaited_writes() -> Iterator[list[Future[None]]]:
    """Collect, in the yielded list, a future for every write queued
    inside the block (in this context): each is set once its write left
    the writer, however it ended (``MapWrite.landed``)."""
    pending: list[Future[None]] = []
    token = _AWAITED.set(pending)
    try:
        yield pending
    finally:
        _AWAITED.reset(token)


class Faults(Protocol):
    """How a write's outcome is counted (``vault.CheckFaults``: a stage, an
    optional counter, and the outage state) — touched on the loop only."""

    counter: Counter[str] | None
    failing: bool
    stage: str

    def failed(self, exc: Exception) -> None: ...

    def succeeded(self) -> None: ...


class Removed(NamedTuple):
    """An overlay verdict: the key's row is deleted where it names one of
    ``sessions`` (a handle superseded through ``replaces``) — the database
    still answers for any other session."""

    sessions: frozenset[str]


# What the overlay says of a key: its session, None (absent), or Removed.
Verdict = str | None | Removed


class _Miss:
    """No pending write touches the key: the database answers."""

    __slots__ = ()


MISS = _Miss()


class MapWrite:
    """One queued write: ``run`` applies it on the writer's connection
    (raising on failure), ``erase`` removes its key instead once its
    session was deleted before it ran, ``faults`` counts its outcome."""

    __slots__ = (
        "abandoned",
        "erase",
        "erase_after",
        "faults",
        "keys",
        "landed",
        "run",
        "session",
    )

    def __init__(
        self,
        session: str,
        run: Callable[[Any], None],
        erase: Callable[[Any], None],
        faults: Faults,
    ) -> None:
        self.session = session
        self.run = run
        self.erase = erase
        self.faults = faults
        # The overlay keys this write set (settled when it lands).
        self.keys: list[tuple[str, str]] = []
        # Its session was deleted while it was in flight: erase it again
        # once it ran (its row may have landed after the delete's COMMIT).
        self.erase_after = False
        # ``close`` gave up on it while in flight (and counted it): its own
        # late outcome is not posted.
        self.abandoned = False
        # Set once it left the writer, however it ended (``awaited_writes``:
        # created only for a write somebody waits for).
        self.landed: Future[None] | None = None

    def release(self) -> None:
        """Release whoever waits for this write (thread-safe; a second
        release, or one after the waiter gave up and cancelled, is a no-op)."""
        if self.landed is not None:
            with suppress(InvalidStateError):
                self.landed.set_result(None)


class _Entry:
    """An overlay entry: the key's verdict once every queued write is
    applied, and the LAST write that set it (None: kept in memory only)."""

    __slots__ = ("value", "write")

    def __init__(self, value: Verdict, write: MapWrite | None) -> None:
        self.value = value
        self.write = write


def _count(faults: Faults) -> None:
    """Count one lost write under its stage (on the loop's thread)."""
    if faults.counter is not None:
        faults.counter[faults.stage] += 1


class MapWriter:
    """The background writer of one vault manager's durable maps (see the
    module docstring). ``open_connection`` runs on the writer thread only,
    as does the ``close()`` of the connection it returns."""

    def __init__(
        self,
        open_connection: Callable[[], Any],
        *,
        max_pending: int = MAX_PENDING_WRITES,
        max_unwritten: int = MAX_UNWRITTEN,
        idle_seconds: float = IDLE_SECONDS,
    ) -> None:
        self._open_connection = open_connection
        self._max_pending = max_pending
        self._max_unwritten = max_unwritten
        self._idle_seconds = idle_seconds
        # Guards everything below; never held across database I/O.
        self._cond = threading.Condition()
        self._queue: deque[MapWrite] = deque()
        self._in_flight: MapWrite | None = None
        self._overlay: dict[tuple[str, str], _Entry] = {}
        # The records kept in the overlay only (oldest first).
        self._unwritten: dict[tuple[str, str], _Entry] = {}
        self._thread: threading.Thread | None = None
        self._closed = False
        # Whole-session deletes inside ``deleting`` (the writer settles a
        # write only while none is).
        self._deletes = 0
        # The event loop outcomes are posted to (the submitter's).
        self._loop: asyncio.AbstractEventLoop | None = None
        # Whether the queue is full (logged once per episode; loop only).
        self._overflowing = False
        # Whether the last drain ran out of time with no write landing
        # since (the writer is stuck: ``close`` then waits no further).
        self._stalled = False

    # -- the submitting side (the event loop) --------------------------------

    def submit(
        self,
        write: MapWrite,
        sets: tuple[tuple[str, str], ...] = (),
        removes: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Queue ``write``: its ``sets`` keys read as its session and its
        ``removes`` keys as deleted where they name its session, until it
        lands. Never waits on the database."""
        with suppress(RuntimeError):
            self._loop = asyncio.get_running_loop()
        with self._cond:
            if self._closed:
                _count(write.faults)
                return
            queued = len(self._queue) < self._max_pending
            owner = write if queued else None
            for key in sets:
                self._set(key, write.session, owner)
            for key in removes:
                self._remove(key, write.session, owner)
            if queued:
                awaited = _AWAITED.get()
                if awaited is not None:
                    write.landed = Future()
                    awaited.append(write.landed)
                self._queue.append(write)
                self._ensure_thread()
                self._cond.notify_all()
        if queued:
            if self._overflowing:
                self._overflowing = False
                logger.info("vault map writer: caught up; writes are queued again")
            return
        _count(write.faults)
        if not self._overflowing:
            self._overflowing = True
            logger.warning(
                "vault map writer: %d writes are waiting; further records are kept in"
                " memory only (after a restart they read as unknown)",
                self._max_pending,
            )

    def _set(self, key: tuple[str, str], session: str, owner: MapWrite | None) -> None:
        self._place(key, _Entry(session, owner))

    def _remove(self, key: tuple[str, str], session: str, owner: MapWrite | None) -> None:
        entry = self._overlay.get(key)
        if entry is None:
            self._place(key, _Entry(Removed(frozenset((session,))), owner))
        elif isinstance(entry.value, Removed):
            self._place(key, _Entry(Removed(entry.value.sessions | {session}), owner))
        elif entry.value == session:
            self._place(key, _Entry(None, owner))
        # A key pending for ANOTHER session (or absent) keeps its verdict:
        # ``replaces`` only ever deletes its own session's rows.

    def _place(self, key: tuple[str, str], entry: _Entry) -> None:
        self._overlay[key] = entry
        if key in self._unwritten:
            del self._unwritten[key]
        if entry.write is not None:
            entry.write.keys.append(key)
            return
        self._unwritten[key] = entry
        while len(self._unwritten) > self._max_unwritten:
            oldest = next(iter(self._unwritten))
            del self._unwritten[oldest]
            del self._overlay[oldest]

    def verdict(self, key: tuple[str, str]) -> Verdict | _Miss:
        """What the overlay says of ``key`` (``MISS``: the database
        answers). A write that lands leaves the overlay only after its
        COMMIT, so a miss always finds it in the database."""
        with self._cond:
            entry = self._overlay.get(key)
            return MISS if entry is None else entry.value

    @contextmanager
    def deleting(self) -> Iterator[Callable[[Iterable[str]], None]]:
        """Mark a whole-session delete active — never waiting for the
        writer (a write may be in flight on its own connection meanwhile).
        Call the yielded function with the sessions the delete removed
        (after its COMMIT): their queued writes become erases of their
        keys, a write of theirs in flight is erased again once it ran, and
        their overlay records read as absent — exactly as if they had
        landed before the delete. The writer settles no write while a
        delete is active, so none slips between the two."""
        with self._cond:
            self._deletes += 1
        try:
            yield self._deleted
        finally:
            with self._cond:
                self._deletes -= 1
                self._cond.notify_all()

    def _deleted(self, sessions: Iterable[str]) -> None:
        gone = set(sessions)
        with self._cond:
            for write in self._queue:
                if write.session in gone:
                    write.run = write.erase
            in_flight = self._in_flight
            if in_flight is not None and in_flight.session in gone:
                in_flight.erase_after = True
            for entry in self._overlay.values():
                if isinstance(entry.value, str) and entry.value in gone:
                    entry.value = None

    def drain(self, timeout: float) -> int:
        """Wait up to ``timeout`` seconds for every queued write to land
        and its outcome to be posted (blocking: the proxy runs it in a
        worker thread); how many have not."""
        with self._cond:
            self._cond.wait_for(self._idle, timeout)
            left = len(self._queue) + (self._in_flight is not None)
            self._stalled = left > 0
            return left

    def _idle(self) -> bool:
        """Whether nothing is left to wait for: no write queued or in
        flight — or no writer thread to apply them (under ``_cond``)."""
        return self._thread is None or (not self._queue and self._in_flight is None)

    def close(self, timeout: float = SHUTDOWN_DRAIN_SECONDS) -> int:
        """Stop. First drain for up to ``timeout`` seconds — a caller
        closing the manager directly never silently loses a write — unless
        the last drain ran out of time with no write landing since (the
        proxy's shutdown drains off the loop first: a stuck writer is not
        waited for twice). What is still queued then is dropped, the write
        in flight given up on when its thread does not finish within
        ``STOP_JOIN_SECONDS`` (a daemon thread: it never holds the exit),
        each counted under its write's stage and logged by NUMBER only, and
        the overlay forgotten. Returns how many were not written."""
        if not self._stalled:
            self.drain(timeout)
        with self._cond:
            self._closed = True
            dropped = list(self._queue)
            self._queue.clear()
            self._overlay.clear()
            self._unwritten.clear()
            thread = self._thread
            self._cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(STOP_JOIN_SECONDS)
        with self._cond:
            if self._in_flight is not None:
                self._in_flight.abandoned = True
                dropped.append(self._in_flight)
        for write in dropped:
            _count(write.faults)
            write.release()
        if dropped:
            logger.warning(
                "vault map writer: %d write(s) were not written at shutdown"
                " (their records read as unknown after the restart)",
                len(dropped),
            )
        return len(dropped)

    # -- the writer thread -------------------------------------------------------

    def _ensure_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._work, name="llm-redact-vault-maps", daemon=True
            )
            self._thread.start()

    def _work(self) -> None:
        connection: Any = None
        try:
            while (write := self._next_write()) is not None:
                connection = self._apply(write, connection)
        finally:
            lost = None
            with self._cond:
                # Only reached with the thread still registered when it died
                # (a BaseException): its write is lost (read as unknown, its
                # waiter released) and the next submit starts another thread.
                if self._thread is threading.current_thread():
                    self._thread = None
                    lost, self._in_flight = self._in_flight, None
                self._cond.notify_all()
            if lost is not None:
                lost.release()
            if connection is not None:
                with suppress(Exception):
                    connection.close()

    def _next_write(self) -> MapWrite | None:
        """Block until a write is queued and take it (in flight from now);
        None to exit (closed, or idle for ``IDLE_SECONDS``) — the thread is
        then forgotten under the lock, so the next submit starts another."""
        with self._cond:
            if not self._queue and not self._closed:
                self._cond.wait(self._idle_seconds)
            if self._queue and not self._closed:
                write = self._queue.popleft()
                self._in_flight = write
                return write
            self._thread = None
            return None

    def _apply(self, write: MapWrite, connection: Any) -> Any:
        """Apply ``write`` — and erase it again when its session was deleted
        while it ran (the erase's outcome is then the write's) — settle its
        overlay keys and post its outcome; returns the connection (opened
        on first use)."""
        try:
            connection, error = self._attempt(write.run, connection)
            if self._deleted_meanwhile(write):
                connection, error = self._attempt(write.erase, connection)
        finally:
            self._settle(write)
        if not write.abandoned:
            faults = write.faults
            if error is not None:
                self._post(faults.failed, error)
            elif faults.failing:
                self._post(faults.succeeded)
        # Its keys are settled (the database answers for them) and its
        # outcome is posted ahead of the waiter's wake-up: release it.
        write.release()
        # Only now is it done: a drain that returns has its outcome posted.
        with self._cond:
            self._in_flight = None
            self._stalled = False
            self._cond.notify_all()
        return connection

    def _attempt(
        self, operation: Callable[[Any], None], connection: Any
    ) -> tuple[Any, Exception | None]:
        """Run ``operation`` on the connection (opened on first use): the
        connection and the exception it raised, if any."""
        try:
            if connection is None:
                connection = self._open_connection()
            operation(connection)
        except Exception as exc:  # noqa: BLE001 — contained: counted, logged by type
            return connection, exc
        return connection, None

    def _deleted_meanwhile(self, write: MapWrite) -> bool:
        """After the write in flight ran: wait until no whole-session delete
        is active (one marks itself active before its COMMIT and reports
        its sessions after), then whether one removed its session — its row
        may have landed after that delete's COMMIT, so it must be erased.
        One erase suffices: a later delete finds the row already gone."""
        with self._cond:
            while self._deletes:
                self._cond.wait()
            return write.erase_after

    def _settle(self, write: MapWrite) -> None:
        """Landed or not, the database answers for ``write``'s keys now (a
        write that failed reads as unknown, as it did written
        synchronously)."""
        with self._cond:
            for key in write.keys:
                entry = self._overlay.get(key)
                if entry is not None and entry.write is write:
                    del self._overlay[key]

    def _post(self, outcome: Callable[..., None], *args: Any) -> None:
        """Run ``outcome`` on the submitting event loop (the counters and
        outage state are only touched there) — directly when there is none
        (a caller outside an event loop) or it has closed."""
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(outcome, *args)
                return
            except RuntimeError:  # the loop closed: nothing else runs on it
                pass
        outcome(*args)


def _nothing(sessions: Iterable[str]) -> None:
    """``holding``'s marker without a background writer: nothing queued."""


@contextmanager
def holding(writer: MapWriter | None) -> Iterator[Callable[[Iterable[str]], None]]:
    """``writer.deleting()`` for a whole-session delete — or, for a manager
    writing synchronously (no background writer), a plain block whose
    marker has nothing to do."""
    if writer is None:
        yield _nothing
        return
    with writer.deleting() as deleted:
        yield deleted
