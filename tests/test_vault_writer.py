"""The durable maps' background writer (``llm_redact.vault_writer``).

The response map rows (Responses chains, stored-object owner records) and
the Live resumption handle map are written after the provider answered.
They used to be synchronous database writes on the event loop — an fsync,
or a remote RDBMS round trip, stalling every other request. The proxy now
switches its vault manager to background writes (``write_maps_in_background``):
one writer thread per manager, with its own connection. Pinned here, for
every persistent backend where it matters:

- read-your-writes: until a write lands, the manager's lookups answer from
  the writer's overlay (a ``previous_response_id`` sent right after the
  answer resumes its own session — through the real app);
- a slow write never delays another request (the requests complete while
  the write is still held up);
- a whole-session delete stays exact: a queued write of a deleted session
  is erased, never resurrected;
- faults are contained and counted under the write's own stage, logged by
  type only;
- the queue is bounded: overflow is counted and kept in memory only;
- shutdown drains before the vault closes, bounded; what is left is
  counted and logged by number only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.proxy as proxy_mod
import llm_redact.registry as registry_mod
import llm_redact.vault_writer as writer_mod
from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.proxy import create_app
from llm_redact.registry import Registry
from llm_redact.vault import SqliteVaultManager
from llm_redact.vault_writer import MISS, MapWriter, Removed, holding
from test_vault_deletion import PERSISTENT, Factory, _factory, _opening

H = "live-handle:"
SECRET_ID = "resp_secret_9f"


@pytest.fixture(params=PERSISTENT)
def open_instance(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Factory]:
    yield from _opening(_factory(request.param, tmp_path, monkeypatch))


class _Parked:
    """Stands in for the writer thread so nothing is written: the writes
    queue up, none in flight (``_paused``)."""

    def join(self, timeout: float | None = None) -> None:
        pass


@contextlib.contextmanager
def _paused(writer: MapWriter) -> Iterator[None]:
    """Hold ``writer`` back: writes queue, none runs, until the block ends
    (a writer thread still running exits first: it idles out)."""
    deadline = time.monotonic() + 10
    while True:
        with writer._cond:
            if writer._thread is None:
                writer._thread = _Parked()  # type: ignore[assignment]
                break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    try:
        yield
    finally:
        with writer._cond:
            writer._thread = None
            if writer._queue and not writer._closed:
                writer._ensure_thread()


def _gate(writer: MapWriter) -> threading.Event:
    """Make the writer's database slow: its connection opens — so its first
    write runs — only once the returned event is set (a write IN FLIGHT)."""
    gate = threading.Event()
    opening = writer._open_connection

    def slow() -> Any:
        assert gate.wait(30)
        return opening()

    writer._open_connection = slow
    return gate


def _background(manager: Any) -> MapWriter:
    manager.write_maps_in_background()
    writer = manager._maps
    assert isinstance(writer, MapWriter)
    return writer


def _records(manager: Any) -> dict[str, str | None]:
    return {
        "resp": manager.lookup_response_session("resp_1"),
        "file": manager.lookup_response_session("file-1"),
        "handle": manager.lookup_handle_session(H + "a"),
    }


async def _until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


# --- read-your-writes, every persistent backend -----------------------------------


def test_lookups_answer_from_the_overlay_until_the_write_lands(open_instance: Factory) -> None:
    manager = open_instance()
    other = open_instance()  # another instance over the same database
    writer = _background(manager)
    gate = _gate(writer)
    manager.record_response_session("resp_1", "s")
    manager.record_object_session("file-1", "s")
    manager.record_handle_session(H + "a", "s")
    expected = {"resp": "s", "file": "s", "handle": "s"}
    assert _records(manager) == expected
    assert manager.lookup_response_sessions(["resp_1", "file-1", "nope"]) == {
        "resp_1": "s",
        "file-1": "s",
    }
    # Nothing has landed yet: the database does not know them.
    assert _records(other) == {"resp": None, "file": None, "handle": None}
    gate.set()
    assert manager.drain_map_writes(10) == 0
    assert _records(other) == expected
    assert _records(manager) == expected
    assert writer._overlay == {}
    # Nothing pending: a batched lookup reads every id from the database.
    assert manager.lookup_response_sessions(["resp_1", "file-1"]) == {
        "resp_1": "s",
        "file-1": "s",
    }


def test_writes_run_on_the_writers_own_thread_and_connection(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    writer = _background(manager)
    seen: list[tuple[str, Any]] = []
    opening = writer._open_connection

    def spy() -> Any:
        conn = opening()
        seen.append((threading.current_thread().name, conn))
        return conn

    writer._open_connection = spy
    manager.record_response_session("resp_1", "s")
    manager.record_response_session("resp_2", "s")
    assert manager.drain_map_writes(10) == 0
    ((thread, conn),) = seen  # one connection for both writes
    assert thread == "llm-redact-vault-maps"
    assert conn is not manager._conn
    manager.close()


def test_a_superseded_handle_reads_as_unknown_before_the_write_lands(
    open_instance: Factory,
) -> None:
    manager = open_instance()
    manager.record_handle_session(H + "old", "s")  # synchronous: on record
    manager.record_handle_session(H + "x", "t")
    writer = _background(manager)
    with _paused(writer):
        manager.record_handle_session(H + "new", "s", replaces=[H + "old"])
        # ``replaces`` only drops the same session's rows: t's stays.
        manager.record_handle_session(H + "y", "s", replaces=[H + "x"])
        manager.record_handle_session(H + "z", "s", replaces=[H + "new"])
        known = {d: manager.lookup_handle_session(H + d) for d in ("old", "new", "x", "z")}
        assert known == {"old": None, "new": None, "x": "t", "z": "s"}
        assert writer.verdict(("handle", H + "old")) == Removed(frozenset({"s"}))
    assert manager.drain_map_writes(10) == 0
    assert {d: manager.lookup_handle_session(H + d) for d in ("old", "new", "x", "z")} == known


def test_a_removal_pending_for_another_session_keeps_the_verdict(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    writer = _background(manager)
    with _paused(writer):
        manager.record_handle_session(H + "a", "t")
        manager.record_handle_session(H + "b", "s", replaces=[H + "a", H + "gone"])
        manager.record_handle_session(H + "c", "u", replaces=[H + "gone"])
        manager.record_handle_session(H + "d", "s", replaces=[H + "a"])
        assert manager.lookup_handle_session(H + "a") == "t"
        assert writer.verdict(("handle", H + "gone")) == Removed(frozenset({"s", "u"}))
        assert writer.verdict(("handle", H + "nothing")) is MISS
    assert manager.drain_map_writes(10) == 0
    assert manager.lookup_handle_session(H + "a") == "t"
    manager.close()


# --- whole-session deletes ---------------------------------------------------------


@pytest.mark.parametrize("how", ["forget", "prune"])
def test_a_session_deleted_before_its_writes_land_stays_unknown(
    how: str, open_instance: Factory
) -> None:
    manager = open_instance()
    other = open_instance()
    manager.get("s").placeholder_for("EMAIL", "ada@corp.example")
    manager.get("keep").placeholder_for("EMAIL", "bob@corp.example")
    manager.record_response_session("resp_moved", "keep")  # synchronous
    manager.record_handle_session(H + "moved", "keep")
    writer = _background(manager)
    with _paused(writer):
        manager.record_response_session("resp_1", "s")
        manager.record_object_session("file-1", "s")
        manager.record_handle_session(H + "a", "s")
        manager.record_response_session("resp_keep", "keep")
        manager.record_response_session("resp_moved", "s")  # re-homed, then deleted
        manager.record_handle_session(H + "moved", "s")
        if how == "forget":
            assert manager.forget_sessions(["s"]) == 1
        else:
            _backdate(manager)
            assert manager.prune_sessions(30, exclude=frozenset({"keep"})) == 1
        assert _records(manager) == {"resp": None, "file": None, "handle": None}
        assert manager.lookup_response_session("resp_moved") is None
        assert manager.lookup_handle_session(H + "moved") is None
        assert manager.lookup_response_session("resp_keep") == "keep"
    assert manager.drain_map_writes(10) == 0
    # The queued writes landed as erases: nothing of s comes back.
    assert _records(other) == {"resp": None, "file": None, "handle": None}
    assert other.lookup_response_session("resp_moved") is None
    assert other.lookup_handle_session(H + "moved") is None
    assert other.lookup_response_session("resp_keep") == "keep"


def _delete(how: str, manager: Any) -> None:
    if how == "forget":
        assert manager.forget_sessions(["s"]) == 1
    else:
        _backdate(manager)
        assert manager.prune_sessions(30, exclude=frozenset({"keep"})) == 1


def _held_after_run(writer: MapWriter) -> tuple[threading.Event, threading.Event]:
    """Hold the OLDEST queued write once it ran (committed) and before the
    writer settles it: ``ran`` is set then; the write finishes once
    ``release`` is set. Call while the writer is ``_paused``."""
    ran, release = threading.Event(), threading.Event()
    write = writer._queue[0]
    inner = write.run

    def run(conn: Any) -> None:
        inner(conn)
        ran.set()
        assert release.wait(30)

    write.run = run
    return ran, release


@pytest.mark.parametrize("how", ["forget", "prune"])
def test_a_delete_never_waits_for_a_write_in_flight(how: str, open_instance: Factory) -> None:
    """A whole-session delete runs on the event loop (the TTL prune, POST
    /sessions/prune, a purge): a write held up on the writer's own
    connection (a slow disk, a hung RDBMS lane) must never hold it — and
    the write, landing after the delete, is erased again."""
    manager = open_instance()
    other = open_instance()
    manager.get("s").placeholder_for("EMAIL", "ada@corp.example")
    manager.get("keep").placeholder_for("EMAIL", "bob@corp.example")
    writer = _background(manager)
    gate = _gate(writer)
    manager.record_response_session("resp_1", "s")
    manager.record_handle_session(H + "a", "s")
    manager.record_response_session("resp_keep", "keep")
    deadline = time.monotonic() + 10
    while writer._in_flight is None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    # A safety net only: before the fix the delete waited for this.
    safety = threading.Timer(15, gate.set)
    safety.start()
    try:
        _delete(how, manager)
        assert not gate.is_set()  # the delete returned while the write was held
        assert _records(manager)["resp"] is None
        assert manager.lookup_handle_session(H + "a") is None
    finally:
        safety.cancel()
        gate.set()
    assert manager.drain_map_writes(10) == 0
    # The held write ran after the delete committed: erased, never resurrected.
    assert other.lookup_response_session("resp_1") is None
    assert other.lookup_handle_session(H + "a") is None
    assert other.lookup_response_session("resp_keep") == "keep"


@pytest.mark.parametrize("how", ["forget", "prune"])
def test_a_session_deleted_after_its_write_ran_stays_deleted(
    how: str, open_instance: Factory
) -> None:
    manager = open_instance()
    other = open_instance()
    manager.get("s").placeholder_for("EMAIL", "ada@corp.example")
    manager.get("keep").placeholder_for("EMAIL", "bob@corp.example")
    writer = _background(manager)
    with _paused(writer):
        manager.record_response_session("resp_1", "s")
        ran, release = _held_after_run(writer)
    safety = threading.Timer(15, release.set)
    safety.start()
    try:
        assert ran.wait(10)  # committed, not yet settled
        _delete(how, manager)
        assert not release.is_set()
        assert manager.lookup_response_session("resp_1") is None
    finally:
        safety.cancel()
        release.set()
    assert manager.drain_map_writes(10) == 0
    assert manager.lookup_response_session("resp_1") is None
    assert other.lookup_response_session("resp_1") is None


def test_the_writer_settles_no_write_while_a_delete_is_active(tmp_path: Path) -> None:
    """A delete is marked active before it commits and reports its sessions
    after: a write that ran meanwhile (its row possibly written AFTER the
    delete's COMMIT) waits for the report, then is erased — never settled
    in between, where its row would outlive the delete."""
    manager = SqliteVaultManager(tmp_path / "vault.db")
    writer = _background(manager)
    with _paused(writer):
        manager.record_response_session("resp_1", "s")
        manager.record_response_session("resp_2", "t")
        ran, release = _held_after_run(writer)
    release.set()
    with writer.deleting() as deleted:
        assert ran.wait(10)
        time.sleep(0.05)  # room for a writer that would not wait
        with writer._cond:
            assert writer._in_flight is not None  # ran, but not settled
            assert ("row", "resp_1") in writer._overlay
        deleted(["s"])
    assert manager.drain_map_writes(10) == 0
    with sqlite3.connect(tmp_path / "vault.db") as conn:
        rows = conn.execute("SELECT response_id FROM response_sessions").fetchall()
    assert rows == [("resp_2",)]
    manager.close()


def _backdate(manager: Any) -> None:
    from test_vault_deletion import _age_all

    _age_all(manager)


def test_holding_without_a_writer_is_a_plain_block(tmp_path: Path) -> None:
    with holding(None) as deleted:
        deleted(["s"])  # nothing queued: nothing to do
    manager = SqliteVaultManager(tmp_path / "vault.db")
    assert manager.drain_map_writes(0) == 0  # no writer: nothing waits
    manager.close()


def test_a_delete_of_no_session_changes_nothing(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    writer = _background(manager)
    with _paused(writer):
        manager.record_response_session("resp_1", "s")
        with writer.deleting() as deleted:
            deleted([])
        assert manager.lookup_response_session("resp_1") == "s"
    assert manager.drain_map_writes(10) == 0
    assert manager.lookup_response_session("resp_1") == "s"
    manager.close()


# --- faults -------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["sqlite", "rdbms-dbapi"])
async def test_a_failed_write_is_counted_under_its_stage_and_reads_as_unknown(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _factory(backend, tmp_path, monkeypatch)()
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    writer = _background(manager)
    opening = writer._open_connection

    def broken() -> Any:
        raise sqlite3.OperationalError("disk I/O error at " + SECRET_ID)

    writer._open_connection = broken
    caplog.set_level(logging.INFO, logger="llm_redact")
    manager.record_response_session(SECRET_ID, "s")
    manager.record_object_session("file-1", "s")
    manager.record_handle_session(H + "a", "s")
    await _until(lambda: sum(counter.values()) == 3)
    assert counter == {"response_id": 1, "object_ids": 1, "handle_map": 1}
    assert manager.lookup_response_session(SECRET_ID) is None  # unknown: fail closed
    assert _records(manager) == {"resp": None, "file": None, "handle": None}
    assert "OperationalError" in caplog.text and SECRET_ID not in caplog.text
    # The database answers again: the outages end, each logged once.
    writer._open_connection = opening
    manager.record_response_session("resp_1", "s")
    manager.record_object_session("file-1", "s")
    manager.record_handle_session(H + "a", "s")
    await _until(lambda: "writes succeed again" in caplog.text and writer._overlay == {})
    await _until(lambda: caplog.text.count("writes succeed again") == 3)
    assert _records(manager) == {"resp": "s", "file": "s", "handle": "s"}
    assert counter == {"response_id": 1, "object_ids": 1, "handle_map": 1}
    manager.close()


# --- the bound ----------------------------------------------------------------------


def test_overflow_is_counted_and_kept_in_memory_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    other = SqliteVaultManager(tmp_path / "vault.db")
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    manager._maps = writer = MapWriter(
        manager._open_map_connection,
        max_pending=2,
        max_unwritten=2,
        idle_seconds=0.01,
    )
    caplog.set_level(logging.INFO, logger="llm_redact")
    with _paused(writer):
        for n in range(1, 6):
            manager.record_response_session(f"resp_{n}", "s")
        assert counter == {"response_id": 3}  # three were never queued
        assert caplog.text.count("writes are waiting") == 1  # once per episode
        # This process still answers for them — the newest two kept.
        known = {n: manager.lookup_response_session(f"resp_{n}") for n in range(1, 6)}
        assert known == {1: "s", 2: "s", 3: None, 4: "s", 5: "s"}
    assert manager.drain_map_writes(10) == 0
    assert {n: manager.lookup_response_session(f"resp_{n}") for n in range(1, 6)} == known
    # Durably only the queued ones: after a restart the others read unknown.
    assert {n: other.lookup_response_session(f"resp_{n}") for n in range(1, 6)} == {
        1: "s",
        2: "s",
        3: None,
        4: None,
        5: None,
    }
    manager.record_response_session("resp_6", "s")  # room again
    assert "caught up" in caplog.text
    # A queued write of an unwritten key takes it over; a deleted session's
    # unwritten records read as absent.
    with _paused(writer):
        manager.record_response_session("resp_4", "s")
        assert "resp_4" not in [key for _, key in writer._unwritten]
        with writer.deleting() as deleted:
            deleted(["s"])
        assert manager.lookup_response_session("resp_5") is None
    assert manager.drain_map_writes(10) == 0
    manager.close()
    other.close()


# --- shutdown -----------------------------------------------------------------------


def test_closing_directly_drains_first(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    gate = _gate(_background(manager))
    manager.record_response_session("resp_1", "s")
    manager.record_response_session("resp_2", "s")
    threading.Timer(0.05, gate.set).start()
    manager.close()  # no drain beforehand: close waits for the writes
    other = SqliteVaultManager(tmp_path / "vault.db")
    assert other.lookup_response_sessions(["resp_1", "resp_2"]) == {"resp_1": "s", "resp_2": "s"}
    other.close()


def test_drain_reports_what_has_not_landed_in_time(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    gate = _gate(_background(manager))
    manager.record_response_session("resp_1", "s")
    assert manager.drain_map_writes(0.05) == 1  # in flight, held up
    gate.set()
    assert manager.drain_map_writes(10) == 0
    manager.close()


def test_closing_drains_again_after_an_earlier_drain_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drain mid-life (a plugin's test helper, an operator tool) never
    stops a later ``close`` from draining what was queued since."""
    monkeypatch.setattr(writer_mod, "STOP_JOIN_SECONDS", 0.01)
    manager = SqliteVaultManager(tmp_path / "vault.db")
    manager._maps = writer = MapWriter(manager._open_map_connection, idle_seconds=0.01)
    manager.record_response_session("resp_0", "s")
    assert manager.drain_map_writes(10) == 0
    gate = _gate(writer)
    with _paused(writer):  # a fresh thread opens a fresh (slow) connection
        manager.record_response_session("resp_1", "s")
    threading.Timer(0.3, gate.set).start()
    manager.close()
    other = SqliteVaultManager(tmp_path / "vault.db")
    assert other.lookup_response_sessions(["resp_0", "resp_1"]) == {
        "resp_0": "s",
        "resp_1": "s",
    }
    other.close()


def test_a_write_landing_after_a_timed_out_drain_lets_close_drain_again(
    tmp_path: Path,
) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    manager._maps = writer = MapWriter(manager._open_map_connection, idle_seconds=0.01)
    gate = _gate(writer)
    manager.record_response_session("resp_1", "s")
    assert manager.drain_map_writes(0.01) == 1  # held up: stalled
    gate.set()
    assert manager.drain_map_writes(10) == 0  # landed: no longer stalled
    with _paused(writer):
        manager.record_response_session("resp_2", "s")
    manager.close()  # drains
    other = SqliteVaultManager(tmp_path / "vault.db")
    assert other.lookup_response_session("resp_2") == "s"
    other.close()


def test_a_write_given_up_on_at_close_is_counted_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The write in flight when ``close`` stops waiting is counted then —
    and its own late outcome (a failure here) is never counted again."""
    monkeypatch.setattr(writer_mod, "STOP_JOIN_SECONDS", 0.01)
    manager = SqliteVaultManager(tmp_path / "vault.db")
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    writer = _background(manager)
    gate = threading.Event()

    def broken() -> Any:
        assert gate.wait(30)
        raise sqlite3.OperationalError("disk I/O error")

    writer._open_connection = broken
    caplog.set_level(logging.WARNING, logger="llm_redact")
    manager.record_handle_session(H + "a", "s")
    assert manager.drain_map_writes(0.01) == 1
    assert writer.close() == 1
    assert counter == {"handle_map": 1}
    assert "1 write(s) were not written at shutdown" in caplog.text
    gate.set()
    deadline = time.monotonic() + 10
    while writer._in_flight is not None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert counter == {"handle_map": 1}
    assert "write failed" not in caplog.text
    manager._conn.close()


def test_close_drops_what_is_queued_counting_each_by_its_stage(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    writer = _background(manager)
    caplog.set_level(logging.WARNING, logger="llm_redact")
    with _paused(writer):
        manager.record_response_session(SECRET_ID, "s")
        manager.record_response_session("resp_2", "s")
        manager.record_object_session("file-1", "s")
        manager.record_handle_session(H + "a", "s")
        assert manager.drain_map_writes(0) == 4  # the shutdown's drain ran out
        manager.close()
    assert counter == {"response_id": 2, "object_ids": 1, "handle_map": 1}
    assert "4 write(s) were not written at shutdown" in caplog.text
    assert SECRET_ID not in caplog.text
    # A late write after close is counted and goes nowhere.
    manager.record_response_session("resp_late", "s")
    assert counter["response_id"] == 3
    assert writer.verdict(("row", "resp_late")) is MISS
    with sqlite3.connect(tmp_path / "vault.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM response_sessions").fetchone() == (0,)


def test_the_writer_thread_exits_when_idle_and_starts_again(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    manager._maps = writer = MapWriter(manager._open_map_connection, idle_seconds=0.02)
    manager.record_response_session("resp_1", "s")
    deadline = time.monotonic() + 10
    while writer._thread is not None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    manager.record_response_session("resp_2", "s")
    assert manager.drain_map_writes(10) == 0
    other = SqliteVaultManager(tmp_path / "vault.db")
    assert other.lookup_response_sessions(["resp_1", "resp_2"]) == {"resp_1": "s", "resp_2": "s"}
    other.close()
    manager.close()


def test_a_writer_thread_that_dies_is_replaced_by_the_next_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    manager = SqliteVaultManager(tmp_path / "vault.db")
    writer = _background(manager)

    class Died(BaseException):
        pass

    opening = writer._open_connection
    calls = 0

    def dying() -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Died
        return opening()

    writer._open_connection = dying
    manager.record_response_session("resp_1", "s")
    deadline = time.monotonic() + 10
    while writer._thread is not None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert manager.lookup_response_session("resp_1") is None  # lost: unknown
    manager.record_response_session("resp_2", "s")
    assert manager.drain_map_writes(10) == 0
    assert manager.lookup_response_session("resp_2") == "s"
    manager.close()


def test_without_an_event_loop_outcomes_are_handled_in_the_writer(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    writer = _background(manager)

    def broken() -> Any:
        raise sqlite3.OperationalError("locked")

    writer._open_connection = broken
    manager.record_response_session("resp_1", "s")
    assert manager.drain_map_writes(10) == 0
    assert counter == {"response_id": 1}
    manager.close()


async def test_an_outcome_after_the_loop_closed_is_handled_in_the_writer(
    tmp_path: Path,
) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    writer = _background(manager)
    closed = asyncio.new_event_loop()
    closed.close()
    writer._loop = closed
    writer._post(counter.update, ["response_id"])
    assert counter == {"response_id": 1}
    manager.close()


# --- through the real app -----------------------------------------------------------

UPSTREAM = "https://upstream.test"


class ChainRouter:
    """A per-conversation router that resolves ``previous_response_id``
    through the vault manager's durable lookup ONLY (as llm-redact-pro's
    does: its absence is the truth), else by the first input."""

    mode = "per-conversation"

    def __init__(self, durable_lookup: Callable[[str], str | None] | None) -> None:
        assert durable_lookup is not None
        self.durable = durable_lookup

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        previous = body.get("previous_response_id") if isinstance(body, dict) else None
        if isinstance(previous, str):
            return self.durable(previous) or f"orphan:{previous}"
        return "conv-a"

    def record_response_id(self, response_id: str, session_id: str) -> bool:
        return True


class Responses:
    """A fake Responses upstream: each answer a new id; bodies captured."""

    def __init__(self) -> None:
        self.bodies: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content.decode())
        n = len(self.bodies)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"id": f"chatcmpl-{n}", "choices": []})
        return httpx.Response(200, json={"id": f"resp_{n}", "object": "response", "output": []})


def _app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, upstream: Responses) -> Any:
    reg = Registry()
    reg.build_session_router = lambda config, **kw: ChainRouter(kw.get("durable_lookup"))
    monkeypatch.setattr(registry_mod, "_registry", reg)
    providers = {**Config().providers, "openai": ProviderConfig(UPSTREAM)}
    vault = VaultConfig(backend="sqlite", path=str(tmp_path / "vault.db"), session="main")
    return create_app(
        Config(providers=providers, vault=vault), upstream_transport=httpx.MockTransport(upstream)
    )


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _durable(tmp_path: Path, response_id: str) -> str | None:
    other = SqliteVaultManager(tmp_path / "vault.db")
    try:
        return other.lookup_response_session(response_id)
    finally:
        other.close()


async def test_a_slow_map_write_never_delays_another_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream = Responses()
    app = _app(monkeypatch, tmp_path, upstream)
    manager = app.state.proxy.vault_manager
    gate = _gate(manager._maps)  # the map write hangs until released
    try:
        async with _client(app) as client:
            chain = client.post("/v1/responses", json={"model": "m", "input": "hi"})
            other = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            )
            answers = await asyncio.wait_for(asyncio.gather(chain, other), 10)
            assert [a.status_code for a in answers] == [200, 200]
            later = await asyncio.wait_for(
                client.post("/v1/responses", json={"model": "m", "input": "again"}), 10
            )
            assert later.status_code == 200
        # Every request was answered while the first map write still hung.
        assert manager._maps._queue or manager._maps._in_flight is not None
        assert _durable(tmp_path, "resp_1") is None
        assert manager.lookup_response_session("resp_1") == "conv-a"
    finally:
        gate.set()
    assert await asyncio.to_thread(manager.drain_map_writes, 10) == 0
    assert _durable(tmp_path, "resp_1") == "conv-a"
    manager.close()


async def test_previous_response_id_right_after_the_answer_resumes_its_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream = Responses()
    app = _app(monkeypatch, tmp_path, upstream)
    manager = app.state.proxy.vault_manager
    gate = _gate(manager._maps)
    try:
        async with _client(app) as client:
            first = await client.post(
                "/v1/responses", json={"model": "m", "input": "mail ada@corp.example"}
            )
            assert first.json()["id"] == "resp_1"
            # The mapping has not landed, yet the chain resolves to conv-a:
            # bob is that session's SECOND address, not an orphan's first.
            assert _durable(tmp_path, "resp_1") is None
            second = await client.post(
                "/v1/responses",
                json={
                    "model": "m",
                    "previous_response_id": "resp_1",
                    "input": "and bob@corp.example",
                },
            )
            assert second.status_code == 200
    finally:
        gate.set()
    assert "«EMAIL_001»" in upstream.bodies[0]
    assert "«EMAIL_002»" in upstream.bodies[1] and "bob@" not in upstream.bodies[1]
    assert await asyncio.to_thread(manager.drain_map_writes, 10) == 0
    assert _durable(tmp_path, "resp_2") == "conv-a"
    manager.close()


async def test_a_failed_map_write_still_delivers_and_is_counted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    upstream = Responses()
    app = _app(monkeypatch, tmp_path, upstream)
    state = app.state.proxy

    def broken() -> Any:
        raise sqlite3.OperationalError("disk I/O error")

    state.vault_manager._maps._open_connection = broken
    caplog.set_level(logging.WARNING, logger="llm_redact")
    async with _client(app) as client:
        answer = await client.post("/v1/responses", json={"model": "m", "input": "hi"})
        assert answer.status_code == 200 and answer.json()["id"] == "resp_1"
        await _until(lambda: state.bookkeeping_errors == {"response_id": 1})
        status = (await client.get("/__llm-redact/status")).json()
    assert status["bookkeeping_errors_total"] == {"response_id": 1}
    assert "OperationalError" in caplog.text and "resp_1" not in caplog.text
    assert state.vault_manager.lookup_response_session("resp_1") is None  # unknown
    state.vault_manager.close()


async def test_shutdown_drains_the_queued_writes_before_the_vault_closes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    upstream = Responses()
    app = _app(monkeypatch, tmp_path, upstream)
    manager = app.state.proxy.vault_manager
    gate = _gate(manager._maps)
    closed: list[str] = []
    real_close = manager.close

    def close() -> None:
        closed.append("drained" if manager._maps._in_flight is None else "pending")
        real_close()

    monkeypatch.setattr(manager, "close", close)
    async with app.router.lifespan_context(app):
        async with _client(app) as client:
            answer = await client.post("/v1/responses", json={"model": "m", "input": "hi"})
            assert answer.status_code == 200
        # The write lands a moment into the shutdown.
        threading.Timer(0.1, gate.set).start()
    assert closed == ["drained"]
    assert _durable(tmp_path, "resp_1") == "conv-a"


async def test_shutdown_waits_a_bounded_time_then_counts_what_is_left(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(proxy_mod, "SHUTDOWN_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(writer_mod, "STOP_JOIN_SECONDS", 0.05)
    upstream = Responses()
    app = _app(monkeypatch, tmp_path, upstream)
    state = app.state.proxy
    gate = _gate(state.vault_manager._maps)  # never lands during the shutdown
    caplog.set_level(logging.WARNING, logger="llm_redact")
    try:
        async with app.router.lifespan_context(app), _client(app) as client:
            writer = state.vault_manager._maps
            for text in ("one", "two"):
                answer = await client.post("/v1/responses", json={"model": "m", "input": text})
                assert answer.status_code == 200
                # resp_1 is in flight (held up) before resp_2 is queued.
                await _until(lambda: writer._in_flight is not None)
    finally:
        gate.set()
    # resp_1 was in flight (held up past the join): given up on; resp_2
    # still queued: dropped. Both counted, their number logged.
    assert state.bookkeeping_errors == {"response_id": 2}
    assert "2 write(s) were not written at shutdown" in caplog.text
    assert "resp_1" not in caplog.text and "resp_2" not in caplog.text


@pytest.mark.parametrize("backend_name", ["postgresql", "mysql", "oracle"])
def test_background_writes_on_a_real_server(backend_name: str) -> None:
    # The writer's own connection (``RdbmsStore.open_lane``), its writes and
    # the erases a whole-session delete turns queued writes into, against a
    # real engine (env-gated like the battery: LLM_REDACT_TEST_PG_DSN /
    # _MYSQL_DSN / _ORACLE_DSN).
    from llm_redact.config import RdbmsConfig
    from llm_redact.vault_rdbms import RdbmsStore, RdbmsVaultManager
    from test_vault_rdbms import _REAL_DSNS, _drop_tables

    dsn = _REAL_DSNS[backend_name]
    if not dsn:
        pytest.skip(f"no real {backend_name} server configured")
    config = VaultConfig(backend=backend_name, rdbms=RdbmsConfig(dsn=dsn))
    _drop_tables(config)
    manager = RdbmsVaultManager(RdbmsStore(config, None))
    other = RdbmsVaultManager(RdbmsStore(config, None))
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    manager.get("s").placeholder_for("EMAIL", "ada@corp.example")
    writer = _background(manager)
    manager.record_response_session("resp_1", "keep")
    manager.record_handle_session(H + "a", "keep")
    assert manager.drain_map_writes(30) == 0
    with _paused(writer):
        manager.record_response_session("resp_1", "s")  # re-homed, then deleted
        manager.record_object_session("file-1", "s")
        manager.record_handle_session(H + "a", "s", replaces=[H + "none"])
        manager.record_handle_session(H + "b", "keep")
        assert manager.forget_sessions(["s"]) == 1
    assert manager.drain_map_writes(30) == 0
    assert other.lookup_response_sessions(["resp_1", "file-1"]) == {}
    assert other.lookup_handle_session(H + "a") is None
    assert other.lookup_handle_session(H + "b") == "keep"
    assert counter == {}
    manager.close()
    other.close()


def test_the_writers_dropped_connection_reconnects_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from llm_redact.vault_rdbms import RdbmsStore, RdbmsVaultManager
    from test_vault_rdbms import _fake_backend_config

    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    manager = RdbmsVaultManager(RdbmsStore(config, None))
    _background(manager)
    manager.record_response_session("resp_1", "s")
    assert manager.drain_map_writes(10) == 0
    assert driver.connect_count == 2  # the store's, and the writer's own
    driver.dead = True  # the writer's idle connection was dropped
    manager.record_handle_session(H + "a", "s")
    assert manager.drain_map_writes(10) == 0
    assert driver.connect_count == 3  # one transparent reconnect, on the writer
    other = RdbmsVaultManager(RdbmsStore(config, None))
    assert other.lookup_handle_session(H + "a") == "s"
    other.close()
    manager.close()


# --- what the docs promise replicas -----------------------------------------------


def test_the_docs_say_another_replica_reads_a_record_only_once_written() -> None:
    """Read-your-writes holds within one process: replicas sharing a vault
    see a record once its background write landed (a follow-up reaching
    another replica sooner is refused or sealed). Every place that tells
    an operator about replicas must say so."""
    root = Path(__file__).resolve().parent.parent
    resilience = (root / "docs" / "resilience.md").read_text()
    assert "Several replicas share one vault" in resilience
    assert "Read-your-writes holds within ONE process only" in resilience
    deployment = (root / "docs" / "deployment.md").read_text()
    assert "reaches ANOTHER replica" in deployment and "session affinity" in deployment
    notes = (root / "deploy" / "helm" / "llm-redact" / "templates" / "NOTES.txt").read_text()
    assert "reaching another pod before its write" in notes
    how = " ".join((root / "docs" / "how-it-works.md").read_text().split())
    assert "on every replica sharing the vault once its write landed" in how
