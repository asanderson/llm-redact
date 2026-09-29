"""One transaction per request for first-sight values (``SqliteVault.batched``).

The sqlite vault runs synchronous=FULL, so every COMMIT is an fsync; a
request carrying many new values used to pay one per value, on the event
loop. ``run_batched`` wraps one request's redaction: BEGIN IMMEDIATE at its
first new value, the request's own lookups see the rows it staged, ONE COMMIT
at the end, the write-through caches updated only after it — and any failure
(a write fault, a failed COMMIT, a blocked value, a swallowed error) rolls the
whole batch back with the caches untouched, so a retry reissues the same
dense numbers. Plain and encrypted (FakeVaultCipher) vaults behave alike.

Faults are injected by wrapping the connection (tests/test_vault_faults.py's
_FlakyConn); cross-process contention is modelled with a second connection
on another thread (sqlite connections are per-thread).
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from fake_cipher import FakeVaultCipher
from llm_redact.vault import (
    CACHE_CHECK_SECONDS,
    InMemoryVault,
    PlaceholderSpaceExhausted,
    SqliteVault,
    SqliteVaultManager,
    open_sqlite_vault,
    run_batched,
)
from test_vault_faults import _FlakyConn


@pytest.fixture(params=["plain", "encrypted"])
def cipher(request: pytest.FixtureRequest) -> FakeVaultCipher | None:
    return FakeVaultCipher() if request.param == "encrypted" else None


@pytest.fixture
def vault(tmp_path: Path, cipher: FakeVaultCipher | None) -> Iterator[SqliteVault]:
    opened = open_sqlite_vault(tmp_path / "vault.db", "s", cipher)
    yield opened
    opened.close()


def _trace(conn: sqlite3.Connection) -> list[str]:
    """Every transaction-control statement the connection runs from now on."""
    seen: list[str] = []
    conn.set_trace_callback(
        lambda sql: seen.append(sql) if sql in ("BEGIN IMMEDIATE", "COMMIT", "ROLLBACK") else None
    )
    return seen


def _rows(vault: SqliteVault) -> int:
    return int(vault._conn.execute("SELECT COUNT(*) FROM mappings").fetchone()[0])


def _emails(count: int, start: int = 0) -> list[str]:
    return [f"user{i}@corp.example" for i in range(start, start + count)]


def _issue_all(vault: SqliteVault, values: list[str]) -> list[str]:
    return [vault.placeholder_for("EMAIL", value) for value in values]


# --- the one transaction ------------------------------------------------------


def test_a_batch_commits_once_for_many_new_values(vault: SqliteVault) -> None:
    seen = _trace(vault._conn)
    tokens = run_batched(vault, lambda: _issue_all(vault, _emails(50)))
    assert tokens == [f"«EMAIL_{n:03d}»" for n in range(1, 51)]  # dense, in order
    assert seen == ["BEGIN IMMEDIATE", "COMMIT"]
    assert _rows(vault) == 50
    # Committed: cached and durable (a fresh connection sees every row).
    assert vault.original_for("«EMAIL_050»") == "user49@corp.example"
    assert len(vault) == 50


def test_outside_a_batch_each_new_value_commits_on_its_own(vault: SqliteVault) -> None:
    seen = _trace(vault._conn)
    _issue_all(vault, _emails(3))
    assert seen == ["BEGIN IMMEDIATE", "COMMIT"] * 3


def test_outside_a_batch_a_failed_write_rolls_back_its_own_transaction(
    vault: SqliteVault,
) -> None:
    real = vault._conn
    vault._conn = _FlakyConn(real, "INSERT INTO mappings")  # type: ignore[assignment]
    seen = _trace(real)
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        vault.placeholder_for("EMAIL", "ada@corp.example")
    assert seen == ["BEGIN IMMEDIATE", "ROLLBACK"]
    vault._conn = real  # type: ignore[assignment]


def test_a_batch_with_nothing_new_writes_nothing(vault: SqliteVault) -> None:
    known = _issue_all(vault, _emails(3))
    seen = _trace(vault._conn)
    assert run_batched(vault, lambda: _issue_all(vault, _emails(3))) == known
    assert seen == []  # no lock taken, no fsync


def test_the_requests_own_lookups_see_its_staged_rows(vault: SqliteVault) -> None:
    def work() -> tuple[str, str, str | None]:
        first = vault.placeholder_for("EMAIL", "ada@corp.example")
        # Not in the caches before the COMMIT...
        assert "«EMAIL_001»" not in vault._reverse
        assert "EMAIL::ada@corp.example" not in vault._forward
        # ...yet the same request sees it: the same value keeps its token,
        # the next value continues after it, and a lookup restores it.
        again = vault.placeholder_for("EMAIL", "ada@corp.example")
        assert vault.placeholder_for("EMAIL", "bob@corp.example") == "«EMAIL_002»"
        return first, again, vault.original_for(first)

    assert run_batched(vault, work) == ("«EMAIL_001»", "«EMAIL_001»", "ada@corp.example")
    assert vault._reverse == {"«EMAIL_001»": "ada@corp.example", "«EMAIL_002»": "bob@corp.example"}
    assert vault._staged_forward == {} and vault._staged_reverse == {}


def test_a_value_repeated_in_a_request_is_served_from_its_staged_row(vault: SqliteVault) -> None:
    statements: list[str] = []

    def work() -> list[str]:
        first = vault.placeholder_for("EMAIL", "ada@corp.example")
        vault._conn.set_trace_callback(statements.append)
        return [first, vault.placeholder_for("EMAIL", "ada@corp.example")]

    assert run_batched(vault, work) == ["«EMAIL_001»", "«EMAIL_001»"]
    assert statements == ["COMMIT"]  # no round trip for the repeat


def test_a_known_value_in_a_batch_keeps_its_token(vault: SqliteVault) -> None:
    ada = vault.placeholder_for("EMAIL", "ada@corp.example")
    tokens = run_batched(
        vault,
        lambda: [
            vault.placeholder_for("EMAIL", "bob@corp.example"),
            vault.placeholder_for("EMAIL", "ada@corp.example"),
        ],
    )
    assert tokens == ["«EMAIL_002»", ada]


def test_nested_batches_share_one_transaction(vault: SqliteVault) -> None:
    seen = _trace(vault._conn)

    def inner() -> str:
        return vault.placeholder_for("EMAIL", "bob@corp.example")

    def outer() -> list[str]:
        first = vault.placeholder_for("EMAIL", "ada@corp.example")
        return [first, run_batched(vault, inner), vault.placeholder_for("EMAIL", "cy@corp.example")]

    assert run_batched(vault, outer) == ["«EMAIL_001»", "«EMAIL_002»", "«EMAIL_003»"]
    assert seen == ["BEGIN IMMEDIATE", "COMMIT"]


def test_several_sessions_share_one_batch(tmp_path: Path) -> None:
    # Views over one connection join the batch whichever of them opened it.
    manager = SqliteVaultManager(tmp_path / "vault.db")
    alpha, beta = manager.get("alpha"), manager.get("beta")
    seen = _trace(manager._conn)

    def work() -> list[str]:
        return [
            alpha.placeholder_for("EMAIL", "ada@corp.example"),
            beta.placeholder_for("EMAIL", "bob@corp.example"),
            beta.placeholder_for("EMAIL", "cy@corp.example"),
        ]

    assert run_batched(alpha, work) == ["«EMAIL_001»", "«EMAIL_001»", "«EMAIL_002»"]
    assert seen == ["BEGIN IMMEDIATE", "COMMIT"]
    assert alpha.original_for("«EMAIL_001»") == "ada@corp.example"
    assert beta.original_for("«EMAIL_001»") == "bob@corp.example"
    assert manager.session_count() == 2
    manager.close()


def test_floors_apply_inside_a_batch(vault: SqliteVault) -> None:
    tokens = run_batched(
        vault,
        lambda: [
            vault.placeholder_for("EMAIL", "ada@corp.example", floor=4),
            vault.placeholder_for("EMAIL", "bob@corp.example"),
        ],
    )
    assert tokens == ["«EMAIL_005»", "«EMAIL_006»"]


def test_run_batched_on_a_vault_without_batches_just_runs_the_work() -> None:
    memory = InMemoryVault()
    assert run_batched(memory, lambda: memory.placeholder_for("EMAIL", "a@corp.example")) == (
        "«EMAIL_001»"
    )
    assert memory.original_for("«EMAIL_001»") == "a@corp.example"


# --- failures roll the whole batch back -----------------------------------------


@pytest.mark.parametrize("failing", [1, 2, 5])
def test_a_fault_on_the_nth_insert_rolls_back_the_whole_batch(
    vault: SqliteVault, failing: int
) -> None:
    real = vault._conn
    vault._conn = _FlakyConn(real, "INSERT INTO mappings", times=0)  # type: ignore[assignment]
    inserts = 0

    def work() -> list[str]:
        nonlocal inserts
        tokens = []
        for value in _emails(5):
            inserts += 1
            if inserts == failing:
                vault._conn._times = 1  # type: ignore[attr-defined]  # this INSERT fails
            tokens.append(vault.placeholder_for("EMAIL", value))
        return tokens

    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        run_batched(vault, work)
    vault._conn = real  # type: ignore[assignment]
    # Every insert of the batch is gone — the ones before the fault too —
    # nothing was cached, and the connection is not wedged.
    assert _rows(vault) == 0
    assert vault._forward == {} and vault._reverse == {}
    assert vault._staged_forward == {} and vault._staged_reverse == {}
    assert not vault._conn.in_transaction
    assert vault.original_for("«EMAIL_001»") is None
    # The retry reissues the SAME dense numbers — never a skip, never a reuse.
    assert run_batched(vault, lambda: _issue_all(vault, _emails(5))) == [
        f"«EMAIL_{n:03d}»" for n in range(1, 6)
    ]


def test_a_fault_on_commit_rolls_back_the_whole_batch(vault: SqliteVault) -> None:
    real = vault._conn
    vault._conn = _FlakyConn(real, "COMMIT")  # type: ignore[assignment]
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        run_batched(vault, lambda: _issue_all(vault, _emails(3)))
    vault._conn = real  # type: ignore[assignment]
    assert _rows(vault) == 0
    assert vault._reverse == {} and vault._staged_reverse == {}
    assert not real.in_transaction
    assert run_batched(vault, lambda: _issue_all(vault, _emails(3))) == [
        "«EMAIL_001»",
        "«EMAIL_002»",
        "«EMAIL_003»",
    ]


def test_a_failed_rollback_still_propagates_the_original_error(vault: SqliteVault) -> None:
    real = vault._conn
    vault._conn = _FlakyConn(  # type: ignore[assignment]
        _FlakyConn(real, "COMMIT"), "ROLLBACK"
    )
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        run_batched(vault, lambda: _issue_all(vault, _emails(2)))
    vault._conn = real  # type: ignore[assignment]
    assert vault._reverse == {}
    real.execute("ROLLBACK")  # the suppressed rollback left it open: clean up


def test_a_swallowed_write_fault_still_rolls_the_batch_back(vault: SqliteVault) -> None:
    real = vault._conn
    vault._conn = _FlakyConn(real, "INSERT INTO mappings", times=0)  # type: ignore[assignment]

    def work() -> str:
        first = vault.placeholder_for("EMAIL", "ada@corp.example")
        vault._conn._times = 1  # type: ignore[attr-defined]
        # A careless caller: the batch must still not commit around it.
        with contextlib.suppress(sqlite3.OperationalError):
            vault.placeholder_for("EMAIL", "bob@corp.example")
        return first

    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        run_batched(vault, work)
    vault._conn = real  # type: ignore[assignment]
    assert _rows(vault) == 0
    assert vault._reverse == {}
    assert not real.in_transaction


class _Refused(Exception):
    """Stands in for BlockedRequest / TooManyStrings: raised by the work."""


def test_any_exception_in_the_work_rolls_the_batch_back(vault: SqliteVault) -> None:
    def work() -> None:
        _issue_all(vault, _emails(3))
        raise _Refused

    with pytest.raises(_Refused):
        run_batched(vault, work)
    assert _rows(vault) == 0
    assert vault._reverse == {} and vault._staged_reverse == {}
    assert not vault._conn.in_transaction
    assert vault.placeholder_for("EMAIL", "user2@corp.example") == "«EMAIL_001»"


def test_an_exhausted_number_space_rolls_the_batch_back(vault: SqliteVault) -> None:
    from llm_redact.placeholders import MAX_TOKEN_NUMBER

    def work() -> None:
        vault.placeholder_for("EMAIL", "ada@corp.example")
        vault.placeholder_for("EMAIL", "bob@corp.example", floor=MAX_TOKEN_NUMBER)

    with pytest.raises(PlaceholderSpaceExhausted):
        run_batched(vault, work)
    assert _rows(vault) == 0
    assert not vault._conn.in_transaction


def test_a_cipher_fault_rolls_the_batch_back(tmp_path: Path) -> None:
    class FailingSecond(FakeVaultCipher):
        calls = 0

        def encrypt(self, original: str) -> bytes:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("cipher unavailable")
            return super().encrypt(original)

    vault = open_sqlite_vault(tmp_path / "vault.db", "s", FailingSecond())
    with pytest.raises(RuntimeError, match="cipher"):
        run_batched(vault, lambda: _issue_all(vault, _emails(3)))
    assert _rows(vault) == 0
    assert not vault._conn.in_transaction
    assert run_batched(vault, lambda: _issue_all(vault, _emails(3))) == [
        "«EMAIL_001»",
        "«EMAIL_002»",
        "«EMAIL_003»",
    ]
    vault.close()


def test_a_later_batch_after_a_rolled_back_one_is_clean(vault: SqliteVault) -> None:
    with pytest.raises(_Refused):
        run_batched(vault, lambda: (_issue_all(vault, _emails(2)), _raise()))
    seen = _trace(vault._conn)
    assert run_batched(vault, lambda: _issue_all(vault, _emails(1, start=7))) == ["«EMAIL_001»"]
    assert seen == ["BEGIN IMMEDIATE", "COMMIT"]


def _raise() -> None:
    raise _Refused


# --- staleness checks wait for the batch ------------------------------------------


def test_the_check_runs_before_a_batch(tmp_path: Path) -> None:
    # The session was deleted by another instance before the request: the
    # check at the batch's start (due exactly now) drops the deleted value
    # from memory first — inside the batch no reload may happen.
    class Clock:
        now = 1000.0

        def __call__(self) -> float:
            return self.now

    clock = Clock()
    here = SqliteVaultManager(tmp_path / "vault.db", clock=clock)
    there = SqliteVaultManager(tmp_path / "vault.db")
    view = here.get("s")
    assert view.placeholder_for("EMAIL", "ada@corp.example") == "«EMAIL_001»"
    assert there.forget_sessions(["s"]) == 1
    clock.now += CACHE_CHECK_SECONDS
    # Numbered above the retired «EMAIL_001», never onto it.
    assert run_batched(view, lambda: view.placeholder_for("EMAIL", "ada@corp.example")) == (
        "«EMAIL_002»"
    )
    assert view.original_for("«EMAIL_001»") is None
    here.close()
    there.close()


def test_no_reload_happens_inside_a_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Another instance deletes the session while this request's batch is
    # open (before its first write, when no lock is held): inside the batch
    # the view keeps serving its cached token — never a wrong one, its number
    # is retired — because a reload there could read the batch's own
    # uncommitted rows as committed; the check after the batch catches up.
    monkeypatch.setattr("llm_redact.vault.CACHE_CHECK_SECONDS", 0.0)
    here = SqliteVaultManager(tmp_path / "vault.db")
    there = SqliteVaultManager(tmp_path / "vault.db")
    view = here.get("s")
    ada = view.placeholder_for("EMAIL", "ada@corp.example")

    def work() -> list[str]:
        assert there.forget_sessions(["s"]) == 1
        cached = view.placeholder_for("EMAIL", "ada@corp.example")
        fresh = view.placeholder_for("EMAIL", "bob@corp.example")
        return [cached, fresh, view.placeholder_for("EMAIL", "ada@corp.example")]

    assert run_batched(view, work) == [
        ada,
        "«EMAIL_002»",  # above the retired «EMAIL_001», never onto it
        ada,
    ]
    # After the batch the check runs: the deleted value is gone from memory,
    # the committed one stays.
    assert view.original_for(ada) is None
    assert view.original_for("«EMAIL_002»") == "bob@corp.example"
    here.close()
    there.close()


# --- another process sharing the file --------------------------------------------


def _hold_batch_open(path: Path, started: threading.Event, release: threading.Event) -> None:
    """In its own thread (its own connection): open a batch, issue a value,
    and keep the write lock until told to commit."""
    vault = open_sqlite_vault(path, "s")

    def work() -> None:
        vault.placeholder_for("EMAIL", "holder@corp.example")
        started.set()
        release.wait(10)

    run_batched(vault, work)
    vault.close()


def test_a_second_process_waits_for_the_batch_then_numbers_densely(tmp_path: Path) -> None:
    path = tmp_path / "vault.db"
    open_sqlite_vault(path, "s").close()
    started, release = threading.Event(), threading.Event()
    holder = threading.Thread(target=_hold_batch_open, args=(path, started, release))
    holder.start()
    assert started.wait(10)
    other = open_sqlite_vault(path, "s")
    threading.Timer(0.2, release.set).start()
    began = time.monotonic()
    # busy_timeout (5 s): the write waits for the batch's COMMIT...
    assert other.placeholder_for("EMAIL", "other@corp.example") == "«EMAIL_002»"
    assert time.monotonic() - began >= 0.15
    holder.join(10)
    # ...and continues above the batch's number, which it can read.
    assert other.original_for("«EMAIL_001»") == "holder@corp.example"
    other.close()


def test_a_second_process_fails_closed_past_the_busy_timeout(tmp_path: Path) -> None:
    path = tmp_path / "vault.db"
    open_sqlite_vault(path, "s").close()
    started, release = threading.Event(), threading.Event()
    holder = threading.Thread(target=_hold_batch_open, args=(path, started, release))
    holder.start()
    assert started.wait(10)
    other = open_sqlite_vault(path, "s")
    other._conn.execute("PRAGMA busy_timeout=50")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.placeholder_for("EMAIL", "other@corp.example")
        # Nothing issued or cached, nothing wedged.
        assert other._reverse == {} and not other._conn.in_transaction
    finally:
        release.set()
        holder.join(10)
    other._conn.execute("PRAGMA busy_timeout=5000")
    assert other.placeholder_for("EMAIL", "other@corp.example") == "«EMAIL_002»"
    other.close()


# --- the batch sees what other connections committed ----------------------------


def test_a_value_another_connection_mapped_is_found_inside_the_batch(tmp_path: Path) -> None:
    path = tmp_path / "vault.db"
    mine = open_sqlite_vault(path, "s")
    theirs = open_sqlite_vault(path, "s")
    ada = theirs.placeholder_for("EMAIL", "ada@corp.example")  # after mine loaded

    def work() -> list[str | None]:
        return [
            mine.placeholder_for("EMAIL", "ada@corp.example"),
            mine.placeholder_for("EMAIL", "bob@corp.example"),
            mine.original_for(ada),
        ]

    assert run_batched(mine, work) == [ada, "«EMAIL_002»", "ada@corp.example"]
    assert mine._forward["EMAIL::ada@corp.example"] == ada  # cached after the COMMIT
    mine.close()
    theirs.close()


def test_a_batch_of_a_manager_view_evicted_while_held(tmp_path: Path) -> None:
    # The view a request holds stays THE view of its session even after the
    # LRU dropped it: the next get() hands it out again, never a second one.
    manager = SqliteVaultManager(tmp_path / "vault.db", view_cache_size=1)
    held = manager.get("a")
    manager.get("b")  # evicts "a" from the LRU; the request still holds it
    assert manager.get("a") is held
    manager.close()


def test_a_token_another_connection_issued_is_cached_both_ways(
    tmp_path: Path, cipher: FakeVaultCipher | None
) -> None:
    mine = open_sqlite_vault(tmp_path / "vault.db", "s", cipher)
    theirs = open_sqlite_vault(tmp_path / "vault.db", "s", cipher)
    token = theirs.placeholder_for("EMAIL", "ada@corp.example")
    assert mine.original_for(token) == "ada@corp.example"
    assert mine._forward == {"EMAIL::ada@corp.example": token}
    seen = _trace(mine._conn)
    assert mine.placeholder_for("EMAIL", "ada@corp.example") == token  # a cache hit
    assert seen == []
    mine.close()
    theirs.close()
