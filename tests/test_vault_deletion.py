"""Whole-session deletes never make a token mean two values (C-R2-03, P-R2-07).

A delete — the TTL prune loop, ``POST /__llm-redact/sessions/prune``, the CLI
prune, an access gate's purge through ``SessionStore.forget`` — used to let
the session's numbering restart at «EMAIL_001» while something still held the
old meaning: another proxy instance's cached view of the session (one sqlite
file or RDBMS shared by several instances is a supported deployment), or a
live view held past the delete (a realtime connection of a purged user).
Both then restored an old token as the NEW value (the wrong-value sin).

The fix, pinned here for every vault:

- a delete RETIRES every number the session held (sqlite
  ``retired_numbers``, RDBMS ``llm_redact_retired``, in the delete's own
  transaction; the in-memory vault keeps its counters) and new values are
  numbered above it — so no (session, type, number) ever carries two values,
  however stale a cache is;
- the retired number doubles as the session's epoch: a view re-reads it
  (``CACHE_CHECK_SECONDS``) and drops a deleted session's values, and the
  deleting manager rebuilds every live view of the session at once (one view
  per session, even when the LRU evicted it);
- the idle check of a prune and its delete are one transaction (sqlite: the
  write lock; RDBMS: the delete sized in the statement that re-checks it).
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from fake_cipher import FakeVaultCipher
from llm_redact.config import Config, ConfigError, ProviderConfig, RdbmsConfig, VaultConfig
from llm_redact.proxy import CSRF_HEADER, create_app
from llm_redact.vault import (
    CACHE_CHECK_SECONDS,
    LOOKUP_CHUNK,
    InMemoryVaultManager,
    SqliteVaultManager,
)
from llm_redact.vault_rdbms import RdbmsStore, RdbmsVaultManager
from test_vault_faults import _FlakyConn
from test_vault_rdbms import _fake_backend_config

PERSISTENT = [
    "sqlite",
    "sqlite-encrypted",
    "rdbms-dbapi",
    "rdbms-dbapi-encrypted",
    "rdbms-postgresql",
    "rdbms-mysql",
    "rdbms-oracle",
]
ALL = ["memory", "memory-encrypted", *PERSISTENT]


class Clock:
    """An injectable monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


Factory = Callable[..., Any]


def _factory(backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Factory:
    """A function opening one more manager (one more proxy instance) over
    the SAME vault, for ``backend``."""
    cipher = FakeVaultCipher() if backend.endswith("-encrypted") else None
    if backend.startswith("memory"):
        return lambda clock=None: InMemoryVaultManager(cipher=cipher)
    if backend.startswith("sqlite"):
        return lambda clock=None: SqliteVaultManager(
            tmp_path / "vault.db", cipher=cipher, **({"clock": clock} if clock else {})
        )
    if backend.startswith("rdbms-dbapi"):
        config = VaultConfig(
            backend="dbapi",
            encryption="fernet" if cipher is not None else "none",
            rdbms=RdbmsConfig(dsn=str(tmp_path / "v.db"), module="sqlite3"),
        )
    else:
        config, _ = _fake_backend_config(monkeypatch, tmp_path, backend.removeprefix("rdbms-"))
    return lambda clock=None: RdbmsVaultManager(
        RdbmsStore(config, cipher), **({"clock": clock} if clock else {})
    )


def _opening(make: Factory) -> Iterator[Factory]:
    """``make``, closing every manager it opened at the test's end."""
    opened: list[Any] = []

    def factory(clock: Clock | None = None) -> Any:
        manager = make(clock)
        opened.append(manager)
        return manager

    yield factory
    for manager in opened:
        manager.close()


@pytest.fixture(params=ALL)
def open_manager(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Factory]:
    yield from _opening(_factory(request.param, tmp_path, monkeypatch))


@pytest.fixture(params=PERSISTENT)
def open_instance(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Factory]:
    yield from _opening(_factory(request.param, tmp_path, monkeypatch))


def _number(token: str) -> int:
    return int(token.rstrip("»").rsplit("_", 1)[1])


def _age_all(manager: Any) -> None:
    """Backdate every mapping so a prune sees the sessions idle."""
    if isinstance(manager, SqliteVaultManager):
        manager._conn.execute("UPDATE mappings SET created_at = '2000-01-01T00:00:00Z'")
        return

    def op(conn: Any) -> None:
        conn.cursor().execute("UPDATE llm_redact_mappings SET created_at = '2000-01-01T00:00:00Z'")
        conn.commit()

    manager._store._run(op)


def _delete(manager: Any, how: str) -> int:
    if how == "forget":
        return int(manager.forget_sessions(["s"]))
    _age_all(manager)
    return int(manager.prune_sessions(30))


def _meanings(*views: Any, tokens: list[str]) -> dict[str, set[str]]:
    """Every value any of the views restores each token to."""
    found: dict[str, set[str]] = {}
    for token in tokens:
        for view in views:
            value = view.original_for(token)
            if value is not None:
                found.setdefault(token, set()).add(value)
    return found


# --- one instance: a delete never lets the numbering restart --------------------


def test_a_forgotten_sessions_numbers_are_never_issued_again(open_manager: Factory) -> None:
    manager = open_manager()
    view = manager.get("s")
    old = [
        view.placeholder_for("EMAIL", "ada@corp.example"),
        view.placeholder_for("EMAIL", "bob@corp.example"),
    ]
    old_phone = view.placeholder_for("PHONE", "+1 555 0100")
    assert manager.forget_sessions(["s"]) == 1
    fresh = manager.get("s")
    assert fresh.original_for(old[0]) is None  # the value is gone...
    carol = fresh.placeholder_for("EMAIL", "carol@corp.example")
    ada_again = fresh.placeholder_for("EMAIL", "ada@corp.example")
    phone = fresh.placeholder_for("PHONE", "+1 555 0199")
    # ...and its number is never handed to a new value.
    assert _number(carol) > 2 and _number(ada_again) > 2
    assert carol not in old and ada_again not in old
    assert phone != old_phone
    assert fresh.original_for(carol) == "carol@corp.example"
    assert fresh.original_for(old[1]) is None


def test_a_live_view_is_emptied_by_the_delete(open_manager: Factory) -> None:
    # P-R2-07: a view held past the delete (a realtime connection's) loses
    # the deleted values at once, and never reissues their numbers.
    manager = open_manager()
    held = manager.get("s")
    alice = held.placeholder_for("EMAIL", "alice.a@corp.example")
    assert manager.forget_sessions(["s"]) == 1
    assert len(held) == 0
    assert held.original_for(alice) is None
    bob = held.placeholder_for("EMAIL", "bob.b@corp.example")
    assert bob != alice
    # Still THE view of its session: the manager hands it out again.
    assert manager.get("s") is held


def test_a_purged_users_open_connection_never_restores_the_wrong_value(
    open_manager: Factory,
) -> None:
    # The P-R2-07 trigger: the open connection sends alice, the user is
    # purged, then bob, then alice again; the echo of alice's token must
    # never come back as bob.
    manager = open_manager()
    connection = manager.get("user:ns:default")
    first_alice = connection.placeholder_for("EMAIL", "alice.a@corp.example")
    assert manager.forget_sessions(["user:ns:default"]) == 1
    bob = connection.placeholder_for("EMAIL", "bob.b@corp.example")
    alice = connection.placeholder_for("EMAIL", "alice.a@corp.example")
    assert len({first_alice, bob, alice}) == 3
    assert connection.original_for(alice) == "alice.a@corp.example"
    assert connection.original_for(bob) == "bob.b@corp.example"
    assert connection.original_for(first_alice) is None  # passes through, never bob


def test_a_pruned_sessions_numbers_are_never_issued_again(open_instance: Factory) -> None:
    manager = open_instance()
    held = manager.get("s")
    ada = held.placeholder_for("EMAIL", "ada@corp.example")
    manager.get("keep").placeholder_for("EMAIL", "kim@corp.example")
    _age_all(manager)
    assert manager.prune_sessions(30, exclude=frozenset({"keep"})) == 1
    assert held.original_for(ada) is None and len(held) == 0
    bob = held.placeholder_for("EMAIL", "bob@corp.example")
    assert _number(bob) > _number(ada)
    assert manager.get("keep").original_for("«EMAIL_001»") == "kim@corp.example"


def test_forgetting_nothing_changes_nothing(open_manager: Factory) -> None:
    manager = open_manager()
    view = manager.get("s")
    ada = view.placeholder_for("EMAIL", "ada@corp.example")
    assert manager.forget_sessions(["missing"]) == 0
    assert manager.forget_sessions([]) == 0
    assert view.original_for(ada) == "ada@corp.example"
    assert view.placeholder_for("EMAIL", "bob@corp.example") == "«EMAIL_002»"


def test_forgetting_several_sessions_counts_each(open_manager: Factory) -> None:
    manager = open_manager()
    for session in ("a", "b", "c"):
        manager.get(session).placeholder_for("EMAIL", f"{session}@corp.example")
    assert manager.forget_sessions(["a", "b", "missing"]) == 2
    assert manager.get("c").original_for("«EMAIL_001»") == "c@corp.example"


def test_deleting_twice_keeps_every_number_retired(open_manager: Factory) -> None:
    manager = open_manager()
    view = manager.get("s")
    issued = [view.placeholder_for("EMAIL", f"v{i}@corp.example") for i in range(3)]
    manager.forget_sessions(["s"])
    issued.append(view.placeholder_for("EMAIL", "w@corp.example"))
    manager.forget_sessions(["s"])
    last = view.placeholder_for("EMAIL", "x@corp.example")
    assert last not in issued
    assert _number(last) > max(_number(token) for token in issued)


# --- two instances over one vault (C-R2-03) ---------------------------------------


@pytest.mark.parametrize("how", ["forget", "prune"])
def test_a_delete_by_another_instance_never_makes_a_token_mean_two_values(
    open_instance: Factory, how: str
) -> None:
    # B's clock is frozen: the assertions below are about the window before
    # B's next staleness check (CACHE_CHECK_SECONDS), which a slow runner
    # could otherwise cross between two lines and rebuild B's caches.
    a, b = open_instance(), open_instance(Clock())
    b_view = b.get("s")
    jane = b_view.placeholder_for("EMAIL", "jane.doe@corp.example")  # B's cache is warm
    assert _delete(a, how) == 1
    # B keeps serving: its cached token (whose number is retired, so it can
    # only ever mean jane) and, for a new value, a number above it.
    assert b_view.placeholder_for("EMAIL", "jane.doe@corp.example") == jane
    bob = b_view.placeholder_for("EMAIL", "bob.builder@corp.example")
    assert bob != jane
    # A, reading the database, never hands jane's old number to anyone.
    a_view = a.get("s")
    a_jane = a_view.placeholder_for("EMAIL", "jane.doe@corp.example")
    carol = a_view.placeholder_for("EMAIL", "carol@corp.example")
    tokens = [jane, bob, a_jane, carol]
    meanings = _meanings(a_view, b_view, tokens=tokens)
    assert all(len(values) == 1 for values in meanings.values()), meanings
    assert meanings[bob] == {"bob.builder@corp.example"}
    assert b_view.original_for(jane) == "jane.doe@corp.example"


def test_a_session_recreated_by_another_instance_never_restores_the_wrong_value(
    open_instance: Factory,
) -> None:
    # The reverse direction: A forgets S and re-creates it; B's stale cache
    # must never restore A's new token as its old value.
    a, b = open_instance(), open_instance()
    b_view = b.get("s")
    ada = b_view.placeholder_for("EMAIL", "ada@corp.example")
    assert a.forget_sessions(["s"]) == 1
    bob = a.get("s").placeholder_for("EMAIL", "bob@corp.example")
    assert bob != ada
    assert b_view.original_for(bob) == "bob@corp.example"
    assert b_view.original_for(ada) in (None, "ada@corp.example")


def test_a_stale_view_drops_the_deleted_values_once_the_check_is_due(
    open_instance: Factory,
) -> None:
    clock = Clock()
    a, b = open_instance(), open_instance(clock)
    b_view = b.get("s")
    ada = b_view.placeholder_for("EMAIL", "ada@corp.example")
    loaded = b_view._reverse
    clock.now += CACHE_CHECK_SECONDS  # a check with nothing deleted...
    assert b_view.original_for(ada) == "ada@corp.example"
    assert b_view._reverse is loaded  # ...keeps the caches as they are
    assert a.forget_sessions(["s"]) == 1
    clock.now += CACHE_CHECK_SECONDS / 2  # not due: still served from memory
    assert b_view.original_for(ada) == "ada@corp.example"
    clock.now += CACHE_CHECK_SECONDS  # due: the deleted value is dropped
    assert b_view.original_for(ada) is None
    assert len(b_view) == 0
    reloaded = b_view._reverse
    clock.now += CACHE_CHECK_SECONDS  # due again, nothing deleted since:
    assert b_view.original_for(ada) is None
    assert b_view._reverse is reloaded  # the caches stay (no reload per check)
    # The forward cache went too: ada gets a fresh number above the old one.
    assert _number(b_view.placeholder_for("EMAIL", "ada@corp.example")) > _number(ada)


def test_a_redaction_checks_the_session_first(open_instance: Factory) -> None:
    # The staleness check also runs on the redaction side (placeholder_for).
    clock = Clock()
    a, b = open_instance(), open_instance(clock)
    b_view = b.get("s")
    ada = b_view.placeholder_for("EMAIL", "ada@corp.example")
    assert a.forget_sessions(["s"]) == 1
    clock.now += CACHE_CHECK_SECONDS  # exactly due
    again = b_view.placeholder_for("EMAIL", "ada@corp.example")
    assert again != ada and _number(again) > _number(ada)


def test_a_new_view_is_first_checked_an_interval_after_its_load(open_instance: Factory) -> None:
    # A view's caches are as fresh as its load: the first check falls due a
    # full interval later, not at the view's first lookup (every new
    # session's first lookup would read the database again).
    clock = Clock()
    a, b = open_instance(), open_instance(clock)
    ada = a.get("s").placeholder_for("EMAIL", "ada@corp.example")
    b_view = b.get("s")  # loaded now, ada included
    assert a.forget_sessions(["s"]) == 1
    clock.now += CACHE_CHECK_SECONDS / 2  # not due: served from the load
    assert b_view.original_for(ada) == "ada@corp.example"
    clock.now += CACHE_CHECK_SECONDS / 2  # due: the delete is noticed
    assert b_view.original_for(ada) is None


# --- the prune's idle check is atomic with its delete -------------------------------


def test_sqlite_prune_checks_idleness_inside_its_write_lock(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    manager.get("s").placeholder_for("EMAIL", "ada@corp.example")
    _age_all(manager)
    statements: list[str] = []
    manager._conn.set_trace_callback(statements.append)
    assert manager.prune_sessions(30) == 1
    begin = statements.index("BEGIN IMMEDIATE")
    idle_check = next(i for i, sql in enumerate(statements) if "HAVING MAX(created_at)" in sql)
    delete = next(i for i, sql in enumerate(statements) if sql.startswith("DELETE FROM mappings"))
    commit = statements.index("COMMIT")
    assert begin < idle_check < delete < commit
    manager.close()


class _Fetched:
    """The rows of a statement already consumed."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return self._rows


class _AfterStatement:
    """Wraps a store's _execute: runs ``action`` once, right after the
    first statement containing ``marker`` (another replica acting there).
    The statement is consumed first — a server's MVCC lets another replica
    commit meanwhile; sqlite's file lock would not while it was pending."""

    def __init__(self, store: RdbmsStore, marker: str, action: Callable[[], None]) -> None:
        self._execute = store._execute
        self._marker = marker
        self._action: Callable[[], None] | None = action

    def __call__(self, conn: Any, sql: str, params: Any = None) -> Any:
        cursor = self._execute(conn, sql, params)
        if self._action is None or self._marker not in sql:
            return cursor
        rows = list(cursor.fetchall())
        del cursor
        action, self._action = self._action, None
        action()
        return _Fetched(rows)


@pytest.mark.parametrize("backend", ["dbapi", "postgresql"])
def test_rdbms_prune_keeps_a_session_used_after_its_candidate_query(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The finding's race: another replica issues a value in the session
    # right after this replica's idle SELECT. The per-session re-check sees
    # it in use and keeps the session whole.
    make = _factory(f"rdbms-{backend}", tmp_path, monkeypatch)
    a, b = make(), make()
    b_view = b.get("s")
    ada = b_view.placeholder_for("EMAIL", "ada@corp.example")
    _age_all(a)
    bob: list[str] = []
    a._store._execute = _AfterStatement(  # type: ignore[method-assign]
        a._store,
        "GROUP BY session_id HAVING MAX(created_at)",
        lambda: bob.append(b_view.placeholder_for("EMAIL", "bob@corp.example")),
    )
    assert a.prune_sessions(30) == 0
    assert bob == ["«EMAIL_002»"]
    fresh = make()
    assert fresh.get("s").original_for(ada) == "ada@corp.example"  # kept whole
    assert fresh.get("s").original_for(bob[0]) == "bob@corp.example"
    for manager in (a, b, fresh):
        manager.close()


@pytest.mark.parametrize("backend", ["dbapi", "postgresql"])
def test_rdbms_prune_spares_a_value_issued_after_its_recheck(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A value issued between the per-session re-check and the delete is
    # numbered above everything the delete removes (and retires): it
    # survives, and nothing is ever issued twice.
    make = _factory(f"rdbms-{backend}", tmp_path, monkeypatch)
    a, b = make(), make()
    b_view = b.get("s")
    ada = b_view.placeholder_for("EMAIL", "ada@corp.example")
    _age_all(a)
    bob: list[str] = []
    a._store._execute = _AfterStatement(  # type: ignore[method-assign]
        a._store,
        "SELECT MAX(n), MAX(created_at)",
        lambda: bob.append(b_view.placeholder_for("EMAIL", "bob@corp.example")),
    )
    assert a.prune_sessions(30) == 1
    fresh = make().get("s")
    assert fresh.original_for(ada) is None  # the idle rows went, retired
    assert fresh.original_for(bob[0]) == "bob@corp.example"  # the new one stayed
    carol = fresh.placeholder_for("EMAIL", "carol@corp.example")
    assert carol not in (ada, bob[0])
    for manager in (a, b):
        manager.close()


# --- the retired number itself ----------------------------------------------------


@pytest.mark.parametrize("how", ["forget", "prune"])
def test_a_failed_delete_and_a_failed_rollback_raise_the_delete_error(
    tmp_path: Path, how: str
) -> None:
    # A disk fault on the delete AND on its rollback: the delete's own error
    # propagates (the rollback's is suppressed), and nothing is deleted.
    manager = SqliteVaultManager(tmp_path / "vault.db")
    view = manager.get("s")
    ada = view.placeholder_for("EMAIL", "ada@corp.example")
    _age_all(manager)
    real = manager._conn
    manager._conn = _FlakyConn(_FlakyConn(real, "DELETE FROM mappings"), "ROLLBACK")
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        _delete(manager, how)
    manager._conn = real
    real.execute("ROLLBACK")  # the suppressed rollback left the transaction open
    assert view.original_for(ada) == "ada@corp.example"
    assert manager.total_entries() == 1
    manager.close()


def test_the_lru_keeps_its_views_alive(tmp_path: Path) -> None:
    # Held only by the LRU, a view survives a collection: a cache that let
    # its views go would reload the session's rows on every request.
    import gc
    import weakref

    manager = SqliteVaultManager(tmp_path / "vault.db")
    alive = weakref.ref(manager.get("s"))
    gc.collect()
    assert alive() is not None
    assert manager.get("s") is alive()
    manager.close()


def test_the_sqlite_retired_number_never_decreases(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    manager.get("s").placeholder_for("EMAIL", "ada@corp.example")
    # A retired number above the live rows (a hand-restored table): a delete
    # keeps the higher one.
    manager._conn.execute("INSERT INTO retired_numbers (session_id, n) VALUES ('s', 50)")
    assert manager.forget_sessions(["s"]) == 1
    assert manager._conn.execute("SELECT n FROM retired_numbers").fetchall() == [(50,)]
    assert manager.get("s").placeholder_for("EMAIL", "bob@corp.example") == "«EMAIL_051»"
    manager.close()


def test_the_rdbms_retired_number_never_decreases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make = _factory("rdbms-dbapi", tmp_path, monkeypatch)
    manager = make()
    store = manager._store
    manager.get("s").placeholder_for("EMAIL", "ada@corp.example")

    def raise_retired(conn: Any) -> None:
        store._retire(conn, "s", 50)
        store._retire(conn, "s", 7)  # lower: ignored
        conn.commit()

    store._run(raise_retired)
    assert store.retired("s") == 50
    assert manager.forget_sessions(["s"]) == 1
    assert store.retired("s") == 50
    assert manager.get("s").placeholder_for("EMAIL", "bob@corp.example") == "«EMAIL_051»"
    manager.close()


def test_rdbms_a_replica_retiring_the_same_session_first_starts_the_delete_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make = _factory("rdbms-dbapi", tmp_path, monkeypatch)
    a, b = make(), make()
    a.get("s").placeholder_for("EMAIL", "ada@corp.example")

    def other_replica_retires() -> None:
        def op(conn: Any) -> None:
            b._store._retire(conn, "s", 3)
            conn.commit()

        b._store._run(op)

    a._store._execute = _AfterStatement(  # type: ignore[method-assign]
        a._store, "SELECT n FROM llm_redact_retired", other_replica_retires
    )
    # A's INSERT of the retired row collides with B's: the delete starts
    # over, finds the row, and never lowers it.
    assert a.forget_sessions(["s"]) == 1
    assert a._store.retired("s") == 3
    assert a.get("s").placeholder_for("EMAIL", "bob@corp.example") == "«EMAIL_004»"
    for manager in (a, b):
        manager.close()


def test_rdbms_a_delete_that_keeps_colliding_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    manager = RdbmsVaultManager(RdbmsStore(config, None))
    view = manager.get("s")
    ada = view.placeholder_for("EMAIL", "ada@corp.example")
    for _ in range(3):
        driver.inject_fault("INSERT INTO llm_redact_retired", sqlite3.IntegrityError("taken"))
    with pytest.raises(RuntimeError, match="kept colliding"):
        manager.forget_sessions(["s"])
    # Rolled back whole: nothing deleted.
    assert view.original_for(ada) == "ada@corp.example"
    assert manager.total_entries() == 1
    manager.close()


def test_rdbms_a_store_that_cannot_create_its_retired_table_refuses_to_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    RdbmsStore(config, None).close()  # an existing schema...
    conn = sqlite3.connect(tmp_path / "postgresql.db")
    conn.execute("DROP TABLE llm_redact_retired")  # ...from before the table
    conn.commit()
    conn.close()
    driver.inject_fault(
        "CREATE TABLE llm_redact_retired", sqlite3.OperationalError("permission denied")
    )
    with pytest.raises(ConfigError) as refused:
        RdbmsStore(config, None)
    message = str(refused.value)
    assert "llm_redact_retired" in message and "CREATE TABLE" in message
    assert "db.corp.example" not in message  # never the DSN
    # A user that may create it gets it on the next start.
    store = RdbmsStore(config, None)
    assert store.retired("s") == 0
    store.close()


def test_the_memory_vault_keeps_one_vault_per_session(tmp_path: Path) -> None:
    manager = InMemoryVaultManager()
    held = manager.get("s")
    held.placeholder_for("EMAIL", "ada@corp.example")
    manager.get("empty")
    assert manager.forget_sessions(["s", "empty"]) == 1  # only "s" held mappings
    assert manager.get("s") is held
    assert manager.sessions_summary() == []  # sessions holding mappings only
    assert manager.session_count() == 0
    held.placeholder_for("EMAIL", "bob@corp.example")
    assert [row["session"] for row in manager.sessions_summary()] == ["s"]
    assert manager.session_count() == 1


def test_a_delete_of_more_sessions_than_one_statement_binds(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    names = [f"s{i:04d}" for i in range(LOOKUP_CHUNK + 2)]
    manager._conn.executemany(
        "INSERT INTO mappings (session_id, detector_type, original, placeholder, n)"
        " VALUES (?, 'EMAIL', ?, '«EMAIL_001»', 1)",
        [(name, f"{name}@corp.example") for name in names],
    )
    held = manager.get(names[-1])
    assert manager.forget_sessions([*names, "missing"]) == len(names)
    assert manager.session_count() == 0
    retired = manager._conn.execute("SELECT COUNT(*), MIN(n), MAX(n) FROM retired_numbers")
    assert retired.fetchone() == (len(names), 1, 1)
    assert held.placeholder_for("EMAIL", "new@corp.example") == "«EMAIL_002»"
    manager.close()


def test_a_delete_binds_one_chunk_per_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Five sessions in chunks of two: three statements of each kind, every
    # session deleted and retired exactly once, the count exact.
    monkeypatch.setattr("llm_redact.vault.LOOKUP_CHUNK", 2)
    manager = SqliteVaultManager(tmp_path / "vault.db")
    for session in ("s1", "s2", "s3", "s4", "s5"):
        manager.get(session).placeholder_for("EMAIL", f"{session}@corp.example")
    statements: list[str] = []
    manager._conn.set_trace_callback(statements.append)
    assert manager.forget_sessions(["s1", "s2", "s3", "s4", "s5", "missing"]) == 5
    deletes = [sql for sql in statements if sql.startswith("DELETE FROM mappings")]
    assert len(deletes) == 3
    # One write transaction for the whole delete.
    assert statements[0] == "BEGIN IMMEDIATE" and statements[-1] == "COMMIT"
    assert manager.total_entries() == 0
    retired = manager._conn.execute("SELECT session_id, n FROM retired_numbers ORDER BY 1")
    assert retired.fetchall() == [(f"s{i}", 1) for i in range(1, 6)]
    manager.close()


def test_a_prune_of_more_sessions_than_one_statement_binds(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    names = [f"s{i:04d}" for i in range(LOOKUP_CHUNK + 2)]
    manager._conn.executemany(
        "INSERT INTO mappings (session_id, detector_type, original, placeholder, n, created_at)"
        " VALUES (?, 'EMAIL', ?, '«EMAIL_001»', 1, '2000-01-01T00:00:00Z')",
        [(name, f"{name}@corp.example") for name in names],
    )
    assert manager.prune_sessions(30) == len(names)
    assert manager.total_entries() == 0
    assert manager._conn.execute("SELECT COUNT(*) FROM retired_numbers").fetchone() == (len(names),)
    manager.close()


# --- end to end ------------------------------------------------------------------

JANE = "jane.doe@corp.example"
BOB = "bob.builder@corp.example"


def _echo(sent: list[str]) -> httpx.MockTransport:
    """An Anthropic upstream answering with the prompt it received."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        text = json.loads(request.content)["messages"][-1]["content"]
        sent.append(text)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
            },
        )

    return httpx.MockTransport(handler)


def _proxy(db: Path, session: str, sent: list[str]) -> Any:
    config = Config(
        providers={**Config().providers, "anthropic": ProviderConfig("http://upstream.test")},
        vault=VaultConfig(backend="sqlite", path=str(db), session=session),
    )
    return create_app(config, upstream_transport=_echo(sent))


async def _ask(app: Any, text: str) -> str:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": text}]},
        )
    assert response.status_code == 200, response.text
    return str(response.json()["content"][0]["text"])


async def test_two_proxies_on_one_vault_never_restore_the_wrong_value_after_a_prune(
    tmp_path: Path,
) -> None:
    # C-R2-03 end to end: proxy A prunes proxy B's live static session
    # (it saw no new value within the TTL); B's next request then carries
    # jane (cached) and bob (new). The upstream must see two DIFFERENT
    # tokens, and B's client must get back exactly what it sent.
    db = tmp_path / "vault.db"
    sent_a: list[str] = []
    sent_b: list[str] = []
    proxy_a = _proxy(db, "work", sent_a)
    proxy_b = _proxy(db, "personal", sent_b)
    assert await _ask(proxy_b, f"mail {JANE}") == f"mail {JANE}"
    assert sent_b == ["mail «EMAIL_001»"]
    _age_all(proxy_b.state.proxy.vault_manager)
    transport = httpx.ASGITransport(app=proxy_a)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        pruned = await client.post(
            "/__llm-redact/sessions/prune",
            headers={CSRF_HEADER: proxy_a.state.proxy.csrf_token},
            json={"older_than_days": 30},
        )
    assert pruned.json() == {"pruned": 1}  # "personal" (A keeps only its own "work")
    answer = await _ask(proxy_b, f"compare {JANE} with {BOB}")
    jane_token, bob_token = sent_b[-1].removeprefix("compare ").split(" with ")
    assert jane_token != bob_token
    assert answer == f"compare {JANE} with {BOB}"
    for proxy in (proxy_a, proxy_b):
        proxy.state.proxy.vault_manager.close()


async def test_a_session_forgotten_under_a_live_view_never_restores_the_wrong_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # P-R2-07, core side: an access gate purges a user's session through the
    # proxy's SessionStore while a realtime connection still holds its view.
    from test_session_ownership_seams import BindingGate, _registry

    gate = BindingGate()
    _registry(monkeypatch, build_access_gate=lambda config, license: gate)
    app = create_app(
        Config(vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"), session="main"))
    )
    state = app.state.proxy
    connection = state.vault_manager.get("user:n1:main")  # the relay's ctx.vault
    alice = connection.placeholder_for("EMAIL", "alice.a@corp.example")
    assert gate.store is not None
    assert gate.store.forget(["user:n1:main"]) == 1
    bob = connection.placeholder_for("EMAIL", "bob.b@corp.example")
    alice_again = connection.placeholder_for("EMAIL", "alice.a@corp.example")
    assert len({alice, bob, alice_again}) == 3
    assert connection.original_for(alice) is None
    assert connection.original_for(alice_again) == "alice.a@corp.example"
    state.vault_manager.close()


# --- a staleness check that cannot read the database ---------------------------------


def _break_retired_reads(manager: Any, patch: pytest.MonkeyPatch) -> type[Exception]:
    """Make every read of the session's retired number fail the way its
    database fails (an RDBMS blip, a locked or failing sqlite file); returns
    the exception type."""
    if isinstance(manager, SqliteVaultManager):
        import llm_redact.vault as vault_mod

        def sqlite_down(conn: Any, session: str) -> int:
            raise sqlite3.OperationalError("disk I/O error at /secret/path")

        patch.setattr(vault_mod, "_retired_number", sqlite_down)
        return sqlite3.OperationalError
    error: type[Exception] = manager._store._module.OperationalError

    def rdbms_down(session: str) -> int:
        raise error("server closed the connection: host db.internal")

    patch.setattr(manager._store, "retired", rdbms_down)
    return error


@pytest.mark.parametrize("bound", [True, False], ids=["counted", "unbound"])
def test_a_failed_staleness_check_keeps_serving_the_cache(
    open_instance: Factory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    bound: bool,
) -> None:
    """A cache hit never needs the database: a due check that cannot read the
    retired number keeps the caches (a cached token only ever restores its
    own value) and is tried again at the next interval — counted per check
    where the manager is bound to a counter (the proxy's), logged once per
    outage by exception TYPE only."""
    clock = Clock()
    manager = open_instance(clock)
    faults: Counter[str] = Counter()
    if bound:
        getattr(manager, "bind_fault_counter", lambda counter: None)(faults)
    view = manager.get("s")
    ada = view.placeholder_for("EMAIL", "ada@corp.example")
    loaded = view._reverse
    caplog.set_level("INFO", logger="llm_redact")
    for outage in (1, 2):
        with monkeypatch.context() as patch:
            error = _break_retired_reads(manager, patch)
            for _ in range(3):
                clock.now += CACHE_CHECK_SECONDS  # due: the check fails
                assert view.original_for(ada) == "ada@corp.example"
                assert view.original_for(ada) == "ada@corp.example"  # not due again yet
                assert view.placeholder_for("EMAIL", "ada@corp.example") == ada
            assert view._reverse is loaded
        assert faults == ({"vault_check": 3 * outage} if bound else {})
        clock.now += CACHE_CHECK_SECONDS  # back up: the check reads again
        assert view.original_for(ada) == "ada@corp.example"
        messages = [(r.levelname, r.getMessage()) for r in caplog.records]
        assert (
            messages
            == [
                (
                    "WARNING",
                    f"vault staleness check failed ({error.__name__}): cached values are served"
                    " until the database answers again",
                ),
                ("INFO", "vault staleness check: the database answers again"),
            ]
            * outage
        )  # once per outage, and once when it ends
        for text in ("ada", "secret", "db.internal", "closed the connection", "disk I/O"):
            assert text not in caplog.text
    # The check works again: a delete by another instance is noticed.
    assert open_instance().forget_sessions(["s"]) == 1
    clock.now += CACHE_CHECK_SECONDS
    assert view.original_for(ada) is None


@pytest.mark.parametrize("backend", ["sqlite", "rdbms-dbapi"])
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streamed"])
async def test_a_database_blip_while_an_answer_is_restored_costs_nothing_cached(
    backend: str, streamed: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the staleness check falls due while the upstream answers
    and the database is down. The answer's cached token is restored anyway
    — it used to fail the check's read (a buffered answer a recorded 502, a
    stream cut) — and the fault is counted in /status."""
    import json

    import llm_redact.registry as registry_mod
    from llm_redact.registry import Registry

    clock = Clock()
    manager = _factory(backend, tmp_path, monkeypatch)(clock)
    registry = Registry()
    registry.build_vault_manager = lambda config: manager
    monkeypatch.setattr(registry_mod, "_registry", registry)
    outage = pytest.MonkeyPatch()  # undone below, before the managers close

    def upstream(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["messages"][-1]["content"]
        if "again" in text:  # the second request: the database goes down now
            clock.now += CACHE_CHECK_SECONDS
            _break_retired_reads(manager, outage)
        if not streamed:
            return httpx.Response(
                200, json={"type": "message", "content": [{"type": "text", "text": text}]}
            )
        events = [
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": text}}),
            ("content_block_stop", {"index": 0}),
            ("message_stop", {}),
        ]
        body = "".join(
            f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"
            for name, data in events
        )
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    config = Config(
        providers={**Config().providers, "anthropic": ProviderConfig("http://upstream.test")},
        vault=VaultConfig(backend="sqlite", path=str(tmp_path / "unused.db"), session="s"),
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            for text in (f"mail {JANE}", f"mail {JANE} again"):
                response = await client.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 5,
                        "stream": streamed,
                        "messages": [{"role": "user", "content": text}],
                    },
                )
                assert response.status_code == 200, response.text
                assert text in response.text  # the token restored, the stream whole
                assert "«EMAIL_" not in response.text
            status = (await client.get("/__llm-redact/status")).json()
        assert status["bookkeeping_errors_total"] == {"vault_check": 1}
    finally:
        outage.undo()
        manager.close()
