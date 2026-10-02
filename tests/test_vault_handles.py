"""The durable Live resumption handle map (optional vault-manager members).

``record_handle_session(handle_digest, session_id, *, replaces=())`` and
``lookup_handle_session(handle_digest)`` let a plugin (llm-redact-pro) vouch
for a Gemini/Vertex Live session-resumption handle across a restart and
across replicas sharing one vault. Pinned here for every persistent backend
(sqlite plain and encrypted, the generic DB-API store over stdlib sqlite3,
the fake psycopg/PyMySQL/oracledb drivers):

- digests map to the session they were issued in; ``replaces`` drops the
  same connection's superseded digests in the SAME transaction (a fault
  leaves both as they were) and only ever the same session's rows;
- the bound follows INSERTION order (a rowid / allocated ``seq``), never a
  timestamp: rows written within one second trim oldest first, and a digest
  written again becomes the newest;
- per session the newest ``MAX_SESSION_HANDLES`` stay; beyond the newest
  ``MAX_HANDLE_ROWS`` in all only rows of sessions holding NO mappings go,
  so another session's traffic never strands a live session's handles;
- every whole-session delete (TTL prune, the prune endpoint, the CLI prune,
  ``SessionStore.forget``) removes the session's rows in its own
  transaction: a handle into a pruned-and-recreated session is unknown;
- a fault is contained (bookkeeping stage ``handle_map``, logged once per
  outage by type): a write records nothing, a read answers None;
- the in-memory manager keeps no durable map (no-op / None).
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ConfigError, VaultConfig
from llm_redact.proxy import CSRF_HEADER, create_app
from llm_redact.vault import (
    HANDLE_FAULT_STAGE,
    MAX_HANDLE_ROWS,
    MAX_SESSION_HANDLES,
    InMemoryVaultManager,
    SqliteVaultManager,
)
from llm_redact.vault_rdbms import RdbmsStore
from test_vault_deletion import PERSISTENT, Factory, _age_all, _factory, _opening
from test_vault_faults import _FlakyConn
from test_vault_rdbms import _fake_backend_config

H = "live-handle:"


@pytest.fixture(params=PERSISTENT)
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def open_instance(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Factory]:
    yield from _opening(_factory(backend, tmp_path, monkeypatch))


def _db_file(backend: str, tmp_path: Path) -> Path:
    """The sqlite file under each backend (the fakes are sqlite underneath)."""
    if backend.startswith("sqlite"):
        return tmp_path / "vault.db"
    if backend.startswith("rdbms-dbapi"):
        return tmp_path / "v.db"
    return tmp_path / f"{backend.removeprefix('rdbms-')}.db"


def _table(backend: str) -> str:
    return "handle_sessions" if backend.startswith("sqlite") else "llm_redact_handle_sessions"


def _bound(
    monkeypatch: pytest.MonkeyPatch, *, per_session: int, total: int, every: int = 1
) -> None:
    for module in ("llm_redact.vault", "llm_redact.vault_rdbms"):
        monkeypatch.setattr(f"{module}.MAX_SESSION_HANDLES", per_session)
        monkeypatch.setattr(f"{module}.MAX_HANDLE_ROWS", total)
        monkeypatch.setattr(f"{module}._RESPONSE_PRUNE_EVERY", every)


def _known(manager: Any, *names: str) -> dict[str, str | None]:
    return {name: manager.lookup_handle_session(H + name) for name in names}


def test_the_bound_is_generous_and_documented() -> None:
    # One session counts recent connections (each keeps its own newest few
    # through ``replaces``); the whole map keeps the response map's bound.
    assert MAX_SESSION_HANDLES == 1024
    assert MAX_HANDLE_ROWS == 10000


def test_a_handle_maps_to_its_session_and_survives_a_restart(open_instance: Factory) -> None:
    first = open_instance()
    first.record_handle_session(H + "a", "user:n1:main")
    first.record_handle_session(H + "b", "main")
    # Another replica (or the same proxy after a restart) over the same vault.
    second = open_instance()
    assert _known(second, "a", "b", "never") == {"a": "user:n1:main", "b": "main", "never": None}


def test_replaces_drops_only_the_same_sessions_superseded_digests(
    open_instance: Factory,
) -> None:
    manager = open_instance()
    manager.record_handle_session(H + "a1", "s")
    manager.record_handle_session(H + "a2", "s")
    manager.record_handle_session(H + "t1", "t")
    manager.record_handle_session(H + "a3", "s", replaces=[H + "a1", H + "a2", H + "t1"])
    # The same connection's older handles go; another session's never do.
    assert _known(manager, "a1", "a2", "a3", "t1") == {
        "a1": None,
        "a2": None,
        "a3": "s",
        "t1": "t",
    }
    # Empty and absent replaces are fine.
    manager.record_handle_session(H + "a4", "s", replaces=[])
    manager.record_handle_session(H + "a5", "s", replaces=[H + "never"])
    assert _known(manager, "a3", "a4", "a5") == {"a3": "s", "a4": "s", "a5": "s"}


def test_replaces_and_the_insert_are_one_transaction(
    open_instance: Factory, backend: str, tmp_path: Path
) -> None:
    manager = open_instance()
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    manager.record_handle_session(H + "old", "s")
    # A write that fails AFTER the replaces delete ran (the insert aborts).
    raw = sqlite3.connect(_db_file(backend, tmp_path))
    raw.execute(
        f"CREATE TRIGGER poison BEFORE INSERT ON {_table(backend)}"
        f" WHEN NEW.handle_digest = '{H}poison' BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    raw.commit()
    manager.record_handle_session(H + "poison", "s", replaces=[H + "old"])  # contained
    assert counter[HANDLE_FAULT_STAGE] == 1
    # Rolled back whole: the superseded digest is still there, nothing new.
    assert _known(manager, "old", "poison") == {"old": "s", "poison": None}
    raw.execute("DROP TRIGGER poison")
    raw.commit()
    raw.close()
    manager.record_handle_session(H + "poison", "s", replaces=[H + "old"])
    assert _known(manager, "old", "poison") == {"old": None, "poison": "s"}
    assert counter[HANDLE_FAULT_STAGE] == 1


def test_a_session_keeps_its_newest_handles_in_insertion_order(
    open_instance: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bound(monkeypatch, per_session=3, total=1000, every=1000)
    manager = open_instance()
    # Five writes within one second: a timestamp would tie; insertion order
    # never does.
    for index in range(5):
        manager.record_handle_session(H + f"h{index}", "s")
    manager.record_handle_session(H + "other", "t")
    assert _known(manager, "h0", "h1", "h2", "h3", "h4", "other") == {
        "h0": None,
        "h1": None,
        "h2": "s",
        "h3": "s",
        "h4": "s",
        "other": "t",  # another session's rows are not this one's bound
    }
    # Written again, a digest is the newest: h3 (now the oldest) goes next.
    manager.record_handle_session(H + "h2", "s")
    manager.record_handle_session(H + "h5", "s")
    assert _known(manager, "h2", "h3", "h4", "h5") == {
        "h2": "s",
        "h3": None,
        "h4": "s",
        "h5": "s",
    }


def test_the_total_bound_never_trims_a_live_sessions_handles(
    open_instance: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bound(monkeypatch, per_session=100, total=2)
    manager = open_instance()
    manager.get("live").placeholder_for("EMAIL", "ada@corp.example")
    manager.record_handle_session(H + "live", "live")  # the OLDEST row
    for index in range(4):
        manager.record_handle_session(H + f"e{index}", "empty")  # holds no mappings
    # Beyond the newest two rows, only rows of sessions without mappings go.
    assert _known(manager, "live", "e0", "e1", "e2", "e3") == {
        "live": "live",
        "e0": None,
        "e1": None,
        "e2": "empty",
        "e3": "empty",
    }


def test_the_total_bound_is_checked_every_interval(
    open_instance: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bound(monkeypatch, per_session=100, total=1, every=3)
    manager = open_instance()
    manager.record_handle_session(H + "a", "empty")
    manager.record_handle_session(H + "b", "empty")
    assert _known(manager, "a", "b") == {"a": "empty", "b": "empty"}  # not yet
    manager.record_handle_session(H + "c", "empty")  # the third write trims
    assert _known(manager, "a", "b", "c") == {"a": None, "b": None, "c": "empty"}
    manager.record_handle_session(H + "d", "empty")
    manager.record_handle_session(H + "e", "empty")
    assert _known(manager, "c", "d", "e") == {"c": "empty", "d": "empty", "e": "empty"}
    manager.record_handle_session(H + "f", "empty")  # the count started over
    assert _known(manager, "d", "e", "f") == {"d": None, "e": None, "f": "empty"}


def test_a_failed_write_does_not_advance_the_trim_interval(
    open_instance: Factory, backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bound(monkeypatch, per_session=100, total=1, every=2)
    manager = open_instance()
    raw = sqlite3.connect(_db_file(backend, tmp_path))
    raw.execute(
        f"CREATE TRIGGER poison BEFORE INSERT ON {_table(backend)}"
        f" WHEN NEW.handle_digest = '{H}poison' BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    raw.commit()
    raw.close()
    manager.record_handle_session(H + "a", "empty")
    manager.record_handle_session(H + "poison", "empty")  # failed: not counted
    manager.record_handle_session(H + "b", "empty")  # the second write trims
    assert _known(manager, "a", "b") == {"a": None, "b": "empty"}


def test_a_pruned_session_takes_its_handles_with_it(
    open_instance: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = open_instance()
    manager.get("old").placeholder_for("EMAIL", "ada@corp.example")
    manager.get("kept").placeholder_for("EMAIL", "bob@corp.example")
    manager.record_handle_session(H + "old", "old")
    manager.record_handle_session(H + "kept", "kept")
    _age_all(manager)
    assert manager.prune_sessions(30, exclude=frozenset({"kept"})) == 1
    # Recreated with a new value, the session is a new epoch: the handle
    # issued before the prune is unknown — never resumed into new values.
    manager.get("old").placeholder_for("EMAIL", "carol@corp.example")
    assert _known(manager, "old", "kept") == {"old": None, "kept": "kept"}
    # Another replica sees the same.
    assert _known(open_instance(), "old", "kept") == {"old": None, "kept": "kept"}


def test_a_forgotten_session_takes_its_handles_with_it(open_instance: Factory) -> None:
    manager = open_instance()
    manager.get("user:n1:main").placeholder_for("EMAIL", "ada@corp.example")
    manager.record_handle_session(H + "mapped", "user:n1:main")
    manager.record_handle_session(H + "bare", "user:n2:main")  # holds no mappings
    manager.record_handle_session(H + "other", "main")
    assert manager.forget_sessions(["user:n1:main", "user:n2:main"]) == 1
    assert _known(manager, "mapped", "bare", "other") == {
        "mapped": None,
        "bare": None,
        "other": "main",
    }


def _break_reads(manager: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    if isinstance(manager, SqliteVaultManager):
        flaky = _FlakyConn(manager._conn, "FROM handle_sessions WHERE handle_digest", times=2)
        manager._shared.conn = flaky  # type: ignore[assignment]
        return

    calls = {"left": 2}
    real = manager._store.lookup_handle

    def lookup(digest: str) -> str | None:
        if calls["left"]:
            calls["left"] -= 1
            raise sqlite3.OperationalError("server closed the connection")
        return real(digest)  # type: ignore[no-any-return]

    monkeypatch.setattr(manager._store, "lookup_handle", lookup)


def test_a_read_fault_reads_as_unknown_counted_and_logged_by_type(
    open_instance: Factory, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    manager = open_instance()
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    manager.record_handle_session(H + "a", "s")
    _break_reads(manager, monkeypatch)
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        assert manager.lookup_handle_session(H + "a") is None  # unknown, never a guess
        assert manager.lookup_handle_session(H + "a") is None
        assert manager.lookup_handle_session(H + "a") == "s"  # answers again
    assert counter[HANDLE_FAULT_STAGE] == 2
    messages = [record.getMessage() for record in caplog.records]
    # One line when the outage starts, one when it ends — by type only.
    assert messages == [
        "vault handle map read failed (OperationalError): Live resumption handles read"
        " as unknown until the database answers again",
        "vault handle map: reads succeed again",
    ]
    assert [record.levelno for record in caplog.records] == [logging.WARNING, logging.INFO]
    assert H not in " ".join(messages) and "closed" not in " ".join(messages)


def test_a_write_fault_records_nothing_and_is_counted(
    open_instance: Factory, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = open_instance()
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    if isinstance(manager, SqliteVaultManager):
        manager._shared.conn = _FlakyConn(manager._conn, "INSERT OR REPLACE INTO handle_sessions")  # type: ignore[assignment]
    else:

        def broken(*args: Any, **kwargs: Any) -> None:
            raise sqlite3.OperationalError("disk full")

        monkeypatch.setattr(manager._store, "_write_handle", broken)
    manager.record_handle_session(H + "a", "s")  # contained: never raises
    assert counter[HANDLE_FAULT_STAGE] == 1
    monkeypatch.undo()
    if isinstance(manager, SqliteVaultManager):
        # The connection is not wedged: the next write and read work.
        manager.record_handle_session(H + "b", "s")
        assert _known(manager, "a", "b") == {"a": None, "b": "s"}
    else:
        assert manager.lookup_handle_session(H + "a") is None


def test_a_write_inside_an_open_batch_is_refused_without_touching_it(tmp_path: Path) -> None:
    # Never reachable from the proxy (a batch's work cannot await), but a
    # write landing inside one must not roll the batch back.
    manager = SqliteVaultManager(tmp_path / "vault.db")
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    view = manager.get("s")
    from llm_redact.vault import run_batched

    def work() -> str:
        token = view.placeholder_for("EMAIL", "ada@corp.example")
        manager.record_handle_session(H + "a", "s")
        return token

    token = run_batched(view, work)
    assert counter[HANDLE_FAULT_STAGE] == 1
    assert view.original_for(token) == "ada@corp.example"
    assert manager.lookup_handle_session(H + "a") is None
    manager.close()


def test_the_memory_manager_keeps_no_durable_handle_map() -> None:
    manager = InMemoryVaultManager()
    assert manager.durable_response_map is False
    manager.record_handle_session(H + "a", "s", replaces=[H + "b"])
    assert manager.lookup_handle_session(H + "a") is None


def test_an_existing_sqlite_vault_gains_the_table(tmp_path: Path) -> None:
    db = tmp_path / "vault.db"
    SqliteVaultManager(db).close()
    raw = sqlite3.connect(db)
    raw.execute("DROP TABLE handle_sessions")  # a vault from before the map
    raw.commit()
    raw.close()
    manager = SqliteVaultManager(db)
    manager.record_handle_session(H + "a", "s")
    assert manager.lookup_handle_session(H + "a") == "s"
    manager.close()


@pytest.mark.parametrize("backend_name", ["postgresql", "mysql", "oracle"])
def test_an_existing_rdbms_schema_gains_the_table_or_refuses_naming_it(
    backend_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, backend_name)
    RdbmsStore(config, None).close()  # an existing schema...
    raw = sqlite3.connect(tmp_path / f"{backend_name}.db")
    raw.execute("DROP TABLE llm_redact_handle_sessions")  # ...from before the map
    raw.commit()
    raw.close()
    driver.inject_fault(
        "CREATE TABLE llm_redact_handle_sessions", sqlite3.OperationalError("permission denied")
    )
    with pytest.raises(ConfigError) as refused:
        RdbmsStore(config, None)
    message = str(refused.value)
    assert "llm_redact_handle_sessions" in message and "CREATE TABLE" in message
    assert "db.corp.example" not in message and "vault@" not in message  # never the DSN
    store = RdbmsStore(config, None)  # a user that may create it gets it
    store.record_handle(H + "a", "s", [], trim_all=False)
    assert store.lookup_handle(H + "a") == "s"
    store.close()


def test_an_rdbms_write_that_collides_starts_over_then_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.vault_rdbms import RdbmsVaultManager

    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    manager = RdbmsVaultManager(RdbmsStore(config, None))
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    # Another replica numbered the same seq first: the write starts over.
    driver.inject_fault("INSERT INTO llm_redact_handle_sessions", sqlite3.IntegrityError("taken"))
    manager.record_handle_session(H + "a", "s")
    assert manager.lookup_handle_session(H + "a") == "s"
    assert counter[HANDLE_FAULT_STAGE] == 0
    # Colliding past the bound: refused (contained), nothing written.
    for _ in range(3):
        driver.inject_fault(
            "INSERT INTO llm_redact_handle_sessions", sqlite3.IntegrityError("taken")
        )
    manager.record_handle_session(H + "b", "s")
    assert counter[HANDLE_FAULT_STAGE] == 1
    assert manager.lookup_handle_session(H + "b") is None
    # Any other database error: rolled back, contained.
    driver.inject_fault("INSERT INTO llm_redact_handle_sessions", sqlite3.DataError("disk full"))
    manager.record_handle_session(H + "c", "s", replaces=[H + "a"])
    assert counter[HANDLE_FAULT_STAGE] == 2
    assert _known(manager, "a", "c") == {"a": "s", "c": None}
    # A dropped connection reconnects once and the write lands.
    driver.dead = True
    manager.record_handle_session(H + "d", "s")
    assert manager.lookup_handle_session(H + "d") == "s"
    manager.close()


def test_the_cli_prune_drops_handles_in_a_vault_predating_the_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from test_cli_vault import _age_session, _run, _seed_plain

    db = tmp_path / "vault.db"
    _seed_plain(db)
    _age_session(db, "conv-aaaa", 120)
    raw = sqlite3.connect(db)
    raw.execute("DROP TABLE handle_sessions")
    raw.commit()
    raw.close()
    assert _run(["sessions", "prune", "--older-than", "90d", "--yes", "--db", str(db)]) == 0
    raw = sqlite3.connect(db)
    assert raw.execute("SELECT COUNT(*) FROM handle_sessions").fetchone() == (0,)
    raw.close()

    manager = SqliteVaultManager(db)
    manager.record_handle_session(H + "b", "conv-bbbb")
    manager.close()
    _age_session(db, "conv-bbbb", 120)
    assert _run(["sessions", "prune", "--older-than", "90d", "--yes", "--db", str(db)]) == 0
    manager = SqliteVaultManager(db)
    assert manager.lookup_handle_session(H + "b") is None
    manager.close()


async def test_the_prune_endpoint_drops_handles_through_the_app(
    tmp_path: Path,
) -> None:
    app = create_app(
        Config(vault=VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"), session="main"))
    )
    state = app.state.proxy
    manager = state.vault_manager
    manager.get("conv-old").placeholder_for("EMAIL", "ada@corp.example")
    manager.get("main").placeholder_for("EMAIL", "bob@corp.example")
    manager.record_handle_session(H + "old", "conv-old")
    manager.record_handle_session(H + "main", "main")
    _age_all(manager)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8787") as client:
        pruned = await client.post(
            "/__llm-redact/sessions/prune",
            headers={CSRF_HEADER: state.csrf_token},
            json={"older_than_days": 30},
        )
    assert pruned.json() == {"pruned": 1}  # the static session is never pruned
    assert _known(manager, "old", "main") == {"old": None, "main": "main"}
    manager.close()


@pytest.mark.parametrize(
    "manager_class",
    [SqliteVaultManager, InMemoryVaultManager, "RdbmsVaultManager"],
    ids=["sqlite", "memory", "rdbms"],
)
def test_every_manager_has_the_documented_call_shapes(manager_class: Any) -> None:
    import inspect

    if isinstance(manager_class, str):
        from llm_redact import vault_rdbms

        manager_class = getattr(vault_rdbms, manager_class)
    record = inspect.signature(manager_class.record_handle_session)
    assert list(record.parameters) == ["self", "handle_digest", "session_id", "replaces"]
    replaces = record.parameters["replaces"]
    assert replaces.kind is inspect.Parameter.KEYWORD_ONLY and replaces.default == ()
    lookup = inspect.signature(manager_class.lookup_handle_session)
    assert list(lookup.parameters) == ["self", "handle_digest"]


class _FailingRollback(_FlakyConn):
    """A write fault whose ROLLBACK fails too (a connection gone bad)."""

    def execute(self, sql: str, *args: object) -> object:
        if sql == "ROLLBACK":
            raise sqlite3.DatabaseError("database disk image is malformed")
        return super().execute(sql, *args)


@pytest.mark.parametrize("rollback_fails", [False, True], ids=["rollback", "rollback-fails"])
def test_a_sqlite_write_fault_is_logged_by_the_writes_own_type(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, rollback_fails: bool
) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    real = manager._conn
    flaky = _FailingRollback if rollback_fails else _FlakyConn
    manager._shared.conn = flaky(real, "INSERT OR REPLACE INTO handle_sessions")  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        manager.record_handle_session(H + "a", "s")
    # The write's own fault (a failing rollback never masks it), by type.
    assert [record.getMessage() for record in caplog.records] == [
        "vault handle map write failed (OperationalError): Live resumption handles go"
        " unrecorded (read as unknown) until a write succeeds again"
    ]
    manager._shared.conn = real
    if real.in_transaction:  # the failed ROLLBACK left it open
        real.execute("ROLLBACK")
    assert manager.lookup_handle_session(H + "a") is None
    manager.close()


def test_replaces_beyond_one_statement_are_all_dropped(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db")
    old = [H + f"old{index}" for index in range(1200)]
    for digest in old:
        manager.record_handle_session(digest, "s")
    manager.record_handle_session(H + "new", "s", replaces=old)
    assert [manager.lookup_handle_session(digest) for digest in old[-3:]] == [None] * 3
    assert manager.lookup_handle_session(H + "new") == "s"
    manager.close()


@pytest.mark.parametrize("backend_name", ["postgresql", "mysql", "oracle", "dbapi"])
def test_the_rdbms_order_column_is_64_bit(backend_name: str) -> None:
    # Review (handles, finding 3): ``seq`` only ever grows (MAX(seq)+1, and
    # the newest row is never trimmed). As a 32-bit INTEGER it overflowed
    # after 2^31 - 1 writes over a deployment's life, and from then on every
    # write failed (contained) — every resumption refused, nothing healing it.
    from llm_redact.vault_rdbms import _ddl

    ddl = _ddl(backend_name)["llm_redact_handle_sessions"]
    width = "NUMBER(19)" if backend_name == "oracle" else "BIGINT"
    assert f"seq {width} NOT NULL" in ddl
    assert "seq INTEGER" not in ddl


def test_an_rdbms_write_numbers_past_32_bits(tmp_path: Path) -> None:
    from llm_redact.vault_rdbms import RdbmsVaultManager
    from test_vault_rdbms import _dbapi_config

    db = tmp_path / "vault.db"
    manager = RdbmsVaultManager(RdbmsStore(_dbapi_config(db), None))
    raw = sqlite3.connect(db)
    raw.execute(
        "INSERT INTO llm_redact_handle_sessions (seq, handle_digest, session_id)"
        " VALUES (?, ?, 's')",
        (2**31 - 1, H + "top"),
    )
    raw.commit()
    manager.record_handle_session(H + "next", "s")
    assert raw.execute(
        "SELECT seq FROM llm_redact_handle_sessions WHERE handle_digest = ?", (H + "next",)
    ).fetchone() == (2**31,)
    raw.close()
    assert _known(manager, "top", "next") == {"top": "s", "next": "s"}
    manager.close()


@pytest.mark.parametrize(
    "table",
    [
        "llm_redact_mappings",
        "llm_redact_response_sessions",
        "llm_redact_meta",
        "llm_redact_retired",
        "llm_redact_handle_sessions",
    ],
)
def test_a_table_the_user_may_not_create_is_refused_with_its_grants(
    table: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review (handles, finding 2): a DBA creating the handle map after the
    # upgrade granted it like the retired-number table (no DELETE), and
    # every whole-session delete then failed — a purged user's values
    # stayed. The refusal now names the privileges the table needs.
    from llm_redact.vault_rdbms import _GRANTS, _ddl

    assert set(_GRANTS) == set(_ddl("postgresql"))
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    driver.inject_fault(f"CREATE TABLE {table} ", sqlite3.OperationalError("permission denied"))
    with pytest.raises(ConfigError) as refused:
        RdbmsStore(config, None)
    message = str(refused.value)
    assert f"could not create its table {table} (OperationalError)" in message
    assert f"grant this database user {_GRANTS[table]} on it" in message
    assert "db.corp.example" not in message and "vault@" not in message  # never the DSN


def test_the_grants_cover_every_statement_a_whole_session_delete_runs() -> None:
    from llm_redact.vault_rdbms import _GRANTS

    # A whole-session delete empties the session from these three tables in
    # one transaction; the retired number is only ever raised.
    for table in ("llm_redact_mappings", "llm_redact_response_sessions"):
        assert "DELETE" in _GRANTS[table]
    assert _GRANTS["llm_redact_handle_sessions"] == "SELECT, INSERT, DELETE"
    assert _GRANTS["llm_redact_retired"] == "SELECT, INSERT, UPDATE"


def _spied_statements(store: RdbmsStore, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    statements: list[str] = []
    execute = store._execute

    def spy(conn: Any, sql: str, params: dict[str, Any] | None = None) -> Any:
        statements.append(sql)
        return execute(conn, sql, params)

    monkeypatch.setattr(store, "_execute", spy)
    return statements


def test_an_rdbms_write_drops_its_digest_and_superseded_ones_in_one_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review (handles, finding 4): each write runs synchronously on the
    # event loop, one round trip per statement. A busy connection supersedes
    # one handle per new one, so the replaces delete and the digest's own
    # delete are one statement (the digest's row whichever session holds it;
    # the superseded ones only this session's).
    from llm_redact.vault_rdbms import RdbmsVaultManager
    from test_vault_rdbms import _dbapi_config

    store = RdbmsStore(_dbapi_config(tmp_path / "vault.db"), None)
    manager = RdbmsVaultManager(store)
    manager.record_handle_session(H + "old", "s")
    manager.record_handle_session(H + "t", "t")
    manager.record_handle_session(H + "moved", "t")
    statements = _spied_statements(store, monkeypatch)
    manager.record_handle_session(H + "new", "s", replaces=[H + "old", H + "t"])
    assert [sql.split()[0] for sql in statements] == ["DELETE", "SELECT", "INSERT", "SELECT"]
    statements.clear()
    manager.record_handle_session(H + "moved", "s")  # no replaces: still one delete
    assert [sql.split()[0] for sql in statements] == ["DELETE", "SELECT", "INSERT", "SELECT"]
    monkeypatch.undo()
    assert _known(manager, "old", "t", "new", "moved") == {
        "old": None,
        "t": "t",  # another session's digest is never superseded
        "new": "s",
        "moved": "s",  # written again: the newest row, in its new session
    }
    manager.close()


def test_an_rdbms_write_drops_superseded_digests_beyond_one_statement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.vault_rdbms import RdbmsVaultManager
    from test_vault_rdbms import _dbapi_config

    monkeypatch.setattr("llm_redact.vault_rdbms.LOOKUP_CHUNK", 2)
    store = RdbmsStore(_dbapi_config(tmp_path / "vault.db"), None)
    manager = RdbmsVaultManager(store)
    old = [f"o{index}" for index in range(5)]
    for name in old:
        manager.record_handle_session(H + name, "s")
    manager.record_handle_session(H + "x", "t")
    statements = _spied_statements(store, monkeypatch)
    manager.record_handle_session(H + "new", "s", replaces=[H + name for name in [*old, "x"]])
    deletes = [sql for sql in statements if sql.startswith("DELETE")]
    assert len(deletes) == 3  # six superseded digests, two per statement
    assert all("IN (:d, :r0, :r1)" in sql for sql in deletes)
    assert _known(manager, *old, "x", "new") == {
        **dict.fromkeys(old, None),
        "x": "t",
        "new": "s",
    }
    manager.close()


@pytest.mark.parametrize("backend_name", ["postgresql", "mysql", "oracle"])
def test_the_handle_map_on_a_real_server(
    backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The handle map's own SQL — the one-statement digest delete, the
    # allocation, both trims (the total one's correlated NOT EXISTS) and the
    # whole-session delete — against a real engine (env-gated like the
    # battery: LLM_REDACT_TEST_PG_DSN / _MYSQL_DSN / _ORACLE_DSN).
    from llm_redact.config import RdbmsConfig
    from llm_redact.vault_rdbms import RdbmsVaultManager
    from test_vault_rdbms import _drop_tables, _real_dsn

    config = VaultConfig(backend=backend_name, rdbms=RdbmsConfig(dsn=_real_dsn(backend_name)))
    _drop_tables(config)
    _bound(monkeypatch, per_session=2, total=2)
    manager = RdbmsVaultManager(RdbmsStore(config, None))
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    manager.get("live").placeholder_for("EMAIL", "ada@corp.example")
    manager.record_handle_session(H + "live", "live")  # the oldest row
    for index in range(4):
        manager.record_handle_session(H + f"e{index}", "empty")
    manager.record_handle_session(H + "f0", "other")
    # Per session the newest two; beyond the newest two in all, only rows
    # of sessions without mappings (e2 goes, the live session's row stays).
    assert _known(manager, "live", "e0", "e1", "e2", "e3", "f0") == {
        "live": "live",
        "e0": None,
        "e1": None,
        "e2": None,
        "e3": "empty",
        "f0": "other",
    }
    _bound(monkeypatch, per_session=2, total=2, every=1000)  # no more total trims
    manager.record_handle_session(H + "l2", "live", replaces=[H + "live", H + "f0"])
    manager.record_handle_session(H + "e3", "live")  # written again: moves
    assert _known(manager, "live", "l2", "e3", "f0") == {
        "live": None,
        "l2": "live",
        "e3": "live",
        "f0": "other",
    }
    assert manager.forget_sessions(["live", "other"]) == 1
    assert _known(manager, "l2", "e3", "f0") == {"l2": None, "e3": None, "f0": None}
    assert counter[HANDLE_FAULT_STAGE] == 0
    manager.close()


def _break_writes(manager: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    if isinstance(manager, SqliteVaultManager):
        real = manager._conn
        manager._shared.conn = _FlakyConn(real, "INSERT OR REPLACE INTO handle_sessions", times=99)  # type: ignore[assignment]
        return

    def broken(*args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError("permission denied for table")

    monkeypatch.setattr(manager._store, "_write_handle", broken)


def test_failing_writes_and_working_reads_are_separate_outages(
    open_instance: Factory, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Review (handles, finding 5): reads and writes shared one fault flag.
    # With writes failing (a missing DELETE grant) and lookups working, each
    # write logged a new WARNING and the next lookup a false "answers again".
    manager = open_instance()
    counter: Counter[str] = Counter()
    manager.bind_fault_counter(counter)
    manager.record_handle_session(H + "a", "s")
    _break_writes(manager, monkeypatch)
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        for index in range(3):
            manager.record_handle_session(H + f"w{index}", "s")  # contained
            assert manager.lookup_handle_session(H + "a") == "s"  # reads still work
    assert counter[HANDLE_FAULT_STAGE] == 3
    # One line for the write outage; no read outage, no false recovery.
    assert [record.getMessage() for record in caplog.records] == [
        "vault handle map write failed (OperationalError): Live resumption handles go"
        " unrecorded (read as unknown) until a write succeeds again"
    ]
    caplog.clear()
    monkeypatch.undo()
    if isinstance(manager, SqliteVaultManager):
        manager._shared.conn = manager._conn._real  # type: ignore[attr-defined]
    _break_reads(manager, monkeypatch)
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        assert manager.lookup_handle_session(H + "a") is None  # read outage starts
        manager.record_handle_session(H + "b", "s")  # the write outage ends
        assert manager.lookup_handle_session(H + "a") is None
        assert manager.lookup_handle_session(H + "b") == "s"  # the read outage ends
    assert [record.getMessage() for record in caplog.records] == [
        "vault handle map read failed (OperationalError): Live resumption handles read"
        " as unknown until the database answers again",
        "vault handle map: writes succeed again",
        "vault handle map: reads succeed again",
    ]
    assert [record.levelno for record in caplog.records] == [
        logging.WARNING,
        logging.INFO,
        logging.INFO,
    ]
