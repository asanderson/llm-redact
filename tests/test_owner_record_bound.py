"""A stored object's owner record is bounded APART from Responses rows.

The vault's durable response map holds two kinds of rows: a Responses
chain's response id -> session, and the session that created a stored
object (the session router's ownership record, mirrored by
``ProxyState.record_object_ids``). Beyond a bound, rows of sessions that hold
no mappings are trimmed — never a live session's (a router reads a missing
row as "that session was pruned"). With ONE shared bound, ordinary Responses
traffic (a row per turn) pushed out the owner record of a file whose
creating session never redacted anything — the object was then attributed
to nobody, and llm-redact-pro refuses such an object under a credential the
proxy holds. Each kind now keeps its own newest rows, in the sqlite store
and the RDBMS store alike; an existing map gains the ``kind`` column (its
rows count as Responses rows), and an RDBMS user that may not ALTER the
table keeps the old, shared bound with a warning.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.vault as vault_mod
import llm_redact.vault_rdbms as rdbms_mod
from llm_redact.config import Config, ProviderConfig, VaultConfig
from llm_redact.proxy import create_app
from llm_redact.vault import InMemoryVaultManager, SqliteVaultManager, _ensure_kind_column
from llm_redact.vault_rdbms import RdbmsStore, RdbmsVaultManager
from test_session_ownership_seams import OwnershipRouter, _files_app, _post_batch
from test_vault_rdbms import _dbapi_config, _fake_backend_config

OLD = "2000-01-01T00:00:00Z"


@pytest.fixture
def tiny_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (vault_mod, rdbms_mod):
        monkeypatch.setattr(module, "_RESPONSE_PRUNE_EVERY", 4)
        monkeypatch.setattr(module, "_MAX_RESPONSE_ROWS", 3)
        monkeypatch.setattr(module, "_MAX_OBJECT_ROWS", 2)


def _sqlite_manager(tmp_path: Path) -> tuple[Any, sqlite3.Connection]:
    manager = SqliteVaultManager(tmp_path / "v.db")
    return manager, manager._conn


def _dbapi_manager(tmp_path: Path) -> tuple[Any, sqlite3.Connection]:
    manager = RdbmsVaultManager(RdbmsStore(_dbapi_config(tmp_path / "v.db"), None))
    return manager, sqlite3.connect(tmp_path / "v.db", isolation_level=None)


def _postgres_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    config, _ = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    manager = RdbmsVaultManager(RdbmsStore(config, None))
    return manager, sqlite3.connect(tmp_path / "postgresql.db", isolation_level=None)


def _table(conn: sqlite3.Connection) -> str:
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return "response_sessions" if "response_sessions" in names else "llm_redact_response_sessions"


def _age(conn: sqlite3.Connection, row_id: str) -> None:
    conn.execute(f"UPDATE {_table(conn)} SET created_at = ? WHERE response_id = ?", (OLD, row_id))


@pytest.fixture(params=["sqlite", "dbapi", "postgresql"])
def store(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, sqlite3.Connection]:
    if request.param == "sqlite":
        return _sqlite_manager(tmp_path)
    if request.param == "dbapi":
        return _dbapi_manager(tmp_path)
    return _postgres_manager(tmp_path, monkeypatch)


def test_responses_never_push_out_an_owner_record(
    store: tuple[Any, sqlite3.Connection], tiny_caps: None
) -> None:
    manager, conn = store
    manager.record_object_session("file-clean", "user:ada:default")  # no mappings there
    _age(conn, "file-clean")  # the OLDEST row: a shared bound would drop it first
    for n in range(12):
        manager.record_response_session(f"resp_{n}", "user:grace:default")
    assert manager.lookup_response_session("file-clean") == "user:ada:default"
    responses = conn.execute(
        f"SELECT COUNT(*) FROM {_table(conn)} WHERE kind = 'response'"
    ).fetchone()[0]
    # The 12th insert ran the trim: exactly the newest 3 are kept (ties on
    # created_at decide WHICH, never how many).
    assert responses == 3


def test_owner_records_keep_their_own_bound(
    store: tuple[Any, sqlite3.Connection], tiny_caps: None
) -> None:
    manager, conn = store
    manager.get("user:ada:live").placeholder_for("EMAIL", "ada@corp.example")
    manager.record_object_session("file-live", "user:ada:live")
    manager.record_response_session("resp_mine", "user:ada:default")
    for row_id in ("file-live", "resp_mine"):
        _age(conn, row_id)
    for n in range(11):
        manager.record_object_session(f"file-{n}", "user:ada:default")
    kinds = dict(
        conn.execute(f"SELECT kind, COUNT(*) FROM {_table(conn)} GROUP BY kind").fetchall()
    )
    # Objects of sessions without mappings are bounded — the 12th insert ran
    # the trim: the newest 2, and the live one kept on top of them — and an
    # object flood never trims a Responses row.
    assert kinds == {"object": 3, "response": 1}
    assert manager.lookup_response_session("file-live") == "user:ada:live"
    assert manager.lookup_response_session("resp_mine") == "user:ada:default"


def test_the_trim_runs_every_prune_every_inserts_of_its_own_kind(
    store: tuple[Any, sqlite3.Connection], tiny_caps: None
) -> None:
    """Amortized: a kind's rows are trimmed on every 4th insert OF THAT KIND
    (the tiny cadence), never sooner — the trim reads the whole map."""
    manager, conn = store
    for n in range(6):  # beyond both bounds, written around the manager
        conn.execute(
            f"INSERT INTO {_table(conn)} (response_id, session_id, created_at, kind)"
            " VALUES (?, 's', ?, ?)",
            (f"old-{n}", OLD, "object" if n % 2 else "response"),
        )
    for n in range(3):
        manager.record_object_session(f"file-{n}", "s")
        manager.record_response_session(f"resp_{n}", "s")

    def count(kind: str) -> int:
        row = conn.execute(f"SELECT COUNT(*) FROM {_table(conn)} WHERE kind = ?", (kind,))
        return int(row.fetchone()[0])

    assert (count("object"), count("response")) == (6, 6)  # three inserts each: no trim
    manager.record_object_session("file-3", "s")  # the 4th object insert trims objects only
    assert (count("object"), count("response")) == (2, 6)
    manager.record_response_session("resp_3", "s")
    assert (count("object"), count("response")) == (2, 3)


def test_a_row_takes_the_kind_it_was_last_recorded_as(
    store: tuple[Any, sqlite3.Connection],
) -> None:
    manager, conn = store
    manager.record_response_session("id-1", "s1")
    manager.record_object_session("id-1", "s2")
    assert conn.execute(
        f"SELECT session_id, kind FROM {_table(conn)} WHERE response_id = 'id-1'"
    ).fetchall() == [("s2", "object")]


def test_the_memory_manager_keeps_no_owner_records() -> None:
    manager = InMemoryVaultManager()
    manager.record_object_session("file-1", "s")
    assert manager.lookup_response_session("file-1") is None


# --- an existing map gains the column ------------------------------------------------

OLD_SQLITE_MAP = """
CREATE TABLE response_sessions (
  response_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
INSERT INTO response_sessions (response_id, session_id) VALUES ('resp_old', 'user:ada:default');
"""


def test_an_existing_sqlite_map_gains_the_kind_column(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "v.db") as conn:
        conn.executescript(OLD_SQLITE_MAP)
    manager = SqliteVaultManager(tmp_path / "v.db")
    try:
        assert manager.lookup_response_session("resp_old") == "user:ada:default"
        manager.record_object_session("file-1", "user:ada:default")
        rows = dict(manager._conn.execute("SELECT response_id, kind FROM response_sessions"))
        assert rows == {"resp_old": "response", "file-1": "object"}
    finally:
        manager.close()
    SqliteVaultManager(tmp_path / "v.db").close()  # reopening finds it in place


class _RacedConnection:
    """A connection whose ALTER loses a race to another opener (or fails)."""

    def __init__(self, conn: sqlite3.Connection, *, other_adds_it: bool) -> None:
        self._conn = conn
        self._other_adds_it = other_adds_it

    def execute(self, sql: str, *args: Any) -> Any:
        if sql.startswith("ALTER TABLE"):
            if self._other_adds_it:
                self._conn.execute(sql)
                raise sqlite3.OperationalError("duplicate column name: kind")
            raise sqlite3.OperationalError("attempt to write a readonly database")
        return self._conn.execute(sql, *args)


def test_a_concurrent_opener_adding_the_column_first_is_fine(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "v.db", isolation_level=None)
    conn.executescript(OLD_SQLITE_MAP)
    _ensure_kind_column(_RacedConnection(conn, other_adds_it=True))  # type: ignore[arg-type]
    columns = {row[1] for row in conn.execute("PRAGMA table_info(response_sessions)")}
    assert "kind" in columns
    fresh = sqlite3.connect(tmp_path / "w.db", isolation_level=None)
    fresh.executescript(OLD_SQLITE_MAP)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        _ensure_kind_column(_RacedConnection(fresh, other_adds_it=False))  # type: ignore[arg-type]


OLD_RDBMS_MAP = """
CREATE TABLE llm_redact_response_sessions (
  response_id VARCHAR(192) NOT NULL,
  session_id VARCHAR(128) NOT NULL,
  created_at VARCHAR(20) NOT NULL,
  PRIMARY KEY (response_id)
);
INSERT INTO llm_redact_response_sessions
  VALUES ('resp_old', 'user:ada:default', '2026-01-01T00:00:00Z');
"""


def test_an_existing_rdbms_map_gains_the_kind_column(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "v.db") as conn:
        conn.executescript(OLD_RDBMS_MAP)
    store = RdbmsStore(_dbapi_config(tmp_path / "v.db"), None)
    manager = RdbmsVaultManager(store)
    manager.record_object_session("file-1", "user:ada:default")
    with sqlite3.connect(tmp_path / "v.db") as conn:
        rows = dict(conn.execute("SELECT response_id, kind FROM llm_redact_response_sessions"))
    assert rows == {"resp_old": "response", "file-1": "object"}
    assert manager.lookup_response_session("resp_old") == "user:ada:default"
    store.close()


def test_a_database_user_that_may_not_alter_keeps_the_shared_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tiny_caps: None,
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    with sqlite3.connect(tmp_path / "postgresql.db") as conn:
        conn.executescript(OLD_RDBMS_MAP)
    driver.inject_fault("ALTER TABLE", sqlite3.OperationalError("permission denied"))
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        store = RdbmsStore(config, None)
    assert "could not add the kind column" in caplog.text and "OperationalError" in caplog.text
    assert "db.corp.example" not in caplog.text  # never the DSN
    manager = RdbmsVaultManager(store)
    manager.record_object_session("file-1", "s")  # still recorded, kind-less
    assert manager.lookup_response_session("file-1") == "s"
    with sqlite3.connect(tmp_path / "postgresql.db") as conn:
        _age(conn, "file-1")
    for n in range(7):  # with file-1, this store's 8th insert runs the trim
        manager.record_response_session(f"resp_{n}", "s")
    with sqlite3.connect(tmp_path / "postgresql.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM llm_redact_response_sessions").fetchone()[0]
    assert count == 3  # ONE shared bound, as before the kind existed ...
    assert manager.lookup_response_session("file-1") is None  # ... the oldest row went
    store.close()


def test_another_replica_adding_the_column_first_is_fine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    RdbmsStore(config, None).close()  # the column exists already
    # This opener's probe misses it; its ALTER then loses to the replica.
    driver.inject_fault("SELECT kind", sqlite3.OperationalError("no such column: kind"))
    driver.inject_fault("ALTER TABLE", sqlite3.OperationalError("duplicate column name: kind"))
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        store = RdbmsStore(config, None)
    assert "kind column" not in caplog.text
    store.record_object_session("file-1", "s")
    with sqlite3.connect(tmp_path / "postgresql.db") as conn:
        kind = conn.execute(
            "SELECT kind FROM llm_redact_response_sessions WHERE response_id = 'file-1'"
        ).fetchone()
    assert kind == ("object",)
    store.close()


# --- the proxy mirrors stored objects as owner records ---------------------------------


async def test_the_proxy_mirrors_stored_objects_as_owner_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _files_app(monkeypatch, tmp_path, OwnershipRouter())
    assert (await _post_batch(app)).status_code == 200
    conn = app.state.proxy.vault_manager._conn
    assert conn.execute(
        "SELECT kind FROM response_sessions WHERE response_id = 'batch_9'"
    ).fetchone() == ("object",)


async def test_a_manager_without_owner_records_mirrors_them_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    router = OwnershipRouter()
    app = _files_app(monkeypatch, tmp_path, router)
    manager = app.state.proxy.vault_manager
    monkeypatch.delattr(SqliteVaultManager, "record_object_session")
    assert (await _post_batch(app)).status_code == 200
    assert manager.lookup_response_session("batch_9") == "user:n1:main"
    assert manager._conn.execute(
        "SELECT kind FROM response_sessions WHERE response_id = 'batch_9'"
    ).fetchone() == ("response",)


async def test_a_memory_vault_proxy_still_reports_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    router = OwnershipRouter()
    import llm_redact.registry as registry_mod
    from llm_redact.registry import Registry

    reg = Registry()
    reg.build_session_router = lambda config, **kw: router
    monkeypatch.setattr(registry_mod, "_registry", reg)
    app = create_app(
        Config(providers={"openai": ProviderConfig("https://upstream.test")}, vault=VaultConfig()),
        upstream_transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"id": "batch_9"})
        ),
    )
    assert (await _post_batch(app)).status_code == 200
    assert router.objects == [("batch_9", "user:n1:main")]


# --- the shared bound is surfaced, value-free ---------------------------------------


def _no_alter_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, alter_refused: bool
) -> tuple[VaultConfig, RdbmsStore]:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    with sqlite3.connect(tmp_path / "postgresql.db") as conn:
        conn.executescript(OLD_RDBMS_MAP)
    if alter_refused:
        driver.inject_fault("ALTER TABLE", sqlite3.OperationalError("permission denied"))
    return config, RdbmsStore(config, None)


@pytest.mark.parametrize("alter_refused", [True, False], ids=["alter-refused", "column-added"])
def test_the_store_and_manager_say_whether_the_bound_is_shared(
    alter_refused: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store = _no_alter_store(tmp_path, monkeypatch, alter_refused=alter_refused)
    assert store.owner_bound_shared is alter_refused
    assert RdbmsVaultManager(store).owner_bound_shared is alter_refused
    store.close()


class _OlderManager:
    """An RDBMS manager from before the member (an older llm-redact-pro's)."""

    def __init__(self, inner: RdbmsVaultManager) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name == "owner_bound_shared":
            raise AttributeError(name)
        return getattr(self._inner, name)


@pytest.mark.parametrize(
    ("alter_refused", "older", "shared"),
    [(True, False, True), (False, False, False), (True, True, False)],
    ids=["alter-refused", "column-added", "manager-without-the-member"],
)
async def test_status_and_its_posture_surface_the_shared_bound(
    alter_refused: bool,
    older: bool,
    shared: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    import llm_redact.registry as registry_mod
    from llm_redact.cli import _print_posture
    from llm_redact.registry import Registry

    config, store = _no_alter_store(tmp_path, monkeypatch, alter_refused=alter_refused)
    manager = RdbmsVaultManager(store)
    reg = Registry()
    reg.build_vault_manager = lambda cfg: _OlderManager(manager) if older else manager
    monkeypatch.setattr(registry_mod, "_registry", reg)
    app = create_app(Config(vault=config))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["vault"]["owner_bound_shared"] is shared
    assert "db.corp.example" not in json.dumps(status)  # value-free: never the DSN
    _print_posture(status)
    out = capsys.readouterr().out
    assert ("owner records share the Responses bound" in out) is shared
    store.close()


@pytest.mark.parametrize("shared", [True, False])
def test_doctor_warns_when_the_running_proxy_shares_the_bound(
    shared: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact import __version__, doctor_cli

    def get(url: str, **kwargs: Any) -> httpx.Response:
        body = {"version": __version__, "vault": {"backend": "postgresql"}}
        if shared:
            body["vault"]["owner_bound_shared"] = True
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    report = doctor_cli._Report(json_mode=True)
    doctor_cli._check_proxy(report, Config())
    vault_rows = [row for row in report.rows if row["area"] == "vault"]
    if shared:
        [row] = vault_rows
        assert row["level"] == "WARN"
        assert "could not add the kind column" in row["message"]
        assert "ALTER TABLE llm_redact_response_sessions" in row["message"]
    else:
        assert vault_rows == []
    assert report.rows[0]["area"] == "proxy" and report.rows[0]["level"] == "PASS"
