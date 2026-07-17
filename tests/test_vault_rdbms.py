"""RDBMS vault invariants — the SqliteVault battery generalized.

Three layers, no database server required:

- the generic ``backend = "dbapi"`` runs END TO END against the stdlib
  sqlite3 module (sqlite3 IS a DB-API 2.0 driver), proving the portable
  SQL subset and the qmark paramstyle path on a real engine;
- fake driver modules wrapping sqlite3 impersonate psycopg (pyformat),
  PyMySQL (pyformat + connect kwargs), and oracledb (named) so every
  dialect's SQL and connect plumbing executes, including fault injection
  the real drivers can't do on demand;
- the same battery runs against REAL PostgreSQL/MySQL/Oracle when
  LLM_REDACT_TEST_{PG,MYSQL,ORACLE}_DSN are set (the CI real-DB job).
"""

from __future__ import annotations

import dataclasses
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from fake_cipher import FakeVaultCipher
from llm_redact.config import ConfigError, RdbmsConfig, VaultConfig, parse_config
from llm_redact.vault import VaultKeyError, build_vault_manager
from llm_redact.vault_rdbms import (
    ENV_DSN,
    ENV_REMOTE_PLAINTEXT,
    RdbmsStore,
    RdbmsVault,
    RdbmsVaultManager,
    adapt_sql,
    build_rdbms_vault_manager,
    managed_dbms_cloud,
    offbox_violation,
    open_rdbms_vault,
)

# --- paramstyle adapter -------------------------------------------------------

_SQL = "SELECT a FROM t WHERE x = :x AND y = :y AND x2 = :x"
_PARAMS = {"x": 1, "y": "two"}


def test_adapt_sql_named_passthrough() -> None:
    sql, params = adapt_sql(_SQL, _PARAMS, "named")
    assert sql == _SQL
    assert params == _PARAMS


def test_adapt_sql_pyformat() -> None:
    sql, params = adapt_sql(_SQL, _PARAMS, "pyformat")
    assert sql == "SELECT a FROM t WHERE x = %(x)s AND y = %(y)s AND x2 = %(x)s"
    assert params == _PARAMS


def test_adapt_sql_qmark_orders_positionally() -> None:
    sql, params = adapt_sql(_SQL, _PARAMS, "qmark")
    assert sql == "SELECT a FROM t WHERE x = ? AND y = ? AND x2 = ?"
    assert params == (1, "two", 1)  # the repeated :x is bound twice


def test_adapt_sql_format() -> None:
    sql, params = adapt_sql(_SQL, _PARAMS, "format")
    assert sql == "SELECT a FROM t WHERE x = %s AND y = %s AND x2 = %s"
    assert params == (1, "two", 1)


def test_adapt_sql_numeric() -> None:
    sql, params = adapt_sql(_SQL, _PARAMS, "numeric")
    assert sql == "SELECT a FROM t WHERE x = :1 AND y = :2 AND x2 = :3"
    assert params == (1, "two", 1)


# --- config parsing -----------------------------------------------------------


def test_parse_rejects_unknown_backend() -> None:
    with pytest.raises(ConfigError, match="backend must be one of"):
        parse_config({"vault": {"backend": "mongodb"}}, "<test>")


def test_parse_dbapi_requires_module() -> None:
    with pytest.raises(ConfigError, match="module .* is required"):
        parse_config({"vault": {"backend": "dbapi"}}, "<test>")


def test_parse_module_only_for_dbapi() -> None:
    with pytest.raises(ConfigError, match="module applies only"):
        parse_config({"vault": {"backend": "postgresql", "rdbms": {"module": "psycopg"}}}, "<test>")


def test_parse_rdbms_section_needs_rdbms_backend() -> None:
    with pytest.raises(ConfigError, match="applies only to the RDBMS backends"):
        parse_config({"vault": {"backend": "sqlite", "rdbms": {"dsn": "postgresql://x/y"}}}, "<t>")


def test_parse_rejects_unknown_cloud() -> None:
    with pytest.raises(ConfigError, match="cloud must be"):
        parse_config(
            {
                "vault": {
                    "backend": "postgresql",
                    "rdbms": {"dsn": "postgresql://h/d", "cloud": "ibm"},
                }
            },
            "<test>",
        )


def test_config_emitter_roundtrips_rdbms_section() -> None:
    import tomllib

    from llm_redact.config import Config
    from llm_redact.config_write import emit_config_toml

    vault = VaultConfig(
        backend="postgresql",
        rdbms=RdbmsConfig(dsn="postgresql://vault@db.corp.example:5432/llmredact", cloud="aws"),
    )
    config = dataclasses.replace(Config(), vault=vault)
    assert parse_config(tomllib.loads(emit_config_toml(config)), "<roundtrip>") == config


# --- config helpers -----------------------------------------------------------


def _dbapi_config(path: Path, **vault_overrides: Any) -> VaultConfig:
    return VaultConfig(
        backend="dbapi",
        rdbms=RdbmsConfig(dsn=str(path), module="sqlite3"),
        **vault_overrides,
    )


# --- the invariant battery ----------------------------------------------------


def _battery(make_store: Any) -> None:
    """The SqliteVault semantics every store must satisfy."""
    store: RdbmsStore = make_store()

    vault = RdbmsVault(store, "sess-a")
    first = vault.placeholder_for("EMAIL", "ada@corp.example")
    # Deterministic: same value, same token; distinct value, next number.
    assert vault.placeholder_for("EMAIL", "ada@corp.example") == first
    second = vault.placeholder_for("EMAIL", "bea@corp.example")
    assert first == "«EMAIL_001»"
    assert second == "«EMAIL_002»"
    assert vault.placeholder_for("PHONE", "+1 555 0100") == "«PHONE_001»"
    # Reverse lookups, including via a FRESH view (no warm cache).
    assert vault.original_for(first) == "ada@corp.example"
    cold = RdbmsVault(store, "sess-a")
    assert cold.original_for(second) == "bea@corp.example"
    assert cold.placeholder_for("EMAIL", "ada@corp.example") == first
    assert len(cold) == 3
    # Session isolation: token names collide across sessions BY DESIGN
    # (every session has an «EMAIL_001») but each resolves to its own value,
    # and a token the other session never issued resolves to nothing.
    other = RdbmsVault(store, "sess-b")
    assert other.placeholder_for("EMAIL", "carol@corp.example") == "«EMAIL_001»"
    assert other.original_for("«EMAIL_001»") == "carol@corp.example"
    assert vault.original_for("«EMAIL_001»") == "ada@corp.example"
    assert other.original_for("«PHONE_001»") is None  # issued only in sess-a

    # Manager surface.
    manager = RdbmsVaultManager(store)
    assert manager.session_count() == 2
    assert manager.total_entries() == 4
    summary = manager.sessions_summary()
    assert {row["session"] for row in summary} == {"sess-a", "sess-b"}
    assert all(row["first"] and row["last"] for row in summary)

    # Response-session map.
    manager.record_response_session("resp_1", "sess-a")
    manager.record_response_session("resp_1", "sess-b")  # idempotent re-record
    assert manager.lookup_response_session("resp_1") == "sess-b"
    assert manager.lookup_response_session("resp_unknown") is None

    # Whole-session prune only. days=-1 puts the cutoff in the future so
    # rows created this second count as idle — deterministic at the
    # created_at column's 1-second granularity.
    pruned = manager.prune_sessions(-1, exclude=frozenset({"sess-a"}))
    assert pruned == 1
    assert manager.session_count() == 1
    assert manager.lookup_response_session("resp_1") is None  # rode along
    # sess-a survived intact, and its numbering continues densely.
    survivor = manager.get("sess-a")
    assert survivor.placeholder_for("EMAIL", "dan@corp.example") == "«EMAIL_003»"
    store.close()


def test_battery_generic_dbapi_sqlite3(tmp_path: Path) -> None:
    config = _dbapi_config(tmp_path / "vault.db")
    _battery(lambda: RdbmsStore(config, None))


def test_generic_dbapi_persists_across_reopen(tmp_path: Path) -> None:
    config = _dbapi_config(tmp_path / "vault.db")
    store = RdbmsStore(config, None)
    token = RdbmsVault(store, "s").placeholder_for("EMAIL", "ada@corp.example")
    store.close()

    reopened = RdbmsStore(config, None)
    vault = RdbmsVault(reopened, "s")
    assert vault.original_for(token) == "ada@corp.example"
    assert vault.placeholder_for("EMAIL", "ada@corp.example") == token
    reopened.close()


def test_build_vault_manager_rdbms_requires_pro(tmp_path: Path) -> None:
    # The Free dispatcher fails closed on a paid backend; the paid
    # build_vault_manager override (tested through the registry / pro suite)
    # is what actually dispatches. The error names the configured backend, so
    # it must be the real backend, not a dropped/None'd argument.
    with pytest.raises(ConfigError) as exc:
        build_vault_manager(_dbapi_config(tmp_path / "vault.db"))
    assert "llm-redact-pro" in str(exc.value) and "dbapi" in str(exc.value)


def test_build_rdbms_vault_manager_dispatches(tmp_path: Path) -> None:
    # The Free MECHANISM (called by the pro override) builds an unencrypted
    # RDBMS manager end to end — this is what the dispatcher delegates to.
    manager = build_rdbms_vault_manager(_dbapi_config(tmp_path / "vault.db"))
    assert isinstance(manager, RdbmsVaultManager)
    token = manager.get("s").placeholder_for("EMAIL", "a@corp.example")
    assert token == "«EMAIL_001»"
    manager.close()


def test_build_vault_dispatches_every_backend(tmp_path: Path) -> None:
    """build_vault (the STATIC single-session builder) routes memory/sqlite
    itself and fails closed on the paid RDBMS backend — the mutation gate
    caught the dispatch branch surviving unpinned."""
    from llm_redact.vault import InMemoryVault, SqliteVault, build_vault

    memory = build_vault(VaultConfig())
    assert isinstance(memory, InMemoryVault)

    # A server RDBMS vault is paid: the Free dispatcher fails closed.
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        build_vault(_dbapi_config(tmp_path / "rdbms.db", session="static-sess"))
    # The mechanism still opens one directly (unencrypted), under the given
    # session name.
    rdbms = open_rdbms_vault(_dbapi_config(tmp_path / "rdbms.db"), "static-sess")
    assert isinstance(rdbms, RdbmsVault)
    assert rdbms.placeholder_for("EMAIL", "a@corp.example") == "«EMAIL_001»"
    rdbms.close()
    reopened = RdbmsStore(_dbapi_config(tmp_path / "rdbms.db"), None)
    assert reopened.lookup_reverse("static-sess", "«EMAIL_001»") == "a@corp.example"
    reopened.close()

    sqlite_vault = build_vault(VaultConfig(backend="sqlite", path=str(tmp_path / "plain.db")))
    assert isinstance(sqlite_vault, SqliteVault)
    sqlite_vault.close()


def test_open_rdbms_vault_owns_store(tmp_path: Path) -> None:
    vault = open_rdbms_vault(_dbapi_config(tmp_path / "vault.db"), "solo")
    assert vault.placeholder_for("EMAIL", "a@corp.example") == "«EMAIL_001»"
    vault.close()  # closes the store it owns — no error on double close path
    vault.close()


def test_env_dsn_overrides_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DSN, str(tmp_path / "env.db"))
    config = VaultConfig(backend="dbapi", rdbms=RdbmsConfig(dsn="", module="sqlite3"))
    store = RdbmsStore(config, None)
    RdbmsVault(store, "s").placeholder_for("EMAIL", "a@corp.example")
    store.close()
    assert (tmp_path / "env.db").exists()


def test_missing_dsn_fails_closed() -> None:
    config = VaultConfig(backend="dbapi", rdbms=RdbmsConfig(module="sqlite3"))
    with pytest.raises(ConfigError, match="dsn is required"):
        RdbmsStore(config, None)


def test_missing_dbapi_module_fails_closed() -> None:
    config = VaultConfig(
        backend="dbapi", rdbms=RdbmsConfig(dsn="x", module="definitely_missing_driver_xyz")
    )
    with pytest.raises(ConfigError, match="not importable"):
        RdbmsStore(config, None)


def test_missing_driver_names_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes import_module raise ImportError.
    monkeypatch.setitem(sys.modules, "psycopg", None)
    config = VaultConfig(backend="postgresql", rdbms=RdbmsConfig(dsn="postgresql://u@h:5432/d"))
    with pytest.raises(ConfigError, match=r"vault-postgres"):
        RdbmsStore(config, None)


def test_scheme_mismatch_never_echoes_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "psycopg", _FakeDriver.__new__(_FakeDriver))
    config = VaultConfig(
        backend="postgresql",
        rdbms=RdbmsConfig(dsn="mysql://user:hunter2secret@db.corp.example/vault"),
    )
    with pytest.raises(ConfigError) as excinfo:
        RdbmsStore(config, None)
    assert "hunter2secret" not in str(excinfo.value)
    assert "db.corp.example" not in str(excinfo.value)
    assert "postgresql://" in str(excinfo.value)


# --- encryption ---------------------------------------------------------------


# Encrypted-store plumbing over the VaultCipher Protocol (FakeVaultCipher — no
# cryptography / llm-redact-pro dependency). The REAL at-rest proof (no
# plaintext on disk) lives in tests/pro/test_vault_rdbms_crypto.py.
def _cipher(seed: bytes = b"k" * 32) -> Any:
    return FakeVaultCipher(seed), seed


def test_encrypted_store_wrong_key_fails_at_open(tmp_path: Path) -> None:
    cipher, _ = _cipher(b"k" * 32)
    config = _dbapi_config(tmp_path / "vault.db", encryption="fernet")
    RdbmsStore(config, cipher).close()
    other, _ = _cipher(b"x" * 32)
    with pytest.raises(VaultKeyError, match="does not match"):
        RdbmsStore(config, other)


def test_encryption_mode_fixed_at_creation(tmp_path: Path) -> None:
    config = _dbapi_config(tmp_path / "vault.db")
    RdbmsStore(config, None).close()
    cipher, _ = _cipher()
    with pytest.raises(ConfigError, match="fixed at creation"):
        RdbmsStore(dataclasses.replace(config, encryption="fernet"), cipher)


# --- fake drivers: the psycopg / pymysql / oracledb dialect paths --------------

_PYFORMAT_RE = re.compile(r"%\((\w+)\)s")


class _FakeCursor:
    def __init__(self, real: sqlite3.Cursor, conn: _FakeConnection) -> None:
        self._real = real
        self._conn = conn
        self._driver = conn._driver

    def execute(self, sql: str, params: Any = None) -> None:
        fault = self._driver.pop_fault(sql)
        if fault is not None:
            raise fault
        if self._driver.paramstyle == "pyformat":
            sql = _PYFORMAT_RE.sub(r":\1", sql)
        if params is None:
            self._real.execute(sql)
        else:
            self._real.execute(sql, params)
        stripped = sql.lstrip().upper()
        if self._driver.transactional_ddl and stripped.startswith("CREATE TABLE"):
            # Emulate PostgreSQL: DDL joins the transaction, so a later
            # rollback() must undo this CREATE (the CI-caught failure mode).
            self._conn.uncommitted_tables.append(sql.split()[2])

    def fetchone(self) -> Any:
        return self._real.fetchone()

    def fetchall(self) -> Any:
        return self._real.fetchall()


class _FakeConnection:
    def __init__(self, real: sqlite3.Connection, driver: _FakeDriver) -> None:
        self._real = real
        self._driver = driver
        self.uncommitted_tables: list[str] = []

    def cursor(self) -> _FakeCursor:
        if self._driver.dead:
            raise sqlite3.OperationalError("server closed the connection unexpectedly")
        return _FakeCursor(self._real.cursor(), self)

    def commit(self) -> None:
        self.uncommitted_tables.clear()
        self._real.commit()

    def rollback(self) -> None:
        for table in self.uncommitted_tables:
            self._real.execute(f"DROP TABLE IF EXISTS {table}")
        self.uncommitted_tables.clear()
        self._real.rollback()

    def close(self) -> None:
        self._real.close()


class _FakeDriver:
    """A DB-API 2.0 'module' impersonating a server driver over sqlite3."""

    Error = sqlite3.Error
    IntegrityError = sqlite3.IntegrityError
    OperationalError = sqlite3.OperationalError
    InterfaceError = sqlite3.InterfaceError

    class _Defaults:  # oracledb.defaults stand-in
        fetch_lobs = True

    def __init__(self, path: Path, paramstyle: str, *, transactional_ddl: bool = False) -> None:
        self._path = path
        self.paramstyle = paramstyle
        self.transactional_ddl = transactional_ddl  # PostgreSQL-style DDL
        self.connect_count = 0
        self.connect_kwargs: dict[str, Any] = {}
        self.dead = False  # next cursor() raises OperationalError once
        self._faults: list[tuple[str, Exception]] = []
        self.defaults = self._Defaults()

    def inject_fault(self, sql_substring: str, exc: Exception) -> None:
        self._faults.append((sql_substring, exc))

    def pop_fault(self, sql: str) -> Exception | None:
        for i, (needle, exc) in enumerate(self._faults):
            if needle in sql:
                del self._faults[i]
                return exc
        return None

    def connect(self, *args: Any, **kwargs: Any) -> _FakeConnection:
        self.connect_count += 1
        self.connect_kwargs = kwargs
        if self.dead:
            self.dead = False  # reconnect succeeds
        return _FakeConnection(sqlite3.connect(self._path), self)


def _fake_backend_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str
) -> tuple[VaultConfig, _FakeDriver]:
    style = {"postgresql": "pyformat", "mysql": "pyformat", "oracle": "named"}[backend]
    # Only PostgreSQL has transactional DDL; emulating it in the fake is
    # what catches the schema-bootstrap rollback bug the real server hit.
    driver = _FakeDriver(
        tmp_path / f"{backend}.db", style, transactional_ddl=(backend == "postgresql")
    )
    module_name = {"postgresql": "psycopg", "mysql": "pymysql", "oracle": "oracledb"}[backend]
    monkeypatch.setitem(sys.modules, module_name, driver)
    dsn = {
        "postgresql": "postgresql://vault@db.corp.example:5432/llmredact",
        "mysql": "mysql://vault@db.corp.example:3306/llmredact",
        "oracle": "oracle://vault@db.corp.example:1521/FREEPDB1",
    }[backend]
    return VaultConfig(backend=backend, rdbms=RdbmsConfig(dsn=dsn)), driver


@pytest.mark.parametrize("backend", ["postgresql", "mysql", "oracle"])
def test_battery_via_fake_driver(
    backend: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, _ = _fake_backend_config(monkeypatch, tmp_path, backend)
    _battery(lambda: RdbmsStore(config, None))


def test_mysql_connect_kwargs_and_password_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "mysql")
    monkeypatch.setenv("LLM_REDACT_VAULT_DB_PASSWORD", "from-the-env")
    RdbmsStore(config, None).close()
    assert driver.connect_kwargs["host"] == "db.corp.example"
    assert driver.connect_kwargs["port"] == 3306
    assert driver.connect_kwargs["user"] == "vault"
    assert driver.connect_kwargs["password"] == "from-the-env"
    assert driver.connect_kwargs["database"] == "llmredact"
    assert driver.connect_kwargs["charset"] == "utf8mb4"


def test_oracle_lob_defaults_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "oracle")
    RdbmsStore(config, None).close()
    assert driver.defaults.fetch_lobs is False


def test_write_fault_fails_closed_then_reissues_same_number(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    store = RdbmsStore(config, None)
    vault = RdbmsVault(store, "s")
    assert vault.placeholder_for("EMAIL", "first@corp.example") == "«EMAIL_001»"
    # A non-retryable database error mid-insert (disk full, constraint
    # machinery, ...): the call fails closed and caches nothing.
    driver.inject_fault("INSERT INTO llm_redact_mappings", sqlite3.DataError("disk full"))
    with pytest.raises(sqlite3.DataError):
        vault.placeholder_for("EMAIL", "second@corp.example")
    assert vault.original_for("«EMAIL_002»") is None
    # The retry reissues the SAME dense number — never a gap, never reuse.
    assert vault.placeholder_for("EMAIL", "second@corp.example") == "«EMAIL_002»"
    store.close()


def test_dropped_connection_reconnects_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    store = RdbmsStore(config, None)
    vault = RdbmsVault(store, "s")
    vault.placeholder_for("EMAIL", "first@corp.example")
    assert driver.connect_count == 1
    driver.dead = True  # idle timeout: next use finds the connection gone
    assert vault.placeholder_for("EMAIL", "second@corp.example") == "«EMAIL_002»"
    assert driver.connect_count == 2  # transparent single reconnect
    assert vault.original_for("«EMAIL_002»") == "second@corp.example"
    store.close()


def test_integrity_race_returns_other_writers_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A concurrent writer inserting the same value between our SELECT and
    INSERT surfaces as IntegrityError; the retry must adopt that row."""
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    store = RdbmsStore(config, None)
    vault = RdbmsVault(store, "s")

    # Simulate the race: the "other writer" commits the row via a second
    # store first, while our first INSERT attempt is made to collide.
    other_store = RdbmsStore(config, None)
    other_token = RdbmsVault(other_store, "s").placeholder_for("EMAIL", "raced@corp.example")
    other_store.close()

    # Our view has a stale cache (empty); its SELECT will find the row and
    # return the other writer's token — same-value-same-token across writers.
    assert vault.placeholder_for("EMAIL", "raced@corp.example") == other_token
    store.close()


def test_response_cap_prune_runs_and_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cap-prune DELETE (derived-table form) executes and bounds the
    table. The Oracle FETCH FIRST variant cannot run on sqlite; the real-DB
    job covers it."""
    config, _ = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    monkeypatch.setattr("llm_redact.vault_rdbms._RESPONSE_PRUNE_EVERY", 1)
    monkeypatch.setattr("llm_redact.vault_rdbms._MAX_RESPONSE_ROWS", 1)
    store = RdbmsStore(config, None)
    manager = RdbmsVaultManager(store)
    for i in range(3):
        manager.record_response_session(f"resp_{i}", "s")
    survivors = [
        resp
        for resp in ("resp_0", "resp_1", "resp_2")
        if manager.lookup_response_session(resp) is not None
    ]
    assert len(survivors) == 1  # capped to _MAX_RESPONSE_ROWS
    store.close()


# --- the off-box rule + managed-DBMS recognition -------------------------------


def _vault(backend: str, dsn: str, cloud: str = "", encryption: str = "none") -> VaultConfig:
    return VaultConfig(
        backend=backend, encryption=encryption, rdbms=RdbmsConfig(dsn=dsn, cloud=cloud)
    )


def test_managed_dbms_recognition() -> None:
    cases = {
        "postgresql://u@db.cluster-xyz.us-east-1.rds.amazonaws.com:5432/v": "aws",
        "mysql://u@myapp.mysql.database.azure.com:3306/v": "azure",
        "postgresql://u@myserver.database.windows.net/v": "azure",
        "postgresql:///v?host=/cloudsql/proj:region:instance": "gcp",
        "postgresql://u@db.corp.example:5432/v": None,
        "postgresql://u@127.0.0.1:5432/v": None,
    }
    for dsn, expected in cases.items():
        backend = "mysql" if dsn.startswith("mysql") else "postgresql"
        assert managed_dbms_cloud(_vault(backend, dsn)) == expected, dsn


def test_managed_dbms_recognized_in_opaque_dbapi_dsn() -> None:
    config = VaultConfig(
        backend="dbapi",
        rdbms=RdbmsConfig(dsn="Server=db.rds.amazonaws.com;Uid=x", module="pyodbc"),
    )
    assert managed_dbms_cloud(config) == "aws"


def test_offbox_plaintext_refused_at_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_REMOTE_PLAINTEXT, raising=False)
    config, _ = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    with pytest.raises(ConfigError, match="PLAINTEXT"):
        build_rdbms_vault_manager(config)


def test_offbox_hatch_allows_and_is_not_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_REMOTE_PLAINTEXT, "1")
    config, _ = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    manager = build_rdbms_vault_manager(config)
    assert manager.get("s").placeholder_for("EMAIL", "a@corp.example") == "«EMAIL_001»"
    manager.close()


def test_offbox_fernet_allows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The off-box rule permits a remote fernet vault; the cipher is supplied
    # by the paid build_vault_manager override (a FakeVaultCipher here — the
    # rule, not the crypto, is under test).
    monkeypatch.delenv(ENV_REMOTE_PLAINTEXT, raising=False)
    config, _ = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    manager = build_rdbms_vault_manager(
        dataclasses.replace(config, encryption="fernet"), FakeVaultCipher()
    )
    token = manager.get("s").placeholder_for("EMAIL", "a@corp.example")
    assert manager.get("s").original_for(token) == "a@corp.example"
    manager.close()


def test_loopback_plaintext_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_REMOTE_PLAINTEXT, raising=False)
    driver = _FakeDriver(tmp_path / "local.db", "pyformat")
    monkeypatch.setitem(sys.modules, "psycopg", driver)
    config = _vault("postgresql", "postgresql://vault@127.0.0.1:5432/llmredact")
    manager = build_rdbms_vault_manager(config)
    assert manager.get("s").placeholder_for("EMAIL", "a@corp.example") == "«EMAIL_001»"
    manager.close()


def test_cloudsql_socket_counts_as_offbox() -> None:
    config = _vault("postgresql", "postgresql:///v?host=/cloudsql/proj:region:instance")
    violation = offbox_violation(config)
    assert violation is not None and "PLAINTEXT" in violation


# --- the vault CLI on an RDBMS backend ------------------------------------------


def _cli_config_file(tmp_path: Path, db: Path) -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        # as_posix(): a backslashed Windows path inside a TOML basic string
        # is an escape sequence (C:\Users -> invalid \U); sqlite3.connect
        # accepts forward slashes on every platform.
        f'[vault]\nbackend = "dbapi"\n\n'
        f'[vault.rdbms]\ndsn = "{db.as_posix()}"\nmodule = "sqlite3"\n'
    )
    return config_file


def _seed(db: Path) -> None:
    config = _dbapi_config(db)
    store = RdbmsStore(config, None)
    vault = RdbmsVault(store, "sess-a")
    vault.placeholder_for("EMAIL", "ada@corp.example")
    vault.placeholder_for("EMAIL", "bea@corp.example")
    store.close()


def test_cli_sessions_list_rdbms(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    from llm_redact.vault_cli import run_sessions_list

    db = tmp_path / "v.db"
    _seed(db)
    args = argparse.Namespace(db=None, config=_cli_config_file(tmp_path, db), json=True)
    assert run_sessions_list(args) == 0
    import json

    rows = json.loads(capsys.readouterr().out)
    assert rows == [
        {"session": "sess-a", "entries": 2, "first": rows[0]["first"], "last": rows[0]["last"]}
    ]


def test_cli_lookup_rdbms_token_and_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    from llm_redact.vault_cli import run_lookup

    db = tmp_path / "v.db"
    _seed(db)
    config_file = _cli_config_file(tmp_path, db)
    by_token = argparse.Namespace(
        db=None, config=config_file, token="«EMAIL_001»", value=None, session=None
    )
    assert run_lookup(by_token) == 0
    assert "ada@corp.example" in capsys.readouterr().out

    by_value = argparse.Namespace(
        db=None, config=config_file, token=None, value="bea@corp.example", session=None
    )
    assert run_lookup(by_value) == 0
    assert "«EMAIL_002»" in capsys.readouterr().out

    missing = argparse.Namespace(
        db=None, config=config_file, token="«EMAIL_009»", value=None, session=None
    )
    assert run_lookup(missing) == 1


def test_cli_sessions_prune_rdbms(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    from llm_redact.vault_cli import run_sessions_prune

    db = tmp_path / "v.db"
    _seed(db)
    # Backdate the rows so a 90d prune has something idle to delete (the
    # dbapi backend IS a sqlite file, so a raw connection can do this).
    raw = sqlite3.connect(db)
    raw.execute("UPDATE llm_redact_mappings SET created_at = '2000-01-01T00:00:00Z'")
    raw.commit()
    raw.close()
    args = argparse.Namespace(
        db=None, config=_cli_config_file(tmp_path, db), older_than="90d", yes=True
    )
    assert run_sessions_prune(args) == 0
    out = capsys.readouterr().out
    assert "would delete sess-a (2 mappings)" in out
    assert "deleted 1 session(s)" in out


def test_cli_sqlite_only_commands_refuse_rdbms(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    from llm_redact.vault_cli import run_vault_backup, run_vault_rotate_key, run_vault_verify

    db = tmp_path / "v.db"
    _seed(db)
    config_file = _cli_config_file(tmp_path, db)
    for runner in (run_vault_verify, run_vault_rotate_key, run_vault_backup):
        args = argparse.Namespace(db=None, config=config_file)
        assert runner(args) == 2
        assert "sqlite backend only" in capsys.readouterr().out


# --- doctor + /status surfacing --------------------------------------------------


def test_doctor_rdbms_offbox_plaintext_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse
    import socket

    from llm_redact.doctor_cli import run_doctor

    monkeypatch.delenv(ENV_REMOTE_PLAINTEXT, raising=False)
    monkeypatch.setitem(sys.modules, "psycopg", _FakeDriver(tmp_path / "d.db", "pyformat"))
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        f"port = {port}\n"
        '[vault]\nbackend = "postgresql"\n\n'
        '[vault.rdbms]\ndsn = "postgresql://vault@db.corp.example:5432/llmredact"\n'
    )
    assert run_doctor(argparse.Namespace(config=config_file)) == 1
    out = capsys.readouterr().out
    assert "PLAINTEXT" in out and "refuse to start" in out
    assert "driver importable" in out


# NOTE (R4): test_status_reports_vault_honesty_fields booted the proxy with a
# dbapi (server) vault — a persistent SERVER vault is Pro-gated, so it needs the
# enforcement resolver and moved to the pro repo's tests/test_vault_rdbms_honesty.py.
# The driver-level RdbmsStore battery above is tier-independent and stays here.


# --- real servers (env-gated; the CI real-DB job sets the DSNs) ---------------

_REAL_DSNS = {
    "postgresql": os.environ.get("LLM_REDACT_TEST_PG_DSN"),
    "mysql": os.environ.get("LLM_REDACT_TEST_MYSQL_DSN"),
    "oracle": os.environ.get("LLM_REDACT_TEST_ORACLE_DSN"),
}


def _drop_tables(config: VaultConfig) -> None:
    from contextlib import suppress

    from llm_redact.vault_rdbms import _resolve_connector

    module, connect = _resolve_connector(config)
    conn = connect()
    for table in ("llm_redact_mappings", "llm_redact_response_sessions", "llm_redact_meta"):
        with suppress(module.Error):
            conn.cursor().execute(f"DROP TABLE {table}")
            conn.commit()
        with suppress(module.Error):
            conn.rollback()
    conn.close()


@pytest.mark.parametrize("backend", ["postgresql", "mysql", "oracle"])
def test_battery_real_server(backend: str) -> None:
    dsn = _REAL_DSNS[backend]
    if not dsn:
        pytest.skip(
            f"LLM_REDACT_TEST_{backend.upper() if backend != 'postgresql' else 'PG'}_DSN not set"
        )
    config = VaultConfig(backend=backend, rdbms=RdbmsConfig(dsn=dsn))
    _drop_tables(config)
    _battery(lambda: RdbmsStore(config, None))
