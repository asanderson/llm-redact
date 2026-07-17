"""Persistent vault on a server RDBMS through DB-API 2.0 (llm-redact-pro).

One store speaks PostgreSQL (psycopg), MySQL (PyMySQL), Oracle (oracledb),
or any DB-API 2.0 module the operator names (``backend = "dbapi"``), via a
paramstyle adapter plus a two-entry dialect table — the vault SQL itself is
a portable subset: no upserts and no SELECT FOR UPDATE. Dense counter
allocation relies on the UNIQUE (session, type, n) constraint plus a
bounded retry, which is the SqliteVault recipe generalized to any engine.

Semantics mirror SqliteVault and are pinned by the same invariant battery
(tests/test_vault_rdbms.py): deterministic (session, type, value) → token,
dense per-(session, type) counters (n is MAX(n)+1 read fresh inside the
transaction, so a rolled-back allocation reissues the SAME number — reuse
is the danger, gaps would come from counters tables, which is why there is
none), caches written only after COMMIT, any write fault rolls back and
fails closed, whole-session prune only.

Two deliberate deltas from the sqlite schema:

- ``original_key`` (64-hex) replaces the raw value in the primary key: it
  is the cipher's HMAC index when encrypted, SHA-256 of the value when not.
  MySQL and Oracle cannot index unbounded text, and the fixed-width key
  makes the plaintext and encrypted layouts one schema.
- The encryption mode is FIXED at schema creation (stored in
  ``llm_redact_meta``); there is no encrypt-in-place migration — old row
  versions linger in server-side MVCC storage where no VACUUM discipline of
  ours could honestly scrub them, so the at-rest claim is only made for
  schemas born encrypted. Changing the mode means a fresh database/schema.

Unlike sqlite, the database may live OFF this machine. With ``encryption =
"fernet"`` only the HMAC index and Fernet ciphertext ever leave the box;
without it plaintext does, so startup refuses a non-local DSN without
fernet (the off-box rule, enforced in the build wiring). DSNs may embed
credentials and are NEVER logged or echoed in error messages.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import os
import re
from collections import OrderedDict
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from llm_redact.placeholders import format_placeholder
from llm_redact.vault import (
    _MAX_RESPONSE_ROWS,
    _RESPONSE_PRUNE_EVERY,
    Vault,
    VaultKeyError,
)

if TYPE_CHECKING:
    from llm_redact.config import VaultConfig
    from llm_redact.plugin_api import VaultCipher

ENV_DSN = "LLM_REDACT_VAULT_DSN"
# The documented hatch for the off-box rule below: set to 1 to run a
# PLAINTEXT vault against a remote database anyway (e.g. a trusted
# same-host container network the hostname check cannot see). Surfaced in
# /status and doctor whenever active — an opt-out is never silent.
ENV_REMOTE_PLAINTEXT = "LLM_REDACT_VAULT_REMOTE_PLAINTEXT"

_DRIVER_MODULES = {"postgresql": "psycopg", "mysql": "pymysql", "oracle": "oracledb"}
_EXTRA_HINTS = {
    "postgresql": "pip install 'llm-redact-proxy[vault-postgres]'",
    "mysql": "pip install 'llm-redact-proxy[vault-mysql]'",
    "oracle": "pip install 'llm-redact-proxy[vault-oracle]'",
}
_SCHEMES = {
    "postgresql": ("postgresql", "postgres"),
    "mysql": ("mysql",),
    "oracle": ("oracle",),
}

# Unbounded-text column type per backend; every other column is a bounded
# VARCHAR so the composite keys index everywhere (MySQL's InnoDB limit).
_LONG_TEXT = {"postgresql": "TEXT", "mysql": "LONGTEXT", "oracle": "CLOB", "dbapi": "TEXT"}

_ALLOCATION_ATTEMPTS = 3

_PARAM_RE = re.compile(r":([a-z_][a-z0-9_]*)")

_KNOWN_PARAMSTYLES = ("named", "pyformat", "qmark", "format", "numeric")


def adapt_sql(
    sql: str, params: dict[str, Any], paramstyle: str
) -> tuple[str, dict[str, Any] | tuple[Any, ...]]:
    """Convert canonical ``:name`` SQL to the driver's paramstyle.

    The vault SQL contains no string literals or casts, so a bare regex
    over ``:name`` sites is sound (pinned by tests for all five styles).
    """
    if paramstyle == "named":
        return sql, dict(params)
    if paramstyle == "pyformat":
        return _PARAM_RE.sub(r"%(\1)s", sql), dict(params)
    order: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        order.append(match.group(1))
        if paramstyle == "qmark":
            return "?"
        if paramstyle == "format":
            return "%s"
        return f":{len(order)}"  # numeric

    out = _PARAM_RE.sub(_sub, sql)
    return out, tuple(params[name] for name in order)


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ddl(backend: str) -> dict[str, str]:
    long_text = _LONG_TEXT[backend]
    return {
        "llm_redact_mappings": f"""CREATE TABLE llm_redact_mappings (
  session_id VARCHAR(128) NOT NULL,
  detector_type VARCHAR(64) NOT NULL,
  original_key VARCHAR(64) NOT NULL,
  original {long_text},
  original_ct {long_text},
  placeholder VARCHAR(96) NOT NULL,
  n INTEGER NOT NULL,
  created_at VARCHAR(20) NOT NULL,
  PRIMARY KEY (session_id, detector_type, original_key),
  CONSTRAINT llmr_uq_placeholder UNIQUE (session_id, placeholder),
  CONSTRAINT llmr_uq_n UNIQUE (session_id, detector_type, n)
)""",
        "llm_redact_response_sessions": """CREATE TABLE llm_redact_response_sessions (
  response_id VARCHAR(192) NOT NULL,
  session_id VARCHAR(128) NOT NULL,
  created_at VARCHAR(20) NOT NULL,
  PRIMARY KEY (response_id)
)""",
        "llm_redact_meta": """CREATE TABLE llm_redact_meta (
  meta_key VARCHAR(32) NOT NULL,
  meta_value VARCHAR(128) NOT NULL,
  PRIMARY KEY (meta_key)
)""",
    }


def resolve_dsn(config: VaultConfig) -> str:
    """Effective DSN: the env override wins so credentials can stay out of
    the config file entirely. Empty = ConfigError (fail closed)."""
    from llm_redact.config import ConfigError

    dsn = os.environ.get(ENV_DSN) or config.rdbms.dsn
    if not dsn:
        raise ConfigError(
            f"[vault.rdbms] dsn is required for backend = {config.backend!r}"
            f" (in the config file or the {ENV_DSN} env var)"
        )
    return dsn


# Managed-DBMS hostname suffixes (best-effort recognition; the deterministic
# channel is the [vault.rdbms] cloud declaration). ".database.azure.com"
# covers postgres/mysql/mariadb.database.azure.com flexible servers;
# ".database.windows.net" is Azure SQL. GCP Cloud SQL has no stable public
# suffix — its signature is the Auth Proxy socket path, matched on the DSN.
_MANAGED_SUFFIXES = (
    (".rds.amazonaws.com", "aws"),
    (".rds.amazonaws.com.cn", "aws"),
    (".database.windows.net", "azure"),
    (".database.azure.com", "azure"),
    (".database.chinacloudapi.cn", "azure"),
)


def _quiet_dsn(config: VaultConfig) -> str | None:
    """The resolved DSN, or None instead of the missing-DSN error (posture
    helpers must not raise where the build gate will)."""
    from llm_redact.config import ConfigError

    try:
        return resolve_dsn(config)
    except ConfigError:
        return None


def dsn_host(config: VaultConfig) -> str | None:
    """Hostname of the resolved DSN for the URL-form backends; None for
    backend = "dbapi" (an opaque connect string — locality unknowable) and
    for host-less DSNs (unix sockets, file paths)."""
    if config.backend == "dbapi":
        return None
    dsn = _quiet_dsn(config)
    if dsn is None:
        return None
    return urlsplit(dsn).hostname


def managed_dbms_cloud(config: VaultConfig) -> str | None:
    """The cloud whose managed-DBMS service the DSN points at, or None.

    Best-effort by design: recognition catches the common hostnames so a
    managed deployment cannot be configured *silently*; the declared
    [vault.rdbms] cloud is the deterministic channel.
    """
    from llm_redact.config import RDBMS_BACKENDS

    if config.backend not in RDBMS_BACKENDS:
        return None
    dsn = _quiet_dsn(config)
    if dsn is None:
        return None
    if "/cloudsql/" in dsn:
        return "gcp"  # the Cloud SQL Auth Proxy socket path
    haystack = (dsn_host(config) or "").lower()
    if not haystack and config.backend == "dbapi":
        haystack = dsn.lower()  # opaque string: substring scan is the best effort
    for suffix, cloud in _MANAGED_SUFFIXES:
        if haystack.endswith(suffix) or (config.backend == "dbapi" and suffix in haystack):
            return cloud
    return None


def offbox_violation(config: VaultConfig) -> str | None:
    """Error text when PLAINTEXT vault rows would leave this machine, else
    None. 'The mapping never leaves the machine' is the vault's founding
    invariant; a remote DSN keeps it only under fernet (HMAC index +
    ciphertext are all that travel)."""
    from llm_redact.config import RDBMS_BACKENDS, _is_loopback_host

    if config.backend not in RDBMS_BACKENDS or config.encryption == "fernet":
        return None
    if os.environ.get(ENV_REMOTE_PLAINTEXT) == "1":
        return None
    remote = managed_dbms_cloud(config) is not None  # a managed DBMS is off-box by definition
    host = dsn_host(config)
    if host is not None and not _is_loopback_host(host):
        remote = True
    if not remote:
        return None
    return (
        "[vault.rdbms] the DSN points off this machine but the vault is"
        ' PLAINTEXT: set [vault] encryption = "fernet" so only the HMAC index'
        " and ciphertext leave the box, or set"
        f" {ENV_REMOTE_PLAINTEXT}=1 to accept plaintext off-box (surfaced, never silent)"
    )


def _resolve_connector(config: VaultConfig) -> tuple[Any, Callable[[], Any]]:
    """(driver module, connect thunk) for the configured backend.

    Errors name the backend, the scheme, or the env var — never the DSN,
    which may embed credentials.
    """
    from llm_redact.config import ConfigError

    backend = config.backend
    dsn = resolve_dsn(config)
    module_name = config.rdbms.module if backend == "dbapi" else _DRIVER_MODULES[backend]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        if backend == "dbapi":
            raise ConfigError(f"[vault.rdbms] module {module_name!r} is not importable") from exc
        raise ConfigError(
            f'[vault] backend = "{backend}" requires the {module_name} driver:'
            f" {_EXTRA_HINTS[backend]}"
        ) from exc

    if backend == "dbapi":
        # Verbatim hand-off: the operator owns the connect-string contract
        # of whatever driver they named (sqlite3 takes a path, pyodbc a
        # connection string, ...).
        return module, lambda: module.connect(dsn)

    parts = urlsplit(dsn)
    if parts.scheme not in _SCHEMES[backend]:
        expected = " / ".join(f"{s}://" for s in _SCHEMES[backend])
        raise ConfigError(
            f"[vault.rdbms] dsn scheme {parts.scheme!r} does not match"
            f' backend = "{backend}" (expected {expected}; the DSN itself is never echoed)'
        )
    password = os.environ.get(config.rdbms.password_env) or parts.password or None

    if backend == "postgresql":
        kwargs: dict[str, Any] = {"password": password} if password else {}
        return module, lambda: module.connect(dsn, **kwargs)
    if backend == "mysql":
        host = parts.hostname or "127.0.0.1"
        port = parts.port or 3306
        user = parts.username or ""
        database = parts.path.lstrip("/")
        return module, lambda: module.connect(
            host=host,
            port=port,
            user=user,
            password=password or "",
            database=database,
            charset="utf8mb4",  # placeholders are non-ASCII; latin1 would mangle them
        )
    # oracle: thin-mode oracledb, host:port/service form.
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 1521
    service = parts.path.lstrip("/")
    user = parts.username or ""
    # CLOB columns must come back as str, not LOB handles.
    with suppress(AttributeError):
        module.defaults.fetch_lobs = False
    return module, lambda: module.connect(
        user=user, password=password or "", dsn=f"{host}:{port}/{service}"
    )


def validate_connector(config: VaultConfig) -> None:
    """Import the driver and validate the DSN shape WITHOUT connecting —
    doctor's read-only check. Raises ConfigError exactly like the build."""
    _resolve_connector(config)


class RdbmsStore:
    """Shared connection + all SQL. Callers (vault views, the manager) hold
    no SQL of their own.

    Every public operation is a self-contained transaction and is wrapped
    in one reconnect-and-retry: idle remote connections drop, and because
    callers cache only after success — and a retried allocation re-reads
    both the mapping and MAX(n)+1 fresh — a retry after a lost commit-ack
    finds the committed row and returns the same token.
    """

    def __init__(self, config: VaultConfig, cipher: VaultCipher | None) -> None:
        from llm_redact.config import ConfigError

        self._backend = config.backend
        self._cipher = cipher
        module, connect = _resolve_connector(config)
        self._module = module
        self._connect = connect
        self._paramstyle = str(getattr(module, "paramstyle", "qmark"))
        if self._paramstyle not in _KNOWN_PARAMSTYLES:
            raise ConfigError(
                f"[vault.rdbms] module paramstyle {self._paramstyle!r} is not a"
                f" DB-API 2.0 style {_KNOWN_PARAMSTYLES}"
            )
        retryable = []
        for name in ("OperationalError", "InterfaceError"):
            exc_type = getattr(module, name, None)
            if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
                retryable.append(exc_type)
        self._retryable: tuple[type[BaseException], ...] = tuple(retryable)
        self._conn = connect()
        self._response_inserts = 0
        self._ensure_schema()

    # -- plumbing ---------------------------------------------------------

    def _execute(self, conn: Any, sql: str, params: dict[str, Any] | None = None) -> Any:
        cursor = conn.cursor()
        if params:
            text, bound = adapt_sql(sql, params, self._paramstyle)
            cursor.execute(text, bound)
        else:
            cursor.execute(sql)
        return cursor

    def _rollback(self, conn: Any) -> None:
        with suppress(Exception):
            conn.rollback()

    def _run(self, op: Callable[[Any], Any]) -> Any:
        try:
            return op(self._conn)
        except self._retryable:
            # Dropped/unusable connection: reconnect once and retry the
            # whole (self-contained) operation. A genuine fault fails again
            # and propagates — fail closed, one extra round trip.
            with suppress(Exception):
                self._conn.close()
            self._conn = self._connect()
            return op(self._conn)

    # -- schema -----------------------------------------------------------

    def _table_missing(self, conn: Any, table: str) -> bool:
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE 1 = 0")
            cursor.fetchall()
            return False
        except self._module.Error:
            # Some engines (psycopg) poison the transaction after any
            # error; roll back so the CREATE that follows can run.
            self._rollback(conn)
            return True

    def _ensure_schema(self) -> None:
        def op(conn: Any) -> None:
            for table, ddl in _ddl(self._backend).items():
                # Probe-then-create instead of IF NOT EXISTS: Oracle only
                # grew the clause in 23ai, and the probe is portable.
                if self._table_missing(conn, table):
                    conn.cursor().execute(ddl)
                    # Commit EACH create: PostgreSQL DDL is transactional,
                    # and the NEXT missing-table probe's rollback would
                    # otherwise undo this CREATE (MySQL/Oracle/sqlite
                    # auto-commit DDL, which is how the bug hid there).
                    conn.commit()
            conn.commit()
            self._check_meta(conn)

        self._run(op)

    def _check_meta(self, conn: Any) -> None:
        from llm_redact.config import ConfigError

        rows = {
            str(key): str(value)
            for key, value in self._execute(
                conn, "SELECT meta_key, meta_value FROM llm_redact_meta"
            ).fetchall()
        }
        configured = "fernet" if self._cipher is not None else "none"
        stored = rows.get("encryption")
        if stored is None:
            now = _utcnow_iso()
            meta = {"schema_version": "1", "encryption": configured, "created_at": now}
            if self._cipher is not None:
                meta["key_check"] = self._cipher.key_check()
            for key, value in meta.items():
                self._execute(
                    conn,
                    "INSERT INTO llm_redact_meta (meta_key, meta_value) VALUES (:k, :v)",
                    {"k": key, "v": value},
                )
            conn.commit()
            return
        conn.commit()  # close the read snapshot
        if stored != configured:
            raise ConfigError(
                f'this RDBMS vault schema was created with encryption = "{stored}"'
                f' but the config says "{configured}"; the mode is fixed at creation'
                " — point the DSN at a fresh database/schema to change it"
            )
        if self._cipher is not None and not hmac.compare_digest(
            rows.get("key_check", ""), self._cipher.key_check()
        ):
            raise VaultKeyError(
                "LLM_REDACT_VAULT_KEY does not match the RDBMS vault at the configured DSN"
            )

    # -- mappings ---------------------------------------------------------

    def _original_key(self, session: str, detector_type: str, original: str) -> str:
        if self._cipher is not None:
            return self._cipher.mac(session, detector_type, original)
        return hashlib.sha256(original.encode("utf-8")).hexdigest()

    def preload(self, session: str) -> list[tuple[str, str, str]]:
        """(detector_type, original, placeholder) rows for one session."""

        def op(conn: Any) -> list[tuple[str, str, str]]:
            rows = self._execute(
                conn,
                "SELECT detector_type, original, original_ct, placeholder"
                " FROM llm_redact_mappings WHERE session_id = :s",
                {"s": session},
            ).fetchall()
            conn.commit()
            out = []
            for detector_type, original, original_ct, placeholder in rows:
                if self._cipher is not None:
                    original = self._cipher.decrypt(str(original_ct).encode("ascii"))
                out.append((str(detector_type), str(original), str(placeholder)))
            return out

        result: list[tuple[str, str, str]] = self._run(op)
        return result

    def get_or_create(self, session: str, detector_type: str, original: str) -> str:
        original_key = self._original_key(session, detector_type, original)

        def op(conn: Any) -> str:
            for _ in range(_ALLOCATION_ATTEMPTS):
                row = self._execute(
                    conn,
                    "SELECT placeholder FROM llm_redact_mappings WHERE session_id = :s"
                    " AND detector_type = :t AND original_key = :k",
                    {"s": session, "t": detector_type, "k": original_key},
                ).fetchone()
                if row is not None:
                    conn.commit()
                    return str(row[0])
                nrow = self._execute(
                    conn,
                    "SELECT COALESCE(MAX(n), 0) + 1 FROM llm_redact_mappings"
                    " WHERE session_id = :s AND detector_type = :t",
                    {"s": session, "t": detector_type},
                ).fetchone()
                n = int(nrow[0])
                placeholder = format_placeholder(detector_type, n)
                params: dict[str, Any] = {
                    "s": session,
                    "t": detector_type,
                    "k": original_key,
                    "o": original if self._cipher is None else None,
                    "c": (
                        self._cipher.encrypt(original).decode("ascii")
                        if self._cipher is not None
                        else None
                    ),
                    "p": placeholder,
                    "n": n,
                    "ts": _utcnow_iso(),
                }
                try:
                    self._execute(
                        conn,
                        "INSERT INTO llm_redact_mappings (session_id, detector_type,"
                        " original_key, original, original_ct, placeholder, n, created_at)"
                        " VALUES (:s, :t, :k, :o, :c, :p, :n, :ts)",
                        params,
                    )
                    conn.commit()
                    return placeholder
                except self._module.IntegrityError:
                    # Another writer (a second proxy instance sharing this
                    # database) inserted this value or claimed this n:
                    # re-read and retry — bounded, never a wrong value.
                    self._rollback(conn)
                    continue
                except self._module.Error:
                    # Any other write failure: roll back and fail closed.
                    # Nothing was cached, and n is MAX(n)+1 read fresh, so
                    # the next attempt reissues the same number.
                    self._rollback(conn)
                    raise
            raise RuntimeError(
                "RDBMS vault allocation kept colliding after"
                f" {_ALLOCATION_ATTEMPTS} attempts; refusing to guess"
            )

        result: str = self._run(op)
        return result

    def lookup_reverse(self, session: str, placeholder: str) -> str | None:
        def op(conn: Any) -> str | None:
            row = self._execute(
                conn,
                "SELECT original, original_ct FROM llm_redact_mappings"
                " WHERE session_id = :s AND placeholder = :p",
                {"s": session, "p": placeholder},
            ).fetchone()
            conn.commit()
            if row is None:
                return None
            if self._cipher is not None:
                return self._cipher.decrypt(str(row[1]).encode("ascii"))
            return str(row[0])

        result: str | None = self._run(op)
        return result

    def lookup_token(self, placeholder: str, session: str | None = None) -> list[tuple[str, str]]:
        """(session_id, original) rows for a placeholder, across sessions
        unless one is named — the CLI `lookup` query."""

        def op(conn: Any) -> list[tuple[str, str]]:
            sql = (
                "SELECT session_id, original, original_ct FROM llm_redact_mappings"
                " WHERE placeholder = :p"
            )
            params: dict[str, Any] = {"p": placeholder}
            if session is not None:
                sql += " AND session_id = :s"
                params["s"] = session
            rows = self._execute(conn, sql, params).fetchall()
            conn.commit()
            out = []
            for session_id, original, original_ct in rows:
                if self._cipher is not None:
                    original = self._cipher.decrypt(str(original_ct).encode("ascii"))
                out.append((str(session_id), str(original)))
            return out

        result: list[tuple[str, str]] = self._run(op)
        return result

    def lookup_value(self, value: str, session: str | None = None) -> list[tuple[str, str, str]]:
        """(session_id, detector_type, placeholder) rows for a value — the
        CLI reverse `lookup --value` query. Encrypted vaults compute the
        (session, type)-domain-separated MAC per stored pair, exactly like
        the sqlite CLI path."""

        def op(conn: Any) -> list[tuple[str, str, str]]:
            out: list[tuple[str, str, str]] = []
            if self._cipher is None:
                sql = (
                    "SELECT session_id, detector_type, placeholder FROM llm_redact_mappings"
                    " WHERE original_key = :k"
                )
                params: dict[str, Any] = {"k": hashlib.sha256(value.encode("utf-8")).hexdigest()}
                if session is not None:
                    sql += " AND session_id = :s"
                    params["s"] = session
                for session_id, detector_type, placeholder in self._execute(
                    conn, sql, params
                ).fetchall():
                    out.append((str(session_id), str(detector_type), str(placeholder)))
                conn.commit()
                return out
            pair_sql = "SELECT DISTINCT session_id, detector_type FROM llm_redact_mappings"
            pair_params: dict[str, Any] = {}
            if session is not None:
                pair_sql += " WHERE session_id = :s"
                pair_params["s"] = session
            pairs = self._execute(conn, pair_sql, pair_params or None).fetchall()
            for session_id, detector_type in pairs:
                mac = self._cipher.mac(str(session_id), str(detector_type), value)
                row = self._execute(
                    conn,
                    "SELECT placeholder FROM llm_redact_mappings WHERE session_id = :s"
                    " AND detector_type = :t AND original_key = :k",
                    {"s": session_id, "t": detector_type, "k": mac},
                ).fetchone()
                if row is not None:
                    out.append((str(session_id), str(detector_type), str(row[0])))
            conn.commit()
            return out

        result: list[tuple[str, str, str]] = self._run(op)
        return result

    # -- manager surface ----------------------------------------------------

    def session_count(self) -> int:
        def op(conn: Any) -> int:
            row = self._execute(
                conn, "SELECT COUNT(DISTINCT session_id) FROM llm_redact_mappings"
            ).fetchone()
            conn.commit()
            return int(row[0])

        result: int = self._run(op)
        return result

    def total_entries(self) -> int:
        def op(conn: Any) -> int:
            row = self._execute(conn, "SELECT COUNT(*) FROM llm_redact_mappings").fetchone()
            conn.commit()
            return int(row[0])

        result: int = self._run(op)
        return result

    def sessions_summary(self) -> list[dict[str, object]]:
        def op(conn: Any) -> list[dict[str, object]]:
            rows = self._execute(
                conn,
                "SELECT session_id, COUNT(*), MIN(created_at), MAX(created_at)"
                " FROM llm_redact_mappings GROUP BY session_id"
                " ORDER BY MAX(created_at) DESC",
            ).fetchall()
            conn.commit()
            return [
                {"session": str(sid), "entries": int(count), "first": first, "last": last}
                for sid, count, first, last in rows
            ]

        result: list[dict[str, object]] = self._run(op)
        return result

    def prune_sessions(self, days: int, *, exclude: frozenset[str] = frozenset()) -> list[str]:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

        def op(conn: Any) -> list[str]:
            rows = self._execute(
                conn,
                "SELECT session_id FROM llm_redact_mappings GROUP BY session_id"
                " HAVING MAX(created_at) < :cutoff",
                {"cutoff": cutoff},
            ).fetchall()
            doomed = [str(row[0]) for row in rows if str(row[0]) not in exclude]
            if not doomed:
                conn.commit()
                return []
            try:
                for session_id in doomed:
                    self._execute(
                        conn,
                        "DELETE FROM llm_redact_mappings WHERE session_id = :s",
                        {"s": session_id},
                    )
                    self._execute(
                        conn,
                        "DELETE FROM llm_redact_response_sessions WHERE session_id = :s",
                        {"s": session_id},
                    )
                conn.commit()
            except self._module.Error:
                self._rollback(conn)
                raise
            return doomed

        result: list[str] = self._run(op)
        return result

    def record_response_session(self, response_id: str, session_id: str) -> None:
        self._response_inserts += 1
        cap_now = self._response_inserts >= _RESPONSE_PRUNE_EVERY
        if cap_now:
            self._response_inserts = 0

        def op(conn: Any) -> None:
            try:
                # DELETE + INSERT instead of an upsert: portable across
                # every dialect, and idempotent under the reconnect retry.
                self._execute(
                    conn,
                    "DELETE FROM llm_redact_response_sessions WHERE response_id = :r",
                    {"r": response_id},
                )
                self._execute(
                    conn,
                    "INSERT INTO llm_redact_response_sessions"
                    " (response_id, session_id, created_at) VALUES (:r, :s, :ts)",
                    {"r": response_id, "s": session_id, "ts": _utcnow_iso()},
                )
                if cap_now:
                    if self._backend == "oracle":
                        keepers = (
                            "SELECT response_id FROM llm_redact_response_sessions"
                            " ORDER BY created_at DESC FETCH FIRST :cap ROWS ONLY"
                        )
                    else:
                        # The derived table both satisfies MySQL's LIMIT-in-IN
                        # restriction and its same-table-delete rule (1093).
                        keepers = (
                            "SELECT response_id FROM (SELECT response_id, created_at"
                            " FROM llm_redact_response_sessions"
                            " ORDER BY created_at DESC LIMIT :cap) keepers"
                        )
                    self._execute(
                        conn,
                        "DELETE FROM llm_redact_response_sessions"
                        f" WHERE response_id NOT IN ({keepers})",
                        {"cap": _MAX_RESPONSE_ROWS},
                    )
                conn.commit()
            except self._module.Error:
                self._rollback(conn)
                raise

        self._run(op)

    def lookup_response_session(self, response_id: str) -> str | None:
        def op(conn: Any) -> str | None:
            row = self._execute(
                conn,
                "SELECT session_id FROM llm_redact_response_sessions WHERE response_id = :r",
                {"r": response_id},
            ).fetchone()
            conn.commit()
            return str(row[0]) if row is not None else None

        result: str | None = self._run(op)
        return result

    def close(self) -> None:
        with suppress(Exception):
            self._conn.close()


class RdbmsVault:
    """Per-session view over a shared RdbmsStore — SqliteVault's caching
    contract: preload at creation, write-through only after COMMIT."""

    def __init__(self, store: RdbmsStore, session: str, *, owns_store: bool = False) -> None:
        self._store = store
        self._session = session
        self._owns_store = owns_store
        self._forward: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        for detector_type, original, placeholder in store.preload(session):
            self._forward[f"{detector_type}::{original}"] = placeholder
            self._reverse[placeholder] = original

    def placeholder_for(self, detector_type: str, original: str) -> str:
        key = f"{detector_type}::{original}"
        existing = self._forward.get(key)
        if existing is not None:
            return existing
        placeholder = self._store.get_or_create(self._session, detector_type, original)
        self._forward[key] = placeholder
        self._reverse[placeholder] = original
        return placeholder

    def original_for(self, placeholder: str) -> str | None:
        cached = self._reverse.get(placeholder)
        if cached is not None:
            return cached
        original = self._store.lookup_reverse(self._session, placeholder)
        if original is not None:
            self._reverse[placeholder] = original
        return original

    def close(self) -> None:
        if self._owns_store:
            self._store.close()

    def __len__(self) -> int:
        return len(self._reverse)


class RdbmsVaultManager:
    """One shared store; per-session views cached in a small LRU (the
    SqliteVaultManager shape — eviction drops only a view's cache)."""

    def __init__(self, store: RdbmsStore, *, view_cache_size: int = 64) -> None:
        self._store = store
        self._views: OrderedDict[str, RdbmsVault] = OrderedDict()
        self._view_cache_size = view_cache_size

    def get(self, session_id: str) -> Vault:
        view = self._views.get(session_id)
        if view is None:
            view = RdbmsVault(self._store, session_id)
            self._views[session_id] = view
        self._views.move_to_end(session_id)
        while len(self._views) > self._view_cache_size:
            self._views.popitem(last=False)
        return view

    def session_count(self) -> int:
        return self._store.session_count()

    def total_entries(self) -> int:
        return self._store.total_entries()

    def sessions_summary(self) -> list[dict[str, object]]:
        return self._store.sessions_summary()

    def prune_sessions(self, days: int, *, exclude: frozenset[str] = frozenset()) -> int:
        doomed = self._store.prune_sessions(days, exclude=exclude)
        for session_id in doomed:
            self._views.pop(session_id, None)
        return len(doomed)

    def record_response_session(self, response_id: str, session_id: str) -> None:
        self._store.record_response_session(response_id, session_id)

    def lookup_response_session(self, response_id: str) -> str | None:
        return self._store.lookup_response_session(response_id)

    def close(self) -> None:
        self._store.close()


def _gated_store(config: VaultConfig, cipher: VaultCipher | None) -> RdbmsStore:
    """Build the store behind the off-box rule — refused BEFORE any
    connection is attempted.

    The cipher is supplied by the caller (the paid ``build_vault_manager``
    override, which resolves it from ``llm-redact-pro``): an encrypted
    config that reaches here without one is a build-time fault, never a
    silent downgrade to plaintext."""
    from llm_redact.config import ConfigError

    violation = offbox_violation(config)
    if violation is not None:
        raise ConfigError(violation)
    if config.encryption == "fernet" and cipher is None:
        raise ConfigError('[vault] encryption = "fernet" requires the llm-redact-pro package')
    return RdbmsStore(config, cipher)


def build_rdbms_vault_manager(
    config: VaultConfig, cipher: VaultCipher | None = None
) -> RdbmsVaultManager:
    return RdbmsVaultManager(_gated_store(config, cipher))


def open_rdbms_vault(
    config: VaultConfig, session: str, cipher: VaultCipher | None = None
) -> RdbmsVault:
    """Standalone single-session vault owning its store (static mode, CLI)."""
    return RdbmsVault(_gated_store(config, cipher), session, owns_store=True)
