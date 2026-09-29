"""Persistent vault on a server RDBMS through DB-API 2.0 (llm-redact-pro).

One store speaks PostgreSQL (psycopg), MySQL (PyMySQL), Oracle (oracledb),
or any DB-API 2.0 module the operator names (``backend = "dbapi"``), via a
paramstyle adapter plus a two-entry dialect table — the vault SQL itself is
a portable subset: no upserts and no SELECT FOR UPDATE. Dense counter
allocation relies on the UNIQUE (session, type, n) constraint plus a
bounded retry, which is the SqliteVault recipe generalized to any engine.

Semantics mirror SqliteVault and are pinned by the same invariant battery
(tests/test_vault_rdbms.py): deterministic (session, type, value) → token,
per-(session, type) counters that never reuse a number (n is
max(MAX(n), retired, floor)+1 with MAX(n) and the session's retired number
read in one statement inside the transaction, so a rolled-back allocation
reissues the SAME number — reuse is the danger; the only gaps are the ones
a request's token floor asks for, and there is no counters table to lose
one), caches written only after COMMIT, any write fault rolls back and fails
closed, whole-session prune only — and a deleted session's numbers retired
(``llm_redact_retired``), never issued again, so a replica still caching
them restores the right value or none.

Allocation stays one self-contained transaction per value (no per-request
batch as on sqlite): the reconnect-retry replays ONE self-contained op, and
the UNIQUE-retry rolls back ONE allocation — inside a shared transaction a
dropped connection or a rollback would take the request's earlier
allocations with it, after their tokens were already substituted.

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
import logging
import os
import re
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from llm_redact.placeholders import format_placeholder
from llm_redact.vault import (
    _MAX_OBJECT_ROWS,
    _MAX_RESPONSE_ROWS,
    _OBJECT_KIND,
    _RESPONSE_KIND,
    _RESPONSE_PRUNE_EVERY,
    CACHE_CHECK_SECONDS,
    LOOKUP_CHUNK,
    PlaceholderSpaceExhausted,
    Vault,
    VaultKeyError,
    next_number,
)

if TYPE_CHECKING:
    from llm_redact.config import VaultConfig
    from llm_redact.plugin_api import DbPasswordProvider, VaultCipher

logger = logging.getLogger("llm_redact")

ENV_DSN = "LLM_REDACT_VAULT_DSN"
# The documented hatch for the off-box rule below: set to 1 to run a
# PLAINTEXT vault against a remote database anyway (e.g. a trusted
# same-host container network the hostname check cannot see). Surfaced in
# /status and doctor whenever active — an opt-out is never silent.
ENV_REMOTE_PLAINTEXT = "LLM_REDACT_VAULT_REMOTE_PLAINTEXT"
# auth = "identity" hatch: accept an UNVERIFIED TLS link to a non-loopback
# database (the token then reaches whoever answers the handshake). Surfaced
# in /status (vault.tls_unverified), doctor and `llm-redact status`.
ENV_TLS_UNVERIFIED = "LLM_REDACT_VAULT_TLS_UNVERIFIED"

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


class RdbmsAllocationError(RuntimeError):
    """A new placeholder's allocation kept colliding with concurrent writers
    past ``_ALLOCATION_ATTEMPTS``: refused, never guessed (one of the store's
    ``fault_types``)."""


# The response map's row kind (a Responses chain's row, or a stored object's
# owner record): each kind is bounded apart (see llm_redact.vault). One
# portable column definition for the create and the upgrade of a schema
# created before it (DEFAULT before NOT NULL: Oracle's order, read by all).
_KIND_COLUMN = "kind VARCHAR(8) DEFAULT 'response' NOT NULL"

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
        "llm_redact_response_sessions": f"""CREATE TABLE llm_redact_response_sessions (
  response_id VARCHAR(192) NOT NULL,
  session_id VARCHAR(128) NOT NULL,
  created_at VARCHAR(20) NOT NULL,
  {_KIND_COLUMN},
  PRIMARY KEY (response_id)
)""",
        "llm_redact_meta": """CREATE TABLE llm_redact_meta (
  meta_key VARCHAR(32) NOT NULL,
  meta_value VARCHAR(128) NOT NULL,
  PRIMARY KEY (meta_key)
)""",
        # A whole-session delete retires every number the session held (see
        # RdbmsStore._retire): one row per session ever deleted, no values.
        "llm_redact_retired": """CREATE TABLE llm_redact_retired (
  session_id VARCHAR(128) NOT NULL,
  n INTEGER NOT NULL,
  PRIMARY KEY (session_id)
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


# PyMySQL before 1.2 silently continues WITHOUT TLS when the server does not
# advertise it, even with ssl= set (a stripped capability flag would then
# carry the cleartext token); 1.2 refuses. auth = "identity" requires it.
_PYMYSQL_TLS_ENFORCING = (1, 2)


class _TlsOnlyClearPassword:
    """PyMySQL ``auth_plugin_map`` handler for ``mysql_clear_password``.

    RDS IAM, Cloud SQL IAM and Entra ID database users all make the server
    switch the client to the cleartext plugin, so the token itself crosses
    the wire. PyMySQL >= 1.2 refuses a non-TLS server before any auth packet
    when ``ssl`` is set; this handler is the second floor — it sends the
    password only when the connection's socket really is TLS.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def authenticate(self, pkt: Any) -> Any:
        import ssl

        del pkt  # the switch request carries no data this plugin uses
        conn = self._conn
        if not isinstance(getattr(conn, "_sock", None), ssl.SSLSocket):
            raise RuntimeError(
                "refusing to send the database token over a MySQL connection without TLS"
            )
        conn.write_packet(conn.password + b"\0")
        reply = conn._read_packet()
        reply.check_error()
        return reply


# libpq connection parameters that could send the identity token somewhere
# other than the DSN's host, or pull in a service file whose settings (a weak
# sslmode, another hostaddr, another root certificate) apply unseen.
_PG_REDIRECTING_PARAMS = ("service", "hostaddr")
_PG_REDIRECTING_ENV = ("PGSERVICE", "PGSERVICEFILE", "PGHOSTADDR")
_PG_VERIFYING_SSLMODES = ("verify-ca", "verify-full")


def _pg_sslmode(dsn: str, environ: Mapping[str, str]) -> str:
    """The sslmode an identity connection runs with: the DSN's, else a
    strong PGSSLMODE, else require (always passed explicitly, so no
    service file or environment value can weaken it)."""
    from llm_redact.config import PG_TLS_SSLMODES

    modes = parse_qs(urlsplit(dsn).query).get("sslmode")
    if modes:
        return modes[-1].lower()
    env_mode = environ.get("PGSSLMODE", "").lower()
    return env_mode if env_mode in PG_TLS_SSLMODES else "require"


def identity_tls_problem(
    backend: str, dsn: str, environ: Mapping[str, str] = os.environ
) -> str | None:
    """Why an ``auth = "identity"`` connection could hand its token to the
    wrong party, else None.

    RDS IAM, Cloud SQL IAM and Entra make the server ask for the token in
    the clear INSIDE TLS, so an unverified link gives it to whoever answers
    the handshake: off loopback the server certificate must be verified
    (PostgreSQL sslmode verify-ca/verify-full, MySQL ``?ssl_ca=``) unless
    the operator sets the surfaced hatch. PostgreSQL also refuses the
    parameters that could redirect the connection behind the DSN's host.
    Messages name the rule, never the DSN."""
    parts = urlsplit(dsn)
    query = parse_qs(parts.query, keep_blank_values=True)
    if backend == "postgresql":
        named = [key for key in _PG_REDIRECTING_PARAMS if key in query]
        hosts = query.get("host", [])
        if any(not host.startswith("/") for host in hosts):
            named.append("host")  # a TCP host overriding the DSN's; a socket dir is fine
        named += [name for name in _PG_REDIRECTING_ENV if environ.get(name)]
        if named:
            return (
                '[vault.rdbms] auth = "identity" refuses ' + ", ".join(named) + ": it could"
                " send the database token past the DSN's host or weaken its TLS settings"
            )
    if environ.get(ENV_TLS_UNVERIFIED) == "1" or _identity_tls_verified(backend, dsn, environ):
        return None
    how = (
        "sslmode=verify-full (verify-ca for Cloud SQL) with sslrootcert=PATH"
        if backend == "postgresql"
        else "?ssl_ca=PATH (the server's CA bundle)"
    )
    return (
        '[vault.rdbms] auth = "identity" sends the database token inside TLS, so the'
        f" server certificate must be verified: add {how} to the DSN, or set"
        f" {ENV_TLS_UNVERIFIED}=1 to accept an unverified link (surfaced, never silent)"
    )


def _identity_tls_verified(backend: str, dsn: str, environ: Mapping[str, str]) -> bool:
    """True when the token cannot reach an unverified server: a unix socket
    or loopback host (it never crosses a network), else a verified TLS
    server certificate."""
    from llm_redact.config import _is_loopback_host

    parts = urlsplit(dsn)
    if parts.hostname is None or _is_loopback_host(parts.hostname):
        return True
    if backend == "postgresql":
        return _pg_sslmode(dsn, environ) in _PG_VERIFYING_SSLMODES
    return bool(parse_qs(parts.query).get("ssl_ca"))


def identity_tls_unverified(config: VaultConfig) -> bool:
    """True when the unverified-TLS hatch is what lets an identity vault
    connect (the /status, doctor and `llm-redact status` honesty signal)."""
    if config.rdbms.auth != "identity" or os.environ.get(ENV_TLS_UNVERIFIED) != "1":
        return False
    dsn = _quiet_dsn(config)
    return dsn is not None and not _identity_tls_verified(config.backend, dsn, os.environ)


def _pg_tls_kwargs(dsn: str) -> dict[str, Any]:
    """Connect kwargs making a PostgreSQL identity connection use TLS: an
    explicit sslmode (explicit parameters beat a service file and the
    environment), after identity_tls_problem has refused the rest."""
    return {"sslmode": _pg_sslmode(dsn, os.environ)}


def _mysql_tls_kwargs(module: Any, dsn: str) -> dict[str, Any]:
    """Connect kwargs making a MySQL identity connection use TLS: an
    SSLContext (``?ssl_ca=PATH`` in the DSN verifies the server certificate
    and hostname; without it the link is encrypted but unverified — libpq's
    sslmode=require) and the TLS-only cleartext-plugin handler."""
    import ssl

    from llm_redact.config import ConfigError

    version = tuple(getattr(module, "VERSION", ()))[:2]
    if version < _PYMYSQL_TLS_ENFORCING:
        raise ConfigError(
            '[vault.rdbms] auth = "identity" on MySQL needs PyMySQL >= 1.2 (older'
            " releases fall back to an unencrypted connection when the server offers"
            " no TLS); upgrade PyMySQL"
        )
    cafiles = parse_qs(urlsplit(dsn).query).get("ssl_ca")
    if cafiles:
        try:
            context = ssl.create_default_context(cafile=cafiles[-1])
        except (OSError, ssl.SSLError) as exc:
            raise ConfigError(
                "[vault.rdbms] the DSN's ssl_ca file could not be loaded as a CA bundle"
            ) from exc
    else:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return {"ssl": context, "auth_plugin_map": {"mysql_clear_password": _TlsOnlyClearPassword}}


def _constant(value: str | None) -> Callable[[], str | None]:
    """The static password (password_env / DSN), read once at build."""
    return lambda: value


def _identity_unavailable() -> str:
    from llm_redact.config import ConfigError

    raise ConfigError('[vault.rdbms] auth = "identity" has no password provider')


def _registry_password(config: VaultConfig) -> DbPasswordProvider | None:
    """The registry's per-connect password for this vault (None = static)."""
    from llm_redact.config import ConfigError
    from llm_redact.registry import get_registry

    provider = get_registry().build_db_password(config)
    if provider is None and config.rdbms.auth == "identity":
        raise ConfigError(
            '[vault.rdbms] auth = "identity": the installed build_db_password'
            " returned no password provider (fail closed)"
        )
    return provider


def _resolve_connector(
    config: VaultConfig,
    password_provider: DbPasswordProvider | None = None,
    *,
    consult_registry: bool = False,
) -> tuple[Any, Callable[[], Any]]:
    """(driver module, connect thunk) for the configured backend.

    The thunk asks for the password at EVERY call (``password_provider``,
    else the registry's when ``consult_registry``, else the static
    password_env/DSN password), so the store's reconnect path gets a fresh
    short-lived token. Errors name the backend, the scheme, or the env var —
    never the DSN, which may embed credentials.
    """
    from llm_redact.config import ConfigError, rdbms_identity_error

    backend = config.backend
    dsn = resolve_dsn(config)
    identity_error = rdbms_identity_error(backend, config.rdbms, dsn)
    if identity_error is not None:
        raise ConfigError(identity_error)
    identity = config.rdbms.auth == "identity"
    if identity and backend in ("postgresql", "mysql"):
        tls_problem = identity_tls_problem(backend, dsn)
        if tls_problem is not None:
            raise ConfigError(tls_problem)
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
    if password_provider is None and consult_registry:
        password_provider = _registry_password(config)

    if backend == "dbapi":
        if password_provider is not None:
            raise ConfigError(
                '[vault.rdbms] backend = "dbapi" hands its DSN to connect() verbatim'
                " and cannot take a per-connect password"
            )
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
    password: Callable[[], str | None]
    if password_provider is not None:
        password = password_provider
    elif identity:
        password = _identity_unavailable  # doctor's validate: never connects
    else:
        password = _constant(os.environ.get(config.rdbms.password_env) or parts.password or None)

    if backend == "postgresql":
        tls = _pg_tls_kwargs(dsn) if identity else {}

        def connect_postgresql() -> Any:
            secret = password()
            kwargs: dict[str, Any] = {**tls, "password": secret} if secret else dict(tls)
            return module.connect(dsn, **kwargs)

        return module, connect_postgresql
    if backend == "mysql":
        host = parts.hostname or "127.0.0.1"
        port = parts.port or 3306
        user = parts.username or ""
        database = parts.path.lstrip("/")
        tls = _mysql_tls_kwargs(module, dsn) if identity else {}
        return module, lambda: module.connect(
            host=host,
            port=port,
            user=user,
            password=password() or "",
            database=database,
            charset="utf8mb4",  # placeholders are non-ASCII; latin1 would mangle them
            **tls,
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
        user=user, password=password() or "", dsn=f"{host}:{port}/{service}"
    )


def validate_connector(config: VaultConfig) -> None:
    """Import the driver and validate the DSN shape (and, for identity auth,
    the TLS and no-static-password rules) WITHOUT connecting or building a
    credential source — doctor's read-only check. Raises ConfigError exactly
    like the build."""
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

    def __init__(
        self,
        config: VaultConfig,
        cipher: VaultCipher | None,
        *,
        password_provider: DbPasswordProvider | None = None,
    ) -> None:
        """``password_provider`` defaults to the registry's
        ``build_db_password`` (None there = the static password). It is
        called at EVERY connect — here and on the reconnect-retry — on the
        calling thread, exactly like the driver's own connect."""
        from llm_redact.config import ConfigError

        self._backend = config.backend
        self._cipher = cipher
        module, connect = _resolve_connector(config, password_provider, consult_registry=True)
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
        self._inserts = {_RESPONSE_KIND: 0, _OBJECT_KIND: 0}
        # False only when a schema from before the row kind could not gain
        # its column (see _ensure_kind_column): one shared bound, as before.
        self._row_kinds = True
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
                    self._create(conn, table, ddl)
                    # Commit EACH create: PostgreSQL DDL is transactional,
                    # and the NEXT missing-table probe's rollback would
                    # otherwise undo this CREATE (MySQL/Oracle/sqlite
                    # auto-commit DDL, which is how the bug hid there).
                    conn.commit()
            conn.commit()
            self._check_meta(conn)
            self._ensure_kind_column(conn)

        self._run(op)

    @property
    def owner_bound_shared(self) -> bool:
        """Whether the response map could not gain its ``kind`` column (the
        database user may not ALTER it, see ``_ensure_kind_column``): stored
        objects' owner records then share the Responses rows' bound. Read at
        startup only — adding the column takes a restart."""
        return not self._row_kinds

    @property
    def fault_types(self) -> tuple[type[BaseException], ...]:
        """What a failed allocation raises: the driver's DB-API ``Error``
        (after the rollback and the one reconnect-retry) and an allocation
        that kept colliding — the proxy refuses such a request 503."""
        driver_error = getattr(self._module, "Error", None)
        driver = (driver_error,) if isinstance(driver_error, type) else ()
        return (*driver, RdbmsAllocationError)

    def _create(self, conn: Any, table: str, ddl: str) -> None:
        """Create a missing table — or refuse to start, naming it and the
        statement a DBA must run, when this database user may not (a table
        added by an upgrade, under a schema created by someone else). Never
        the DSN."""
        from llm_redact.config import ConfigError

        try:
            conn.cursor().execute(ddl)
        except self._module.Error as exc:
            self._rollback(conn)
            raise ConfigError(
                f"the RDBMS vault could not create its table {table}"
                f" ({type(exc).__name__}); create it, then start again: {ddl}"
            ) from exc

    def _kind_missing(self, conn: Any) -> bool:
        """Whether the response map lacks the ``kind`` column (the portable
        probe of ``_table_missing``, for a column)."""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT kind FROM llm_redact_response_sessions WHERE 1 = 0")
            cursor.fetchall()
            return False
        except self._module.Error:
            self._rollback(conn)
            return True

    def _ensure_kind_column(self, conn: Any) -> None:
        """Give a response map created before its rows had a kind the
        column (existing rows default to Responses rows). A database user
        that may not ALTER the table keeps the store working as before —
        stored-object records then share the Responses bound — with a
        warning naming the table, never the DSN. Another replica adding it
        first is fine."""
        if not self._kind_missing(conn):
            conn.commit()  # close the probe's read snapshot
            return
        try:
            conn.cursor().execute(f"ALTER TABLE llm_redact_response_sessions ADD {_KIND_COLUMN}")
            conn.commit()
        except self._module.Error as exc:
            self._rollback(conn)
            if self._kind_missing(conn):
                self._row_kinds = False
                logger.warning(
                    "vault: could not add the kind column to llm_redact_response_sessions"
                    " (%s); stored objects' owner records share the Responses bound until"
                    " it is added (ALTER TABLE llm_redact_response_sessions ADD %s) and the"
                    " proxy restarted",
                    type(exc).__name__,
                    _KIND_COLUMN,
                )

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

    def _retired_of(self, conn: Any, session: str) -> int:
        row = self._execute(
            conn, "SELECT n FROM llm_redact_retired WHERE session_id = :s", {"s": session}
        ).fetchone()
        return 0 if row is None else int(row[0])

    def load(self, session: str) -> tuple[int, list[tuple[str, str, str]]]:
        """The session's retired number, then its (detector_type, original,
        placeholder) rows — in that order (see SqliteVault._load)."""

        def op(conn: Any) -> tuple[int, list[tuple[str, str, str]]]:
            retired = self._retired_of(conn, session)
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
            return retired, out

        result: tuple[int, list[tuple[str, str, str]]] = self._run(op)
        return result

    def retired(self, session: str) -> int:
        """The highest number ``session`` held in rows since deleted (0:
        none) — a view's staleness check."""

        def op(conn: Any) -> int:
            value = self._retired_of(conn, session)
            conn.commit()
            return value

        result: int = self._run(op)
        return result

    def _retire(self, conn: Any, session: str, highest: int) -> None:
        """Raise ``session``'s retired number to ``highest`` — never lower
        it (a replica retiring it concurrently may have gone higher). No
        upsert: a concurrent first INSERT surfaces as IntegrityError, and
        the caller's op starts over."""
        existing = self._execute(
            conn, "SELECT n FROM llm_redact_retired WHERE session_id = :s", {"s": session}
        ).fetchone()
        if existing is None:
            self._execute(
                conn,
                "INSERT INTO llm_redact_retired (session_id, n) VALUES (:s, :n)",
                {"s": session, "n": highest},
            )
            return
        self._execute(
            conn,
            "UPDATE llm_redact_retired SET n = :n WHERE session_id = :s AND n < :n",
            {"s": session, "n": highest},
        )

    def _delete_session(self, conn: Any, session: str, highest: int) -> None:
        """Delete one session's rows numbered up to ``highest`` (its MAX(n),
        read by the caller) and its response rows, retiring the numbers
        first. A row another replica allocated meanwhile is numbered above
        ``highest`` (its allocation read MAX(n) or the retired number) and
        survives: every number deleted is retired, none ever reissued."""
        self._retire(conn, session, highest)
        self._execute(
            conn,
            "DELETE FROM llm_redact_mappings WHERE session_id = :s AND n <= :n",
            {"s": session, "n": highest},
        )
        self._execute(
            conn,
            "DELETE FROM llm_redact_response_sessions WHERE session_id = :s",
            {"s": session},
        )

    def get_or_create(
        self, session: str, detector_type: str, original: str, *, floor: int = 0
    ) -> str:
        """The value's token in ``session``; a NEW value is numbered above
        the session's numbers and above ``floor`` (vault.next_number)."""
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
                # The live numbers and the retired number in ONE statement
                # (one snapshot): a concurrent whole-session delete is seen
                # either before (its rows) or after (its retired number) —
                # never neither.
                nrow = self._execute(
                    conn,
                    "SELECT COALESCE(MAX(n), 0), (SELECT COALESCE(MAX(n), 0)"
                    " FROM llm_redact_retired WHERE session_id = :s)"
                    " FROM llm_redact_mappings WHERE session_id = :s AND detector_type = :t",
                    {"s": session, "t": detector_type},
                ).fetchone()
                try:
                    n = next_number(max(int(nrow[0]), int(nrow[1])), floor, detector_type)
                except PlaceholderSpaceExhausted:
                    self._rollback(conn)  # close the read transaction; nothing written
                    raise
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
                    # Nothing was cached, and n comes from MAX(n) read fresh,
                    # so the next attempt reissues the same number.
                    self._rollback(conn)
                    raise
            raise RdbmsAllocationError(
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
        """Delete (``_delete_session``) the sessions that issued no new value
        in the last ``days`` days, but those in ``exclude``; their ids.

        Each candidate's idle check is re-read in the statement that sizes
        its delete (MAX(created_at) and MAX(n) in one snapshot): the delete
        removes exactly the rows seen idle, and a value another replica
        issued since keeps its row (numbered above them)."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

        def attempt(conn: Any) -> list[str]:
            candidates = self._execute(
                conn,
                "SELECT session_id FROM llm_redact_mappings GROUP BY session_id"
                " HAVING MAX(created_at) < :cutoff",
                {"cutoff": cutoff},
            ).fetchall()
            doomed = []
            for (session_id,) in candidates:
                if str(session_id) in exclude:
                    continue
                highest, last = self._execute(
                    conn,
                    "SELECT MAX(n), MAX(created_at) FROM llm_redact_mappings WHERE session_id = :s",
                    {"s": str(session_id)},
                ).fetchone()
                if highest is None or str(last) >= cutoff:
                    continue  # emptied, or used, since the candidate query
                self._delete_session(conn, str(session_id), int(highest))
                doomed.append(str(session_id))
            return doomed

        result: list[str] = self._run(lambda conn: self._deleting(conn, attempt))
        return result

    def forget_sessions(self, session_ids: list[str]) -> int:
        """Delete whole named sessions and their response rows in one
        transaction, retiring their numbers; how many held mappings."""

        def attempt(conn: Any) -> int:
            present = 0
            for session_id in session_ids:
                (highest,) = self._execute(
                    conn,
                    "SELECT MAX(n) FROM llm_redact_mappings WHERE session_id = :s",
                    {"s": session_id},
                ).fetchone()
                if highest is not None:
                    present += 1
                    self._delete_session(conn, session_id, int(highest))
                else:
                    self._execute(
                        conn,
                        "DELETE FROM llm_redact_response_sessions WHERE session_id = :s",
                        {"s": session_id},
                    )
            return present

        result: int = self._run(lambda conn: self._deleting(conn, attempt))
        return result

    def _deleting(self, conn: Any, attempt: Callable[[Any], Any]) -> Any:
        """Run a delete ``attempt`` as one transaction: committed, or rolled
        back whole. A replica retiring the same session first (the retired
        row's INSERT collides) starts it over — bounded, like allocation."""
        for _ in range(_ALLOCATION_ATTEMPTS):
            try:
                result = attempt(conn)
                conn.commit()
                return result
            except self._module.IntegrityError:
                self._rollback(conn)
                continue
            except self._module.Error:
                self._rollback(conn)
                raise
        raise RuntimeError(
            f"RDBMS vault session delete kept colliding after {_ALLOCATION_ATTEMPTS} attempts"
        )

    def record_response_session(self, response_id: str, session_id: str) -> None:
        """Map a Responses chain's response id to its session."""
        self._record(response_id, session_id, _RESPONSE_KIND, _MAX_RESPONSE_ROWS)

    def record_object_session(self, object_id: str, session_id: str) -> None:
        """Record the session that created a stored object; bounded apart
        from the Responses rows (see llm_redact.vault)."""
        self._record(object_id, session_id, _OBJECT_KIND, _MAX_OBJECT_ROWS)

    def _record(self, row_id: str, session_id: str, kind: str, cap: int) -> None:
        if not self._row_kinds:  # a schema that could not gain the column
            kind, cap = _RESPONSE_KIND, _MAX_RESPONSE_ROWS
        self._inserts[kind] += 1
        cap_now = self._inserts[kind] >= _RESPONSE_PRUNE_EVERY
        if cap_now:
            self._inserts[kind] = 0
        # Only the columns this schema has: every row is a Responses row
        # to a map that could not gain the kind column.
        insert = (
            "INSERT INTO llm_redact_response_sessions"
            " (response_id, session_id, created_at, kind) VALUES (:r, :s, :ts, :k)"
            if self._row_kinds
            else "INSERT INTO llm_redact_response_sessions"
            " (response_id, session_id, created_at) VALUES (:r, :s, :ts)"
        )
        of_kind = " WHERE kind = :k" if self._row_kinds else ""

        def op(conn: Any) -> None:
            try:
                # DELETE + INSERT instead of an upsert: portable across
                # every dialect, and idempotent under the reconnect retry.
                self._execute(
                    conn,
                    "DELETE FROM llm_redact_response_sessions WHERE response_id = :r",
                    {"r": row_id},
                )
                params = {"r": row_id, "s": session_id, "ts": _utcnow_iso()}
                self._execute(conn, insert, {**params, "k": kind} if self._row_kinds else params)
                if cap_now:
                    if self._backend == "oracle":
                        keepers = (
                            f"SELECT response_id FROM llm_redact_response_sessions{of_kind}"
                            " ORDER BY created_at DESC FETCH FIRST :cap ROWS ONLY"
                        )
                    else:
                        # The derived table both satisfies MySQL's LIMIT-in-IN
                        # restriction and its same-table-delete rule (1093).
                        keepers = (
                            "SELECT response_id FROM (SELECT response_id, created_at"
                            f" FROM llm_redact_response_sessions{of_kind}"
                            " ORDER BY created_at DESC LIMIT :cap) keepers"
                        )
                    # Only rows of sessions without mappings (see the
                    # sqlite store): a live session's chain must resolve.
                    # Each kind keeps its own newest rows.
                    self._execute(
                        conn,
                        "DELETE FROM llm_redact_response_sessions"
                        f" WHERE {'kind = :k AND ' if self._row_kinds else ''}"
                        f"response_id NOT IN ({keepers})"
                        " AND NOT EXISTS (SELECT 1 FROM llm_redact_mappings m"
                        " WHERE m.session_id = llm_redact_response_sessions.session_id)",
                        {"cap": cap, "k": kind} if self._row_kinds else {"cap": cap},
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

    def lookup_response_sessions(self, response_ids: list[str]) -> dict[str, str]:
        """Many ids in one round trip per ``LOOKUP_CHUNK`` (the sqlite
        store's batched lookup): recorded ids with their sessions."""

        def op(conn: Any) -> dict[str, str]:
            found: dict[str, str] = {}
            for start in range(0, len(response_ids), LOOKUP_CHUNK):
                chunk = response_ids[start : start + LOOKUP_CHUNK]
                params = {f"r{index}": value for index, value in enumerate(chunk)}
                marks = ", ".join(f":{name}" for name in params)
                rows = self._execute(
                    conn,
                    "SELECT response_id, session_id FROM llm_redact_response_sessions"
                    f" WHERE response_id IN ({marks})",
                    params,
                ).fetchall()
                found.update((str(row[0]), str(row[1])) for row in rows)
            conn.commit()
            return found

        result: dict[str, str] = self._run(op)
        return result

    def close(self) -> None:
        with suppress(Exception):
            self._conn.close()


class RdbmsVault:
    """Per-session view over a shared RdbmsStore — SqliteVault's caching
    contract: loaded at creation, write-through only after COMMIT, rebuilt
    when the session's retired number moved (another replica deleted the
    session; checked at most every CACHE_CHECK_SECONDS)."""

    # Set by _load (SqliteVault's fields).
    _forward: dict[str, str]
    _reverse: dict[str, str]
    _retired: int
    _next_check: float

    def __init__(
        self,
        store: RdbmsStore,
        session: str,
        *,
        owns_store: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._session = session
        self._owns_store = owns_store
        self._clock = clock
        self._load()

    def _load(self) -> None:
        retired, rows = self._store.load(self._session)
        self._forward = {
            f"{detector_type}::{original}": placeholder
            for detector_type, original, placeholder in rows
        }
        self._reverse = {placeholder: original for _, original, placeholder in rows}
        self._retired = retired
        self._next_check = self._clock() + CACHE_CHECK_SECONDS

    def _revalidate(self) -> None:
        """SqliteVault._revalidate: one indexed read once the check is due."""
        if self._store.retired(self._session) != self._retired:
            self._load()
            return
        self._next_check = self._clock() + CACHE_CHECK_SECONDS

    def placeholder_for(self, detector_type: str, original: str, *, floor: int = 0) -> str:
        if self._clock() >= self._next_check:
            self._revalidate()
        key = f"{detector_type}::{original}"
        existing = self._forward.get(key)
        if existing is not None:
            return existing
        placeholder = self._store.get_or_create(self._session, detector_type, original, floor=floor)
        self._forward[key] = placeholder
        self._reverse[placeholder] = original
        return placeholder

    def original_for(self, placeholder: str) -> str | None:
        if self._clock() >= self._next_check:
            self._revalidate()
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
    SqliteVaultManager shape — eviction drops only a view's cache, a view
    still held elsewhere is handed out again, and a delete rebuilds every
    live view of the session)."""

    def __init__(
        self,
        store: RdbmsStore,
        *,
        view_cache_size: int = 64,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._clock = clock
        self._views: OrderedDict[str, RdbmsVault] = OrderedDict()
        self._view_cache_size = view_cache_size
        self._live: weakref.WeakValueDictionary[str, RdbmsVault] = weakref.WeakValueDictionary()

    def get(self, session_id: str) -> Vault:
        # Every view the LRU holds is live, so the registry answers for both.
        view = self._live.get(session_id)
        if view is None:
            view = RdbmsVault(self._store, session_id, clock=self._clock)
            self._live[session_id] = view
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

    @property
    def owner_bound_shared(self) -> bool:
        """``RdbmsStore.owner_bound_shared`` (surfaced in /status and doctor)."""
        return self._store.owner_bound_shared

    @property
    def fault_types(self) -> tuple[type[BaseException], ...]:
        """``RdbmsStore.fault_types`` (the proxy's vault-fault refusal)."""
        return self._store.fault_types

    def prune_sessions(self, days: int, *, exclude: frozenset[str] = frozenset()) -> int:
        doomed = self._store.prune_sessions(days, exclude=exclude)
        self._drop_views(doomed)
        return len(doomed)

    def forget_sessions(self, session_ids: Iterable[str]) -> int:
        wanted = sorted(set(session_ids))
        if not wanted:
            return 0
        present = self._store.forget_sessions(wanted)
        self._drop_views(wanted)
        return present

    def _drop_views(self, session_ids: Iterable[str]) -> None:
        """SqliteVaultManager._drop_views: evicted, and rebuilt where held."""
        for session_id in session_ids:
            self._views.pop(session_id, None)
            view = self._live.get(session_id)
            if view is not None:
                view._load()

    def record_response_session(self, response_id: str, session_id: str) -> None:
        self._store.record_response_session(response_id, session_id)

    def record_object_session(self, object_id: str, session_id: str) -> None:
        self._store.record_object_session(object_id, session_id)

    def lookup_response_session(self, response_id: str) -> str | None:
        return self._store.lookup_response_session(response_id)

    def lookup_response_sessions(self, response_ids: Iterable[str]) -> dict[str, str]:
        wanted = list(dict.fromkeys(response_ids))
        return self._store.lookup_response_sessions(wanted) if wanted else {}

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
