"""Placeholder-to-original mapping store. The mapping never leaves the machine."""

import hmac
import os
import sqlite3
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol

from llm_redact.placeholders import MAX_TOKEN_NUMBER, format_placeholder

if TYPE_CHECKING:
    from llm_redact.config import VaultConfig
    from llm_redact.plugin_api import VaultCipher


class VaultKeyError(RuntimeError):
    """The vault encryption key is missing, malformed, or wrong.

    Defined here (dependency-free) so callers can catch it without the
    ``crypto`` extra installed."""


class PlaceholderSpaceExhausted(RuntimeError):
    """A new placeholder would need a number above MAX_TOKEN_NUMBER: the
    request carries a token numbered at the limit (or, in theory, the
    session issued that many). Refused, never wrapped around or reused —
    the redactor turns it into a refused request. The message names the
    detector type only, never a value."""

    def __init__(self, detector_type: str) -> None:
        super().__init__(
            f"no {detector_type} placeholder number above the ones this request carries"
            f" is left to issue (the limit is {MAX_TOKEN_NUMBER})"
        )
        self.detector_type = detector_type


def next_number(issued: int, floor: int, detector_type: str) -> int:
    """The number a NEW placeholder takes: above every number the session
    has issued (``issued`` is their maximum, 0 when none) AND above
    ``floor``, the highest same-type number the request being redacted
    already carries. Without a floor that is the dense MAX(n)+1; with one,
    the numbers in between are skipped — a gap, never a reuse."""
    n = max(issued, floor) + 1
    if n > MAX_TOKEN_NUMBER:
        raise PlaceholderSpaceExhausted(detector_type)
    return n


class Vault(Protocol):
    def placeholder_for(self, detector_type: str, original: str, *, floor: int = 0) -> str:
        """Get or create the placeholder for an original value.

        A value the session already mapped keeps its token whatever the
        floor. A NEW value is numbered ``next_number``: above the session's
        own numbers and above ``floor`` — the highest same-type token number
        the request carries (placeholders.token_floors), so a token the
        session never issued (a compacted history, a pasted answer) never
        gets a second meaning. Callers pass ``floor`` only when it is
        non-zero, so a vault predating the keyword still serves every
        request that carries no tokens."""
        ...

    def original_for(self, placeholder: str) -> str | None:
        """Reverse lookup; None when the placeholder is unknown."""
        ...

    def close(self) -> None: ...

    def __len__(self) -> int:
        """Number of mappings held (for /status; never the values)."""
        ...


class InMemoryVault:
    """Session-scoped vault: deterministic within one proxy process.

    The same (detector_type, original) pair always yields the same
    placeholder, so entity identity stays coherent across a conversation.
    """

    def __init__(self) -> None:
        self._forward: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def placeholder_for(self, detector_type: str, original: str, *, floor: int = 0) -> str:
        key = f"{detector_type}::{original}"
        existing = self._forward.get(key)
        if existing is not None:
            return existing
        n = next_number(self._counters.get(detector_type, 0), floor, detector_type)
        self._counters[detector_type] = n
        placeholder = format_placeholder(detector_type, n)
        self._forward[key] = placeholder
        self._reverse[placeholder] = original
        return placeholder

    def original_for(self, placeholder: str) -> str | None:
        return self._reverse.get(placeholder)

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self._reverse)


class EncryptedInMemoryVault:
    """In-memory vault holding originals Fernet-encrypted (crypto extra).

    The RAM mapping holds ciphertext plus a domain-separated HMAC index
    instead of plaintext, narrowing core-dump/swap exposure. It does NOT
    change the threat model's same-UID stance: the key — and whatever
    plaintext is being substituted right now — still lives in process
    memory, and the docs say so. Reverse lookups decrypt per hit
    (deliberately no plaintext cache — that would defeat the point).
    """

    def __init__(self, cipher: "VaultCipher", session_id: str) -> None:
        self._cipher = cipher
        self._session_id = session_id  # domain-separates the HMAC index
        self._forward: dict[str, str] = {}  # HMAC(index key, value) -> placeholder
        self._reverse: dict[str, bytes] = {}  # placeholder -> Fernet token
        self._counters: dict[str, int] = {}

    def placeholder_for(self, detector_type: str, original: str, *, floor: int = 0) -> str:
        mac = self._cipher.mac(self._session_id, detector_type, original)
        existing = self._forward.get(mac)
        if existing is not None:
            return existing
        n = next_number(self._counters.get(detector_type, 0), floor, detector_type)
        # Encrypt BEFORE any state changes: a failing cipher must leave no
        # forward entry whose token could never be restored.
        ciphertext = self._cipher.encrypt(original)
        self._counters[detector_type] = n
        placeholder = format_placeholder(detector_type, n)
        self._forward[mac] = placeholder
        self._reverse[placeholder] = ciphertext
        return placeholder

    def original_for(self, placeholder: str) -> str | None:
        token = self._reverse.get(placeholder)
        return None if token is None else self._cipher.decrypt(token)

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self._reverse)


# v2: plaintext originals. v3: HMAC index + Fernet ciphertext (crypto extra).
_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS mappings (
  session_id TEXT NOT NULL,
  detector_type TEXT NOT NULL,
  original TEXT NOT NULL,
  placeholder TEXT NOT NULL,
  n INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (session_id, detector_type, original),
  UNIQUE (session_id, placeholder),
  UNIQUE (session_id, detector_type, n)
);
CREATE TABLE IF NOT EXISTS response_sessions (
  response_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
"""

_MAPPINGS_V3_COLUMNS = """
  session_id TEXT NOT NULL,
  detector_type TEXT NOT NULL,
  original_mac TEXT NOT NULL,
  original_ct BLOB NOT NULL,
  placeholder TEXT NOT NULL,
  n INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  PRIMARY KEY (session_id, detector_type, original_mac),
  UNIQUE (session_id, placeholder),
  UNIQUE (session_id, detector_type, n)
"""

_SCHEMA_V3 = f"""
CREATE TABLE IF NOT EXISTS mappings ({_MAPPINGS_V3_COLUMNS});
CREATE TABLE IF NOT EXISTS response_sessions (
  response_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE IF NOT EXISTS vault_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# The response-id map keeps at most this many rows of sessions that hold no
# mappings; rows of live sessions are removed with their session (prune).
_MAX_RESPONSE_ROWS = 10000
_RESPONSE_PRUNE_EVERY = 256
# How many ids one batched response-map lookup binds per query (well under
# every engine's parameter limit: sqlite's, and Oracle's 1000-item IN list).
LOOKUP_CHUNK = 500


def default_vault_path() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    return Path(xdg) / "llm-redact" / "vault.db"


def _open_connection(path: Path, cipher: "VaultCipher | None" = None) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    # Pre-create with tight permissions before SQLite touches it.
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    # Wait on a briefly-held write lock instead of failing immediately — the
    # multi-process case (two proxies over one DB) contends on the WAL writer.
    conn.execute("PRAGMA busy_timeout=5000")
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])

    if cipher is None:
        if version >= 3:
            conn.close()
            # Fail closed BEFORE any request: opening an encrypted vault
            # without the key must never silently issue fresh tokens.
            from llm_redact.config import ConfigError

            raise ConfigError(
                f'the vault at {path} is encrypted; set [vault] encryption = "fernet" '
                "and LLM_REDACT_VAULT_KEY (the migration is one-way)"
            )
        conn.executescript(_SCHEMA_V2)
        if version < 2:
            conn.execute("PRAGMA user_version = 2")
        return conn

    if version >= 3:
        conn.executescript(_SCHEMA_V3)
        _verify_key(conn, cipher, path)
        return conn
    # v0 (fresh) and v2 (plaintext) both migrate: an empty v2 rebuild is
    # exactly fresh-v3 creation, so one path covers both.
    conn.executescript(_SCHEMA_V2)
    _migrate_to_v3(conn, cipher)
    return conn


def _verify_key(conn: sqlite3.Connection, cipher: "VaultCipher", path: Path) -> None:
    row = conn.execute("SELECT value FROM vault_meta WHERE key = 'key_check'").fetchone()
    if row is None:  # pragma: no cover - vault_meta always written at migration
        conn.execute(
            "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('key_check', ?)",
            (cipher.key_check(),),
        )
        return
    if not hmac.compare_digest(str(row[0]), cipher.key_check()):
        conn.close()
        raise VaultKeyError(f"LLM_REDACT_VAULT_KEY does not match the vault at {path}")


def _migrate_to_v3(conn: sqlite3.Connection, cipher: "VaultCipher") -> None:
    """Encrypt-in-place, one transaction: a crash rolls back to intact v2.

    Refuse-to-mix was rejected — it would strand every existing sqlite user
    behind a manual export step. The migration is one-way.
    """
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(f"CREATE TABLE mappings_v3 ({_MAPPINGS_V3_COLUMNS})")
    rows = conn.execute(
        "SELECT session_id, detector_type, original, placeholder, n, created_at FROM mappings"
    ).fetchall()
    for session_id, detector_type, original, placeholder, n, created_at in rows:
        conn.execute(
            "INSERT INTO mappings_v3"
            " (session_id, detector_type, original_mac, original_ct, placeholder, n, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                detector_type,
                cipher.mac(session_id, detector_type, original),
                cipher.encrypt(original),
                placeholder,
                n,
                created_at,
            ),
        )
    conn.execute("DROP TABLE mappings")
    conn.execute("ALTER TABLE mappings_v3 RENAME TO mappings")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vault_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('key_check', ?)",
        (cipher.key_check(),),
    )
    conn.execute("PRAGMA user_version = 3")
    conn.execute("COMMIT")
    # Plaintext lingers in the WAL and freed pages after the rebuild;
    # checkpoint + VACUUM scrub both, or the at-rest claim is false.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")


def rotate_vault_key(
    conn: sqlite3.Connection, old_cipher: "VaultCipher", new_cipher: "VaultCipher"
) -> int:
    """Re-encrypt every mapping under new_cipher, one transaction. Returns count.

    Both HKDF subkeys change on rotation, so BOTH stored columns must change:
    original_ct (decrypt-old, encrypt-new) and original_mac (recompute over
    the same plaintext — the MAC is a deterministic PK-component index, which
    is exactly why lazy/MultiFernet rotation cannot work here). placeholder,
    n, and created_at are copied VERBATIM, so token identity and the dense
    per-(session,type) counter are invariant — no number is ever reused or
    shifted, and the (session,type,value)->token function is unchanged.

    key_check is bumped INSIDE the transaction: a crash rolls back to the
    fully-old-key state (reopens fine with the old key), success is fully
    new-key — there is no observable mixed state, so the "lost write reissues
    a live number" sin cannot occur. The final checkpoint + VACUUM scrub the
    old ciphertext from the WAL and freed pages, or the at-rest claim is false.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(f"CREATE TABLE mappings_rot ({_MAPPINGS_V3_COLUMNS})")
        rows = conn.execute(
            "SELECT session_id, detector_type, original_ct, placeholder, n, created_at"
            " FROM mappings"
        ).fetchall()
        for session_id, detector_type, original_ct, placeholder, n, created_at in rows:
            original = old_cipher.decrypt(original_ct)
            conn.execute(
                "INSERT INTO mappings_rot (session_id, detector_type, original_mac,"
                " original_ct, placeholder, n, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    detector_type,
                    new_cipher.mac(session_id, detector_type, original),
                    new_cipher.encrypt(original),
                    placeholder,
                    n,
                    created_at,
                ),
            )
        conn.execute("DROP TABLE mappings")
        conn.execute("ALTER TABLE mappings_rot RENAME TO mappings")
        conn.execute(
            "INSERT OR REPLACE INTO vault_meta (key, value) VALUES ('key_check', ?)",
            (new_cipher.key_check(),),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    return len(rows)


class SqliteVault:
    """Per-session view over a shared persistent database.

    Within a named session, (detector_type, original) yields the same
    placeholder across proxy restarts — tokens issued before a restart keep
    rehydrating after it, and provider prompt caches (which key on exact
    prefix bytes) stay coherent.

    The database holds real secrets: the file is created 0600 in a 0700
    directory (WAL sidecars inherit the file's mode). synchronous=FULL is
    deliberate — a mapping lost to power failure would let MAX(n) reissue an
    old placeholder for a *different* value, silently rehydrating history to
    the wrong secret. Inserts happen only on first sight of a value, so the
    fsync cost is negligible.

    A write-through cache preloaded per session keeps ``original_for`` off
    the database on the streaming hot path. Views share one connection
    (single-process asyncio; point ops are microseconds).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        session: str,
        *,
        owns_connection: bool,
        cipher: "VaultCipher | None" = None,
    ) -> None:
        self._conn = conn
        self._session = session
        self._owns_connection = owns_connection
        self._cipher = cipher
        # Caches are keyed by plaintext either way, so the hot path (a cache
        # hit) is identical with and without encryption; decryption happens
        # once per row at preload.
        self._forward: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        if cipher is None:
            preload = self._conn.execute(
                "SELECT detector_type, original, placeholder FROM mappings WHERE session_id = ?",
                (session,),
            )
            for detector_type, original, placeholder in preload:
                self._forward[f"{detector_type}::{original}"] = placeholder
                self._reverse[placeholder] = original
        else:
            preload = self._conn.execute(
                "SELECT detector_type, original_ct, placeholder FROM mappings WHERE session_id = ?",
                (session,),
            )
            for detector_type, original_ct, placeholder in preload:
                original = cipher.decrypt(original_ct)
                self._forward[f"{detector_type}::{original}"] = placeholder
                self._reverse[placeholder] = original

    def placeholder_for(self, detector_type: str, original: str, *, floor: int = 0) -> str:
        key = f"{detector_type}::{original}"
        existing = self._forward.get(key)
        if existing is not None:
            return existing
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            # MAX(n) read fresh inside the write lock: a retry after a
            # rolled-back write computes the same max(MAX(n), floor) + 1.
            row = self._conn.execute(
                "SELECT COALESCE(MAX(n), 0) FROM mappings"
                " WHERE session_id = ? AND detector_type = ?",
                (self._session, detector_type),
            ).fetchone()
            n = next_number(int(row[0]), floor, detector_type)
            placeholder = format_placeholder(detector_type, n)
            if self._cipher is None:
                self._conn.execute(
                    "INSERT INTO mappings (session_id, detector_type, original, placeholder, n)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (self._session, detector_type, original, placeholder, n),
                )
            else:
                self._conn.execute(
                    "INSERT INTO mappings"
                    " (session_id, detector_type, original_mac, original_ct, placeholder, n)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        self._session,
                        detector_type,
                        self._cipher.mac(self._session, detector_type, original),
                        self._cipher.encrypt(original),
                        placeholder,
                        n,
                    ),
                )
            self._conn.execute("COMMIT")
        except sqlite3.IntegrityError:
            # Another process sharing the DB inserted this original first.
            self._conn.execute("ROLLBACK")
            if self._cipher is None:
                row = self._conn.execute(
                    "SELECT placeholder FROM mappings"
                    " WHERE session_id = ? AND detector_type = ? AND original = ?",
                    (self._session, detector_type, original),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT placeholder FROM mappings"
                    " WHERE session_id = ? AND detector_type = ? AND original_mac = ?",
                    (
                        self._session,
                        detector_type,
                        self._cipher.mac(self._session, detector_type, original),
                    ),
                ).fetchone()
            if row is None:  # pragma: no cover - constraint failed another way
                raise
            placeholder = str(row[0])
        except BaseException:
            # Any other failure (disk full, I/O error, lock timeout, a cipher
            # fault, an exhausted number space): roll back so the open BEGIN
            # IMMEDIATE can't wedge the connection for the next request, and
            # fail closed. Nothing was cached (the caches are written only
            # after a successful commit below), and n is computed from MAX(n)
            # read fresh on every call, so the next attempt reissues the same
            # number — never a skipped number, never a reused token.
            with suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            raise
        self._forward[key] = placeholder
        self._reverse[placeholder] = original
        return placeholder

    def original_for(self, placeholder: str) -> str | None:
        cached = self._reverse.get(placeholder)
        if cached is not None:
            return cached
        # Cache miss can only mean another process issued the token.
        if self._cipher is None:
            row = self._conn.execute(
                "SELECT original FROM mappings WHERE session_id = ? AND placeholder = ?",
                (self._session, placeholder),
            ).fetchone()
            if row is None:
                return None
            original = str(row[0])
        else:
            row = self._conn.execute(
                "SELECT original_ct FROM mappings WHERE session_id = ? AND placeholder = ?",
                (self._session, placeholder),
            ).fetchone()
            if row is None:
                return None
            original = self._cipher.decrypt(row[0])
        self._reverse[placeholder] = original
        return original

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def __len__(self) -> int:
        return len(self._reverse)


def open_sqlite_vault(path: Path, session: str, cipher: "VaultCipher | None" = None) -> SqliteVault:
    """Standalone single-session vault owning its connection (static mode,
    tests)."""
    return SqliteVault(_open_connection(path, cipher), session, owns_connection=True, cipher=cipher)


class VaultManager(Protocol):
    def get(self, session_id: str) -> Vault: ...

    def session_count(self) -> int: ...

    def total_entries(self) -> int: ...

    def sessions_summary(self) -> list[dict[str, object]]: ...

    def prune_sessions(self, days: int, *, exclude: frozenset[str] = ...) -> int: ...

    def forget_sessions(self, session_ids: Iterable[str]) -> int: ...

    def record_response_session(self, response_id: str, session_id: str) -> None: ...

    def lookup_response_session(self, response_id: str) -> str | None: ...

    def close(self) -> None: ...


class InMemoryVaultManager:
    # No durable response-session map: the session router's in-memory map
    # is the only truth (ProxyState then hands the router no durable lookup).
    durable_response_map = False

    def __init__(self, cipher: "VaultCipher | None" = None) -> None:
        self._cipher = cipher
        self._vaults: dict[str, Vault] = {}

    def get(self, session_id: str) -> Vault:
        vault = self._vaults.get(session_id)
        if vault is None:
            vault = (
                EncryptedInMemoryVault(self._cipher, session_id)
                if self._cipher is not None
                else InMemoryVault()
            )
            self._vaults[session_id] = vault
        return vault

    def has_session(self, session_id: str) -> bool:
        """Whether the session holds mappings, WITHOUT creating it (``get``
        would add an empty session to this manager)."""
        vault = self._vaults.get(session_id)
        return vault is not None and len(vault) > 0

    def session_count(self) -> int:
        return len(self._vaults)

    def total_entries(self) -> int:
        return sum(len(v) for v in self._vaults.values())

    def sessions_summary(self) -> list[dict[str, object]]:
        # Memory mappings carry no timestamps: counts only, insertion order.
        return [
            {"session": session_id, "entries": len(vault), "first": None, "last": None}
            for session_id, vault in self._vaults.items()
        ]

    def prune_sessions(self, days: int, *, exclude: frozenset[str] = frozenset()) -> int:
        # Nothing to prune by age: memory sessions have no timestamps and
        # die with the process anyway.
        return 0

    def forget_sessions(self, session_ids: Iterable[str]) -> int:
        """Drop whole sessions (a purged user's); how many existed."""
        dropped = [self._vaults.pop(session_id, None) for session_id in set(session_ids)]
        return sum(vault is not None for vault in dropped)

    def record_response_session(self, response_id: str, session_id: str) -> None:
        pass  # the SessionRouter's in-memory map is authoritative here

    def lookup_response_session(self, response_id: str) -> str | None:
        return None

    def lookup_response_sessions(self, response_ids: Iterable[str]) -> dict[str, str]:
        return {}

    def close(self) -> None:
        pass


class SqliteVaultManager:
    """One shared connection; per-session views cached in a small LRU.

    Eviction only drops a view's write-through cache — every mapping lives in
    the database, and a re-created view preloads its session's rows (small
    for per-conversation sessions).
    """

    def __init__(
        self, path: Path, *, cipher: "VaultCipher | None" = None, view_cache_size: int = 64
    ) -> None:
        self._conn = _open_connection(path, cipher)
        self._cipher = cipher
        self._views: OrderedDict[str, SqliteVault] = OrderedDict()
        self._view_cache_size = view_cache_size
        self._response_inserts = 0

    def get(self, session_id: str) -> Vault:
        view = self._views.get(session_id)
        if view is None:
            view = SqliteVault(self._conn, session_id, owns_connection=False, cipher=self._cipher)
            self._views[session_id] = view
        self._views.move_to_end(session_id)
        while len(self._views) > self._view_cache_size:
            self._views.popitem(last=False)
        return view

    def session_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(DISTINCT session_id) FROM mappings").fetchone()
        return int(row[0])

    def total_entries(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM mappings").fetchone()[0])

    def sessions_summary(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT session_id, COUNT(*), MIN(created_at), MAX(created_at)"
            " FROM mappings GROUP BY session_id ORDER BY MAX(created_at) DESC"
        ).fetchall()
        return [
            {"session": str(session_id), "entries": int(count), "first": first, "last": last}
            for session_id, count, first, last in rows
        ]

    def prune_sessions(self, days: int, *, exclude: frozenset[str] = frozenset()) -> int:
        """Delete whole idle sessions in one transaction, then drop their
        cached views.

        Whole sessions only — the same rule as the CLI: deleting individual
        rows would let MAX(n)+1 reissue a still-referenced placeholder
        number for a different value. Unlike the CLI (which requires a
        proxy restart), this runs inside the live process, so evicting the
        views keeps the proxy from serving pruned mappings from cache;
        callers exclude the always-live static session.
        """
        doomed = [
            str(row[0])
            for row in self._conn.execute(
                "SELECT session_id FROM mappings GROUP BY session_id"
                " HAVING MAX(created_at) < strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?)",
                (f"-{days} days",),
            )
            if str(row[0]) not in exclude
        ]
        if not doomed:
            return 0
        marks = ",".join("?" * len(doomed))
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(f"DELETE FROM mappings WHERE session_id IN ({marks})", doomed)
            self._conn.execute(
                f"DELETE FROM response_sessions WHERE session_id IN ({marks})", doomed
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        for session_id in doomed:
            self._views.pop(session_id, None)
        return len(doomed)

    def forget_sessions(self, session_ids: Iterable[str]) -> int:
        """Delete whole named sessions (mappings and response rows) in one
        transaction and drop their cached views; how many held mappings.
        Whole sessions only, like prune — never a partial delete."""
        wanted = sorted(set(session_ids))
        if not wanted:
            return 0
        marks = ",".join("?" * len(wanted))
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            present = self._conn.execute(
                f"SELECT COUNT(DISTINCT session_id) FROM mappings WHERE session_id IN ({marks})",
                wanted,
            ).fetchone()[0]
            self._conn.execute(f"DELETE FROM mappings WHERE session_id IN ({marks})", wanted)
            self._conn.execute(
                f"DELETE FROM response_sessions WHERE session_id IN ({marks})", wanted
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        for session_id in wanted:
            self._views.pop(session_id, None)
        return int(present)

    def record_response_session(self, response_id: str, session_id: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO response_sessions (response_id, session_id) VALUES (?, ?)",
            (response_id, session_id),
        )
        self._response_inserts += 1
        if self._response_inserts >= _RESPONSE_PRUNE_EVERY:
            self._response_inserts = 0
            # Beyond the cap, only rows whose session holds no mappings go
            # (a pruned session, or one that never redacted anything): a
            # router reads a missing row as "that session was pruned", and a
            # chain into a LIVE session resumed in a fresh one would reissue
            # «EMAIL_001» for a new value while the provider's history still
            # means the old one. Live sessions' rows leave with the session.
            self._conn.execute(
                "DELETE FROM response_sessions WHERE response_id NOT IN"
                " (SELECT response_id FROM response_sessions ORDER BY created_at DESC LIMIT ?)"
                " AND NOT EXISTS"
                " (SELECT 1 FROM mappings m WHERE m.session_id = response_sessions.session_id)",
                (_MAX_RESPONSE_ROWS,),
            )

    def lookup_response_session(self, response_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT session_id FROM response_sessions WHERE response_id = ?", (response_id,)
        ).fetchone()
        return str(row[0]) if row is not None else None

    def lookup_response_sessions(self, response_ids: Iterable[str]) -> dict[str, str]:
        """``lookup_response_session`` for many ids at once — one query per
        ``LOOKUP_CHUNK`` distinct ids, not one per id (a listing names
        thousands): the recorded ids with their sessions, an unknown id
        simply absent."""
        wanted = list(dict.fromkeys(response_ids))
        found: dict[str, str] = {}
        for start in range(0, len(wanted), LOOKUP_CHUNK):
            chunk = wanted[start : start + LOOKUP_CHUNK]
            marks = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                "SELECT response_id, session_id FROM response_sessions"
                f" WHERE response_id IN ({marks})",
                chunk,
            )
            found.update((str(response_id), str(session_id)) for response_id, session_id in rows)
        return found

    def close(self) -> None:
        self._conn.close()


def build_cipher(config: "VaultConfig") -> "VaultCipher | None":
    """The configured cipher, or None when encryption is off.

    The concrete cipher (Fernet + HKDF) is a paid subsystem in the
    ``llm-redact-pro`` package. This Free default returns None for a
    plaintext vault and fails closed when encryption is requested without
    the paid package — never a silent downgrade to plaintext. When
    llm-redact-pro is installed it overrides this factory (and
    ``build_vault_manager``) via the registry with a real cipher.
    """
    if config.encryption != "fernet":
        return None
    from llm_redact.config import ConfigError

    raise ConfigError(
        '[vault] encryption = "fernet" requires the llm-redact-pro package '
        "(pip install llm-redact-pro); the Free tier keeps the in-memory and "
        "unencrypted sqlite vaults"
    )


def cipher_from_key(master_key: bytes) -> "VaultCipher":
    """Build a cipher from a raw master key (rotate-key's new key).

    Paid, like ``build_cipher``: the Free default fails closed and
    llm-redact-pro overrides it via the registry."""
    from llm_redact.config import ConfigError

    raise ConfigError("vault key operations require the llm-redact-pro package")


def _rdbms_backend_requires_pro(backend: str) -> NoReturn:
    from llm_redact.config import ConfigError

    raise ConfigError(
        f"[vault] backend = {backend!r} (a server RDBMS vault) requires the "
        "llm-redact-pro package (pip install llm-redact-pro)"
    )


def build_vault(config: "VaultConfig") -> Vault:
    """Single-session vault for the configured (static) session."""
    if config.backend == "memory":
        return InMemoryVault()
    from llm_redact.config import RDBMS_BACKENDS

    if config.backend in RDBMS_BACKENDS:
        _rdbms_backend_requires_pro(config.backend)
    path = Path(config.path).expanduser() if config.path else default_vault_path()
    return open_sqlite_vault(path, config.session, build_cipher(config))


def build_vault_manager(config: "VaultConfig") -> VaultManager:
    if config.backend == "memory":
        # encryption = "fernet" applies here too (previously it was
        # silently ignored for the memory backend): same key resolution,
        # same fail-closed behavior when the key or package is absent.
        return InMemoryVaultManager(cipher=build_cipher(config))
    from llm_redact.config import RDBMS_BACKENDS

    if config.backend in RDBMS_BACKENDS:
        _rdbms_backend_requires_pro(config.backend)
    path = Path(config.path).expanduser() if config.path else default_vault_path()
    return SqliteVaultManager(path, cipher=build_cipher(config))
