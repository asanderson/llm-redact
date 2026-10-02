"""Placeholder-to-original mapping store. The mapping never leaves the machine."""

import hmac
import logging
import os
import sqlite3
import time
import weakref
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, TypeVar

from llm_redact.placeholders import MAX_TOKEN_NUMBER, format_placeholder
from llm_redact.vault_writer import MISS, MapWrite, MapWriter, Removed, holding

if TYPE_CHECKING:
    from llm_redact.config import VaultConfig
    from llm_redact.plugin_api import VaultCipher

T = TypeVar("T")

logger = logging.getLogger("llm_redact")

# How long a persistent view (sqlite, RDBMS) serves from its caches before it
# re-reads its session's retired number: another proxy instance sharing the
# database may have deleted the session meanwhile. Stale caches can never
# restore a WRONG value (a deleted session's numbers are never issued again —
# see ``delete_sessions``); this bounds how long a deleted session's values
# stay restorable from another instance's memory.
CACHE_CHECK_SECONDS = 1.0
# The proxy's bookkeeping stage a failed staleness check is counted under.
CHECK_FAULT_STAGE = "vault_check"


class CheckFaults:
    """The failed staleness checks of every view over one database (one
    manager's). A check that cannot read the database keeps the view's
    caches as they are — a cached token only ever restores its own value —
    and runs again at the next interval, so a cache hit never needs the
    database. Each failure is counted in ``counter`` (the proxy's
    ``bookkeeping_errors``, bound by the manager's ``bind_fault_counter``);
    an outage is logged when it starts and when the database answers again,
    by exception TYPE only (a driver's message can name a host, a file or a
    value)."""

    __slots__ = ("counter", "failing")

    # The bookkeeping stage a failure is counted under, and the outage's two
    # log lines (a subclass names its own).
    stage = CHECK_FAULT_STAGE
    outage_message = (
        "vault staleness check failed (%s): cached values are served until the"
        " database answers again"
    )
    recovery_message = "vault staleness check: the database answers again"

    def __init__(self) -> None:
        self.counter: Counter[str] | None = None
        self.failing = False

    def failed(self, exc: Exception) -> None:
        if self.counter is not None:
            self.counter[self.stage] += 1
        if not self.failing:
            self.failing = True
            logger.warning(self.outage_message, type(exc).__name__)

    def succeeded(self) -> None:
        if self.failing:
            self.failing = False
            logger.info(self.recovery_message)


# The proxy's bookkeeping stage a failed Live resumption handle-map read or
# write is counted under (``HandleWriteFaults``/``HandleReadFaults``).
HANDLE_FAULT_STAGE = "handle_map"


class HandleWriteFaults(CheckFaults):
    """The failed writes of one manager's durable Live resumption handle map
    (``record_handle_session``). Contained: a write that fails leaves the
    handle unrecorded, so it reads as unknown — refused by the plugin that
    asks, never resumed into a session the vault no longer vouches for.
    Counted and logged like ``CheckFaults`` (by type only), as an outage of
    its own: writes can fail while reads work (a missing grant), and only a
    write that succeeds ends it."""

    __slots__ = ()

    stage = HANDLE_FAULT_STAGE
    outage_message = (
        "vault handle map write failed (%s): Live resumption handles go unrecorded"
        " (read as unknown) until a write succeeds again"
    )
    recovery_message = "vault handle map: writes succeed again"


class HandleReadFaults(CheckFaults):
    """The failed reads of one manager's durable Live resumption handle map
    (``lookup_handle_session``): each answers None, so the handle reads as
    unknown (fail closed). An outage of its own (``HandleWriteFaults``):
    only a read that succeeds ends it."""

    __slots__ = ()

    stage = HANDLE_FAULT_STAGE
    outage_message = (
        "vault handle map read failed (%s): Live resumption handles read as unknown"
        " until the database answers again"
    )
    recovery_message = "vault handle map: reads succeed again"


class ResponseMapWriteFaults(CheckFaults):
    """Failed BACKGROUND writes of Responses chain rows (``vault_writer``;
    stage ``response_id``, the proxy's own for a lost mapping): the response
    id reads as unknown — an orphan session, never a wrong value."""

    __slots__ = ()

    stage = "response_id"
    outage_message = (
        "vault response map write failed (%s): Responses chains go unrecorded"
        " (read as unknown) until a write succeeds again"
    )
    recovery_message = "vault response map: writes succeed again"


class ObjectMapWriteFaults(CheckFaults):
    """Failed BACKGROUND writes of stored-object owner records
    (``vault_writer``; stage ``object_ids``): the object reads as unknown
    (the router's unknown-object case)."""

    __slots__ = ()

    stage = "object_ids"
    outage_message = (
        "vault owner record write failed (%s): stored objects go unrecorded"
        " (read as unknown) until a write succeeds again"
    )
    recovery_message = "vault owner records: writes succeed again"


# The overlay keyspaces of a manager's background writer (``vault_writer``):
# a response map row (Responses chain or owner record: one primary key) and
# a handle map row.
ROW = "row"
HANDLE = "handle"


def row_verdict(writer: MapWriter | None, row_id: str) -> "str | None | object":
    """What a manager's background writer says of a response map row:
    its pending session, None (deleted with its session), or ``MISS``
    (nothing pending: the database answers)."""
    if writer is None:
        return MISS
    return writer.verdict((ROW, row_id))


def overlaid_rows(
    writer: MapWriter | None, row_ids: list[str], read: Callable[[list[str]], dict[str, str]]
) -> dict[str, str]:
    """A batched response map lookup: the ids a pending write answers for
    from the writer's overlay, the rest from ``read`` (the database)."""
    found: dict[str, str] = {}
    rest = row_ids
    if writer is not None:
        rest = []
        for row_id in row_ids:
            verdict = writer.verdict((ROW, row_id))
            if verdict is MISS:
                rest.append(row_id)
            elif isinstance(verdict, str):
                found[row_id] = verdict
    if rest:
        found.update(read(rest))
    return found


def handle_answer(verdict: object, read: Callable[[], str | None]) -> str | None:
    """A handle map lookup under a background writer's ``verdict``: a
    pending session (or None: its row is gone) answers alone; otherwise the
    database (``read``) does — None where the overlay holds its row as
    superseded (``Removed``) for the session it names."""
    if verdict is None or isinstance(verdict, str):
        return verdict
    session = read()
    if isinstance(verdict, Removed) and session in verdict.sessions:
        return None
    return session


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

    # Optional, read with getattr (``run_batched``): ``batched(work)`` runs
    # ``work`` — one request's redaction — with every NEW value it issues
    # written in ONE transaction, committed once when it returns and rolled
    # back whole if anything in it fails. A vault without it issues per call.


def run_batched(vault: Vault, work: Callable[[], T]) -> T:
    """Run ``work`` — one request's whole redaction — with every new value
    it issues written in one transaction where the vault offers that (its
    optional ``batched``: the sqlite vault), else per call as before.

    ``work`` is synchronous by construction: it cannot await, so no other
    request can reach the vault's shared connection while the transaction
    is open, and nothing it issued can be forwarded before it committed."""
    batched = getattr(vault, "batched", None)
    if batched is None:
        return work()
    result: T = batched(work)
    return result


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

    def forget_mappings(self) -> None:
        """Drop every mapping but keep the numbering: a new value is
        numbered above everything this session ever issued, so a dropped
        token still in a provider's history (or a live connection's frames)
        never gains a second meaning."""
        self._forward.clear()
        self._reverse.clear()

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

    def forget_mappings(self) -> None:
        """InMemoryVault.forget_mappings: the mappings go, the numbering
        stays."""
        self._forward.clear()
        self._reverse.clear()

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self._reverse)


# A whole-session delete (prune, forget) retires every number the session
# held: the session's new values are numbered above it, forever (see
# ``delete_sessions``). One row per session ever deleted; no values.
_RETIRED_TABLE = (
    "CREATE TABLE IF NOT EXISTS retired_numbers"
    " (session_id TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID"
)

# The durable Live resumption handle map (``record_handle_session``): a
# handle's DIGEST (the plugin's domain-separated hash; never a raw handle) ->
# the session it was issued in. ``seq`` is the rowid: a new row is numbered
# above every row present, so the trims below follow INSERTION order even
# for rows written within one second (``created_at``'s resolution). Rows
# leave with their session on every whole-session delete.
_HANDLE_TABLE = (
    "CREATE TABLE IF NOT EXISTS handle_sessions (seq INTEGER PRIMARY KEY,"
    " handle_digest TEXT NOT NULL UNIQUE, session_id TEXT NOT NULL)"
)
_HANDLE_INDEX = (
    "CREATE INDEX IF NOT EXISTS handle_sessions_by_session ON handle_sessions (session_id, seq)"
)

# v2: plaintext originals. v3: HMAC index + Fernet ciphertext (crypto extra).
_SCHEMA_V2 = f"""
{_RETIRED_TABLE};
{_HANDLE_TABLE};
{_HANDLE_INDEX};
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
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  kind TEXT NOT NULL DEFAULT 'response'
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
{_RETIRED_TABLE};
{_HANDLE_TABLE};
{_HANDLE_INDEX};
CREATE TABLE IF NOT EXISTS mappings ({_MAPPINGS_V3_COLUMNS});
CREATE TABLE IF NOT EXISTS response_sessions (
  response_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
  kind TEXT NOT NULL DEFAULT 'response'
);
CREATE TABLE IF NOT EXISTS vault_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# The response-id map keeps at most this many rows of sessions that hold no
# mappings; rows of live sessions are removed with their session (prune).
# Its two kinds of rows are bounded APART: a Responses chain's rows
# (``record_response_session``) and a stored object's owner record
# (``record_object_session``) — every Responses turn writes one, so a shared
# bound let ordinary traffic push out who created a file whose creating
# session never redacted anything (then attributed to nobody: refused).
_MAX_RESPONSE_ROWS = 10000
_MAX_OBJECT_ROWS = 10000
_RESPONSE_PRUNE_EVERY = 256
_RESPONSE_KIND = "response"
_OBJECT_KIND = "object"
# The handle map's bound (``write_handle``): each session keeps its newest
# MAX_SESSION_HANDLES rows (a client resumes with the NEWEST handle its
# connection was sent, and the plugin supersedes a connection's older ones
# through ``replaces`` — so this counts recent connections, not handles), and
# beyond the newest MAX_HANDLE_ROWS rows in all only rows of sessions holding
# no mappings go (checked every _RESPONSE_PRUNE_EVERY writes), so a live
# session's newest handles are never trimmed by other sessions' traffic.
# Rows in all: at most MAX_HANDLE_ROWS (+ one check interval) plus
# MAX_SESSION_HANDLES per session holding mappings; a session's rows leave
# with it (TTL prune, forget).
MAX_SESSION_HANDLES = 1024
MAX_HANDLE_ROWS = 10000
# How many ids one batched response-map lookup binds per query (well under
# every engine's parameter limit: sqlite's, and Oracle's 1000-item IN list).
LOOKUP_CHUNK = 500


def default_vault_path() -> Path:
    # An empty XDG_DATA_HOME is unset (the XDG spec), never the current dir.
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
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


def _has_kind_column(conn: sqlite3.Connection) -> bool:
    return any(row[1] == "kind" for row in conn.execute("PRAGMA table_info(response_sessions)"))


def _ensure_kind_column(conn: sqlite3.Connection) -> None:
    """Give a response map created before its rows had a ``kind`` the
    column (every existing row is a Responses row as far as the bound goes:
    the default). A concurrent opener that added it first is fine."""
    if _has_kind_column(conn):
        return
    try:
        conn.execute(
            "ALTER TABLE response_sessions ADD COLUMN kind TEXT NOT NULL DEFAULT 'response'"
        )
    except sqlite3.OperationalError:
        if not _has_kind_column(conn):
            raise


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


def _retired_number(conn: sqlite3.Connection, session: str) -> int:
    """The highest number ``session`` held in rows since deleted (0: none)."""
    row = conn.execute("SELECT n FROM retired_numbers WHERE session_id = ?", (session,)).fetchone()
    return 0 if row is None else int(row[0])


def delete_sessions(conn: sqlite3.Connection, session_ids: Sequence[str]) -> int:
    """Inside an open write transaction: delete whole sessions — their
    mappings, response-map rows and Live resumption handle rows (a handle
    issued in a deleted session must never resume into the session a later
    value recreates) — after raising each one's retired number
    to the highest number it holds. Returns how many of them held mappings.

    New values are always numbered above the retired number, so no number a
    deleted session issued is ever issued again there: a provider's history,
    another proxy instance's cache or a live connection still holding one of
    its tokens can only ever restore it to its own value, or not at all —
    never to a value issued after the delete. Whole sessions only, as
    before; the retired row is all that stays (one per session, no values).
    """
    present = 0
    for start in range(0, len(session_ids), LOOKUP_CHUNK):
        chunk = list(session_ids[start : start + LOOKUP_CHUNK])
        marks = ",".join("?" * len(chunk))
        present += conn.execute(
            f"SELECT COUNT(DISTINCT session_id) FROM mappings WHERE session_id IN ({marks})",
            chunk,
        ).fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO retired_numbers (session_id, n)"
            " SELECT m.session_id, max(MAX(m.n), COALESCE(r.n, 0)) FROM mappings m"
            " LEFT JOIN retired_numbers r ON r.session_id = m.session_id"
            f" WHERE m.session_id IN ({marks}) GROUP BY m.session_id",
            chunk,
        )
        conn.execute(f"DELETE FROM mappings WHERE session_id IN ({marks})", chunk)
        conn.execute(f"DELETE FROM response_sessions WHERE session_id IN ({marks})", chunk)
        conn.execute(f"DELETE FROM handle_sessions WHERE session_id IN ({marks})", chunk)
    return int(present)


def ensure_side_tables(conn: sqlite3.Connection) -> None:
    """Create the tables a whole-session delete writes or empties beside
    ``mappings`` — retired numbers and the handle map — in a database no
    proxy of this version has opened yet (the CLI's prune)."""
    conn.execute(_RETIRED_TABLE)
    conn.execute(_HANDLE_TABLE)
    conn.execute(_HANDLE_INDEX)


def write_handle(
    conn: sqlite3.Connection,
    handle_digest: str,
    session_id: str,
    replaces: Sequence[str],
    *,
    trim_all: bool,
) -> None:
    """Inside an open write transaction: map ``handle_digest`` to
    ``session_id`` as the NEWEST row (a digest written again moves up), drop
    the ``replaces`` digests this connection's newer handle supersedes —
    only the same session's — and trim the session to its newest
    ``MAX_SESSION_HANDLES`` rows; with ``trim_all``, also every row beyond
    the newest ``MAX_HANDLE_ROWS`` whose session holds no mappings. Both
    trims follow insertion order (``seq``)."""
    conn.executemany(
        "DELETE FROM handle_sessions WHERE session_id = ? AND handle_digest = ?",
        [(session_id, digest) for digest in replaces],
    )
    conn.execute(
        "INSERT OR REPLACE INTO handle_sessions (handle_digest, session_id) VALUES (?, ?)",
        (handle_digest, session_id),
    )
    conn.execute(
        "DELETE FROM handle_sessions WHERE session_id = ? AND seq <="
        " (SELECT seq FROM handle_sessions WHERE session_id = ?"
        " ORDER BY seq DESC LIMIT 1 OFFSET ?)",
        (session_id, session_id, MAX_SESSION_HANDLES),
    )
    if trim_all:
        conn.execute(
            "DELETE FROM handle_sessions WHERE seq <="
            " (SELECT seq FROM handle_sessions ORDER BY seq DESC LIMIT 1 OFFSET ?)"
            " AND NOT EXISTS"
            " (SELECT 1 FROM mappings m WHERE m.session_id = handle_sessions.session_id)",
            (MAX_HANDLE_ROWS,),
        )


def prune_idle_sessions(
    conn: sqlite3.Connection, days: int, *, exclude: frozenset[str] = frozenset()
) -> list[str]:
    """Delete (``delete_sessions``) every session that issued no new value in
    the last ``days`` days, but those in ``exclude``; returns their ids.

    The idle check runs INSIDE the delete's write transaction: no writer —
    another proxy instance sharing the file included — can issue a value in
    a session between the check that found it idle and its delete."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        doomed = [
            str(row[0])
            for row in conn.execute(
                "SELECT session_id FROM mappings GROUP BY session_id"
                " HAVING MAX(created_at) < strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?)",
                (f"-{days} days",),
            ).fetchall()
            if str(row[0]) not in exclude
        ]
        delete_sessions(conn, doomed)
        conn.execute("COMMIT")
    except BaseException:
        with suppress(sqlite3.Error):
            conn.execute("ROLLBACK")
        raise
    return doomed


class _Batch:
    """The write transaction one request's new values share (``batched``)."""

    __slots__ = ("begun", "failure", "views")

    def __init__(self) -> None:
        # BEGIN IMMEDIATE is issued at the first new value, not before: a
        # request whose values are all known writes (and locks) nothing.
        self.begun = False
        # A write that failed inside the batch, even if the work swallowed
        # it: the batch then rolls back instead of committing around it.
        self.failure: BaseException | None = None
        # The views holding staged rows, settled at the batch's end.
        self.views: list[SqliteVault] = []


class _Connection:
    """What every view over one database connection shares: the connection
    and the batch open on it — at most one, since ``batched`` runs its work
    synchronously."""

    __slots__ = ("conn", "batch", "check_faults")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.batch: _Batch | None = None
        self.check_faults = CheckFaults()


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
    the wrong secret. The fsync that costs is paid once per request, not
    once per value: ``batched`` writes a request's new values in one
    transaction.

    A number is never issued twice in a session: a new value is numbered
    above the session's live numbers AND its retired number (every number a
    whole-session delete removed), read inside the write lock. So the
    write-through caches — loaded at creation, written only after COMMIT —
    can go stale (another instance deleted the session) without ever
    restoring a wrong value; ``_revalidate`` re-reads the retired number
    (``CACHE_CHECK_SECONDS``) and rebuilds them when it moved. Views share
    one connection (single-process asyncio; point ops are microseconds).
    """

    # Set by _load. Caches are keyed by plaintext either way, so the hot path
    # (a cache hit) is identical with and without encryption; decryption
    # happens once per row at load.
    _forward: dict[str, str]
    _reverse: dict[str, str]
    # The session's retired number when the caches were loaded (its epoch),
    # and when to re-read it.
    _retired: int
    _next_check: float

    def __init__(
        self,
        shared: _Connection,
        session: str,
        *,
        owns_connection: bool,
        cipher: "VaultCipher | None" = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._shared = shared
        self._session = session
        self._owns_connection = owns_connection
        self._cipher = cipher
        self._clock = clock
        # Rows written inside the open batch and not committed yet: visible
        # to the request's own lookups, entering the caches only after the
        # COMMIT (dropped by a rollback).
        self._staged_forward: dict[str, str] = {}
        self._staged_reverse: dict[str, str] = {}
        self._load()

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._shared.conn

    @_conn.setter
    def _conn(self, conn: sqlite3.Connection) -> None:
        self._shared.conn = conn

    def _load(self) -> None:
        """(Re)build the caches from the database: the retired number FIRST,
        then the rows. A delete landing between the two reads leaves the
        recorded number behind the database's, so the next check reloads;
        the other order could record the new number over the old rows and
        never notice. Never run while a batch is open (``_revalidate``), and
        a manager keeps one view per session, so no staged row of this
        session can be read here as if committed."""
        conn = self._conn
        retired = _retired_number(conn, self._session)
        forward: dict[str, str] = {}
        reverse: dict[str, str] = {}
        if self._cipher is None:
            rows = conn.execute(
                "SELECT detector_type, original, placeholder FROM mappings WHERE session_id = ?",
                (self._session,),
            )
            for detector_type, original, placeholder in rows:
                forward[f"{detector_type}::{original}"] = placeholder
                reverse[placeholder] = original
        else:
            rows = conn.execute(
                "SELECT detector_type, original_ct, placeholder FROM mappings WHERE session_id = ?",
                (self._session,),
            )
            for detector_type, original_ct, placeholder in rows:
                original = self._cipher.decrypt(original_ct)
                forward[f"{detector_type}::{original}"] = placeholder
                reverse[placeholder] = original
        # All or nothing: a load that fails part-way changes nothing, so the
        # next check sees the retired number still moved and loads again.
        self._retired = retired
        self._forward = forward
        self._reverse = reverse
        self._next_check = self._clock() + CACHE_CHECK_SECONDS

    def _revalidate(self) -> None:
        """Re-read the session's retired number — called once the check is
        due (``self._clock() >= self._next_check``, compared at each call
        site so a cache hit pays a clock read, not a call), never while a
        batch is open. If it moved, the session was deleted (by another
        instance) since the caches were loaded: they are rebuilt, so its
        values stop being restorable here too. A check that cannot read the
        database keeps the caches as they are (``CheckFaults``)."""
        shared = self._shared
        if shared.batch is not None:
            return
        try:
            if _retired_number(self._conn, self._session) != self._retired:
                self._load()
        except Exception as exc:  # noqa: BLE001 — a cached token restores only its own value
            shared.check_faults.failed(exc)
        else:
            shared.check_faults.succeeded()
        self._next_check = self._clock() + CACHE_CHECK_SECONDS

    def placeholder_for(self, detector_type: str, original: str, *, floor: int = 0) -> str:
        if self._clock() >= self._next_check:
            self._revalidate()
        key = f"{detector_type}::{original}"
        existing = self._forward.get(key) or self._staged_forward.get(key)
        if existing is not None:
            return existing
        batch = self._shared.batch
        if batch is None:
            placeholder = self._issue_alone(detector_type, original, floor)
        else:
            placeholder = self._issue_in(batch, detector_type, original, floor)
        self._remember(key, placeholder, original)
        return placeholder

    def _issue_alone(self, detector_type: str, original: str, floor: int) -> str:
        """Outside a batch: one write transaction for this one value."""
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            placeholder = self._insert(detector_type, original, floor)
            self._conn.execute("COMMIT")
        except BaseException:
            # Any failure (disk full, I/O error, lock timeout, a cipher
            # fault, an exhausted number space): roll back so the open BEGIN
            # IMMEDIATE can't wedge the connection for the next request, and
            # fail closed. Nothing was cached (the caches are written only
            # after a successful commit), and n is computed from MAX(n) read
            # fresh on every call, so the next attempt reissues the same
            # number — never a skipped number, never a reused token.
            with suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            raise
        return placeholder

    def _issue_in(self, batch: _Batch, detector_type: str, original: str, floor: int) -> str:
        """Inside a batch: the batch's transaction, begun at its first new
        value and committed (or rolled back) once, by ``batched``."""
        try:
            if not batch.begun:
                self._conn.execute("BEGIN IMMEDIATE")
                batch.begun = True
            return self._insert(detector_type, original, floor)
        except BaseException as exc:
            # Whatever failed — the lock, a write, the cipher, the number
            # space — the batch must not commit: even a failure the work
            # swallows rolls it back whole (a sqlite I/O error may already
            # have rolled the transaction back under the staged rows).
            batch.failure = exc
            raise

    def _insert(self, detector_type: str, original: str, floor: int) -> str:
        """The value's placeholder, inserting a NEW row when it has none —
        inside an open write transaction, so nothing another connection
        does can come between the reads and the insert."""
        conn = self._conn
        cipher = self._cipher
        # The row's key: the value itself, or its HMAC index when encrypted.
        if cipher is None:
            column, index = "original", original
        else:
            column, index = "original_mac", cipher.mac(self._session, detector_type, original)
        row = conn.execute(
            "SELECT placeholder FROM mappings"
            f" WHERE session_id = ? AND detector_type = ? AND {column} = ?",
            (self._session, detector_type, index),
        ).fetchone()
        if row is not None:
            # Another connection (a second proxy instance) mapped it after
            # this view's caches were loaded: its token, never a second one.
            return str(row[0])
        # The live numbers and the retired number, read fresh inside the
        # write lock: a retry after a rolled-back write computes the same
        # number, and nothing a deleted incarnation issued is issued again.
        issued = conn.execute(
            "SELECT COALESCE(MAX(n), 0),"
            " (SELECT COALESCE(MAX(n), 0) FROM retired_numbers WHERE session_id = ?)"
            " FROM mappings WHERE session_id = ? AND detector_type = ?",
            (self._session, self._session, detector_type),
        ).fetchone()
        n = next_number(max(int(issued[0]), int(issued[1])), floor, detector_type)
        placeholder = format_placeholder(detector_type, n)
        if cipher is None:
            conn.execute(
                "INSERT INTO mappings (session_id, detector_type, original, placeholder, n)"
                " VALUES (?, ?, ?, ?, ?)",
                (self._session, detector_type, index, placeholder, n),
            )
        else:
            conn.execute(
                "INSERT INTO mappings"
                " (session_id, detector_type, original_mac, original_ct, placeholder, n)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (self._session, detector_type, index, cipher.encrypt(original), placeholder, n),
            )
        return placeholder

    def _remember(self, key: str, placeholder: str, original: str) -> None:
        """Cache a row read or written: straight into the caches outside a
        batch (its transaction committed), staged inside one."""
        batch = self._shared.batch
        if batch is None:
            self._forward[key] = placeholder
            self._reverse[placeholder] = original
            return
        if not self._staged_forward:
            batch.views.append(self)
        self._staged_forward[key] = placeholder
        self._staged_reverse[placeholder] = original

    def _keep_staged(self) -> None:
        """The batch committed: its staged rows join the caches."""
        self._forward.update(self._staged_forward)
        self._reverse.update(self._staged_reverse)
        self._drop_staged()

    def _drop_staged(self) -> None:
        self._staged_forward.clear()
        self._staged_reverse.clear()

    def batched(self, work: Callable[[], T]) -> T:
        """Run ``work`` with every new value it issues — in this view or in
        any other over the same connection — written in ONE transaction:
        BEGIN IMMEDIATE at the first, one COMMIT when ``work`` returns, the
        caches updated only after it. Any exception (a write fault, a failed
        COMMIT, a blocked value, a cap refusal, a bug) rolls the whole batch
        back and leaves the caches untouched — so no token issued inside a
        rolled-back batch is ever restored, and its number is issued again
        only to the same value (dense, from MAX(n) read fresh). Re-entrant:
        a nested call joins the open batch. ``work`` must not await (see
        ``run_batched``); holding the write lock for its duration, another
        process sharing the file waits (busy_timeout) and fails closed past
        it."""
        shared = self._shared
        if shared.batch is not None:
            return work()
        if self._clock() >= self._next_check:
            self._revalidate()  # before anything is staged
        batch = shared.batch = _Batch()
        try:
            result = work()
            if batch.failure is not None:
                raise batch.failure
            if batch.begun:
                shared.conn.execute("COMMIT")
        except BaseException:
            if batch.begun:
                with suppress(sqlite3.Error):
                    shared.conn.execute("ROLLBACK")
            for view in batch.views:
                view._drop_staged()
            raise
        finally:
            shared.batch = None
        for view in batch.views:
            view._keep_staged()
        return result

    def original_for(self, placeholder: str) -> str | None:
        if self._clock() >= self._next_check:
            self._revalidate()
        cached = self._reverse.get(placeholder)
        if cached is not None:
            return cached
        # A cache miss can only mean another connection issued the token —
        # or, inside a batch, this one did (the read sees the batch's own
        # rows; what it finds is staged like them).
        if self._cipher is None:
            row = self._conn.execute(
                "SELECT detector_type, original FROM mappings"
                " WHERE session_id = ? AND placeholder = ?",
                (self._session, placeholder),
            ).fetchone()
            if row is None:
                return None
            original = str(row[1])
        else:
            row = self._conn.execute(
                "SELECT detector_type, original_ct FROM mappings"
                " WHERE session_id = ? AND placeholder = ?",
                (self._session, placeholder),
            ).fetchone()
            if row is None:
                return None
            original = self._cipher.decrypt(row[1])
        self._remember(f"{row[0]}::{original}", placeholder, original)
        return original

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def __len__(self) -> int:
        return len(self._reverse)


def open_sqlite_vault(path: Path, session: str, cipher: "VaultCipher | None" = None) -> SqliteVault:
    """Standalone single-session vault owning its connection (static mode,
    tests)."""
    return SqliteVault(
        _Connection(_open_connection(path, cipher)), session, owns_connection=True, cipher=cipher
    )


class VaultManager(Protocol):
    """Per-session vaults over one store, plus the durable maps a session
    router reads (re-exported by ``llm_redact.plugin_api``).

    OPTIONAL members, read with ``getattr`` (a manager written before them
    keeps working):

    - OPTIONAL ``record_object_session(object_id, session_id) -> None`` — a
      stored object's owner record, bounded apart from the Responses rows.
      The proxy calls it; a manager without it records owners through
      ``record_response_session`` (one shared bound).
    - OPTIONAL ``record_handle_session(handle_digest, session_id, *,
      replaces=()) -> None`` — the durable Live resumption handle map, for
      a plugin (llm-redact-pro) to call: the session a handle was issued
      in, keyed by the plugin's domain-separated DIGEST of it (never the
      raw handle), the ``replaces`` digests (the same connection's older
      handles; only rows of the same session) deleted in the same
      transaction. Bounded per session and in all in insertion order,
      never trimming a live session's newest rows for another session's
      traffic; a session's rows leave with it on every whole-session
      delete, so a handle into a pruned-and-recreated session is unknown.
      A fault is contained (bookkeeping stage ``handle_map``, logged by
      type): nothing is written.
    - OPTIONAL ``lookup_handle_session(handle_digest) -> str | None`` — the
      session a recorded digest names, None when absent (the truth: a
      deleted session's rows are gone) or when the read fails (unknown,
      fail closed).
    - OPTIONAL ``write_maps_in_background() -> None`` — called once by the
      proxy at startup: from then on the three durable maps above (and
      ``record_response_session``) are written off the event loop
      (``vault_writer.MapWriter``: one thread, its own connection, bounded),
      every lookup answering from the writer's overlay until each write
      lands (read-your-writes), a whole-session delete erasing its
      sessions' queued writes. A manager without it writes synchronously.
    - OPTIONAL ``drain_map_writes(timeout) -> int`` — blocking: wait up to
      ``timeout`` seconds for the queued writes to land; how many have not.
      The proxy's shutdown runs it (in a worker thread) before ``close``,
      which counts and drops what is still queued (and a write still in
      flight); a drain earlier in the manager's life never stops ``close``
      from draining again.

    A manager whose ``durable_response_map`` is False (the in-memory one)
    keeps no durable map: its handle-map members record nothing and answer
    None, and a caller must not read that None as "pruned" — the proxy
    hands its session router no durable lookup, so the router keeps its own
    in-process records instead.
    """

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
        # One vault per session for the process lifetime: a forgotten
        # session keeps its vault, emptied, so its numbering never restarts.
        self._vaults: dict[str, InMemoryVault | EncryptedInMemoryVault] = {}

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
        """Sessions holding mappings (the persistent managers' count)."""
        return sum(len(vault) > 0 for vault in self._vaults.values())

    def total_entries(self) -> int:
        return sum(len(v) for v in self._vaults.values())

    def sessions_summary(self) -> list[dict[str, object]]:
        # Sessions holding mappings, like the persistent managers. Memory
        # mappings carry no timestamps: counts only, insertion order.
        return [
            {"session": session_id, "entries": len(vault), "first": None, "last": None}
            for session_id, vault in self._vaults.items()
            if len(vault) > 0
        ]

    def prune_sessions(self, days: int, *, exclude: frozenset[str] = frozenset()) -> int:
        # Nothing to prune by age: memory sessions have no timestamps and
        # die with the process anyway.
        return 0

    def forget_sessions(self, session_ids: Iterable[str]) -> int:
        """Drop whole sessions' mappings (a purged user's); how many held
        mappings. Each vault is emptied IN PLACE and kept: a holder of it (a
        live realtime connection) sees the mappings go too, and a new value
        is numbered above every number the session issued — a token of the
        dropped mappings never gains a second meaning."""
        forgotten = 0
        for session_id in set(session_ids):
            vault = self._vaults.get(session_id)
            if vault is not None and len(vault) > 0:
                vault.forget_mappings()
                forgotten += 1
        return forgotten

    def record_response_session(self, response_id: str, session_id: str) -> None:
        pass  # the SessionRouter's in-memory map is authoritative here

    def record_object_session(self, object_id: str, session_id: str) -> None:
        pass  # likewise: no durable map

    def record_handle_session(
        self, handle_digest: str, session_id: str, *, replaces: Sequence[str] = ()
    ) -> None:
        pass  # no durable map: a plugin vouches for handles in its own process

    def lookup_handle_session(self, handle_digest: str) -> str | None:
        return None  # unknown — never "pruned" (durable_response_map is False)

    def lookup_response_session(self, response_id: str) -> str | None:
        return None

    def lookup_response_sessions(self, response_ids: Iterable[str]) -> dict[str, str]:
        return {}

    def close(self) -> None:
        pass


class SqliteVaultManager:
    """One shared connection; per-session views cached in a small LRU.

    Eviction only drops a view's write-through cache — every mapping lives in
    the database, and a re-created view loads its session's rows (small for
    per-conversation sessions). A view evicted while still held elsewhere
    (an in-flight request, a realtime connection) is handed out again, not
    duplicated: one view per session, so a delete reaches every live one.
    """

    def __init__(
        self,
        path: Path,
        *,
        cipher: "VaultCipher | None" = None,
        view_cache_size: int = 64,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._shared = _Connection(_open_connection(path, cipher))
        _ensure_kind_column(self._conn)
        self._cipher = cipher
        self._clock = clock
        self._views: OrderedDict[str, SqliteVault] = OrderedDict()
        self._view_cache_size = view_cache_size
        # Every live view, wherever it is held (weakly: the LRU above and the
        # holders keep them alive).
        self._live: weakref.WeakValueDictionary[str, SqliteVault] = weakref.WeakValueDictionary()
        self._path = path
        self._inserts = {_RESPONSE_KIND: 0, _OBJECT_KIND: 0}
        self._handle_writes = 0
        self._handle_write_faults = HandleWriteFaults()
        self._handle_read_faults = HandleReadFaults()
        self._response_write_faults = ResponseMapWriteFaults()
        self._object_write_faults = ObjectMapWriteFaults()
        # The durable maps' background writer (``write_maps_in_background``),
        # None while they are written synchronously (the CLI, tests).
        self._maps: MapWriter | None = None

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._shared.conn

    @_conn.setter
    def _conn(self, conn: sqlite3.Connection) -> None:
        self._shared.conn = conn

    def bind_fault_counter(self, counter: Counter[str]) -> None:
        """Count this manager's failed staleness checks, handle-map reads
        and writes and background map writes in ``counter`` (the proxy's
        ``bookkeeping_errors``; optional, read via getattr)."""
        self._shared.check_faults.counter = counter
        self._handle_write_faults.counter = counter
        self._handle_read_faults.counter = counter
        self._response_write_faults.counter = counter
        self._object_write_faults.counter = counter

    def write_maps_in_background(self) -> None:
        """From now on, write the durable maps (response rows, owner
        records, handles) on a background thread with its own connection
        (``vault_writer.MapWriter``), answering lookups from its overlay
        until each write lands. The proxy calls it once at startup."""
        if self._maps is None:
            self._maps = MapWriter(self._open_map_connection)

    def _open_map_connection(self) -> sqlite3.Connection:
        """The background writer's own connection (opened on its thread).
        ``synchronous=NORMAL``: a map row lost to a power failure reads as
        unknown (refused or sealed), never as a wrong value — unlike a
        mapping, whose loss could reissue a number."""
        conn = sqlite3.connect(self._path, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def drain_map_writes(self, timeout: float) -> int:
        """Wait up to ``timeout`` seconds for the background writes to land
        (blocking); how many have not."""
        return self._maps.drain(timeout) if self._maps is not None else 0

    def get(self, session_id: str) -> Vault:
        # Every view the LRU holds is live, so the registry answers for both.
        view = self._live.get(session_id)
        if view is None:
            view = SqliteVault(
                self._shared,
                session_id,
                owns_connection=False,
                cipher=self._cipher,
                clock=self._clock,
            )
            self._live[session_id] = view
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
        """Delete whole idle sessions (``prune_idle_sessions``: the idle
        check and the delete are one write transaction), then rebuild every
        live view of them.

        Whole sessions only — the same rule as the CLI — and each one's
        numbers are retired (``delete_sessions``): an instance or a live
        view still holding one of its tokens can never see the number issued
        to another value. Callers exclude the always-live static session.
        """
        with holding(self._maps) as deleted:
            doomed = prune_idle_sessions(self._conn, days, exclude=exclude)
            deleted(doomed)
        self._drop_views(doomed)
        return len(doomed)

    def forget_sessions(self, session_ids: Iterable[str]) -> int:
        """Delete whole named sessions (mappings and response rows, their
        numbers retired) in one transaction and rebuild every live view of
        them; how many held mappings. Whole sessions only, like prune."""
        wanted = sorted(set(session_ids))
        if not wanted:
            return 0
        with holding(self._maps) as deleted:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                present = delete_sessions(self._conn, wanted)
                self._conn.execute("COMMIT")
            except BaseException:
                with suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise
            deleted(wanted)
        self._drop_views(wanted)
        return present

    def _drop_views(self, session_ids: Iterable[str]) -> None:
        """After a delete: evict the sessions' cached views, and rebuild any
        still held elsewhere (an in-flight request, a realtime connection)
        from the database — empty now, so the deleted values stop being
        restorable there at once."""
        for session_id in session_ids:
            self._views.pop(session_id, None)
            view = self._live.get(session_id)
            if view is not None:
                view._load()

    def record_response_session(self, response_id: str, session_id: str) -> None:
        """Map a Responses chain's response id to its session."""
        self._record_row(response_id, session_id, _RESPONSE_KIND)

    def record_object_session(self, object_id: str, session_id: str) -> None:
        """Record the session that created a stored object (the router's
        ownership record); bounded apart from the Responses rows."""
        self._record_row(object_id, session_id, _OBJECT_KIND)

    def _record_row(self, row_id: str, session_id: str, kind: str) -> None:
        """One response map row: written now, or — with a background writer
        — queued (its lookups answered from the writer's overlay meanwhile;
        a fault counted under the kind's stage)."""
        maps = self._maps
        if maps is None:
            self._record(self._conn, row_id, session_id, kind)
            return

        def run(conn: sqlite3.Connection) -> None:
            self._record(conn, row_id, session_id, kind)

        def erase(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM response_sessions WHERE response_id = ?", (row_id,))

        faults = (
            self._response_write_faults if kind == _RESPONSE_KIND else self._object_write_faults
        )
        maps.submit(MapWrite(session_id, run, erase, faults), sets=((ROW, row_id),))

    def _record(self, conn: sqlite3.Connection, row_id: str, session_id: str, kind: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO response_sessions (response_id, session_id, kind)"
            " VALUES (?, ?, ?)",
            (row_id, session_id, kind),
        )
        self._inserts[kind] += 1
        if self._inserts[kind] >= _RESPONSE_PRUNE_EVERY:
            self._inserts[kind] = 0
            # Beyond the cap, only rows whose session holds no mappings go
            # (a pruned session, or one that never redacted anything): a
            # router reads a missing row as "that session was pruned", and a
            # chain into a LIVE session resumed in a fresh one would reissue
            # «EMAIL_001» for a new value while the provider's history still
            # means the old one. Live sessions' rows leave with the session.
            # Each kind keeps its own newest rows.
            conn.execute(
                "DELETE FROM response_sessions WHERE kind = ? AND response_id NOT IN"
                " (SELECT response_id FROM response_sessions WHERE kind = ?"
                " ORDER BY created_at DESC LIMIT ?)"
                " AND NOT EXISTS"
                " (SELECT 1 FROM mappings m WHERE m.session_id = response_sessions.session_id)",
                (kind, kind, _MAX_OBJECT_ROWS if kind == _OBJECT_KIND else _MAX_RESPONSE_ROWS),
            )

    def lookup_response_session(self, response_id: str) -> str | None:
        verdict = row_verdict(self._maps, response_id)
        if verdict is not MISS:
            return verdict  # type: ignore[return-value]  # a row is never Removed
        row = self._conn.execute(
            "SELECT session_id FROM response_sessions WHERE response_id = ?", (response_id,)
        ).fetchone()
        return str(row[0]) if row is not None else None

    def record_handle_session(
        self, handle_digest: str, session_id: str, *, replaces: Sequence[str] = ()
    ) -> None:
        """Record the session a Live resumption handle was issued in, by the
        handle's digest, dropping the ``replaces`` digests it supersedes in
        the same transaction (``write_handle``: bounded per session and in
        all, in insertion order). A fault is contained (``HandleWriteFaults``):
        nothing is written, and the handle reads as unknown. With a
        background writer the write is queued (its lookups answered from the
        writer's overlay until it lands)."""
        superseded = tuple(replaces)
        maps = self._maps
        if maps is None:
            try:
                self._write_handle(self._conn, handle_digest, session_id, superseded)
            except Exception as exc:  # noqa: BLE001 — contained: the handle stays unknown
                self._handle_write_faults.failed(exc)
                return
            self._handle_write_faults.succeeded()
            return

        def run(conn: sqlite3.Connection) -> None:
            self._write_handle(conn, handle_digest, session_id, superseded)

        def erase(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM handle_sessions WHERE handle_digest = ?", (handle_digest,))

        maps.submit(
            MapWrite(session_id, run, erase, self._handle_write_faults),
            sets=((HANDLE, handle_digest),),
            removes=tuple((HANDLE, digest) for digest in superseded),
        )

    def _write_handle(
        self,
        conn: sqlite3.Connection,
        handle_digest: str,
        session_id: str,
        replaces: Sequence[str],
    ) -> None:
        """One handle write as one transaction (raising on a fault, rolled
        back); the total trim runs every ``_RESPONSE_PRUNE_EVERY`` writes
        that succeeded."""
        trim_all = self._handle_writes + 1 >= _RESPONSE_PRUNE_EVERY
        conn.execute("BEGIN IMMEDIATE")
        try:
            write_handle(conn, handle_digest, session_id, replaces, trim_all=trim_all)
            conn.execute("COMMIT")
        except BaseException:
            with suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        self._handle_writes = 0 if trim_all else self._handle_writes + 1

    def lookup_handle_session(self, handle_digest: str) -> str | None:
        """The session a recorded handle digest was issued in, or None — its
        absence is the truth (a deleted session's rows left with it). A
        fault reads as None: unknown, never a guess. A write still queued
        answers from the background writer's overlay."""
        verdict = MISS if self._maps is None else self._maps.verdict((HANDLE, handle_digest))
        return handle_answer(verdict, lambda: self._read_handle(handle_digest))

    def _read_handle(self, handle_digest: str) -> str | None:
        try:
            row = self._conn.execute(
                "SELECT session_id FROM handle_sessions WHERE handle_digest = ?",
                (handle_digest,),
            ).fetchone()
        except Exception as exc:  # noqa: BLE001 — fail closed: unknown
            self._handle_read_faults.failed(exc)
            return None
        self._handle_read_faults.succeeded()
        return str(row[0]) if row is not None else None

    def lookup_response_sessions(self, response_ids: Iterable[str]) -> dict[str, str]:
        """``lookup_response_session`` for many ids at once — one query per
        ``LOOKUP_CHUNK`` distinct ids, not one per id (a listing names
        thousands): the recorded ids with their sessions, an unknown id
        simply absent (a write still queued answers from the overlay)."""
        return overlaid_rows(self._maps, list(dict.fromkeys(response_ids)), self._read_rows)

    def _read_rows(self, wanted: list[str]) -> dict[str, str]:
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
        """Stop the background writer (``MapWriter.close``: drained first,
        bounded, unless the last drain ran out of time with nothing landing
        since; what is left — the write in flight included — is counted and
        dropped),
        then close the connection."""
        if self._maps is not None:
            self._maps.close()
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
