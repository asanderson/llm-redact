"""Refusal overrides: the requester's one-time or every-time approval.

Off by default (``[overrides] enabled = true`` opts in, restart-only):
while they are off the proxy builds no store, so no refusal carries a code
or a hint, nothing is written, and the records a store already holds are
inert. With them on, a DETECTION refusal — a block-mode value, values found inside a binary
upload, a verbatim identifier field that would be redacted, a body or a
binary upload part the proxy cannot scan — is answered with a short
single-use CODE and the text ``to allow: llm-redact override CODE --once |
--always`` (``override --config PATH CODE …`` when the proxy was started with
an explicit config file the CLI's default search would not find:
``hint_config``). A HUMAN approves it: ``llm-redact override CODE`` reads a
confirmation typed on the terminal (``/dev/tty``, never stdin), or the
llm-redact-pro dashboard's buttons (the guarded POST chain). Then:

- ``--once``: the NEXT request of the same requester that would be refused
  the same way (the same values, or the same kind on the same route) passes
  that refusal once, within ``ttl``; the grant is consumed atomically as the
  request passes (``OverrideScope.commit``: exactly one of two parallel
  requests uses it).
- ``--always``: for a value refusal, an allowlist entry for each exact value
  and detector type (per requester); for a route refusal, that kind on that
  route for that requester. Until ``llm-redact override revoke``.

An override never turns detection off. Detection runs as always and decides
a refusal; only then is the requester's override asked, and an overridden
value is FORWARDED UPSTREAM UNREDACTED, exactly like a warn-mode value (an
overridden body or binary part is forwarded unscanned).

The requester is the subject the access gate (llm-redact-pro named users)
admitted, else the local operator (the empty subject): a record is only
ever matched by, and approved for, its own subject.

Storage: one sqlite file (0600, its directory 0700) under the XDG data
directory. Values are never stored: each refused value is an HMAC-SHA256
under a random per-install key kept in the same file; codes are stored as
SHA-256 hashes. The running proxy reads the file whenever a refusal is about
to be decided (a small indexed read, on the refusal path only), so an
approval made by the CLI in another process applies to the very next
request — no reload. Nothing here logs, stores or reports a value, a digest
or a code (logs name the exception TYPE only)."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import shlex
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_redact.config import ETC_CONFIG_PATH, default_config_path

logger = logging.getLogger(__name__)

# Refusals whose record matches VALUES (each refused (type, value) as an HMAC):
# an approval covers exactly those values wherever they would refuse a
# request (a block-mode value, a value inside a binary upload, a value in a
# verbatim field).
VALUE_KINDS = ("block", "binary_values", "verbatim_field")
# Refusals whose record matches (kind, provider, method, route): a body the
# proxy cannot read as JSON, a binary upload part under
# `binary_uploads = "refuse"`.
ROUTE_KINDS = ("unscanned_body", "binary_upload")
OVERRIDE_KINDS = VALUE_KINDS + ROUTE_KINDS
SCOPES = ("once", "always")

# 12 Crockford base32 characters: 60 bits, single-use, short-lived.
CODE_LENGTH = 12
_CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_CODE_ALIASES = str.maketrans({"O": "0", "I": "1", "L": "1"})

DEFAULT_TTL_SECONDS = 900
# Bounds: unapproved codes (a client retrying a refused request mints one per
# attempt) PER REQUESTER — so one requester's retries never drop another's
# codes — and across the file, and approved one-time grants; each dropped
# oldest first.
MAX_PENDING = 256
MAX_PENDING_TOTAL = 4096
MAX_ONCE_GRANTS = 256
# How long a write waits for another process's write lock before it fails
# closed (the refusal stands). The proxy waits briefly — it waits on the
# event loop, and CLI transactions are a handful of rows — the CLI longer.
PROXY_BUSY_TIMEOUT_MS = 200
CLI_BUSY_TIMEOUT_MS = 5000
# Every-time rules' use counts are kept in memory and written at most this
# often (and at close): a request passing on an every-time rule never
# writes the store on the event loop.
USE_FLUSH_SECONDS = 60.0

_VALUE_DOMAIN = b"llm-redact override value v1\x00"
_CODE_DOMAIN = b"llm-redact override code v1\x00"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value BLOB NOT NULL)",
    """CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code_hash TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL,
        subject TEXT NOT NULL,
        provider TEXT NOT NULL,
        method TEXT NOT NULL,
        route TEXT NOT NULL,
        items TEXT NOT NULL,
        created REAL NOT NULL,
        expires REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope TEXT NOT NULL,
        kind TEXT NOT NULL,
        subject TEXT NOT NULL,
        provider TEXT NOT NULL,
        method TEXT NOT NULL,
        route TEXT NOT NULL,
        items TEXT NOT NULL,
        created REAL NOT NULL,
        expires REAL,
        uses INTEGER NOT NULL DEFAULT 0,
        consumed INTEGER NOT NULL DEFAULT 0)""",
    "CREATE INDEX IF NOT EXISTS rules_subject ON rules (subject, consumed)",
)


def default_overrides_path() -> Path:
    # An empty XDG_DATA_HOME is unset (the XDG spec), never the current dir.
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "llm-redact" / "overrides.db"


def new_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(CODE_LENGTH))


def normalize_code(text: str) -> str | None:
    """A code as typed (any case, dashes and spaces ignored, O/I/L read as
    0/1), or None when it cannot be one."""
    code = "".join(text.split()).replace("-", "").upper().translate(_CODE_ALIASES)
    if len(code) != CODE_LENGTH or any(c not in _CODE_ALPHABET for c in code):
        return None
    return code


def allow_hint(code: str, config_arg: str | None = None) -> str:
    """The CLI hint for the local operator. ``config_arg`` (``hint_config``)
    names the config file the proxy was started with when the CLI's own
    default search would not find it: the command then reads the same
    config — and so the same store — as the running proxy."""
    if config_arg is None:
        return f"to allow: llm-redact override {code} --once | --always"
    return f"to allow: llm-redact override --config {config_arg} {code} --once | --always"


def hint_config(loaded: Path | None, env: Mapping[str, str]) -> str | None:
    """The ``--config`` argument a refusal hint carries (shell-quoted,
    absolute), or None when the plain hint reaches the same config.

    The proxy was started with an EXPLICIT config file — ``serve --config
    PATH`` (``loaded``), else ``LLM_REDACT_CONFIG`` (the operator's own shell
    may not set it) — that is not what ``llm-redact override`` finds by its
    default search without that variable (the XDG file, else
    /etc/llm-redact/config.toml). A path that does not encode as UTF-8 is
    left out (the hint could not carry it; the plain hint stands). Computed
    once at startup: overrides are restart-only.

    The default search is this process's own (its HOME/XDG_CONFIG_HOME, as
    ``serve`` searched); ``env`` supplies ``LLM_REDACT_CONFIG`` only. A
    search this process cannot check (EACCES on an unreadable HOME) never
    stops startup — ``serve --config`` never needed it: the explicit file is
    named (a redundant ``--config`` is harmless, a missing one is not)."""
    env_path = env.get("LLM_REDACT_CONFIG")
    explicit = loaded if loaded is not None else (Path(env_path) if env_path else None)
    if explicit is None:
        return None
    if _searched_config(explicit):
        return None
    try:
        text = str(explicit.absolute())
    except OSError:
        # A working directory gone from under a relative path leaves no
        # absolute path to print: the plain hint (the off note names
        # --config).
        return None
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return shlex.quote(text)


def _searched_config(explicit: Path) -> bool:
    """Whether the CLI's default search (without LLM_REDACT_CONFIG) finds
    ``explicit``. A search this process cannot check — a candidate it may
    not stat, a path it cannot resolve — reads as another file: the hint
    then names the explicit one, which the CLI reads without searching."""
    try:
        for candidate in (default_config_path(), ETC_CONFIG_PATH):
            if candidate.exists():
                return candidate.resolve() == explicit.resolve()
    except OSError:
        pass
    return False


# Refusal overrides are OFF by default: an operator opts in with this
# setting (restart-only). While they are off no refusal carries a code or a
# hint, nothing is written to the store, and the records it already holds
# are inert — never applied. Every surface that answers while they are off
# names the setting.
ENABLE_SETTING = "[overrides] enabled = true"
DISABLED_REASON = (
    "refusal overrides are off ([overrides] enabled = false, the default):"
    " no refusal carries a code and no approval is applied"
)
TO_ENABLE = f"set {ENABLE_SETTING} in the proxy's config and restart it to use them"


# A named user's refusal (an access gate admitted them): approved signed in
# to the dashboard, where it is listed — the code is the CLI's, not theirs.
DASHBOARD_HINT = (
    "to allow: Allow once | Always allow under Refusal overrides in the llm-redact dashboard"
)


# At most this many characters of a route or requester are listed (``shown``).
SHOWN_CHARS = 120


def shown(text: str) -> str:
    """``text`` as it is safe to print to a terminal or a page: every
    character that is not printable — C0 and C1 controls, DEL, bidi and
    other format characters, line and paragraph separators — escaped
    (``\\x1b``, ``\\u202e``), and at most ``SHOWN_CHARS`` of it, a longer
    one cut with its length named. A route and a requester come from the
    request: raw, an escape sequence in a path could rewrite or hide the
    kind and types a person is asked to confirm, and a long one push them
    off the screen."""
    escaped = "".join(
        char
        if char.isprintable()
        else f"\\x{ord(char):02x}"
        if ord(char) < 0x100
        else f"\\u{ord(char):04x}"
        if ord(char) < 0x10000
        else f"\\U{ord(char):08x}"
        for char in text
    )
    if len(escaped) > SHOWN_CHARS:
        return f"{escaped[:SHOWN_CHARS]}… ({len(escaped)} characters)"
    return escaped


def _code_hash(code: str) -> str:
    return hashlib.sha256(_CODE_DOMAIN + code.encode("ascii")).hexdigest()


def _digest(key: bytes, detector_type: str, value: str) -> str:
    message = (
        _VALUE_DOMAIN
        + detector_type.encode("utf-8", "surrogatepass")
        + b"\x00"
        + value.encode("utf-8", "surrogatepass")
    )
    return hmac.new(key, message, hashlib.sha256).hexdigest()


class OverrideError(Exception):
    """An approval, listing or revocation that cannot be done: the message
    names the reason only (never a code, a value or a digest)."""


@dataclass(frozen=True)
class OverrideEntry:
    """One record as listed (value-free): ``id`` is ``p<N>`` for a pending
    code, ``r<N>`` for an approved rule; ``state`` is pending / once /
    always; ``types`` the detector types a value record covers; ``route``
    "METHOD provider path" (for an every-time value rule: "any route").
    ``route`` and ``subject`` are as printable (``shown``): they come from
    a request."""

    id: str
    state: str
    kind: str
    types: tuple[str, ...]
    route: str
    subject: str
    created: float
    expires: float | None
    uses: int

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "state": self.state,
            "kind": self.kind,
            "types": list(self.types),
            "route": self.route,
            "subject": self.subject or None,
            "created": self.created,
            "expires": self.expires,
            "uses": self.uses,
        }


@dataclass(frozen=True)
class _RouteKey:
    kind: str
    provider: str
    method: str
    route: str


@dataclass
class _Snapshot:
    """One subject's live rules, read once per request."""

    always_values: dict[tuple[str, str], int]
    once_values: list[tuple[int, frozenset[tuple[str, str]]]]
    always_routes: dict[_RouteKey, int]
    once_routes: dict[_RouteKey, int]


_EMPTY = _Snapshot({}, [], {}, {})


def _items(raw: str) -> frozenset[tuple[str, str]]:
    return frozenset((str(t), str(d)) for t, d in json.loads(raw))


def _types(raw: str) -> tuple[str, ...]:
    return tuple(sorted({str(t) for t, _ in json.loads(raw)}))


class OverrideStore:
    """The override records of one install (one sqlite file). Used by the
    proxy on the event loop (refusal path only) and by the CLI; the file is
    opened lazily and CREATED only when a first code is minted (or the CLI
    needs it), so a proxy that never refuses never writes one."""

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
        max_pending: int = MAX_PENDING,
        max_pending_total: int = MAX_PENDING_TOTAL,
        busy_timeout_ms: int = CLI_BUSY_TIMEOUT_MS,
        read_only: bool = False,
    ) -> None:
        self.path = path
        # A reader (doctor, `override list`): opens an existing file
        # read-only — no schema, no key, no chmod, no write of any kind.
        self.read_only = read_only
        self.ttl = ttl_seconds
        self._clock = clock
        self._max_pending = max_pending
        self._max_pending_total = max_pending_total
        self._busy_timeout_ms = busy_timeout_ms
        # Every-time rules' uses not yet written (count_uses), by rule id.
        self._unflushed: Counter[int] = Counter()
        self._last_flush = clock()
        self._conn: sqlite3.Connection | None = None
        self._key: bytes | None = None
        self._lock = threading.Lock()
        # Overrides used by requests of THIS process, by scope (once/always).
        self.used: Counter[str] = Counter()

    # -- connection --

    def _open(self, *, create: bool) -> sqlite3.Connection | None:
        if self._conn is not None:
            return self._conn
        if self.read_only:
            return self._open_read_only()
        if not create and not self.path.exists():
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(self.path, 0o600)
        conn = sqlite3.connect(
            self.path,
            isolation_level=None,
            check_same_thread=False,
            timeout=self._busy_timeout_ms / 1000,
        )
        try:
            conn.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
            conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL under WAL: a commit never waits for the disk (the proxy
            # writes here on its event loop). A power loss may drop the last
            # commits: a pending code, an approval or a one-time use — the
            # refusal then stands, or a one-time grant passes once more.
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("BEGIN IMMEDIATE")
            for statement in _SCHEMA:
                conn.execute(statement)
            conn.execute(
                "INSERT OR IGNORE INTO meta (name, value) VALUES ('hmac_key', ?)",
                (secrets.token_bytes(32),),
            )
            conn.execute("COMMIT")
            row = conn.execute("SELECT value FROM meta WHERE name = 'hmac_key'").fetchone()
        except BaseException:
            conn.close()
            raise
        self._key = bytes(row[0])
        self._conn = conn
        return conn

    def _open_read_only(self) -> sqlite3.Connection | None:
        """The existing store, read-only (``mode=ro``), or None when there is
        none — no file, or a file without the store's tables (never created
        here)."""
        if not self.path.exists():
            return None
        conn = sqlite3.connect(
            self.path.resolve().as_uri() + "?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=self._busy_timeout_ms / 1000,
        )
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        except BaseException:
            conn.close()
            raise
        if not {"pending", "rules"} <= tables:
            conn.close()
            return None
        self._conn = conn
        return conn

    def close(self) -> None:
        try:
            self.flush_uses()
        except Exception as exc:  # noqa: BLE001 — counts only; closing goes on
            logger.warning("overrides: use counts not written (%s)", type(exc).__name__)
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _transaction(self, conn: sqlite3.Connection) -> _Tx:
        return _Tx(conn)

    # -- the proxy's side --

    def snapshot(self, subject: str) -> _Snapshot:
        """``subject``'s live rules (approved, not consumed, not expired)."""
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return _EMPTY
            rows = conn.execute(
                "SELECT id, scope, kind, provider, method, route, items FROM rules"
                " WHERE subject = ? AND consumed = 0 AND (expires IS NULL OR expires > ?)",
                (subject, self._clock()),
            ).fetchall()
        snap = _Snapshot({}, [], {}, {})
        for rule_id, scope, kind, provider, method, route, items in rows:
            if kind in ROUTE_KINDS:
                key = _RouteKey(kind, provider, method, route)
                (snap.always_routes if scope == "always" else snap.once_routes)[key] = rule_id
            elif scope == "always":
                for item in _items(items):
                    snap.always_values[item] = rule_id
            else:
                snap.once_values.append((rule_id, _items(items)))
        return snap

    def digest(self, detector_type: str, value: str) -> str:
        assert self._key is not None
        return _digest(self._key, detector_type, value)

    def record_pending(
        self,
        kind: str,
        subject: str,
        provider: str,
        method: str,
        route: str,
        values: Iterable[tuple[str, str]] = (),
    ) -> str:
        """Mint the code of one refusal (the file is created if needed):
        what an approval will match — each refused (type, value) as an
        HMAC, or the route — bound to ``subject`` for ``ttl``."""
        code = new_code()
        now = self._clock()
        with self._lock:
            conn = self._open(create=True)
            assert conn is not None and self._key is not None
            key = self._key
            items = sorted({(t, _digest(key, t, v)) for t, v in values})
            with self._transaction(conn):
                conn.execute("DELETE FROM pending WHERE expires <= ?", (now,))
                conn.execute(
                    "INSERT INTO pending (code_hash, kind, subject, provider, method, route,"
                    " items, created, expires) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _code_hash(code),
                        kind,
                        subject,
                        provider,
                        method,
                        route,
                        json.dumps(items),
                        now,
                        now + self.ttl,
                    ),
                )
                conn.execute(
                    "DELETE FROM pending WHERE subject = ? AND id NOT IN"
                    " (SELECT id FROM pending WHERE subject = ? ORDER BY id DESC LIMIT ?)",
                    (subject, subject, self._max_pending),
                )
                conn.execute(
                    "DELETE FROM pending WHERE id NOT IN"
                    " (SELECT id FROM pending ORDER BY id DESC LIMIT ?)",
                    (self._max_pending_total,),
                )
        return code

    def consume(self, once: Iterable[int]) -> bool:
        """Use the one-time grants ``once`` (each exactly once: a grant
        another request consumed first, or one that expired meanwhile, fails
        the whole use — nothing is consumed). Atomic across processes. A
        consumed grant stays (inert: never matched, listed or counted) until
        it expires, so ``release`` can hand it back to a request that was
        refused before it reached the upstream."""
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return False
            now = self._clock()
            with self._transaction(conn) as tx:
                for rule_id in once:
                    cursor = conn.execute(
                        "UPDATE rules SET consumed = 1, uses = uses + 1"
                        " WHERE id = ? AND scope = 'once' AND consumed = 0 AND expires > ?",
                        (rule_id, now),
                    )
                    if cursor.rowcount != 1:
                        tx.rollback()
                        return False
                # Expired one-time grants are spent: dropped.
                conn.execute("DELETE FROM rules WHERE scope = 'once' AND expires <= ?", (now,))
        return True

    def release(self, once: Iterable[int]) -> None:
        """Hand back the one-time grants ``once`` that ``consume`` took for a
        request then refused before it reached the upstream (an audit
        refusal, a missing upstream, the authorizer, a routed 402 …): the
        next matching request uses them, within their lifetime."""
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return
            with self._transaction(conn):
                for rule_id in once:
                    conn.execute(
                        "UPDATE rules SET consumed = 0, uses = uses - 1"
                        " WHERE id = ? AND scope = 'once' AND consumed = 1",
                        (rule_id,),
                    )

    def count_uses(self, always: Iterable[int]) -> None:
        """Count a use of each every-time rule in ``always``: in memory,
        written at most every ``USE_FLUSH_SECONDS`` (and at close) — never a
        write per request."""
        with self._lock:
            self._unflushed.update(always)
        if self._clock() - self._last_flush >= USE_FLUSH_SECONDS:
            self.flush_uses()

    def flush_uses(self) -> None:
        """Write the every-time rules' uses counted since the last flush."""
        with self._lock:
            self._last_flush = self._clock()
            pending, self._unflushed = self._unflushed, Counter()
            conn = self._open(create=False) if pending else None
            if conn is None:
                return
            with self._transaction(conn):
                for rule_id, uses in sorted(pending.items()):
                    conn.execute("UPDATE rules SET uses = uses + ? WHERE id = ?", (uses, rule_id))

    # -- approving, listing, revoking (the CLI and the dashboard) --

    def describe(self, code: str) -> OverrideEntry:
        """The pending record a code names (for the confirmation prompt)."""
        with self._lock:
            conn = self._open(create=False)
            row = (
                None
                if conn is None
                else conn.execute(
                    "SELECT id, kind, subject, provider, method, route, items, created, expires"
                    " FROM pending WHERE code_hash = ? AND expires > ?",
                    (_code_hash(code), self._clock()),
                ).fetchone()
            )
        if row is None:
            raise OverrideError("no pending refusal has that code (it is unknown, used or expired)")
        return _pending_entry(row)

    def approve(
        self,
        scope: str,
        *,
        approver: str | None,
        code: str | None = None,
        pending_id: str | None = None,
    ) -> OverrideEntry:
        """Turn one pending refusal (named by its ``code`` or its listed
        ``pending_id``) into a one-time grant or an every-time rule. Only
        its own subject approves it (``approver``: the admitted subject,
        None for the local operator). The code is single-use: the pending
        record is deleted in the same transaction."""
        if scope not in SCOPES:
            raise OverrideError("the scope must be once or always")
        if code is not None:
            where, key = "code_hash = ?", _code_hash(code)
        else:
            number = _entry_number(pending_id, "p")
            where, key = "id = ?", str(number)
        now = self._clock()
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                raise OverrideError("no pending refusal matches (it is unknown, used or expired)")
            with self._transaction(conn) as tx:
                row = conn.execute(
                    "SELECT id, kind, subject, provider, method, route, items, created, expires"
                    f" FROM pending WHERE {where} AND expires > ?",
                    (key, now),
                ).fetchone()
                if row is None:
                    tx.rollback()
                    raise OverrideError(
                        "no pending refusal matches (it is unknown, used or expired)"
                    )
                subject = row[2]
                if subject != (approver or ""):
                    tx.rollback()
                    raise OverrideError(
                        "that refusal belongs to another requester; only they can approve it"
                    )
                conn.execute("DELETE FROM pending WHERE id = ?", (row[0],))
                _, kind, _, provider, method, route, items, _, _ = row
                if scope == "once":
                    conn.execute(
                        "INSERT INTO rules (scope, kind, subject, provider, method, route,"
                        " items, created, expires) VALUES ('once', ?, ?, ?, ?, ?, ?, ?, ?)",
                        (kind, subject, provider, method, route, items, now, now + self.ttl),
                    )
                    conn.execute(
                        "DELETE FROM rules WHERE scope = 'once' AND id NOT IN (SELECT id FROM"
                        " rules WHERE scope = 'once' ORDER BY id DESC LIMIT ?)",
                        (MAX_ONCE_GRANTS,),
                    )
                elif kind in ROUTE_KINDS:
                    _insert_always(conn, kind, subject, provider, method, route, "[]", now)
                else:
                    for item in sorted(_items(items)):
                        _insert_always(
                            conn, kind, subject, provider, method, route, json.dumps([item]), now
                        )
        entry = _pending_entry(row)
        return OverrideEntry(
            entry.id,
            scope,
            entry.kind,
            entry.types,
            entry.route,
            entry.subject,
            now,
            now + self.ttl if scope == "once" else None,
            0,
        )

    def entries(self, subject: str | None = None) -> list[OverrideEntry]:
        """Pending codes and live rules (value-free), of ``subject`` only or
        (None) of every subject."""
        now = self._clock()
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return []
            filter_sql, params = (
                ("AND subject = ?", (subject,)) if subject is not None else ("", ())
            )
            pending = conn.execute(
                "SELECT id, kind, subject, provider, method, route, items, created, expires"
                f" FROM pending WHERE expires > ? {filter_sql} ORDER BY id",
                (now, *params),
            ).fetchall()
            rules = conn.execute(
                "SELECT id, scope, kind, subject, provider, method, route, items, created,"
                " expires, uses FROM rules WHERE consumed = 0 AND (expires IS NULL OR"
                f" expires > ?) {filter_sql} ORDER BY id",
                (now, *params),
            ).fetchall()
        listed = [_pending_entry(row) for row in pending]
        for rule_id, scope, kind, subj, provider, method, route, items, created, exp, uses in rules:
            where = (
                "any route"
                if scope == "always" and kind in VALUE_KINDS
                else shown(f"{method} {provider} {route}")
            )
            uses += self._unflushed.get(rule_id, 0)  # this process's, not yet written
            listed.append(
                OverrideEntry(
                    f"r{rule_id}",
                    scope,
                    kind,
                    _types(items),
                    where,
                    shown(subj),
                    created,
                    exp,
                    uses,
                )
            )
        return listed

    def revoke(self, entry_id: str, *, subject: str | None = None) -> None:
        """Delete one rule (``r<N>``) or pending code (``p<N>``) — of
        ``subject`` only, or (None: the local operator's file access) any."""
        table = "rules" if entry_id[:1] == "r" else "pending"
        number = _entry_number(entry_id, entry_id[:1] if entry_id[:1] in ("r", "p") else "r")
        with self._lock:
            conn = self._open(create=False)
            deleted = 0
            if conn is not None:
                if subject is None:
                    cursor = conn.execute(f"DELETE FROM {table} WHERE id = ?", (number,))
                else:
                    cursor = conn.execute(
                        f"DELETE FROM {table} WHERE id = ? AND subject = ?", (number, subject)
                    )
                deleted = cursor.rowcount
        if deleted != 1:
            raise OverrideError(f"no override {entry_id} to revoke")

    def counts(self) -> dict[str, int]:
        """Live record counts, for /status and doctor (value-free)."""
        now = self._clock()
        with self._lock:
            conn = self._open(create=False)
            if conn is None:
                return {"pending": 0, "once": 0, "always": 0}
            pending = conn.execute(
                "SELECT count(*) FROM pending WHERE expires > ?", (now,)
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT scope, count(*) FROM rules WHERE consumed = 0 AND"
                " (expires IS NULL OR expires > ?) GROUP BY scope",
                (now,),
            ).fetchall()
        by_scope = {str(scope): int(n) for scope, n in rows}
        return {
            "pending": int(pending),
            "once": by_scope.get("once", 0),
            "always": by_scope.get("always", 0),
        }


class _Tx:
    """``BEGIN IMMEDIATE`` … ``COMMIT``, rolled back on any exception (or
    explicitly, once)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._open = False

    def __enter__(self) -> _Tx:
        self._conn.execute("BEGIN IMMEDIATE")
        self._open = True
        return self

    def rollback(self) -> None:
        if self._open:
            self._open = False
            self._conn.execute("ROLLBACK")

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._open:
            return
        self._open = False
        self._conn.execute("ROLLBACK" if exc_type is not None else "COMMIT")


def _insert_always(
    conn: sqlite3.Connection,
    kind: str,
    subject: str,
    provider: str,
    method: str,
    route: str,
    items: str,
    now: float,
) -> None:
    # One every-time rule per value (or per route): an existing identical
    # one is kept (its use count with it).
    exists = conn.execute(
        "SELECT 1 FROM rules WHERE scope = 'always' AND subject = ? AND items = ? AND"
        " (kind IN ('block', 'binary_values', 'verbatim_field') OR"
        " (kind = ? AND provider = ? AND method = ? AND route = ?))",
        (subject, items, kind, provider, method, route),
    ).fetchone()
    if exists is None:
        conn.execute(
            "INSERT INTO rules (scope, kind, subject, provider, method, route, items, created)"
            " VALUES ('always', ?, ?, ?, ?, ?, ?, ?)",
            (kind, subject, provider, method, route, items, now),
        )


def _pending_entry(row: Sequence[object]) -> OverrideEntry:
    pending_id, kind, subject, provider, method, route, items, created, expires = row
    return OverrideEntry(
        f"p{pending_id}",
        "pending",
        str(kind),
        _types(str(items)),
        shown(f"{method} {provider} {route}"),
        shown(str(subject)),
        float(created),  # type: ignore[arg-type]
        float(expires),  # type: ignore[arg-type]
        0,
    )


def _entry_number(entry_id: str | None, prefix: str) -> int:
    if (
        not isinstance(entry_id, str)
        or entry_id[:1] != prefix
        or not entry_id[1:].isdigit()
        or len(entry_id) > 20
    ):
        raise OverrideError(f"an override id looks like {prefix}12")
    return int(entry_id[1:])


class OverrideScope:
    """One request's (or one realtime frame's) view of its requester's
    overrides: asked only once detection has decided a refusal. It keeps
    the values it refused (to mint the code) and the rules it used (to
    consume them in ``commit`` once the request passes). Any store fault
    reads as "no override" — the refusal stands (fail closed)."""

    def __init__(
        self,
        store: OverrideStore,
        subject: str,
        *,
        approvable: bool | Callable[[], bool] = True,
        owner: Callable[[], str | None] | None = None,
        config_arg: str | None = None,
    ) -> None:
        self._store = store
        # The ``--config`` argument the local operator's CLI hint carries
        # (``hint_config``), None when the plain hint reaches the proxy's
        # config.
        self.config_arg = config_arg
        # The requester as admitted ("" = the local operator): which hint a
        # refusal names (the CLI's or the dashboard's).
        self.subject = subject
        # Whether this requester can approve a refusal at all (the local
        # operator always — the CLI; a named user only when the access gate
        # says it can, in the dashboard). No code is minted, and no hint
        # promised, for one that cannot; its approved rules still apply. A
        # callable is asked once, and only when a code is about to be minted.
        self._approvable = approvable
        # The key the requester's records are stored under (``owner``,
        # asked once, on the refusal path only): the access gate's stable id
        # for the subject when it supplies one, else the subject itself;
        # None — the gate could not name it — means no override and no code.
        self._owner_of = owner
        self._owner: str | None = subject if owner is None else None
        self._owner_known = owner is None
        self._snap: _Snapshot | None = None
        # (type, value) pairs refused: held in memory for this request only
        # (the request body holds them anyway), hashed when a code is minted.
        self._refused: dict[tuple[str, str], None] = {}
        self._once: set[int] = set()
        self._always: set[int] = set()
        # Whether a finding no approval passes (a deny string in text the
        # proxy cannot rewrite) refuses this request: no value code then.
        self._final = False
        # Whether ``commit`` failed on a store fault (not a lost race).
        self.fault = False
        # What ``commit`` took, until ``settle``: (once, always, marker).
        self._settle: tuple[list[int], list[int], str] | None = None

    @property
    def approvable(self) -> bool:
        if callable(self._approvable):
            self._approvable = self._approvable() is True
        return self._approvable

    @property
    def owner(self) -> str | None:
        """The store key of this requester's records (see ``__init__``)."""
        if not self._owner_known:
            assert self._owner_of is not None
            self._owner = self._owner_of()
            self._owner_known = True
        return self._owner

    def _snapshot(self) -> _Snapshot:
        if self._snap is None:
            owner = self.owner
            try:
                self._snap = _EMPTY if owner is None else self._store.snapshot(owner)
            except Exception as exc:  # noqa: BLE001 — a store fault never overrides
                logger.warning("overrides: the store could not be read (%s)", type(exc).__name__)
                self._snap = _EMPTY
        return self._snap

    def allows(self, detector_type: str, value: str) -> bool:
        """Whether an approved override lets this value through where it
        would refuse the request (the value is then forwarded as sent)."""
        snap = self._snapshot()
        if snap.always_values or snap.once_values:
            item = (detector_type, self._store.digest(detector_type, value))
            rule = snap.always_values.get(item)
            if rule is not None:
                self._always.add(rule)
                return True
            for grant, items in snap.once_values:
                if item in items:
                    self._once.add(grant)
                    return True
        self._refused[(detector_type, value)] = None
        return False

    def unoverridable(self) -> None:
        """This request is refused for a finding no approval passes (a
        deny string where the proxy cannot redact it): a value refusal's
        code could never let it through, so none is minted."""
        self._final = True

    def route_rule(
        self, kind: str, provider: str, method: str, route: str
    ) -> tuple[str, int] | None:
        """The requester's approved override of refusal ``kind`` on this
        route, if any (not yet used: ``use_route``)."""
        key = _RouteKey(kind, provider, method, route)
        snap = self._snapshot()
        if key in snap.always_routes:
            return ("always", snap.always_routes[key])
        if key in snap.once_routes:
            return ("once", snap.once_routes[key])
        return None

    def use_route(self, rule: tuple[str, int]) -> None:
        (self._always if rule[0] == "always" else self._once).add(rule[1])

    def refusal_code(self, kind: str, provider: str, method: str, route: str) -> str | None:
        """Mint the code for the refusal this request is about to be
        answered with, or None when it cannot carry one (a value refusal
        that refused no value through this scope, a requester who cannot
        approve it, a store fault)."""
        if kind in VALUE_KINDS and (self._final or not self._refused):
            return None
        if not self.approvable:
            return None
        owner = self.owner
        if owner is None:
            return None
        try:
            return self._store.record_pending(kind, owner, provider, method, route, self._refused)
        except Exception as exc:  # noqa: BLE001 — the refusal stands without a code
            logger.warning("overrides: no code minted (%s)", type(exc).__name__)
            return None

    def refusal_hint(self, kind: str, provider: str, method: str, route: str) -> str | None:
        """The text telling this requester how to allow the refusal (its code
        minted), or None when there is none: the local operator runs the CLI
        with the code; a named user approves in the dashboard."""
        code = self.refusal_code(kind, provider, method, route)
        if code is None:
            return None
        return allow_hint(code, self.config_arg) if not self.subject else DASHBOARD_HINT

    def commit(self) -> tuple[bool, str | None]:
        """Consume what this request used, as it passes: ``(ok, marker)``,
        marker once / always / None (nothing used). ``ok`` is False when a
        one-time grant was consumed by another request first (or the store
        failed): the request must be refused. The use is settled once the
        request's fate is known (``settle``): a request refused before it
        reaches the upstream hands its one-time grants back."""
        if not self._once and not self._always:
            return True, None
        try:
            # Every-time rules alone need no write: their use is counted in
            # memory once the request is sent (settle).
            ok = self._store.consume(sorted(self._once)) if self._once else True
        except Exception as exc:  # noqa: BLE001
            logger.warning("overrides: a use could not be recorded (%s)", type(exc).__name__)
            self.fault = True
            ok = False
        if not ok:
            return False, None
        marker = "once" if self._once else "always"
        self._settle = (sorted(self._once), sorted(self._always), marker)
        self._once.clear()
        self._always.clear()
        return True, marker

    def settle(self, sent: bool) -> None:
        """The fate of what ``commit`` took: ``sent`` (handed to the
        upstream — a send that then fails in transit included) counts the
        use; otherwise the one-time grants go back to the store, unused.
        Once; a store fault is logged by type only."""
        pending, self._settle = self._settle, None
        if pending is None:
            return
        once, always, marker = pending
        try:
            if sent:
                self._store.used[marker] += 1
                if always:
                    self._store.count_uses(always)
            elif once:
                self._store.release(once)
        except Exception as exc:  # noqa: BLE001 — the request's fate is already decided
            logger.warning("overrides: a use could not be settled (%s)", type(exc).__name__)


OVERRIDE_RACED = (
    "llm-redact: the one-time override this request relied on was already used by"
    " another request; the request was not forwarded"
)
OVERRIDE_FAULT = (
    "llm-redact: the one-time override this request relied on could not be recorded"
    " (the override store is busy or unreadable); the request was not forwarded"
)


def raced_message(scope: OverrideScope) -> str:
    """The refusal text when ``commit`` failed: a lost race or a store fault."""
    return OVERRIDE_FAULT if scope.fault else OVERRIDE_RACED
