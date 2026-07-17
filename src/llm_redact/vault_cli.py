"""Vault lifecycle commands: sessions list/prune, lookup, gen-key.

These operate directly on the sqlite database. Reads (list, lookup) are safe
against a running proxy (WAL allows concurrent readers). Prune is not: a
running proxy's write-through caches still hold pruned mappings, and a
pruned session's counters restart — stop or restart the proxy afterwards.

``lookup`` deliberately prints a secret to the terminal: that is the tool.
The value is never logged anywhere.
"""

import argparse
import hmac
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from llm_redact.config import apply_env_overrides, load_config
from llm_redact.placeholders import canonicalize
from llm_redact.vault import VaultKeyError, default_vault_path

if TYPE_CHECKING:
    # Type-only import against the plugin-API Protocol: the concrete cipher is
    # resolved lazily through the registry (paid subsystem) so a plaintext
    # vault never reaches for llm-redact-pro or the cryptography extra.
    from llm_redact.plugin_api import VaultCipher

_OLDER_THAN_RE = re.compile(r"^(\d+)d$")

if TYPE_CHECKING:
    from llm_redact.config import Config
    from llm_redact.vault_rdbms import RdbmsStore


def _rdbms_config(args: argparse.Namespace) -> "Config | None":
    """The effective Config when it selects an RDBMS vault backend and no
    explicit --db file was given (an explicit file always means sqlite);
    None otherwise. RDBMS lifecycle commands are config-driven — there is
    no file path to point at."""
    if getattr(args, "db", None) is not None:
        return None
    from llm_redact.config import RDBMS_BACKENDS

    config = apply_env_overrides(load_config(args.config))
    if config.vault.backend in RDBMS_BACKENDS:
        return config
    return None


def _open_rdbms_store(config: "Config") -> "RdbmsStore":
    # Deliberately NOT behind the off-box gate: these commands read data
    # the operator already owns (the same stance as the license gates —
    # local tools are never blocked from your own vault). The cipher comes
    # through the registry so an encrypted vault resolves the paid cipher
    # (llm-redact-pro) and a plaintext one stays dependency-free.
    from llm_redact.registry import get_registry
    from llm_redact.vault_rdbms import RdbmsStore

    return RdbmsStore(config.vault, get_registry().build_cipher(config.vault))


def _resolve_db(args: argparse.Namespace) -> Path:
    if args.db is not None:
        return Path(args.db).expanduser()
    config = apply_env_overrides(load_config(args.config))
    if config.vault.path:
        return Path(config.vault.path).expanduser()
    return default_vault_path()


def _readonly_uri(path: Path) -> str:
    """A read-only sqlite `file:` URI with the path percent-encoded.

    Without encoding, a `?` or `#` in the vault path would be parsed as the
    URI's query/fragment — letting later text inject parameters (e.g. `vfs=`)
    or defeat `mode=ro`. `/` and `:` stay literal (POSIX separators, Windows
    drive letters)."""
    from urllib.parse import quote

    return f"file:{quote(path.resolve().as_posix(), safe='/:')}?mode=ro"


def _connect(path: Path, *, readonly: bool) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(path)
    if readonly:
        return sqlite3.connect(_readonly_uri(path), uri=True)
    return sqlite3.connect(path, isolation_level=None)


def _cipher_for(conn: sqlite3.Connection, path: Path) -> "VaultCipher | None":
    """A verified VaultCipher for a v3 database, None for v2. Fails closed.

    The concrete cipher is the paid subsystem, resolved through the registry;
    a missing llm-redact-pro surfaces as a VaultKeyError (the caller already
    prints it and exits) rather than a traceback."""
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version < 3:
        return None
    from llm_redact.config import ConfigError, VaultConfig
    from llm_redact.registry import get_registry

    try:
        cipher = get_registry().build_cipher(VaultConfig(encryption="fernet"))
    except ConfigError as exc:
        raise VaultKeyError(str(exc)) from exc
    assert cipher is not None  # a fernet config always yields a cipher or raises
    row = conn.execute("SELECT value FROM vault_meta WHERE key = 'key_check'").fetchone()
    if row is not None and str(row[0]) != cipher.key_check():
        raise VaultKeyError(f"LLM_REDACT_VAULT_KEY does not match the vault at {path}")
    return cipher


def run_sessions_list(args: argparse.Namespace) -> int:
    rdbms = _rdbms_config(args)
    if rdbms is not None:
        store = _open_rdbms_store(rdbms)
        try:
            summary = store.sessions_summary()
        finally:
            store.close()
        rows = [(row["session"], row["entries"], row["first"], row["last"]) for row in summary]
        path: Path | str = f"the {rdbms.vault.backend} vault"
    else:
        try:
            path = _resolve_db(args)
            conn = _connect(path, readonly=True)
        except FileNotFoundError as exc:
            print(f"no vault database at {exc}")
            return 1
        rows = conn.execute(
            "SELECT session_id, COUNT(*), MIN(created_at), MAX(created_at)"
            " FROM mappings GROUP BY session_id ORDER BY MAX(created_at) DESC"
        ).fetchall()
        conn.close()
    if args.json:
        print(
            json.dumps(
                [
                    {"session": s, "entries": c, "first": first, "last": last}
                    for s, c, first, last in rows
                ],
                indent=2,
            )
        )
        return 0
    if not rows:
        print(f"no sessions in {path}")
        return 0
    width = max(len(str(row[0])) for row in rows)
    print(f"{'SESSION':<{width}}  ENTRIES  FIRST                 LAST")
    for session_id, count, first, last in rows:
        print(f"{session_id:<{width}}  {count:>7}  {first}  {last}")
    return 0


def _rdbms_sessions_prune(config: "Config", days: int, *, assume_yes: bool) -> int:
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    store = _open_rdbms_store(config)
    try:
        doomed = [
            row for row in store.sessions_summary() if row["last"] and str(row["last"]) < cutoff
        ]
        if not doomed:
            print(f"nothing to prune: no session idle longer than {days} days")
            return 0
        for row in doomed:
            print(f"would delete {row['session']} ({row['entries']} mappings)")
        if not assume_yes:
            answer = input(f"delete {len(doomed)} session(s)? [y/N] ").strip().lower()
            if answer != "y":
                print("aborted")
                return 1
        deleted = len(store.prune_sessions(days))
    finally:
        store.close()
    print(f"deleted {deleted} session(s)")
    print(
        "NOTE: stop or restart a running proxy — its caches still hold the "
        "pruned mappings, and pruned sessions' counters restart."
    )
    return 0


def run_sessions_prune(args: argparse.Namespace) -> int:
    match = _OLDER_THAN_RE.fullmatch(args.older_than)
    if match is None:
        print("--older-than must look like '90d' (whole days)")
        return 2
    days = int(match.group(1))
    rdbms = _rdbms_config(args)
    if rdbms is not None:
        return _rdbms_sessions_prune(rdbms, days, assume_yes=args.yes)
    try:
        path = _resolve_db(args)
        conn = _connect(path, readonly=False)
    except FileNotFoundError as exc:
        print(f"no vault database at {exc}")
        return 1
    # Whole sessions only: deleting individual rows would let MAX(n)+1
    # reissue a still-referenced placeholder number for a different value.
    doomed = conn.execute(
        "SELECT session_id, COUNT(*) FROM mappings GROUP BY session_id"
        " HAVING MAX(created_at) < strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?)",
        (f"-{days} days",),
    ).fetchall()
    if not doomed:
        print(f"nothing to prune: no session idle longer than {days} days")
        conn.close()
        return 0
    for session_id, count in doomed:
        print(f"would delete {session_id} ({count} mappings)")
    if not args.yes:
        answer = input(f"delete {len(doomed)} session(s)? [y/N] ").strip().lower()
        if answer != "y":
            print("aborted")
            conn.close()
            return 1
    ids = [session_id for session_id, _count in doomed]
    marks = ",".join("?" * len(ids))
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(f"DELETE FROM mappings WHERE session_id IN ({marks})", ids)
    conn.execute(f"DELETE FROM response_sessions WHERE session_id IN ({marks})", ids)
    conn.execute("COMMIT")
    conn.close()
    print(f"deleted {len(doomed)} session(s)")
    print(
        "NOTE: stop or restart a running proxy — its caches still hold the "
        "pruned mappings, and pruned sessions' counters restart."
    )
    return 0


def _rdbms_lookup(config: "Config", args: argparse.Namespace) -> int:
    store = _open_rdbms_store(config)
    try:
        if args.value is not None:
            value_rows = store.lookup_value(args.value, args.session)
            for session_id, detector_type, placeholder in value_rows:
                print(f"{session_id}\t{detector_type}\t{placeholder}")
            if not value_rows:
                print("value not found")
                return 1
            return 0
        canonical = canonicalize(args.token) or args.token  # accept pasted mangles
        token_rows = store.lookup_token(canonical, args.session)
        for session_id, original in token_rows:
            print(f"{session_id}\t{original}")
        if not token_rows:
            print(f"{canonical} not found")
            return 1
        return 0
    finally:
        store.close()


def run_lookup(args: argparse.Namespace) -> int:
    rdbms = _rdbms_config(args)
    if rdbms is not None:
        return _rdbms_lookup(rdbms, args)
    try:
        path = _resolve_db(args)
        conn = _connect(path, readonly=True)
    except FileNotFoundError as exc:
        print(f"no vault database at {exc}")
        return 2
    try:
        cipher = _cipher_for(conn, path)
    except VaultKeyError as exc:
        print(str(exc))
        conn.close()
        return 2
    try:
        if args.value is not None:
            return _lookup_by_value(conn, cipher, args.value, args.session)
        return _lookup_by_token(conn, cipher, args.token, args.session)
    finally:
        conn.close()


def _lookup_by_token(
    conn: sqlite3.Connection, cipher: "VaultCipher | None", token: str, session: str | None
) -> int:
    canonical = canonicalize(token) or token  # accept pasted mangles
    column = "original" if cipher is None else "original_ct"
    if session is not None:
        rows = conn.execute(
            f"SELECT session_id, {column} FROM mappings"  # noqa: S608 - column is ours
            " WHERE session_id = ? AND placeholder = ?",
            (session, canonical),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT session_id, {column} FROM mappings WHERE placeholder = ?",  # noqa: S608
            (canonical,),
        ).fetchall()
    if not rows:
        print(f"{canonical} not found")
        return 1
    for session_id, stored in rows:
        original = stored if cipher is None else cipher.decrypt(stored)
        print(f"{session_id}\t{original}")
    return 0


def _lookup_by_value(
    conn: sqlite3.Connection, cipher: "VaultCipher | None", value: str, session: str | None
) -> int:
    found = False
    if cipher is None:
        where = "original = ?"
        params: tuple[str, ...] = (value,)
        if session is not None:
            where += " AND session_id = ?"
            params = (value, session)
        for session_id, detector_type, placeholder in conn.execute(
            f"SELECT session_id, detector_type, placeholder FROM mappings WHERE {where}", params
        ):
            print(f"{session_id}\t{detector_type}\t{placeholder}")
            found = True
    else:
        # The MAC is (session, type)-domain-separated: compute it per pair.
        pair_sql = "SELECT DISTINCT session_id, detector_type FROM mappings"
        pair_params: tuple[str, ...] = ()
        if session is not None:
            pair_sql += " WHERE session_id = ?"
            pair_params = (session,)
        for session_id, detector_type in conn.execute(pair_sql, pair_params).fetchall():
            mac = cipher.mac(session_id, detector_type, value)
            row = conn.execute(
                "SELECT placeholder FROM mappings"
                " WHERE session_id = ? AND detector_type = ? AND original_mac = ?",
                (session_id, detector_type, mac),
            ).fetchone()
            if row is not None:
                print(f"{session_id}\t{detector_type}\t{row[0]}")
                found = True
    if not found:
        print("value not found")
        return 1
    return 0


def run_vault_verify(args: argparse.Namespace) -> int:
    """Read-only integrity sweep. Checks the cardinal counter invariant
    (n is exactly 1..N per session/type — a gap would let MAX(n)+1 reissue a
    live number for a different value) and, for encrypted vaults, that every
    ciphertext decrypts and every MAC index matches its plaintext. Never
    prints a stored value — sessions, types, and counts only. Safe against a
    running proxy (WAL readers)."""
    rdbms = _rdbms_config(args)
    if rdbms is not None:
        print(
            f"vault verify supports the sqlite backend only (configured:"
            f" {rdbms.vault.backend!r}); a server RDBMS enforces the same"
            " constraints natively — use the engine's own integrity tooling"
        )
        return 2
    try:
        path = _resolve_db(args)
        conn = _connect(path, readonly=True)
    except FileNotFoundError as exc:
        print(f"no vault database at {exc}")
        return 2
    try:
        cipher = _cipher_for(conn, path)
    except VaultKeyError as exc:
        print(str(exc))
        conn.close()
        return 2

    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    total = conn.execute("SELECT COUNT(*) FROM mappings").fetchone()[0]
    sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM mappings").fetchone()[0]
    print(f"vault verify: {path} (schema v{version}, {'encrypted' if cipher else 'plaintext'})")
    print(f"  rows: {total} across {sessions} session(s)")

    failed = False

    # Counter density: UNIQUE(session,type,n) already bars duplicates; this
    # catches GAPS (a hand-deleted row) that break the 1..N invariant.
    bad_density = conn.execute(
        "SELECT session_id, detector_type, MIN(n), MAX(n), COUNT(*) FROM mappings"
        " GROUP BY session_id, detector_type HAVING MAX(n) != COUNT(*) OR MIN(n) != 1"
    ).fetchall()
    if bad_density:
        failed = True
        print(f"  FAIL counter density: {len(bad_density)} (session,type) group(s) not 1..N")
        for session_id, detector_type, lo, hi, count in bad_density[:10]:
            print(f"       {session_id} / {detector_type}: n in [{lo}..{hi}], count {count}")
    else:
        print("  PASS counter density (n is 1..N per session/type)")

    if cipher is not None:
        checked = decrypt_fail = mac_fail = 0
        for session_id, detector_type, original_ct, original_mac in conn.execute(
            "SELECT session_id, detector_type, original_ct, original_mac FROM mappings"
        ):
            checked += 1
            try:
                original = cipher.decrypt(original_ct)
            except VaultKeyError:
                # Key already matched key_check, so this is genuine ciphertext
                # corruption, not a wrong key.
                decrypt_fail += 1
                continue
            expected = cipher.mac(session_id, detector_type, original)
            if not hmac.compare_digest(expected, str(original_mac)):
                mac_fail += 1
        if decrypt_fail:
            failed = True
            print(f"  FAIL ciphertext: {decrypt_fail}/{checked} row(s) did not decrypt")
        else:
            print(f"  PASS ciphertext decrypts ({checked}/{checked})")
        if mac_fail:
            failed = True
            print(f"  FAIL MAC index: {mac_fail}/{checked} row(s) inconsistent with plaintext")
        else:
            print(f"  PASS MAC index consistent ({checked}/{checked})")

    conn.close()
    print("verify FAILED" if failed else "verify OK")
    return 1 if failed else 0


def _proxy_reachable(args: argparse.Namespace) -> bool:
    """Best-effort probe: is a proxy answering /status on the configured
    host:port? Rotation must not run against a live proxy — its in-memory
    cipher would keep writing old-key rows after the DB flips to the new key.
    A failed probe (no config, mTLS with no client cert, down) returns False;
    the confirmation prompt is the backstop."""
    try:
        config = apply_env_overrides(load_config(args.config))
    except Exception:
        return False
    import httpx

    from llm_redact.proxy import RESERVED_PREFIX

    scheme = "https" if config.tls.enabled else "http"
    url = f"{scheme}://{config.host}:{config.port}{RESERVED_PREFIX}/status"
    try:
        httpx.get(url, timeout=1.0).raise_for_status()
        return True
    except httpx.HTTPError:
        return False


def run_vault_rotate_key(args: argparse.Namespace) -> int:
    """Re-encrypt the vault under a new key (offline, one transaction).

    The CURRENT key resolves via the normal env->keyring path; the NEW key
    comes from LLM_REDACT_NEW_VAULT_KEY or an interactive getpass (never
    argv/history). Preflight — fail before touching state: proxy must be
    stopped, vault must be encrypted, current key must match, new key must
    decode and differ, and every row must decrypt with the current key.
    Afterwards, point LLM_REDACT_VAULT_KEY (or the keychain) at the new key
    before restarting; until then serve fails closed at open."""
    rdbms = _rdbms_config(args)
    if rdbms is not None:
        print(
            f"vault rotate-key supports the sqlite backend only (configured:"
            f" {rdbms.vault.backend!r}): a server engine's MVCC keeps old row"
            " versions no rewrite of ours could honestly scrub — rotate by"
            " standing up a fresh schema under the new key and re-pointing the DSN"
        )
        return 2
    from llm_redact.config import ConfigError, VaultConfig
    from llm_redact.registry import get_registry
    from llm_redact.vault_crypto import ENV_KEY, NEW_ENV_KEY, decode_master_key

    registry = get_registry()

    try:
        path = _resolve_db(args)
    except FileNotFoundError as exc:
        print(f"no vault database at {exc}")
        return 2
    if not path.exists():
        print(f"no vault database at {path}")
        return 2
    if _proxy_reachable(args):
        print(
            "a proxy is running — stop it before rotating"
            " (its cached key would keep writing old-key rows)"
        )
        return 3

    conn = _connect(path, readonly=False)
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version < 3:
            print("this vault is not encrypted; there is no key to rotate")
            return 2
        try:
            old_cipher = registry.build_cipher(VaultConfig(encryption="fernet"))
        except (VaultKeyError, ConfigError) as exc:
            print(str(exc))
            return 2
        assert old_cipher is not None  # a fernet config yields a cipher or raises
        row = conn.execute("SELECT value FROM vault_meta WHERE key = 'key_check'").fetchone()
        if row is not None and str(row[0]) != old_cipher.key_check():
            print(f"the current {ENV_KEY} does not match the vault at {path}")
            return 2

        raw_new = os.environ.get(NEW_ENV_KEY, "").strip()
        source = NEW_ENV_KEY
        if not raw_new:
            import getpass

            raw_new = getpass.getpass("new vault key (from `llm-redact vault gen-key`): ").strip()
            source = "the entered key"
        if not raw_new:
            print("no new key entered; generate one first: llm-redact vault gen-key")
            return 2
        try:
            new_cipher = registry.cipher_from_key(decode_master_key(raw_new, source))
        except (VaultKeyError, ConfigError) as exc:
            print(str(exc))
            return 2
        if new_cipher.key_check() == old_cipher.key_check():
            print("the new key is identical to the current key; nothing to rotate")
            return 2

        # Preflight decrypt-sweep: a corrupt row aborts before any write.
        bad = sum(
            _decrypt_fails(old_cipher, ct)
            for (ct,) in conn.execute("SELECT original_ct FROM mappings")
        )
        if bad:
            print(
                f"aborted: {bad} row(s) did not decrypt with the current key (run `vault verify`)"
            )
            return 1

        count = conn.execute("SELECT COUNT(*) FROM mappings").fetchone()[0]
        if not args.yes:
            answer = (
                input(
                    f"re-encrypt {count} mapping(s) under the new key? "
                    "ensure the proxy is stopped. [y/N] "
                )
                .strip()
                .lower()
            )
            if answer != "y":
                print("aborted")
                return 1

        from llm_redact.vault import rotate_vault_key

        rotated = rotate_vault_key(conn, old_cipher, new_cipher)
    finally:
        conn.close()
    print(
        f"rotated {rotated} mapping(s). Set {ENV_KEY} (or the keychain via "
        "`vault set-key`) to the NEW key before restarting the proxy."
    )
    return 0


def _decrypt_fails(cipher: "VaultCipher", ct: bytes) -> int:
    try:
        cipher.decrypt(ct)
    except VaultKeyError:
        return 1
    return 0


def run_vault_backup(args: argparse.Namespace) -> int:
    """Consistent single-file snapshot via the SQLite online-backup API.

    The vault is live WAL: a naive `cp vault.db` misses the -wal sidecar and
    can capture a torn/stale copy — and a mapping present in WAL but absent
    from the copy would later let MAX(n)+1 reissue that number for a DIFFERENT
    value (the cardinal sin) on restore. The backup API reads THROUGH the WAL,
    so the destination is a coherent standalone database. Safe against a
    running proxy (it copies committed state)."""
    rdbms = _rdbms_config(args)
    if rdbms is not None:
        print(
            f"vault backup supports the sqlite backend only (configured:"
            f" {rdbms.vault.backend!r}); back up a server RDBMS with the"
            " engine's own tooling (pg_dump, mysqldump, RMAN, ...)"
        )
        return 2
    try:
        src_path = _resolve_db(args)
    except FileNotFoundError as exc:
        print(f"no vault database at {exc}")
        return 2
    if not src_path.exists():
        print(f"no vault database at {src_path}")
        return 2
    dest = Path(args.dest).expanduser()
    if dest.exists() and not args.force:
        print(f"{dest} already exists; pass --force to overwrite")
        return 2
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Pre-create the destination 0600: it holds the same secrets/ciphertext.
    # O_NOFOLLOW (no-op on Windows) refuses to write THROUGH a pre-planted
    # symlink at `dest` — a broken symlink slips past the exists() guard above,
    # so without this the vault secrets could be truncated onto its target.
    flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(dest, flags, 0o600)
    except OSError as exc:
        # ELOOP (symlink refused by O_NOFOLLOW) or any other create failure.
        print(f"cannot write backup to {dest}: {type(exc).__name__}")
        return 2
    os.close(fd)
    src = sqlite3.connect(_readonly_uri(src_path), uri=True)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    print(f"backed up {src_path} -> {dest}")
    return 0


def run_vault_gen_key(_args: argparse.Namespace) -> int:
    try:
        from llm_redact.vault_crypto import generate_key
    except ImportError:
        print("gen-key requires the crypto extra: pip install 'llm-redact-proxy[crypto]'")
        return 2
    print(generate_key())
    return 0


def run_vault_set_key(_args: argparse.Namespace) -> int:
    """Store the vault key in the OS keychain (keyring extra).

    The key is read via getpass — never echoed, never in shell history —
    and validated before storing, so a truncated paste fails here rather
    than at the next `serve`. Deliberately no generate-on-empty: gen-key
    prints a key the user can back up first; a key that only ever lived
    in one keychain would make the vault unrecoverable if that keychain
    is lost.
    """
    try:
        import keyring
    except ImportError:
        print("set-key requires the keyring extra: pip install 'llm-redact-proxy[keyring]'")
        return 2
    try:
        from llm_redact.vault_crypto import KEYRING_ITEM, KEYRING_SERVICE, decode_master_key
    except ImportError:
        print("set-key requires the crypto extra: pip install 'llm-redact-proxy[crypto]'")
        return 2
    import getpass

    from llm_redact.vault import VaultKeyError

    raw = getpass.getpass("vault key (from `llm-redact vault gen-key`): ").strip()
    if not raw:
        print("no key entered; generate one first: llm-redact vault gen-key")
        return 2
    try:
        decode_master_key(raw, "the entered key")
    except VaultKeyError as problem:
        print(str(problem))
        return 2
    keyring.set_password(KEYRING_SERVICE, KEYRING_ITEM, raw)
    print(f"stored in the OS keychain as {KEYRING_SERVICE}/{KEYRING_ITEM}")
    return 0
