"""Vault durability under fault. The vault is the one piece of state whose
corruption is unacceptable: a lost or reused token number silently rehydrates
the WRONG secret. These prove the write path fails closed on a disk/IO fault
without wedging the connection or skipping a counter, that interleaved
multi-view writes keep the counter dense, and that a corrupted at-rest
ciphertext fails closed rather than yielding a wrong plaintext.

Faults are injected by wrapping the connection / cipher — no real disk is
filled, no real key is used.
"""

import sqlite3
from pathlib import Path

import pytest

from llm_redact.vault import open_sqlite_vault


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "vault.db"


class _FlakyConn:
    """Delegates to a real sqlite connection but raises OperationalError the
    first `times` executions whose SQL contains `fail_on` — a disk-full write."""

    def __init__(self, real: sqlite3.Connection, fail_on: str, times: int = 1) -> None:
        self._real = real
        self._fail_on = fail_on
        self._times = times

    def execute(self, sql: str, *args: object) -> object:
        if self._fail_on in sql and self._times > 0:
            self._times -= 1
            raise sqlite3.OperationalError("database or disk is full")
        return self._real.execute(sql, *args)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def test_write_fault_fails_closed_without_wedging_or_skipping(db_path: Path) -> None:
    vault = open_sqlite_vault(db_path, "s")
    vault._conn = _FlakyConn(vault._conn, "INSERT INTO mappings", times=1)  # type: ignore[assignment]

    # The write raises (fail closed) instead of returning a half-issued token.
    with pytest.raises(sqlite3.OperationalError):
        vault.placeholder_for("EMAIL", "jane@example.com")

    # Nothing was cached from the failed write — no poisoned mapping.
    assert vault.original_for("«EMAIL_001»") is None

    # The connection is NOT wedged (the open transaction was rolled back) and
    # the counter was NOT consumed: the retry issues the very same dense number
    # and resolves to the right value.
    token = vault.placeholder_for("EMAIL", "jane@example.com")
    assert token == "«EMAIL_001»"
    assert vault.original_for(token) == "jane@example.com"

    # A subsequent distinct value continues densely — 002, never a gap.
    assert vault.placeholder_for("EMAIL", "john@example.com") == "«EMAIL_002»"
    vault.close()


def test_double_fault_propagates_the_original_error(db_path: Path) -> None:
    # Disk-full on the INSERT *and* on the recovery ROLLBACK: the original
    # write error must still propagate (the rollback failure is suppressed),
    # never a TypeError from a broken suppress clause. Nothing may be cached.
    vault = open_sqlite_vault(db_path, "s")
    vault._conn = _FlakyConn(  # type: ignore[assignment]
        _FlakyConn(vault._conn, "INSERT INTO mappings", times=1), "ROLLBACK", times=1
    )
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        vault.placeholder_for("EMAIL", "jane@example.com")
    assert vault.original_for("«EMAIL_001»") is None


def test_interleaved_multi_view_writes_keep_counter_dense(db_path: Path) -> None:
    a = open_sqlite_vault(db_path, "s")
    b = open_sqlite_vault(db_path, "s")
    tokens = []
    for i in range(10):
        view = a if i % 2 == 0 else b
        tokens.append(view.placeholder_for("EMAIL", f"user{i}@example.com"))
    # Ten distinct values issued across two interleaved views over one DB must
    # occupy exactly 001..010 — no gap (a lost number), no reuse (a collision).
    assert sorted(tokens) == [f"«EMAIL_{n:03d}»" for n in range(1, 11)]
    assert len(set(tokens)) == 10
    a.close()
    b.close()


class _FakeCipher:
    """A VaultCipher stand-in with no crypto dependency: decrypt raises on a
    ciphertext it did not produce, modelling at-rest corruption."""

    def key_check(self) -> str:
        return "fake-key-check"

    def mac(self, session_id: str, detector_type: str, original: str) -> str:
        return f"{session_id}:{detector_type}:{original}"

    def encrypt(self, original: str) -> bytes:
        return b"ct:" + original.encode()

    def decrypt(self, token: bytes) -> str:
        if not token.startswith(b"ct:"):
            raise ValueError("ciphertext failed authentication")
        return token[3:].decode()


def test_corrupted_ciphertext_fails_closed_never_wrong_value(db_path: Path) -> None:
    cipher = _FakeCipher()
    vault = open_sqlite_vault(db_path, "s", cipher=cipher)  # type: ignore[arg-type]
    token = vault.placeholder_for("EMAIL", "jane@example.com")
    assert vault.original_for(token) == "jane@example.com"

    # Corrupt the stored ciphertext and drop the plaintext cache entry, so the
    # next lookup takes the cold-cache DB path (another view / post-restart).
    vault._conn.execute(
        "UPDATE mappings SET original_ct = ? WHERE placeholder = ?", (b"CORRUPT", token)
    )
    vault._reverse.pop(token)

    # Fail closed: a value that cannot be authenticated raises, never returns a
    # wrong or partial plaintext.
    with pytest.raises(ValueError):
        vault.original_for(token)
    vault.close()
