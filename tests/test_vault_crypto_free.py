"""Mutation-killing coverage for vault.py's ENCRYPTED arms, driven by the
Free-side FakeVaultCipher.

vault.py stays in the Free mutmut `only_mutate` set, but the tests that killed
its crypto-path mutants (the memory-encrypted vault, the cipher-threading
manager, key rotation's rollback path, and `open_sqlite_vault`'s ownership)
used the REAL Fernet cipher and moved to the llm-redact-pro repo in the R4
open-core split. These reproduce that mutation coverage with the fake cipher, so
a weakened crypto arm still fails a Free-side test (docs/assurance.md). The
real-Fernet at-rest guarantees stay proven in the pro repo.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fake_cipher import FakeVaultCipher
from llm_redact.vault import (
    EncryptedInMemoryVault,
    InMemoryVaultManager,
    VaultKeyError,
    open_sqlite_vault,
    rotate_vault_key,
)

# --- EncryptedInMemoryVault: forward dedup + reverse decrypt round trip -------


def test_encrypted_memory_vault_dedup_counter_and_roundtrip() -> None:
    vault = EncryptedInMemoryVault(FakeVaultCipher(b"k" * 32), "sess-a")

    first = vault.placeholder_for("EMAIL", "jane@corp.example")
    assert first == "«EMAIL_001»"
    # Same (type, value) → the SAME token (forward HMAC index dedups; a dropped
    # cache or a null forward-write would reissue «EMAIL_002»).
    assert vault.placeholder_for("EMAIL", "jane@corp.example") == first
    # A distinct value advances the per-type counter.
    assert vault.placeholder_for("EMAIL", "bob@corp.example") == "«EMAIL_002»"
    # Reverse lookup decrypts the stored ciphertext back to the original (a
    # null reverse-write or a dropped lookup would return None / the wrong value).
    assert vault.original_for(first) == "jane@corp.example"
    assert vault.original_for("«EMAIL_002»") == "bob@corp.example"
    assert vault.original_for("«EMAIL_404»") is None
    assert len(vault) == 2


def test_encrypted_memory_vault_holds_ciphertext_not_plaintext() -> None:
    # The RAM reverse map holds Fernet-shaped ciphertext, not the plaintext —
    # kills a mutant that stores the original (or None) instead of encrypt(...).
    vault = EncryptedInMemoryVault(FakeVaultCipher(b"k" * 32), "sess-a")
    token = vault.placeholder_for("EMAIL", "jane@corp.example")
    stored = vault._reverse[token]
    assert isinstance(stored, bytes) and stored != b"jane@corp.example"
    assert vault.original_for(token) == "jane@corp.example"


# --- InMemoryVaultManager threads the cipher into each encrypted view ---------


def test_memory_manager_hands_out_encrypted_views() -> None:
    manager = InMemoryVaultManager(cipher=FakeVaultCipher(b"k" * 32))
    view = manager.get("conv-1")
    # A cipher-configured manager MUST build EncryptedInMemoryVault views (a
    # dropped cipher, a null session id, or a mis-arity construction all break
    # this) — and those views round-trip.
    assert isinstance(view, EncryptedInMemoryVault)
    token = view.placeholder_for("EMAIL", "jane@corp.example")
    assert view.original_for(token) == "jane@corp.example"
    # Same session id → same view instance (identity cache).
    assert manager.get("conv-1") is view
    # Distinct sessions collide on token NAME but not on value.
    other = manager.get("conv-2")
    assert other.placeholder_for("EMAIL", "carol@corp.example") == "«EMAIL_001»"
    assert other.original_for("«EMAIL_001»") == "carol@corp.example"


def test_memory_manager_without_cipher_is_plaintext() -> None:
    # The other arm of the dispatch: no cipher → a plain in-memory vault.
    from llm_redact.vault import InMemoryVault

    manager = InMemoryVaultManager()
    assert isinstance(manager.get("conv-1"), InMemoryVault)


# --- rotate_vault_key: the except-arm ROLLBACK actually rolls back -----------


def test_rotate_vault_key_rolls_back_on_decrypt_failure(tmp_path: Path) -> None:
    db = tmp_path / "vault.db"
    old = FakeVaultCipher(b"o" * 32)
    seed = open_sqlite_vault(db, "sess-a", old)
    seed.placeholder_for("EMAIL", "jane@corp.example")
    seed.close()

    # Corrupt the ciphertext so the rotation's decrypt of the old value fails
    # partway through the transaction — exercising the except: ROLLBACK; raise
    # arm. A mutated rollback (a bad SQL string / a None argument) raises a
    # DIFFERENT error type instead of re-raising the decrypt failure.
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("UPDATE mappings SET original_ct = X'00'")
    conn.commit()
    new = FakeVaultCipher(b"n" * 32)
    with pytest.raises(VaultKeyError):
        rotate_vault_key(conn, old, new)
    conn.close()


def test_rotate_vault_key_reencrypts_and_preserves_identity(tmp_path: Path) -> None:
    db = tmp_path / "vault.db"
    old = FakeVaultCipher(b"o" * 32)
    va = open_sqlite_vault(db, "sess-a", old)
    token = va.placeholder_for("EMAIL", "jane@corp.example")
    seed_conn = sqlite3.connect(db)
    before = seed_conn.execute("SELECT placeholder, n, original_mac FROM mappings").fetchone()
    seed_conn.close()
    va.close()

    conn = sqlite3.connect(db, isolation_level=None)
    new = FakeVaultCipher(b"n" * 32)
    assert rotate_vault_key(conn, old, new) == 1
    # The MAC index is re-derived under the new key (it is seed-dependent);
    # placeholder + counter are copied verbatim (token identity invariant).
    after = conn.execute("SELECT placeholder, n, original_mac FROM mappings").fetchone()
    conn.close()
    assert after[0] == before[0] and after[1] == before[1]
    assert after[2] != before[2]
    # Reopen under the NEW key: the rotation bumped key_check, so the token
    # still restores the same value under the new cipher.
    reopened = open_sqlite_vault(db, "sess-a", new)
    assert reopened.original_for(token) == "jane@corp.example"
    reopened.close()


# --- open_sqlite_vault owns (and closes) its connection ----------------------


def test_open_sqlite_vault_owns_and_closes_connection(tmp_path: Path) -> None:
    db = tmp_path / "vault.db"
    vault = open_sqlite_vault(db, "sess-a", FakeVaultCipher(b"k" * 32))
    conn = vault._conn
    vault.close()
    # owns_connection=True ⇒ close() really closed it; a falsy ownership flag
    # would leave the connection open.
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
