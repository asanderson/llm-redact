import os
import sqlite3
import stat
from pathlib import Path

import pytest

from fake_cipher import FakeVaultCipher
from llm_redact.config import VaultConfig
from llm_redact.vault import (
    InMemoryVault,
    SqliteVault,
    SqliteVaultManager,
    Vault,
    build_vault,
    open_sqlite_vault,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "vault.db"


def _open(db_path: Path, session: str = "default") -> SqliteVault:
    return open_sqlite_vault(db_path, session)


# The InMemoryVault behavioral contract must hold identically for SQLite.
@pytest.fixture(params=["memory", "sqlite"])
def any_vault(request: pytest.FixtureRequest, db_path: Path) -> Vault:
    if request.param == "memory":
        return InMemoryVault()
    return _open(db_path)


def test_contract_deterministic(any_vault: Vault) -> None:
    a = any_vault.placeholder_for("EMAIL", "jane@example.com")
    b = any_vault.placeholder_for("EMAIL", "jane@example.com")
    assert a == b == "«EMAIL_001»"


def test_contract_distinct_values(any_vault: Vault) -> None:
    any_vault.placeholder_for("EMAIL", "jane@example.com")
    assert any_vault.placeholder_for("EMAIL", "john@example.com") == "«EMAIL_002»"


def test_contract_type_namespaces(any_vault: Vault) -> None:
    assert any_vault.placeholder_for("EMAIL", "x") != any_vault.placeholder_for("SECRET", "x")


def test_contract_reverse_lookup(any_vault: Vault) -> None:
    token = any_vault.placeholder_for("EMAIL", "jane@example.com")
    assert any_vault.original_for(token) == "jane@example.com"
    assert any_vault.original_for("«EMAIL_999»") is None


def test_restart_determinism(db_path: Path) -> None:
    vault = _open(db_path)
    token = vault.placeholder_for("EMAIL", "jane@example.com")
    vault.close()

    reopened = _open(db_path)
    assert reopened.placeholder_for("EMAIL", "jane@example.com") == token
    assert reopened.original_for(token) == "jane@example.com"
    reopened.close()


def test_counters_continue_after_restart(db_path: Path) -> None:
    vault = _open(db_path)
    vault.placeholder_for("EMAIL", "a@example.com")  # 001
    vault.placeholder_for("EMAIL", "b@example.com")  # 002
    vault.close()

    reopened = _open(db_path)
    # New value must get 003, never reuse an existing n.
    assert reopened.placeholder_for("EMAIL", "c@example.com") == "«EMAIL_003»"
    reopened.close()


def test_session_isolation(db_path: Path) -> None:
    a = _open(db_path, "session-a")
    token_a = a.placeholder_for("EMAIL", "jane@example.com")
    a.close()

    b = _open(db_path, "session-b")
    # session-b has its own counter/token space: «EMAIL_001» is reissued
    # there for a different original, and resolves to session-b's value.
    token_b = b.placeholder_for("EMAIL", "other@example.com")
    assert token_a == token_b == "«EMAIL_001»"
    assert b.original_for(token_b) == "other@example.com"
    b.close()


def test_file_permissions(db_path: Path) -> None:
    vault = _open(db_path)
    vault.close()
    if os.name == "posix":
        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(db_path.parent.stat().st_mode) == 0o700


def test_wal_mode(db_path: Path) -> None:
    vault = _open(db_path)
    mode = vault._conn.execute("PRAGMA journal_mode").fetchone()[0]
    sync = vault._conn.execute("PRAGMA synchronous").fetchone()[0]
    assert mode == "wal"
    assert sync == 2  # FULL
    vault.close()


def test_concurrent_insert_race(db_path: Path) -> None:
    # Two connections to the same DB/session: the loser of the unique-key
    # race must recover the winner's placeholder.
    first = _open(db_path)
    second = _open(db_path)
    token_first = first.placeholder_for("EMAIL", "jane@example.com")
    # `second` has a stale cache (opened before the insert), forcing the
    # INSERT → IntegrityError → re-SELECT path.
    token_second = second.placeholder_for("EMAIL", "jane@example.com")
    assert token_first == token_second
    first.close()
    second.close()


def test_reverse_lookup_from_other_process(db_path: Path) -> None:
    first = _open(db_path)
    second = _open(db_path)
    token = first.placeholder_for("EMAIL", "jane@example.com")
    # Not in second's preloaded cache; must fall back to the DB.
    assert second.original_for(token) == "jane@example.com"
    first.close()
    second.close()


def test_build_vault_dispatch(db_path: Path) -> None:
    assert isinstance(build_vault(VaultConfig()), InMemoryVault)
    vault = build_vault(VaultConfig(backend="sqlite", path=str(db_path)))
    assert isinstance(vault, SqliteVault)
    vault.close()


def test_raw_db_has_original_only_with_optin(db_path: Path) -> None:
    vault = _open(db_path)
    vault.placeholder_for("EMAIL", "jane@example.com")
    vault.close()
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT session_id, detector_type, original, n FROM mappings").fetchall()
    conn.close()
    assert rows == [("default", "EMAIL", "jane@example.com", 1)]


# ---- managers ----


def test_manager_views_share_one_db(db_path: Path) -> None:
    manager = SqliteVaultManager(db_path)
    a = manager.get("conv-a")
    b = manager.get("conv-b")
    token_a = a.placeholder_for("EMAIL", "jane@example.com")
    token_b = b.placeholder_for("EMAIL", "john@example.com")
    assert token_a == token_b == "«EMAIL_001»"  # independent counters
    assert a.original_for("«EMAIL_001»") == "jane@example.com"
    assert b.original_for("«EMAIL_001»") == "john@example.com"
    assert manager.session_count() == 2
    assert manager.total_entries() == 2
    manager.close()


def test_manager_view_cache_eviction_preserves_data(db_path: Path) -> None:
    manager = SqliteVaultManager(db_path, view_cache_size=2)
    token = manager.get("conv-1").placeholder_for("EMAIL", "jane@example.com")
    manager.get("conv-2")
    manager.get("conv-3")  # evicts conv-1's view (LRU)
    # A fresh view reloads the session's rows from the DB.
    revived = manager.get("conv-1")
    assert revived.original_for(token) == "jane@example.com"
    assert revived.placeholder_for("EMAIL", "jane@example.com") == token
    manager.close()


def test_manager_response_sessions_persist(db_path: Path) -> None:
    manager = SqliteVaultManager(db_path)
    manager.record_response_session("resp_1", "conv-abc")
    assert manager.lookup_response_session("resp_1") == "conv-abc"
    assert manager.lookup_response_session("resp_missing") is None
    manager.close()

    reopened = SqliteVaultManager(db_path)
    assert reopened.lookup_response_session("resp_1") == "conv-abc"
    reopened.close()


def test_manager_same_session_returns_same_view(db_path: Path) -> None:
    manager = SqliteVaultManager(db_path)
    assert manager.get("conv-x") is manager.get("conv-x")
    manager.close()


# ---- encryption at rest: Free vault-class plumbing over the VaultCipher
# Protocol (FakeVaultCipher — no cryptography / llm-redact-pro dependency).
# The REAL-crypto proofs (no plaintext on disk, build_cipher/from_env key
# resolution) live in tests/pro/test_vault_at_rest.py. ----


def _cipher(seed: bytes = b"k" * 32):
    return FakeVaultCipher(seed)


def test_migration_preserves_mappings_and_counters(tmp_path):
    path = tmp_path / "vault.db"
    plain = open_sqlite_vault(path, "s1")
    token_1 = plain.placeholder_for("EMAIL", "jane@corp.example")
    token_2 = plain.placeholder_for("EMAIL", "bob@corp.example")
    plain.close()

    cipher = _cipher()
    encrypted = open_sqlite_vault(path, "s1", cipher)
    # Both directions survive with identical placeholders...
    assert encrypted.placeholder_for("EMAIL", "jane@corp.example") == token_1
    assert encrypted.original_for(token_2) == "bob@corp.example"
    # ...and the counter continues rather than reissuing numbers.
    assert encrypted.placeholder_for("EMAIL", "new@corp.example") == "«EMAIL_003»"
    encrypted.close()

    # Idempotent reopen with the same key.
    again = open_sqlite_vault(path, "s1", cipher)
    assert again.original_for(token_1) == "jane@corp.example"
    again.close()


def test_encrypted_vault_requires_key_on_reopen(tmp_path):
    path = tmp_path / "vault.db"
    vault = open_sqlite_vault(path, "s1", _cipher())
    vault.placeholder_for("EMAIL", "jane@corp.example")
    vault.close()

    from llm_redact.config import ConfigError

    with pytest.raises(ConfigError, match="encrypted"):
        open_sqlite_vault(path, "s1")  # no cipher against a v3 vault


def test_encrypted_vault_wrong_key_fails_at_open(tmp_path):
    from llm_redact.vault import VaultKeyError

    path = tmp_path / "vault.db"
    vault = open_sqlite_vault(path, "s1", _cipher(b"k" * 32))
    vault.placeholder_for("EMAIL", "jane@corp.example")
    vault.close()

    with pytest.raises(VaultKeyError, match="does not match"):
        open_sqlite_vault(path, "s1", _cipher(b"x" * 32))


def test_encrypted_two_connection_race(tmp_path):
    path = tmp_path / "vault.db"
    cipher = _cipher()
    a = open_sqlite_vault(path, "s1", cipher)
    b = open_sqlite_vault(path, "s1", cipher)
    token = a.placeholder_for("EMAIL", "jane@corp.example")
    # b's cache doesn't have it; the IntegrityError fallback resolves by MAC.
    assert b.placeholder_for("EMAIL", "jane@corp.example") == token
    a.close()
    b.close()


def test_manager_hands_out_encrypted_view(tmp_path):
    # SqliteVaultManager threads the cipher into each view (kills the cipher
    # argument being dropped in the manager -> SqliteVault construction).
    path = tmp_path / "v.db"
    manager = SqliteVaultManager(path, cipher=_cipher())
    vault = manager.get("conv-1")
    token = vault.placeholder_for("EMAIL", "jane@corp.example")
    assert manager.get("conv-1").original_for(token) == "jane@corp.example"
    manager.close()
