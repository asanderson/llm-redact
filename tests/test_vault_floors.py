"""The vault's token floor: ``placeholder_for(type, value, floor=N)``.

A NEW value is numbered above the session's own numbers AND above the
request's floor — max(MAX(n), floor) + 1 — so a token the request carries
but the session never issued (a compacted history, a pasted answer) never
gets a second meaning. The battery runs against EVERY vault: in-memory,
encrypted in-memory, sqlite (plain and encrypted), and the RDBMS store over
stdlib sqlite3 (the generic DB-API backend) and the fake server drivers.
Pinned for each: the floor is respected, a mapped value keeps its token
whatever the floor, floor 0 is the old dense numbering, a gap is only ever
left below a floored number, a rolled-back allocation reissues the SAME
number, and the number space ends at MAX_TOKEN_NUMBER with a refusal that
leaves nothing written and nothing wedged.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fake_cipher import FakeVaultCipher
from llm_redact.config import RdbmsConfig, VaultConfig
from llm_redact.placeholders import MAX_TOKEN_NUMBER
from llm_redact.vault import (
    EncryptedInMemoryVault,
    InMemoryVault,
    PlaceholderSpaceExhausted,
    SqliteVaultManager,
    Vault,
    next_number,
    open_sqlite_vault,
)
from llm_redact.vault_rdbms import RdbmsStore, RdbmsVault
from test_vault_faults import _FlakyConn
from test_vault_rdbms import _fake_backend_config

_BACKENDS = [
    "memory",
    "memory-encrypted",
    "sqlite",
    "sqlite-encrypted",
    "rdbms-dbapi",
    "rdbms-dbapi-encrypted",
    "rdbms-postgresql",
    "rdbms-mysql",
    "rdbms-oracle",
]


def _open(backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Vault:
    cipher = FakeVaultCipher() if backend.endswith("-encrypted") else None
    if backend.startswith("memory"):
        return InMemoryVault() if cipher is None else EncryptedInMemoryVault(cipher, "s")
    if backend.startswith("sqlite"):
        return open_sqlite_vault(tmp_path / "vault.db", "s", cipher)
    if backend.startswith("rdbms-dbapi"):
        config = VaultConfig(
            backend="dbapi", rdbms=RdbmsConfig(dsn=str(tmp_path / "v.db"), module="sqlite3")
        )
    else:
        config, _ = _fake_backend_config(monkeypatch, tmp_path, backend.removeprefix("rdbms-"))
    return RdbmsVault(RdbmsStore(config, cipher), "s", owns_store=True)


@pytest.fixture(params=_BACKENDS)
def vault(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Vault]:
    opened = _open(request.param, tmp_path, monkeypatch)
    yield opened
    opened.close()


def test_a_new_value_is_numbered_above_the_floor(vault: Vault) -> None:
    assert vault.placeholder_for("EMAIL", "bob@corp.example", floor=3) == "«EMAIL_004»"
    assert vault.original_for("«EMAIL_004»") == "bob@corp.example"
    # The skipped numbers stay unissued: a token the request carried keeps
    # resolving to nothing here, so an echo of it passes through verbatim.
    for skipped in ("«EMAIL_001»", "«EMAIL_002»", "«EMAIL_003»"):
        assert vault.original_for(skipped) is None


def test_floor_zero_is_the_dense_numbering(vault: Vault) -> None:
    assert vault.placeholder_for("EMAIL", "a@corp.example", floor=0) == "«EMAIL_001»"
    assert vault.placeholder_for("EMAIL", "b@corp.example") == "«EMAIL_002»"
    assert vault.placeholder_for("EMAIL", "c@corp.example", floor=0) == "«EMAIL_003»"


def test_a_floor_below_the_sessions_own_numbers_changes_nothing(vault: Vault) -> None:
    vault.placeholder_for("EMAIL", "a@corp.example")
    vault.placeholder_for("EMAIL", "b@corp.example")
    # A request whose history carries the session's OWN tokens: MAX(n) wins.
    assert vault.placeholder_for("EMAIL", "c@corp.example", floor=2) == "«EMAIL_003»"
    assert vault.placeholder_for("EMAIL", "d@corp.example", floor=1) == "«EMAIL_004»"


def test_a_mapped_value_keeps_its_token_whatever_the_floor(vault: Vault) -> None:
    token = vault.placeholder_for("EMAIL", "a@corp.example")
    assert vault.placeholder_for("EMAIL", "a@corp.example", floor=40) == token
    # Even at the very end of the number space: nothing new is needed.
    assert vault.placeholder_for("EMAIL", "a@corp.example", floor=MAX_TOKEN_NUMBER) == token


def test_numbering_continues_from_the_highest_issued_after_a_gap(vault: Vault) -> None:
    assert vault.placeholder_for("EMAIL", "a@corp.example", floor=5) == "«EMAIL_006»"
    # A later request with no floor never reaches back into the gap.
    assert vault.placeholder_for("EMAIL", "b@corp.example") == "«EMAIL_007»"


def test_the_floor_is_per_type(vault: Vault) -> None:
    assert vault.placeholder_for("EMAIL", "a@corp.example", floor=8) == "«EMAIL_009»"
    assert vault.placeholder_for("PHONE", "+1 555 0100") == "«PHONE_001»"


def test_the_number_space_ends_with_a_refusal_not_a_reuse(vault: Vault) -> None:
    with pytest.raises(PlaceholderSpaceExhausted) as refused:
        vault.placeholder_for("EMAIL", "bob@corp.example", floor=MAX_TOKEN_NUMBER)
    # The type is named; the value never is.
    assert refused.value.detector_type == "EMAIL"
    assert "EMAIL" in str(refused.value) and "bob" not in str(refused.value)
    assert str(MAX_TOKEN_NUMBER) in str(refused.value)
    # Nothing was written, and the store is not wedged: the next request
    # allocates normally.
    assert len(vault) == 0
    assert vault.placeholder_for("EMAIL", "bob@corp.example") == "«EMAIL_001»"


def test_the_last_number_is_issuable(vault: Vault) -> None:
    last = vault.placeholder_for("EMAIL", "a@corp.example", floor=MAX_TOKEN_NUMBER - 1)
    assert last == f"«EMAIL_{MAX_TOKEN_NUMBER}»"
    assert vault.original_for(last) == "a@corp.example"
    with pytest.raises(PlaceholderSpaceExhausted):
        vault.placeholder_for("EMAIL", "b@corp.example")


def test_next_number() -> None:
    assert next_number(0, 0, "EMAIL") == 1
    assert next_number(4, 0, "EMAIL") == 5
    assert next_number(4, 9, "EMAIL") == 10
    assert next_number(9, 4, "EMAIL") == 10
    assert next_number(MAX_TOKEN_NUMBER - 1, 0, "EMAIL") == MAX_TOKEN_NUMBER
    with pytest.raises(PlaceholderSpaceExhausted, match="PHONE"):
        next_number(MAX_TOKEN_NUMBER, 0, "PHONE")
    with pytest.raises(PlaceholderSpaceExhausted):
        next_number(0, MAX_TOKEN_NUMBER, "PHONE")


# --- write faults: a rolled-back floored allocation reissues the SAME number ---


def test_sqlite_write_fault_with_a_floor_reissues_the_same_number(tmp_path: Path) -> None:
    vault = open_sqlite_vault(tmp_path / "vault.db", "s")
    vault._conn = _FlakyConn(vault._conn, "INSERT INTO mappings", times=1)  # type: ignore[assignment]
    with pytest.raises(sqlite3.OperationalError):
        vault.placeholder_for("EMAIL", "bob@corp.example", floor=3)
    assert vault.original_for("«EMAIL_004»") is None  # nothing cached
    # The retry (same request, same floor) gets exactly the number the
    # failed attempt computed: never a skip on top of the floor's gap.
    assert vault.placeholder_for("EMAIL", "bob@corp.example", floor=3) == "«EMAIL_004»"
    assert vault.placeholder_for("EMAIL", "carol@corp.example") == "«EMAIL_005»"
    vault.close()


def test_sqlite_exhaustion_rolls_back_the_open_transaction(tmp_path: Path) -> None:
    # The refusal is raised INSIDE BEGIN IMMEDIATE: without the rollback the
    # shared connection would stay in a write transaction for every later
    # request (and a second connection would wait on its lock).
    db = tmp_path / "vault.db"
    vault = open_sqlite_vault(db, "s")
    with pytest.raises(PlaceholderSpaceExhausted):
        vault.placeholder_for("EMAIL", "bob@corp.example", floor=MAX_TOKEN_NUMBER)
    assert not vault._conn.in_transaction
    other = sqlite3.connect(db, timeout=0.1)
    other.execute("BEGIN IMMEDIATE")  # the write lock is free
    other.execute("ROLLBACK")
    other.close()
    vault.close()


def test_sqlite_views_of_one_session_never_reuse_a_floored_number(tmp_path: Path) -> None:
    # Two views of one session (a second proxy instance over the same file):
    # MAX(n) is read fresh inside the write lock, so the floor-free view
    # continues ABOVE the other's floored number.
    first_instance = SqliteVaultManager(tmp_path / "vault.db")
    second_instance = SqliteVaultManager(tmp_path / "vault.db")
    first = first_instance.get("s")
    second = second_instance.get("s")  # loaded before the floored value existed
    assert first.placeholder_for("EMAIL", "a@corp.example", floor=6) == "«EMAIL_007»"
    assert second is not first
    assert second.placeholder_for("EMAIL", "b@corp.example") == "«EMAIL_008»"
    first_instance.close()
    second_instance.close()


@pytest.mark.parametrize("backend", ["postgresql", "mysql", "oracle"])
def test_rdbms_write_fault_with_a_floor_reissues_the_same_number(
    backend: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, backend)
    store = RdbmsStore(config, None)
    vault = RdbmsVault(store, "s")
    driver.inject_fault("INSERT INTO llm_redact_mappings", sqlite3.DataError("disk full"))
    with pytest.raises(sqlite3.DataError):
        vault.placeholder_for("EMAIL", "bob@corp.example", floor=3)
    assert vault.original_for("«EMAIL_004»") is None
    assert vault.placeholder_for("EMAIL", "bob@corp.example", floor=3) == "«EMAIL_004»"
    store.close()


def test_rdbms_reconnect_retry_keeps_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    store = RdbmsStore(config, None)
    vault = RdbmsVault(store, "s")
    driver.dead = True  # the idle connection dropped: one reconnect, same op
    assert vault.placeholder_for("EMAIL", "bob@corp.example", floor=4) == "«EMAIL_005»"
    assert driver.connect_count == 2
    store.close()


def test_rdbms_exhaustion_rolls_back_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, driver = _fake_backend_config(monkeypatch, tmp_path, "postgresql")
    store = RdbmsStore(config, None)
    rolled_back: list[bool] = []
    real_rollback = store._rollback

    def spy(conn: Any) -> None:
        rolled_back.append(True)
        real_rollback(conn)

    store._rollback = spy  # type: ignore[method-assign]
    with pytest.raises(PlaceholderSpaceExhausted):
        store.get_or_create("s", "EMAIL", "bob@corp.example", floor=MAX_TOKEN_NUMBER)
    assert rolled_back == [True]
    assert store.total_entries() == 0
    assert store.get_or_create("s", "EMAIL", "bob@corp.example") == "«EMAIL_001»"
    store.close()


def test_encrypted_memory_vault_cipher_fault_leaves_no_orphan_token() -> None:
    # The ciphertext is computed before any state changes: a failing cipher
    # must not leave a forward entry whose token could never be restored,
    # nor consume a number.
    class FailingOnce(FakeVaultCipher):
        failed = False

        def encrypt(self, original: str) -> bytes:
            if not self.failed:
                self.failed = True
                raise RuntimeError("cipher unavailable")
            return super().encrypt(original)

    vault = EncryptedInMemoryVault(FailingOnce(), "s")
    with pytest.raises(RuntimeError):
        vault.placeholder_for("EMAIL", "bob@corp.example", floor=2)
    assert len(vault) == 0
    token = vault.placeholder_for("EMAIL", "bob@corp.example", floor=2)
    assert token == "«EMAIL_003»"
    assert vault.original_for(token) == "bob@corp.example"


def test_sqlite_cipher_fault_rolls_back(tmp_path: Path) -> None:
    # Any failure inside the write transaction — not only a sqlite error —
    # rolls back, so the connection is never left wedged.
    class FailingOnce(FakeVaultCipher):
        failed = False

        def encrypt(self, original: str) -> bytes:
            if not self.failed:
                self.failed = True
                raise RuntimeError("cipher unavailable")
            return super().encrypt(original)

    vault = open_sqlite_vault(tmp_path / "vault.db", "s", FailingOnce())
    with pytest.raises(RuntimeError):
        vault.placeholder_for("EMAIL", "bob@corp.example", floor=2)
    assert not vault._conn.in_transaction
    assert vault.placeholder_for("EMAIL", "bob@corp.example", floor=2) == "«EMAIL_003»"
    vault.close()


class _LegacyVault:
    """A third-party Vault predating the ``floor`` keyword."""

    def __init__(self) -> None:
        self.inner = InMemoryVault()

    def placeholder_for(self, detector_type: str, original: str) -> str:
        return self.inner.placeholder_for(detector_type, original)

    def original_for(self, placeholder: str) -> str | None:
        return self.inner.original_for(placeholder)

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.inner)


def test_a_vault_predating_the_keyword_keeps_serving_token_free_requests() -> None:
    from llm_redact.detection.engine import DetectionConfig, build_allowlist, build_detectors
    from llm_redact.redactor import Redactor

    config = DetectionConfig()
    legacy: Any = _LegacyVault()
    redactor = Redactor(build_detectors(config), legacy, build_allowlist(config))
    # No floor: the keyword is never passed.
    assert redactor.redact_text("mail a@corp.example") == "mail «EMAIL_001»"
    # A floor it cannot honor fails closed instead of numbering onto the
    # request's own token.
    with pytest.raises(TypeError):
        redactor.with_floors({"EMAIL": 4}).redact_text("mail b@corp.example")
