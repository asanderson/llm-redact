"""Assertions that kill rehydrate/vault mutation survivors (Phase 20).

Each test pins a behavior whose mutation survived the existing suites —
mostly boundaries the sweeps never hit (escape holdback edges, constructor
defaults, the write-through cache contract, LRU eviction edges, admin-path
error messages). Grouped by module; the mutant class each test kills is
named in its comment. Provably-equivalent survivors are recorded in
scripts/mutation_equivalents.py instead.
"""

import re
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

import llm_redact.vault as vault_mod
from fake_cipher import FakeVaultCipher
from llm_redact.config import ConfigError, parse_config
from llm_redact.rehydrate import (
    Rehydrator,
    RehydratorPool,
    StreamingRehydrator,
    _escape_body_is_guillemet_prefix,
    escape_prefix_start,
)
from llm_redact.vault import (
    EncryptedInMemoryVault,
    InMemoryVault,
    InMemoryVaultManager,
    SqliteVaultManager,
    VaultKeyError,
    build_cipher,
    build_vault,
    default_vault_path,
    open_sqlite_vault,
)

# ---------------------------------------------------------------------------
# rehydrate.py — escape holdback boundaries
# ---------------------------------------------------------------------------


def test_escape_body_prefix_boundaries() -> None:
    # Non-'u' first char is rejected via the OR's second arm (kills or->and
    # and the early-return flip); a COMPLETE 5-char body is past the bound
    # (kills > 4 -> > 5); wrong hex positions reject (kills their flips).
    assert not _escape_body_is_guillemet_prefix("x")
    assert not _escape_body_is_guillemet_prefix("u00ab")  # complete: not a prefix
    assert not _escape_body_is_guillemet_prefix("u1")
    assert not _escape_body_is_guillemet_prefix("u00c")
    assert _escape_body_is_guillemet_prefix("u00a")  # genuine partial


def test_escape_prefix_start_boundaries() -> None:
    # A trailing backslash at index 1 is held (kills i == -1 -> i == +1).
    assert escape_prefix_start("a\\") == 1
    # A backslash followed by a non-prefix body is NOT held (kills the
    # body-check call being bypassed).
    assert escape_prefix_start("a\\x") is None
    # No backslash at all: never held, even when the text itself looks like
    # an escape body (kills i == -1 -> i == -2, which would treat the whole
    # text as an escape tail).
    assert escape_prefix_start("u00a") is None
    # Four backslashes are two complete literal pairs — nothing partial
    # (kills the parity modulus % 2 -> % 3).
    assert escape_prefix_start("a" + "\\" * 4) is None
    # Three backslashes: one pair plus a genuinely partial trailing one.
    assert escape_prefix_start("a" + "\\" * 3) == 3


# ---------------------------------------------------------------------------
# rehydrate.py — constructor defaults, counts threading, flush/reuse state
# ---------------------------------------------------------------------------


@pytest.fixture
def stocked_vault() -> InMemoryVault:
    vault = InMemoryVault()
    vault.placeholder_for("EMAIL", 'jane"quoted"@x.example')
    vault.placeholder_for("PHONE", "+1 415 555 0100")
    return vault


def test_rehydrator_default_is_not_fuzzy(stocked_vault: InMemoryVault) -> None:
    # The fuzzy default is False everywhere: a mangled token passes through
    # verbatim unless fuzzy is explicitly enabled (kills default flips on
    # Rehydrator/StreamingRehydrator/RehydratorPool).
    assert Rehydrator(stocked_vault).rehydrate_text("«email_001»") == "«email_001»"
    channel = StreamingRehydrator(stocked_vault)
    assert channel.feed("«email_001»") + channel.flush() == "«email_001»"
    pool = RehydratorPool(stocked_vault)
    assert pool.get("k").feed("«email_001»") + pool.flush("k") == "«email_001»"


def test_streaming_channel_json_source_dispatch(stocked_vault: InMemoryVault) -> None:
    # Default channels are NOT json_source: a restored value keeps its raw
    # quote. json_source=True must re-escape it. Kills the default flip AND
    # the json_source argument being dropped/None'd in streaming_channel,
    # pool.get, and rehydrate_whole.
    r = Rehydrator(stocked_vault)
    raw = r.streaming_channel()
    out = raw.feed("«EMAIL_001»") + raw.flush()
    assert '"quoted"' in out
    escaped = r.streaming_channel(json_source=True)
    out = escaped.feed("«EMAIL_001»") + escaped.flush()
    assert '\\"quoted\\"' in out
    assert '"quoted"' in RehydratorPool(stocked_vault).rehydrate_whole("«EMAIL_001»")
    pool = RehydratorPool(stocked_vault)
    assert '"quoted"' in pool.get("k").feed("«EMAIL_001»") + pool.flush("k")


def test_streaming_channel_inherits_fuzzy(stocked_vault: InMemoryVault) -> None:
    # A channel from a fuzzy Rehydrator restores mangled tokens (kills the
    # fuzzy argument being dropped/None'd in streaming_channel).
    channel = Rehydrator(stocked_vault, fuzzy=True).streaming_channel()
    assert "jane" in channel.feed("«email_001»") + channel.flush()


def test_json_source_text_threads_counts(stocked_vault: InMemoryVault) -> None:
    # rehydrate_json_source_text reports restores into the shared counter —
    # the audit/status metadata path (kills counts=None / dropped).
    counts: Counter[str] = Counter()
    r = Rehydrator(stocked_vault, counts=counts)
    r.rehydrate_json_source_text('{"a": "«PHONE_001»"}')
    assert counts == {"PHONE": 1}


def test_flush_emits_buffer_plus_unresolved_escape_tail(
    stocked_vault: InMemoryVault,
) -> None:
    # At flush, a held partial token AND a held partial escape are both
    # emitted — the tail is APPENDED to the buffer, never replaces it (kills
    # += -> = / -= and the normalize argument being None'd).
    channel = StreamingRehydrator(stocked_vault, json_source=True)
    assert channel.feed("«EMAIL_\\u00") == ""  # both parts held back
    out = channel.flush()
    assert "«EMAIL_" in out and "\\u00" in out


def test_streaming_rehydrator_reusable_after_flush(
    stocked_vault: InMemoryVault,
) -> None:
    # flush() resets to a clean, reusable state EVEN when it had to resolve a
    # held escape tail: the next feed/flush cycle starts empty — no None, no
    # junk, no leftover tail prepended (kills the buffer/escape_tail reset
    # mutants, which only fire on the tail-resolving branch).
    channel = StreamingRehydrator(stocked_vault, json_source=True)
    assert channel.feed("x\\u00") == "x"  # the partial escape tail is held
    assert channel.flush() == "\\u00"  # ...and resolved as plain text
    assert channel.feed("«PHONE_001»") + channel.flush() == "+1 415 555 0100"
    assert channel.flush() == ""


# ---------------------------------------------------------------------------
# vault.py — the write-through cache contract (the streaming hot path)
# ---------------------------------------------------------------------------


def test_cache_serves_hot_path_without_touching_db(tmp_path: Path) -> None:
    # The write-through cache is a documented contract, not a nicety: repeat
    # lookups must come off the in-process cache. Killing the DB connection
    # proves it — a mutant that skips the cache write (or reads it wrongly)
    # hits the closed connection and errors. Kills the forward/reverse cache
    # mutations in placeholder_for/original_for, which are otherwise masked
    # by the DB fallback.
    vault = open_sqlite_vault(tmp_path / "v.db", "s")
    token = vault.placeholder_for("EMAIL", "a@b.example")
    vault._conn.close()
    assert vault.placeholder_for("EMAIL", "a@b.example") == token
    assert vault.original_for(token) == "a@b.example"


def test_preload_populates_cache_for_both_directions(tmp_path: Path) -> None:
    # Same contract across a restart: the per-session preload fills BOTH
    # cache directions (kills the preload-loop mutations in __init__).
    path = tmp_path / "v.db"
    seed = open_sqlite_vault(path, "s")
    token = seed.placeholder_for("EMAIL", "a@b.example")
    seed.close()
    vault = open_sqlite_vault(path, "s")
    vault._conn.close()
    assert vault.placeholder_for("EMAIL", "a@b.example") == token
    assert vault.original_for(token) == "a@b.example"


def test_db_fallback_lookup_backfills_the_cache(tmp_path: Path) -> None:
    # A cache-miss lookup resolved from the database (another process issued
    # the token) must backfill the reverse cache so repeats stay off the DB
    # (kills the fallback cache-write mutation).
    path = tmp_path / "v.db"
    issuer = open_sqlite_vault(path, "s")
    reader = open_sqlite_vault(path, "s")  # preloaded while empty
    token = issuer.placeholder_for("EMAIL", "a@b.example")
    assert reader.original_for(token) == "a@b.example"  # miss -> DB -> backfill
    reader._conn.close()
    assert reader.original_for(token) == "a@b.example"  # now a pure cache hit
    issuer.close()


def test_encrypted_preload_populates_cache(tmp_path: Path) -> None:
    # Encrypted preload decrypts once per row into the same plaintext-keyed
    # caches (kills the decrypt/None mutations in the encrypted preload arm).
    # FakeVaultCipher stands in for the paid cipher — the Free SqliteVault
    # only depends on the VaultCipher Protocol.
    path = tmp_path / "v.db"
    cipher = FakeVaultCipher(b"k" * 32)
    seed = open_sqlite_vault(path, "s", cipher)
    token = seed.placeholder_for("EMAIL", "enc@x.example")
    seed.close()
    vault = open_sqlite_vault(path, "s", FakeVaultCipher(b"k" * 32))
    vault._conn.close()
    assert vault.placeholder_for("EMAIL", "enc@x.example") == token
    assert vault.original_for(token) == "enc@x.example"


def test_encrypted_memory_counters_stay_dense() -> None:
    # Two distinct values of one type get consecutive numbers — the counter
    # is per-type state, not a constant (kills the counter get/set mutants;
    # the prior suite only ever issued one value per type).
    vault = EncryptedInMemoryVault(FakeVaultCipher(b"k" * 32), "s")
    assert vault.placeholder_for("EMAIL", "one@x.example") == "«EMAIL_001»"
    assert vault.placeholder_for("EMAIL", "two@x.example") == "«EMAIL_002»"
    # A reverse miss returns None, and __len__ counts stored mappings.
    assert vault.original_for("«EMAIL_404»") is None
    assert len(vault) == 2


# ---------------------------------------------------------------------------
# vault.py — open/verify: wrong key on an EMPTY vault, messages, deep paths
# ---------------------------------------------------------------------------


def test_wrong_key_fails_at_open_even_on_empty_vault(tmp_path: Path) -> None:
    # An EMPTY encrypted vault has no ciphertext whose decryption could fail
    # later, so the key_check comparison is the ONLY guard — a mutant that
    # skips it (or re-stamps key_check with the wrong key) silently accepts
    # and re-keys the vault. The error names the vault path.
    path = tmp_path / "v.db"
    open_sqlite_vault(path, "s", FakeVaultCipher(b"k" * 32)).close()
    with pytest.raises(VaultKeyError, match=re.escape(str(path))):
        open_sqlite_vault(path, "s", FakeVaultCipher(b"x" * 32))
    # And the check must not have been weakened by the failed attempt: the
    # right key still opens.
    open_sqlite_vault(path, "s", FakeVaultCipher(b"k" * 32)).close()


def test_encrypted_vault_without_key_names_the_remedy(tmp_path: Path) -> None:
    # The fail-closed ConfigError spells out the exact remedy; the message is
    # the UX (kills the message-text mutants).
    path = tmp_path / "v.db"
    open_sqlite_vault(path, "s", FakeVaultCipher(b"k" * 32)).close()
    with pytest.raises(
        ConfigError,
        match=re.escape('"fernet" and LLM_REDACT_VAULT_KEY (the migration is one-way)'),
    ):
        open_sqlite_vault(path, "s", None)


def test_open_creates_nested_parent_directories(tmp_path: Path) -> None:
    # parents=True on the 0700 mkdir (kills parents dropped/False/None).
    path = tmp_path / "a" / "b" / "c" / "vault.db"
    open_sqlite_vault(path, "s").close()
    assert path.exists()


def test_default_vault_path_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    # The documented data location, both with and without XDG_DATA_HOME
    # (kills the env-var-name and path-segment mutants).
    monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")
    assert default_vault_path() == Path("/custom/data/llm-redact/vault.db")
    monkeypatch.delenv("XDG_DATA_HOME")
    # as_posix(): the layout is what matters, not the platform separator.
    assert default_vault_path().as_posix().endswith(".local/share/llm-redact/vault.db")


# ---------------------------------------------------------------------------
# vault.py — manager: summary contract, LRU edges, prune, response map
# ---------------------------------------------------------------------------


def test_sessions_summary_row_contract(tmp_path: Path) -> None:
    # The /sessions endpoint's row shape, pinned for BOTH backends: exactly
    # these keys, correct entry counts, sqlite timestamps present (kills the
    # dict-key mutants).
    manager = SqliteVaultManager(tmp_path / "m.db")
    manager.get("s1").placeholder_for("EMAIL", "a@b.example")
    (row,) = manager.sessions_summary()
    assert set(row) == {"session", "entries", "first", "last"}
    assert row["session"] == "s1" and row["entries"] == 1
    assert row["first"] is not None and row["last"] is not None
    manager.close()

    memory = InMemoryVaultManager()
    memory.get("s1").placeholder_for("EMAIL", "a@b.example")
    (row,) = memory.sessions_summary()
    assert set(row) == {"session", "entries", "first", "last"}
    assert row["session"] == "s1" and row["entries"] == 1
    assert row["first"] is None and row["last"] is None


def test_view_cache_boundary_and_lru_direction(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "m.db", view_cache_size=2)
    view_a = manager.get("a")
    manager.get("b")
    # Exactly at capacity: nothing evicted, views are cached instances
    # (kills > -> >= on the eviction bound).
    assert len(manager._views) == 2
    assert manager.get("a") is view_a  # refreshes a's recency
    view_c = manager.get("c")  # evicts the LRU entry, which is now b
    # The newest view must never be the eviction victim (kills popitem
    # last=False -> last=True) and a stays cached after its refresh.
    assert manager.get("c") is view_c
    assert manager.get("a") is view_a
    # Manager views never own the shared connection: a view-close path would
    # otherwise kill every other session's vault (white-box, load-bearing).
    assert manager.get("a")._owns_connection is False
    manager.close()


def test_default_view_cache_size_is_64(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "m.db")
    for i in range(70):
        manager.get(f"s{i}")
    assert len(manager._views) == 64
    manager.close()


def test_prune_multiple_sessions_and_view_eviction(tmp_path: Path) -> None:
    # Two idle sessions pruned in ONE call (the IN (?,?) placeholder list —
    # kills the join-separator mutant) and their cached views dropped so the
    # proxy cannot keep serving pruned mappings from cache (kills
    # views.pop(None)).
    manager = SqliteVaultManager(tmp_path / "m.db")
    for name in ("old1", "old2", "fresh"):
        manager.get(name).placeholder_for("EMAIL", f"{name}@x.example")
    manager._conn.execute(
        "UPDATE mappings SET created_at = '2020-01-01T00:00:00Z'"
        " WHERE session_id IN ('old1', 'old2')"
    )
    stale_view = manager.get("old1")
    assert manager.prune_sessions(days=30) == 2
    assert manager.session_count() == 1
    assert manager.get("old1") is not stale_view
    manager.close()


def test_prune_failure_rolls_back_and_propagates(tmp_path: Path) -> None:
    # A mid-transaction failure (second DELETE hits a missing table) must
    # roll back the first DELETE and propagate the ORIGINAL error — never a
    # TypeError from a broken rollback call (kills the ROLLBACK-arg mutants).
    manager = SqliteVaultManager(tmp_path / "m.db")
    manager.get("idle").placeholder_for("EMAIL", "i@x.example")
    manager._conn.execute("UPDATE mappings SET created_at = '2020-01-01T00:00:00Z'")
    manager._conn.execute("DROP TABLE response_sessions")
    with pytest.raises(sqlite3.OperationalError, match="response_sessions"):
        manager.prune_sessions(days=30)
    assert manager.total_entries() == 1  # the rollback restored the mappings
    manager.close()


def test_response_session_map_stays_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The response_id -> session map is a bounded buffer: after many inserts
    # the row count must sit at the cap, and lookups still resolve. Tiny
    # constants make the cap reachable in-test (kills the never-prune and
    # prune-SQL mutants; a shifted-by-one prune cadence overshoots the cap
    # by the end of the loop and dies here too).
    monkeypatch.setattr(vault_mod, "_RESPONSE_PRUNE_EVERY", 4)
    monkeypatch.setattr(vault_mod, "_MAX_RESPONSE_ROWS", 3)
    manager = SqliteVaultManager(tmp_path / "m.db")
    for i in range(12):
        manager.record_response_session(f"resp_{i:03d}", "s")
    count = manager._conn.execute("SELECT COUNT(*) FROM response_sessions").fetchone()[0]
    assert count <= 3
    # Whichever rows survive still resolve (created_at ties make the exact
    # keep-set order-dependent; the bound is the contract, not the picks).
    survivors = manager._conn.execute("SELECT response_id FROM response_sessions").fetchall()
    assert all(manager.lookup_response_session(r[0]) == "s" for r in survivors)
    manager.close()


# ---------------------------------------------------------------------------
# vault.py — build wiring
# ---------------------------------------------------------------------------


def test_build_vault_uses_configured_session(tmp_path: Path) -> None:
    # The configured session name reaches the database rows (kills the
    # session argument being dropped/None'd in build_vault).
    path = tmp_path / "bv.db"
    config = parse_config(
        {"vault": {"backend": "sqlite", "path": str(path), "session": "mysess"}}, "t"
    )
    vault = build_vault(config.vault)
    vault.placeholder_for("EMAIL", "bv@x.example")
    vault.close()
    raw = sqlite3.connect(path)
    assert raw.execute("SELECT DISTINCT session_id FROM mappings").fetchall() == [("mysess",)]
    raw.close()


def test_build_vault_fernet_requires_pro(tmp_path: Path) -> None:
    # The concrete cipher is the paid subsystem: build_vault's sqlite+fernet
    # path fails closed through the Free build_cipher (kills the fernet-branch
    # and message mutants), never a silent unencrypted vault.
    config = parse_config(
        {"vault": {"backend": "sqlite", "path": str(tmp_path / "bv.db"), "encryption": "fernet"}},
        "t",
    )
    with pytest.raises(ConfigError, match=re.escape("llm-redact-pro")):
        build_vault(config.vault)


def test_build_vault_rdbms_requires_pro() -> None:
    # A server RDBMS vault is a paid backend; the Free dispatcher fails closed
    # (kills the RDBMS_BACKENDS dispatch branch surviving unpinned). The error
    # names the configured backend, so it must be the real backend, not a
    # dropped/None'd argument (kills the config.backend -> None mutant).
    config = parse_config(
        {"vault": {"backend": "dbapi", "rdbms": {"dsn": "x", "module": "sqlite3"}}}, "t"
    )
    with pytest.raises(ConfigError) as exc:
        build_vault(config.vault)
    assert "llm-redact-pro" in str(exc.value) and "dbapi" in str(exc.value)


def test_build_cipher_off_returns_none_fernet_requires_pro() -> None:
    # Plaintext -> None (no cipher); fernet -> a fail-closed ConfigError that
    # names the paid package — the message is the UX (kills its text mutants).
    assert build_cipher(parse_config({"vault": {}}, "t").vault) is None
    with pytest.raises(ConfigError, match=re.escape("llm-redact-pro")):
        build_cipher(parse_config({"vault": {"encryption": "fernet"}}, "t").vault)


def test_build_vault_manager_plaintext_wiring(tmp_path: Path) -> None:
    # The Free build_vault_manager threads its own arguments: memory builds an
    # InMemoryVaultManager, sqlite builds a real SqliteVaultManager at the
    # configured path (kills the path/cipher wiring mutants — path->None,
    # dropped kwargs, build_cipher(None) — which only fire once the manager is
    # actually exercised end to end).
    from llm_redact.vault import (
        InMemoryVaultManager,
        SqliteVaultManager,
        build_vault_manager,
    )

    memory = build_vault_manager(parse_config({"vault": {}}, "t").vault)
    assert isinstance(memory, InMemoryVaultManager)
    assert memory.get("s").placeholder_for("EMAIL", "a@x.example") == "«EMAIL_001»"

    path = tmp_path / "bvm.db"
    manager = build_vault_manager(
        parse_config({"vault": {"backend": "sqlite", "path": str(path)}}, "t").vault
    )
    assert isinstance(manager, SqliteVaultManager)
    token = manager.get("conv-1").placeholder_for("EMAIL", "b@x.example")
    assert manager.get("conv-1").original_for(token) == "b@x.example"
    assert path.exists()  # the configured path was used, not None
    manager.close()


def test_build_vault_manager_fernet_requires_pro(tmp_path: Path) -> None:
    # Both encrypted paths fail closed through the Free build_cipher (kills the
    # cipher=build_cipher(config) -> None / dropped-kwarg mutants, which would
    # otherwise silently return an unencrypted manager for a fernet config).
    from llm_redact.vault import build_vault_manager

    for cfg in (
        {"vault": {"backend": "memory", "encryption": "fernet"}},
        {"vault": {"backend": "sqlite", "path": str(tmp_path / "e.db"), "encryption": "fernet"}},
    ):
        with pytest.raises(ConfigError, match=re.escape("llm-redact-pro")):
            build_vault_manager(parse_config(cfg, "t").vault)
