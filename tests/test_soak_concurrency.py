"""Concurrency & soak checks (deselected by default; run with -m soak).

The per-conversation cross-session isolation soak tests (many concurrent
conversations sharing token NAMES but never each other's values) need the Pro
session router, so they moved to the llm-redact-pro repo in the R4 open-core
split. What stays here is tier-independent: the sqlite VaultManager driven past
its view-cache so LRU eviction is exercised without losing a mapping.
"""

from pathlib import Path

import pytest

from llm_redact.vault import SqliteVaultManager

pytestmark = pytest.mark.soak


def test_lru_eviction_preserves_all_mappings(tmp_path: Path) -> None:
    manager = SqliteVaultManager(tmp_path / "vault.db", view_cache_size=4)
    n = 50
    for i in range(n):
        token = manager.get(f"conv-{i}").placeholder_for("EMAIL", f"user{i}@corp.example")
        assert token == "«EMAIL_001»"

    # Far more sessions than the cache holds: the view cache stayed bounded...
    assert len(manager._views) <= 4
    # ...but every mapping persisted, and each (mostly evicted) session still
    # rehydrates its OWN value — eviction dropped caches, never data.
    assert manager.session_count() == n
    for i in range(n):
        assert manager.get(f"conv-{i}").original_for("«EMAIL_001»") == f"user{i}@corp.example"
    manager.close()
