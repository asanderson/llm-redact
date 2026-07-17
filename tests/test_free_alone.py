"""Free-alone: the proxy with the llm_redact_pro package genuinely absent.

The open-core split (llm-redact-pro docs/licensing.md) keeps the paid subsystems in
llm_redact_pro, discovered via the ``llm_redact.plugins`` entry point. This
module proves the end-to-end Free-tier reality that ``tests/test_registry.py``
can only *simulate* with a bare ``Registry()``: when the pro package is NOT
installed, ``get_registry()`` discovers no plugin, the Free defaults stand, the
proxy boots keyless on the memory and unencrypted-sqlite vaults, and every paid
config fails closed with a "requires the llm-redact-pro package" ``ConfigError``
— never a silent downgrade.

It runs only in the R2 "Free-alone" CI job, which removes src/llm_redact_pro
from the tree and sets ``LLM_REDACT_TEST_FREE_ALONE=1`` (so conftest does not
force a licensed tier). With the pro package importable it skips entirely, so
the normal suite is unaffected.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_redact.config import AuditConfig, Config, ConfigError, OtelConfig, UsersConfig, VaultConfig
from llm_redact.proxy import create_app
from llm_redact.registry import get_registry, loaded_plugins
from llm_redact.vault import InMemoryVaultManager, SqliteVaultManager

# The pro package is co-located in this repo; the Free-alone CI job hides it
# (moves src/llm_redact_pro out of the tree) so these assertions run only there.
# In the normal suite it is importable, so the whole module skips.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("llm_redact_pro") is not None,
    reason="Free-alone assertions require the llm_redact_pro package to be absent",
)


@pytest.fixture(autouse=True)
def _fresh_registry() -> Iterator[None]:
    # get_registry() caches process-wide; reset so discovery re-runs with the
    # pro package genuinely absent for every assertion here.
    import llm_redact.registry as registry_mod

    registry_mod._registry = None
    registry_mod._loaded_plugins[:] = []
    yield
    registry_mod._registry = None
    registry_mod._loaded_plugins[:] = []


def test_no_pro_plugin_is_discovered() -> None:
    # The real entry-point scan finds nothing to register — this is what
    # test_registry.py's fake/bare Registry cannot exercise.
    get_registry()
    assert loaded_plugins() == []


def test_free_vaults_build_keyless(tmp_path: Path) -> None:
    reg = get_registry()
    assert isinstance(reg.build_vault_manager(VaultConfig(backend="memory")), InMemoryVaultManager)
    sqlite = reg.build_vault_manager(VaultConfig(backend="sqlite", path=str(tmp_path / "vault.db")))
    assert isinstance(sqlite, SqliteVaultManager)


def test_static_session_router_builds() -> None:
    router = get_registry().build_session_router(VaultConfig(session_mode="static", session="s"))
    assert router.mode == "static"
    assert router.resolve("anthropic", "POST", "/v1/messages", {}) == "s"


def test_encrypted_vault_fails_closed() -> None:
    reg = get_registry()
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_cipher(VaultConfig(encryption="fernet"))
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_vault_manager(VaultConfig(backend="memory", encryption="fernet"))


def test_rdbms_vault_fails_closed() -> None:
    reg = get_registry()
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_vault_manager(VaultConfig(backend="postgresql"))


def test_per_conversation_sessions_fail_closed() -> None:
    reg = get_registry()
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_session_router(VaultConfig(session_mode="per-conversation", session="s"))


def test_audit_and_otel_fail_closed() -> None:
    reg = get_registry()
    with pytest.raises(ConfigError, match="audit"):
        reg.build_audit(AuditConfig(enabled=True))
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_telemetry(OtelConfig(enabled=True))


def test_named_users_fail_closed() -> None:
    reg = get_registry()
    # Free → no registry; Pro+ needs the pro package, so it fails closed.
    assert reg.build_users_store(UsersConfig(), "free") is None
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_users_store(UsersConfig(path="x"), "pro")


def test_license_key_without_pro_resolves_free_with_notice() -> None:
    # R3: the enforcement core (verify/parse/resolve) lives in llm_redact_pro.
    # Without it a configured key CANNOT be verified, so the proxy runs Free —
    # loudly (never a silent claim of a paid tier). The delegator routes to the
    # Free default resolver because the pro plugin is absent.
    from llm_redact.licensing import resolve_license

    resolved = resolve_license(env={"LLM_REDACT_LICENSE_KEY": "llmr1.some.key"})
    assert resolved.tier == "free"
    assert resolved.source == "env"
    assert resolved.warnings and "llm-redact-pro" in resolved.warnings[0]


def test_proxy_boots_on_free_config() -> None:
    # End-to-end: a fully-Free config (memory vault, static sessions, keyless)
    # constructs the app without any license or pro package.
    app = create_app(Config())
    assert app is not None
