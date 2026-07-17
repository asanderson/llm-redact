"""The plugin registry: Free defaults, entry-point discovery, fail-safe load.

The open-core split (llm-redact-pro docs/licensing.md) builds swappable subsystems
through this registry so the paid package can override them. These tests pin
the Free defaults, the one-time caching, and — critically — that a broken
plugin is logged and skipped rather than taking the proxy down.
"""

from __future__ import annotations

import pytest

import llm_redact.registry as registry_mod
from llm_redact.config import AuditConfig, ConfigError, OtelConfig, UsersConfig, VaultConfig
from llm_redact.registry import Registry, get_registry, load_plugins, loaded_plugins
from llm_redact.sessions import StaticSessionRouter
from llm_redact.vault import InMemoryVaultManager


@pytest.fixture(autouse=True)
def _reset_registry_cache() -> None:
    registry_mod._registry = None
    registry_mod._loaded_plugins[:] = []
    yield
    registry_mod._registry = None
    registry_mod._loaded_plugins[:] = []


class _FakeEntryPoint:
    def __init__(self, name: str, loader: object) -> None:
        self.name = name
        self._loader = loader

    def load(self) -> object:
        return self._loader


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, eps: list[_FakeEntryPoint]) -> None:
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: eps)


def test_defaults_are_the_free_factories() -> None:
    reg = Registry()
    # Instance attributes, so bare functions stay plain callables (never
    # descriptor-bound as methods).
    assert reg.build_vault_manager.__name__ == "build_vault_manager"
    assert reg.build_session_router.__name__ == "build_session_router"
    assert reg.build_telemetry.__name__ == "build_telemetry"


def test_default_factories_build_free_subsystems() -> None:
    reg = Registry()
    assert isinstance(reg.build_vault_manager(VaultConfig(backend="memory")), InMemoryVaultManager)
    assert reg.build_telemetry(OtelConfig()) is None  # disabled → None, no pro package needed
    # OTel is a paid feature: enabling it without the pro package fails closed
    # in the Free default (the load-bearing open-core rule), never downgrades.
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_telemetry(OtelConfig(enabled=True))


def test_build_cipher_default_factory() -> None:
    reg = Registry()
    # Plaintext → no cipher; fernet and raw-key construction are the paid
    # subsystem, so the Free default fails closed rather than downgrading.
    assert reg.build_cipher(VaultConfig()) is None
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_cipher(VaultConfig(encryption="fernet"))
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.cipher_from_key(b"k" * 32)


def test_build_session_router_default_is_static_and_fails_closed() -> None:
    # After R2's sessions hard split the Free registry default builds only the
    # static router (threading the session name) and fails closed on
    # per-conversation rather than downgrading to the shared namespace. The
    # per-conversation router — and its argument threading — is the pro
    # factory, pinned in tests/pro/test_sessions_pro.py.
    reg = Registry()
    router = reg.build_session_router(VaultConfig(session_mode="static", session="mysession"))
    assert isinstance(router, StaticSessionRouter)
    assert router.mode == "static"
    assert router._fallback == "mysession"  # session arg threaded, not the mode
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_session_router(VaultConfig(session_mode="per-conversation", session="x"))


def test_resolve_license_default_is_free_with_notice() -> None:
    # License verification (the "what did the vendor sign" enforcement core) is
    # a paid subsystem as of R3; the Free default resolves to Free — silently
    # when no key is configured, with a loud "package not installed" notice when
    # one is, never a silent claim of a paid tier.
    reg = Registry()
    assert reg.resolve_license(env={}).tier == "free"
    with_key = reg.resolve_license(env={"LLM_REDACT_LICENSE_KEY": "llmr1.x.y"})
    assert with_key.tier == "free"
    assert with_key.source == "env"
    assert with_key.warnings and "llm-redact-pro" in with_key.warnings[0]


def test_build_audit_default_factory() -> None:
    reg = Registry()
    assert reg.build_audit(AuditConfig(enabled=False)) is None  # disabled → None
    # The audit log (and its tamper chain) is a paid subsystem; the Free default
    # fails closed rather than silently dropping the audit trail.
    with pytest.raises(ConfigError, match="audit"):
        reg.build_audit(AuditConfig(enabled=True))


def test_build_audit_sinks_default_factory() -> None:
    reg = Registry()
    assert reg.build_audit_sinks(AuditConfig()) == (None, None)  # both disabled
    # Enabling either off-machine sink without the pro package fails closed.
    from llm_redact.config import S3AuditConfig

    with pytest.raises(ConfigError, match="sink"):
        reg.build_audit_sinks(AuditConfig(s3=S3AuditConfig(enabled=True, bucket="b")))


def test_build_users_store_default_factory() -> None:
    reg = Registry()
    assert reg.build_users_store(UsersConfig(), "free") is None  # Free → no registry
    # On Pro+ the named-user registry is a paid subsystem; the Free default
    # fails closed rather than running enforcement without a store.
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_users_store(UsersConfig(path="x"), "pro")


def test_get_registry_is_cached_and_discovers_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _counting(group: str) -> list[_FakeEntryPoint]:
        calls["n"] += 1
        return []

    monkeypatch.setattr("importlib.metadata.entry_points", _counting)
    first = get_registry()
    second = get_registry()
    assert first is second
    assert calls["n"] == 1  # the entry-point scan runs exactly once per process


def test_no_plugins_leaves_free_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [])
    reg = get_registry()
    assert reg.build_vault_manager.__name__ == "build_vault_manager"
    assert loaded_plugins() == []


def test_plugin_hook_overrides_a_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()

    def register(reg: Registry) -> None:
        reg.build_telemetry = lambda config: sentinel  # type: ignore[assignment,return-value]

    _patch_entry_points(monkeypatch, [_FakeEntryPoint("pro", register)])
    reg = get_registry()
    assert reg.build_telemetry(OtelConfig()) is sentinel
    assert loaded_plugins() == ["pro"]


def test_broken_plugin_is_skipped_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def good(reg: Registry) -> None:
        reg.build_telemetry = lambda config: "ok"  # type: ignore[assignment,return-value]

    def boom(reg: Registry) -> None:
        raise RuntimeError("plugin exploded")

    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("boom", boom), _FakeEntryPoint("good", good)],
    )
    reg = Registry()
    loaded = load_plugins(reg)
    # The exploding plugin is dropped; the healthy one still registered.
    assert loaded == ["good"]
    assert reg.build_telemetry(OtelConfig()) == "ok"


def test_plugin_that_fails_to_import_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingEP:
        name = "missing"

        def load(self) -> object:
            raise ModuleNotFoundError("no pro package")

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [_FailingEP()])
    reg = Registry()
    assert load_plugins(reg) == []  # a missing import is logged and skipped


def test_pro_package_installed_is_a_pure_import_spec_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    from llm_redact.registry import PRO_PACKAGE, pro_package_installed

    # Free suite: the paid package is genuinely absent, so the probe is False
    # without any patching (the honest default).
    assert pro_package_installed() is False

    # It probes the paid package by name and never imports it: a truthy spec
    # stands in for a real ModuleSpec, None means absent.
    queried: list[str] = []

    def _spec(name: str) -> object | None:
        queried.append(name)
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", _spec)
    assert pro_package_installed() is True
    assert queried == [PRO_PACKAGE]

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert pro_package_installed() is False
