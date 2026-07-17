"""The FOSS core enforces nothing: no tier gates, no seat caps, no cloud
entitlements (the AGPL-3.0 relicense removed `features.check_license`).

Both directions are pinned, mirroring the old gate suite's discipline:

* every capability implemented IN THIS REPOSITORY builds keyless — the
  configs the old tier matrix refused on Free (non-loopback serving,
  Kubernetes, the cloud LLM adapters) now start cleanly;
* a config that requests a subsystem only llm-redact-pro implements
  (vault encryption, RDBMS vaults, the audit log, OTel, per-conversation
  sessions) still fails closed, but from the registry/factory seams,
  naming the PACKAGE — package presence (distribution) is the boundary,
  never a license key.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from llm_redact.config import (
    AuditConfig,
    Config,
    ConfigError,
    OtelConfig,
    RdbmsConfig,
    VaultConfig,
)
from llm_redact.proxy import create_app


def _with(**kwargs: object) -> Config:
    return dataclasses.replace(Config(), **kwargs)


# --- formerly tier-gated, implemented here: now keyless ----------------------


def test_non_loopback_app_builds_keyless() -> None:
    # The bind POLICY (mTLS trio or the confined-container hatch) is
    # validate_bind_security's job at serve time and is unchanged; what is
    # gone is the licensing gate that refused a non-loopback host outright.
    create_app(_with(host="0.0.0.0"))


@pytest.mark.parametrize("provider", ["bedrock", "azure", "vertex"])
def test_cloud_llm_adapters_route_keyless(provider: str) -> None:
    # Formerly Team-gated with a per-cloud entitlement; the adapters live in
    # this repository, so they are FOSS like everything else here.
    providers = dict(Config().providers)
    base = providers[provider]
    providers[provider] = dataclasses.replace(base, upstream_base_url="https://cloud.example/api")
    create_app(_with(providers=providers))


# --- pro-only subsystems: fail closed on package absence, naming it ----------


def test_vault_encryption_fails_closed_naming_package(tmp_path: Path) -> None:
    vault = VaultConfig(backend="sqlite", path=str(tmp_path / "v.db"), encryption="fernet")
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        create_app(_with(vault=vault))


def test_audit_log_fails_closed_naming_package(tmp_path: Path) -> None:
    config = _with(audit=AuditConfig(enabled=True, path=str(tmp_path / "audit.db")))
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        create_app(config)


def test_rdbms_vault_fails_closed_naming_package(tmp_path: Path) -> None:
    vault = VaultConfig(
        backend="dbapi",
        rdbms=RdbmsConfig(module="sqlite3", dsn=str(tmp_path / "vault.db")),
    )
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        create_app(_with(vault=vault))


def test_per_conversation_fails_closed_naming_package() -> None:
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        create_app(_with(vault=VaultConfig(session_mode="per-conversation")))


def test_otel_fails_closed_naming_package() -> None:
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        create_app(_with(otel=OtelConfig(enabled=True)))
