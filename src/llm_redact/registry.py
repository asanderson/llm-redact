"""Extension registry: the construction seams the paid package overrides.

The open-core split (llm-redact-pro docs/licensing.md) keeps the Free tier in this
public package and moves the paid subsystems (vault encryption, RDBMS vault,
audit, OTel, per-conversation sessions, named users) to a separately-distributed
``llm-redact-pro`` package. This module is the seam between them: the proxy
builds its swappable subsystems through a ``Registry`` of factories rather than
importing concrete implementations, and an installed plugin — discovered via
the ``llm_redact.plugins`` entry-point group — replaces the factories for the
capabilities it provides.

Wiring only. This module holds no policy: the license gate stays the single
``features.check_license`` chokepoint, and the fail-closed rule (paid config
without the paid package is a ``ConfigError``, never a silent downgrade) lives
in the factories the registry dispatches to.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from .audit import build_audit as _build_audit
from .audit_s3 import build_audit_sinks as _build_audit_sinks
from .free_defaults import build_telemetry as _build_telemetry
from .free_defaults import resolve_license as _resolve_license
from .sessions import build_session_router as _build_session_router
from .users import build_users_store as _build_users_store
from .vault import build_cipher as _build_cipher
from .vault import build_vault_manager as _build_vault_manager
from .vault import cipher_from_key as _cipher_from_key

if TYPE_CHECKING:
    from .audit import AuditLog
    from .audit_s3 import AzureAuditSink, S3AuditSink
    from .config import AuditConfig, OtelConfig, UsersConfig, VaultConfig
    from .licensing import ResolvedLicense
    from .plugin_api import SessionRouter, Telemetry, VaultCipher
    from .users import UsersStore
    from .vault import VaultManager

logger = logging.getLogger("llm_redact")

ENTRY_POINT_GROUP = "llm_redact.plugins"

# The paid distribution's import package (open-core split, llm-redact-pro docs/licensing.md).
PRO_PACKAGE = "llm_redact_pro"


def pro_package_installed() -> bool:
    """True when the paid ``llm-redact-pro`` package is importable.

    The honest "licensed-features package: installed / not installed" signal
    surfaced by ``doctor``, ``/status``, and the dashboard (llm-redact-pro docs/licensing.md).
    A pure import-spec probe — it does NOT import the package, trigger plugin
    discovery, or consult any license: package presence and license *tier* are
    independent (a tier is enforced by ``features.check_license``). "Installed"
    means only that the paid code is on the path; whether its plugin actually
    registered is a separate question ``loaded_plugins()`` answers.
    """
    return importlib.util.find_spec(PRO_PACKAGE) is not None


class Registry:
    """Factories the proxy calls to build its swappable subsystems.

    Attributes default to the in-tree Free implementations. A plugin's
    ``register(registry)`` hook mutates this object to install its own
    factories; the proxy never learns which implementation it holds.
    """

    build_vault_manager: Callable[[VaultConfig], VaultManager]
    build_cipher: Callable[[VaultConfig], VaultCipher | None]
    cipher_from_key: Callable[[bytes], VaultCipher]
    build_session_router: Callable[..., SessionRouter]
    resolve_license: Callable[..., ResolvedLicense]
    build_telemetry: Callable[[OtelConfig], Telemetry | None]
    build_audit: Callable[[AuditConfig], AuditLog | None]
    build_audit_sinks: Callable[[AuditConfig], tuple[S3AuditSink | None, AzureAuditSink | None]]
    build_users_store: Callable[[UsersConfig, str], UsersStore | None]

    def __init__(self) -> None:
        # Assigned as INSTANCE attributes (not class attributes) so a bare
        # function is stored as a plain callable, never bound as a method.
        self.build_vault_manager = _build_vault_manager
        # The cipher factories are their own seam so the CLI (vault lookup /
        # rotate-key / doctor key-match) can reach the paid cipher through
        # the registry without importing llm-redact-pro directly; the Free
        # defaults fail closed on an encrypted vault.
        self.build_cipher = _build_cipher
        self.cipher_from_key = _cipher_from_key
        self.build_session_router = _build_session_router
        # License verification (the "what did the vendor sign" enforcement core)
        # is a paid subsystem (R3); the Free default resolves to Free with a
        # "package not installed" notice, and llm-redact-pro overrides it.
        self.resolve_license = _resolve_license
        self.build_telemetry = _build_telemetry
        self.build_audit = _build_audit
        self.build_audit_sinks = _build_audit_sinks
        self.build_users_store = _build_users_store


def load_plugins(registry: Registry) -> list[str]:
    """Let each installed plugin override registry factories.

    Returns the names of plugins that registered (surfaced in /status for
    honesty). A plugin whose hook fails to import or raises is logged and
    skipped — a broken optional package must never take the proxy down.
    """
    loaded: list[str] = []
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            register = entry_point.load()
            register(registry)
        except Exception:
            logger.exception("failed to load llm-redact plugin %r", entry_point.name)
            continue
        loaded.append(entry_point.name)
    return loaded


_registry: Registry | None = None
_loaded_plugins: list[str] = []


def get_registry() -> Registry:
    """The process-wide registry, built once; plugins are discovered on first
    use so the entry-point scan happens exactly one time per process."""
    global _registry
    if _registry is None:
        registry = Registry()
        _loaded_plugins[:] = load_plugins(registry)
        _registry = registry
    return _registry


def loaded_plugins() -> list[str]:
    """Names of the plugins that registered against ``get_registry()``."""
    return list(_loaded_plugins)
