"""Vault session routing (Free core: static namespace only).

The Free tier serves one shared placeholder namespace: every request resolves
to the configured session name, so redaction and rehydration always agree at
zero hot-path cost. ``StaticSessionRouter`` is that router.

Per-conversation routing — each conversation gets its own namespace, derived
from the immutable first-message anchor (with orphan sessions for unmapped
OpenAI Responses chains) — is a paid feature and lives in the
``llm-redact-pro`` package (``llm_redact_pro.sessions``). When
``[vault] session_mode = "per-conversation"`` is configured without that
package installed, ``build_session_router`` fails closed with a
``ConfigError`` rather than silently downgrading to the shared static namespace
(open-core split, llm-redact-pro docs/licensing.md).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import VaultConfig
    from .plugin_api import SessionRouter

logger = logging.getLogger("llm_redact")


class StaticSessionRouter:
    """Resolves every request to the one configured session name.

    This is the whole Free routing surface: ``mode`` is always ``"static"``,
    so the proxy takes its prebuilt-context hot path and never calls
    ``resolve``/``record_response_id`` — they are implemented as inert stubs to
    satisfy the ``plugin_api.SessionRouter`` contract.
    """

    mode = "static"

    def __init__(self, fallback_session: str) -> None:
        self._fallback = fallback_session

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return self._fallback

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None


def build_session_router(
    config: VaultConfig,
    durable_lookup: Callable[[str], str | None] | None = None,
) -> SessionRouter:
    """Construction seam for the session router (registry-dispatched).

    The Free core ships only the static router; ``session_mode =
    "per-conversation"`` requires the paid ``llm-redact-pro`` package, which
    registers a router that supports it. Absent that package, per-conversation
    fails closed here — never a silent downgrade to the shared static
    namespace (which would place every conversation's secrets in one namespace,
    the opposite of what per-conversation isolation promises).
    """
    if config.session_mode == "static":
        return StaticSessionRouter(config.session)
    from .config import ConfigError

    raise ConfigError(
        '[vault] session_mode = "per-conversation" requires the llm-redact-pro '
        "package (pip install llm-redact-pro); the Free tier serves one static "
        "session namespace"
    )
