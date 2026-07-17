"""Stable contract for out-of-tree plugins (the paid ``llm-redact-pro``
package registers against these names).

The open-core split (llm-redact-pro docs/licensing.md) lets a separately-distributed
package supply the paid vault/audit/session/telemetry/users implementations.
Those implementations must bind to a *supported* surface, not to private
internals that move between releases. This module is that surface: re-exports
promoted from the Free core with public names. Everything here is part of the
plugin API contract — change it deliberately, never incidentally.
"""

from __future__ import annotations

from typing import Any, Protocol

from .vault import (
    _MAX_RESPONSE_ROWS as MAX_RESPONSE_ROWS,
)
from .vault import (
    _RESPONSE_PRUNE_EVERY as RESPONSE_PRUNE_EVERY,
)
from .vault import (
    Vault,
    VaultKeyError,
    VaultManager,
)


class Telemetry(Protocol):
    """The telemetry recorder surface the proxy depends on.

    The concrete implementation (OpenTelemetry export) lives in the paid
    ``llm-redact-pro`` package; the Free core holds only this structural
    contract so ``proxy.py`` stays type-checked without importing pro.
    """

    def record(
        self, row: dict[str, Any], duration_seconds: float, *, traceparent: str | None = None
    ) -> None: ...

    def shutdown(self) -> None: ...


class VaultCipher(Protocol):
    """The at-rest vault cipher surface the Free vault code branches on.

    The concrete implementation (Fernet + HKDF, env/keyring key resolution)
    lives in the paid ``llm-redact-pro`` package; the Free vault classes accept
    a ``VaultCipher | None`` and are inert (unencrypted) when it is None, which
    is always the case unless the pro package supplies one.
    """

    def mac(self, session_id: str, detector_type: str, original: str) -> str: ...

    def encrypt(self, original: str) -> bytes: ...

    def decrypt(self, token: bytes) -> str: ...

    def key_check(self) -> str: ...


class SessionRouter(Protocol):
    """The session-routing surface the proxy depends on.

    Static-mode routing — one shared placeholder namespace, ``mode ==
    "static"`` — lives in the Free core (``StaticSessionRouter``); the
    per-conversation router (which derives a per-conversation session id from
    the conversation anchor) lives in the paid ``llm-redact-pro`` package. The
    proxy holds only this structural contract and reads ``mode`` to keep the
    static hot path (a prebuilt ``RequestContext``, no per-request resolution).

    ``resolve`` and ``record_response_id`` are only invoked when ``mode`` is not
    ``"static"``, so the Free static router implements them as inert stubs.
    """

    mode: str

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str: ...

    def record_response_id(self, response_id: str, session_id: str) -> None: ...


__all__ = [
    "MAX_RESPONSE_ROWS",
    "RESPONSE_PRUNE_EVERY",
    "SessionRouter",
    "Telemetry",
    "Vault",
    "VaultCipher",
    "VaultKeyError",
    "VaultManager",
]
