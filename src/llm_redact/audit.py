"""Audit-log contract (the Free side of the open-core split).

The audit log records metadata only — detector types and counts, paths,
timestamps, durations. Never values, never placeholder ids, never headers or
bodies. The concrete SQLite log, its tamper-evident hash chain, and the
off-machine sinks are a paid subsystem (``llm_redact_pro.audit`` /
``llm_redact_pro.audit_s3``). This module holds only what the Free core needs
at the seam:

- ``AuditRecord`` — the metadata row ``record_request`` builds and hands to the
  log (kept here so ``proxy.py`` is unchanged),
- the ``AuditLog`` **Protocol** the proxy calls,
- the env-var name + key-resolution helper ``doctor`` uses for its posture
  checks (generic env-reading glue, no paid secrecy value), and
- the fail-closed ``build_audit`` default.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .config import ConfigError

if TYPE_CHECKING:
    from .config import AuditConfig

AUDIT_HMAC_ENV = "LLM_REDACT_AUDIT_HMAC_KEY"

_PRO_HINT = "install the llm-redact-pro package to enable it"


@dataclass
class AuditRecord:
    ts: str
    session: str
    provider: str | None
    method: str
    path: str
    status: int | None
    duration_ms: float
    streamed: bool
    detections: dict[str, int] = field(default_factory=dict)
    rehydrations: dict[str, int] = field(default_factory=dict)
    # Named-user attribution (2.0): the user NAME only, never keys. None on
    # single-user deployments and rows recorded before 2.0.
    user: str | None = None
    # Warn-mode hits attributed to this request (3.3): types+counts of values
    # FORWARDED upstream unredacted. None when nothing warned and on rows
    # recorded before the column existed.
    warned: dict[str, int] | None = None
    # A refusal this request passed on its requester's approved override
    # (overrides.py): "once" | "always"; None otherwise and on older rows.
    override: str | None = None


class AuditWriteError(RuntimeError):
    """A required-mode audit row could not be durably committed.

    Raised by write-ahead audit implementations (``[audit] required = true``,
    llm-redact-pro) from ``begin``/``finalize``/``record`` when the row cannot
    be committed. On the ``begin`` path the proxy answers a provider-shaped
    503 WITHOUT contacting the upstream — no durably committed audit row, no
    upstream contact; after a response has been committed the proxy can only
    log the failure loudly. Never raised in the default fail-open mode, where
    the concrete log swallows write faults by design.
    """


class AuditLog(Protocol):
    """The audit-log surface the proxy depends on.

    The concrete SQLite implementation (with the tamper-evident chain) is
    ``llm_redact_pro.audit.AuditLog``; the Free core holds only this structural
    contract. ``verify()`` — used by the offline ``audit verify`` CLI, not the
    request path — lives on the concrete class only. The write-ahead pair for
    ``[audit] required`` mode is the :class:`WriteAheadAudit` sub-Protocol —
    kept out of this base so a pro package predating it still satisfies every
    fail-open configuration, type contract included.
    """

    def record(self, entry: AuditRecord) -> None: ...

    def recent(self, limit: int) -> list[dict[str, object]]: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


@runtime_checkable
class WriteAheadAudit(AuditLog, Protocol):
    """An audit log with the ``[audit] required`` write-ahead pair.

    ``begin`` durably commits a START row BEFORE any upstream contact
    (raising :class:`AuditWriteError` on failure) and returns an opaque
    non-None token; ``finalize`` commits the matching END row at completion.
    Both are SYNCHRONOUS: each must make its row durable before it returns.
    A member declared ``async def`` (``begin``, ``finalize`` or the optional
    ``amend``) refuses required mode at startup — a ConfigError naming it,
    which ``serve --check`` reports. A synchronous member answering an
    awaitable anyway (which the core never awaits on the request path) has
    it closed unrun — a pending asyncio Task or Future cancelled, so a write
    it scheduled never lands later — and treated as
    :class:`AuditWriteError`, never as a committed row: from ``begin`` the
    request is refused with the provider-shaped 503 of a START row that
    cannot be committed (HTTP and realtime, before any upstream contact,
    recorded); from ``finalize`` it is an END-row fault, logged CRITICAL by
    type (the answer is already committed). ``ProxyState`` resolves the
    capability once at startup (``isinstance`` checks method presence, the
    runtime-checkable contract) and refuses required mode when the built log
    lacks the pair.

    OPTIONAL ``amend(token, entry) -> None``, read once at startup via
    ``getattr`` (it is not a Protocol method, so a log predating it still
    satisfies the pair): an upload whose binary parts go to the upload
    inspector commits its START row BEFORE the inspection, when the
    redaction has not run and the row carries no detections. When the
    redaction then finds values (warn-mode ones are FORWARDED) or the
    request passes a refusal on an approved override, the core hands
    ``amend`` that START row's token and an entry shaped like a START row
    carrying the request's ``detections``/``warned``/``override``, before
    any upstream contact, and the request keeps ONE START row: its END row
    finalizes the same token. ``amend`` is synchronous and must make the
    amendment durable before it returns, or raise :class:`AuditWriteError`
    (an awaitable answer is closed unrun and treated as that error, never as
    a durable amendment); the core then refuses the request with
    the provider-shaped 503 of a START row that cannot be committed (no
    upstream contact), and that refusal's END row closes the START row. A
    tamper-evident log never rewrites a chained row in place (it appends an
    amendment record of its own). Without the member the core commits a
    second START row carrying the counts and ends the first one with no
    status: two START rows for one request.
    """

    def begin(self, entry: AuditRecord) -> object: ...

    def finalize(self, token: object, entry: AuditRecord) -> None: ...


def default_audit_path() -> Path:
    # An empty XDG_DATA_HOME is unset (the XDG spec), never the current dir.
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "llm-redact" / "audit.db"


def audit_hmac_key_from_env() -> bytes | None:
    """The tamper-evidence HMAC key, derived from LLM_REDACT_AUDIT_HMAC_KEY.

    Env-only (never the config file, matching the vault key and S3
    credentials). Any-length passphrase is hashed to 32 bytes. Returns None
    when unset so the caller can fail closed with a clear message.
    """
    raw = os.environ.get(AUDIT_HMAC_ENV, "")
    if not raw:
        return None
    return hashlib.sha256(raw.encode("utf-8")).digest()


def build_audit(config: AuditConfig) -> AuditLog | None:
    """Fail-closed Free default for the audit log.

    The audit log — and its tamper-evident chain — is a paid subsystem whose
    implementation lives in llm-redact-pro. Disabled is None; enabled without
    the pro package fails closed rather than silently dropping the audit trail.
    """
    if not config.enabled:
        return None
    raise ConfigError(f"[audit] enabled = true requires an audit log; {_PRO_HINT}")
