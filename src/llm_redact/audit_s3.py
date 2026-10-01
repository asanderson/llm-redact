"""Off-machine audit-sink contract (the Free side of the open-core split).

The concrete S3/GCS and Azure Blob sinks — the hand-rolled SigV4 and Azure
SharedKey signers, batch buffering, and client-side Fernet encryption — are a
paid (Pro-tier) subsystem in ``llm_redact_pro.audit_s3``. This module holds
only the seam:

- the ``S3AuditSink`` / ``AzureAuditSink`` Protocols the proxy runs and reads
  counters from,
- the env-var names + key/credential helpers ``doctor`` uses for its posture
  checks (generic env-reading glue, no paid secrecy value; credentials NEVER
  come from the config file), and
- the fail-closed ``build_audit_sinks`` default.
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import TYPE_CHECKING, Any, Protocol

from .config import ConfigError

if TYPE_CHECKING:
    from .config import AuditConfig

_PRO_HINT = "install the llm-redact-pro package to enable it"

# Client-side batch encryption ([audit.s3]/[audit.azure] encryption = "fernet").
AUDIT_ENC_KEY_ENV = "LLM_REDACT_AUDIT_ENC_KEY"
AZURE_STORAGE_KEY_ENV = "AZURE_STORAGE_KEY"
# [audit.azure] auth = "sas": the SAS token (a bearer secret appended to the
# blob URL's query — never logged, never in /status).
AZURE_STORAGE_SAS_ENV = "AZURE_STORAGE_SAS_TOKEN"


def audit_enc_key_from_env() -> bytes | None:
    """SHA-256 of the env passphrase, urlsafe-base64 — a valid Fernet key
    (the audit_hmac_key_from_env recipe, shaped for Fernet)."""
    raw = os.environ.get(AUDIT_ENC_KEY_ENV, "")
    if not raw:
        return None
    return base64.urlsafe_b64encode(hashlib.sha256(raw.encode("utf-8")).digest())


# Per-provider credential environment variables. GCS is reached through its
# S3-compatible XML API (interoperability) with HMAC interoperability keys read
# from GCS-specific vars. Credentials NEVER come from the config file.
_CREDENTIAL_ENV: dict[str, tuple[str, str, str | None]] = {
    "aws": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"),
    "minio": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"),
    "ceph": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"),
    "gcs": ("GCS_HMAC_ACCESS_ID", "GCS_HMAC_SECRET", None),
}


def credential_env_names(provider: str) -> tuple[str, str, str | None]:
    """The (access, secret, session-token) env var names for a provider's
    static keys (``[audit.s3] auth = "keys"``). ``required_credential_env``
    is the auth-aware form."""
    return _CREDENTIAL_ENV.get(provider, _CREDENTIAL_ENV["aws"])


def required_credential_env(
    sink: str, provider: str = "aws", auth: str = "keys"
) -> tuple[str, ...]:
    """The env vars that must be PRESENT for one sink's auth mode (doctor's
    presence check — values are never read here).

    Empty for ``auth = "identity"``: those credentials resolve at runtime
    from the workload's cloud identity (llm-redact-pro), which an offline
    check cannot verify without network calls.
    """
    if auth == "identity":
        return ()
    if sink == "azure":
        return (AZURE_STORAGE_SAS_ENV,) if auth == "sas" else (AZURE_STORAGE_KEY_ENV,)
    access_env, secret_env, _ = credential_env_names(provider)
    return access_env, secret_env


class S3AuditSink(Protocol):
    """The S3/GCS audit-sink surface the proxy runs (concrete impl in pro).

    Lifecycle (both sinks): ``run()`` is the flush loop, started as a task in
    the lifespan; ``add(row)`` is handed each request's row. At shutdown,
    after the server has drained its in-flight requests (every request's END
    row written), the core cancels ``run()`` and then awaits ``aclose()`` —
    the final flush — while the audit database is STILL OPEN, so rows
    spooled since the last upload ship now; the audit database closes after
    it, the vault last. ``aclose()`` of both sinks runs concurrently under
    one deadline (``proxy._SINK_CLOSE_TIMEOUT_SECONDS``, 20 s): a flush still
    running then is cancelled (its unshipped spooled rows wait in the
    database for the next start), one ignoring the cancellation is abandoned
    after a short grace, and an exception is logged by type only.
    """

    batches_uploaded: int
    rows_dropped: int

    def add(self, row: dict[str, Any]) -> None: ...

    async def run(self) -> None: ...

    async def aclose(self) -> None: ...


class AzureAuditSink(Protocol):
    """The Azure Blob audit-sink surface the proxy runs (concrete impl in pro).

    Same lifecycle as :class:`S3AuditSink`: the bounded final flush
    (``aclose()``) runs before the audit database closes."""

    batches_uploaded: int
    rows_dropped: int

    def add(self, row: dict[str, Any]) -> None: ...

    async def run(self) -> None: ...

    async def aclose(self) -> None: ...


def build_audit_sinks(
    config: AuditConfig,
) -> tuple[S3AuditSink | None, AzureAuditSink | None]:
    """Fail-closed Free default for the off-machine audit sinks.

    Both sinks disabled is ``(None, None)``. Enabling either without the pro
    package fails closed rather than silently dropping every batch — the sinks,
    signers, and batch encryption are a paid subsystem.
    """
    if not config.s3.enabled and not config.azure.enabled:
        return None, None
    raise ConfigError(f"audit backup sinks require an off-machine sink; {_PRO_HINT}")
