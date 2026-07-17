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
    """The (access, secret, session-token) env var names for a provider."""
    return _CREDENTIAL_ENV.get(provider, _CREDENTIAL_ENV["aws"])


class S3AuditSink(Protocol):
    """The S3/GCS audit-sink surface the proxy runs (concrete impl in pro)."""

    batches_uploaded: int
    rows_dropped: int

    def add(self, row: dict[str, Any]) -> None: ...

    async def run(self) -> None: ...

    async def aclose(self) -> None: ...


class AzureAuditSink(Protocol):
    """The Azure Blob audit-sink surface the proxy runs (concrete impl in pro)."""

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
