"""License data shapes and the tier map (open-core split, llm-redact-pro docs/licensing.md).

A key is ``llmr1.<base64url payload JSON>.<base64url signature>``. The
verification of that signature — the "what did the vendor sign" enforcement
core — lives in the paid ``llm-redact-pro`` package (``_enforcement``), so a
Free-only deployment cannot verify (and therefore cannot claim) a paid tier:
distribution control is the primary protection, the signed key the second gate.
This module keeps only the parts that are pure *data*, not secret and not
tamper-sensitive: the ``License`` / ``ResolvedLicense`` dataclasses, the tier
map (``TIER_ORDER`` / ``TIER_USER_CAPS`` / ``CLOUDS`` / ``LEGACY_TIER_ALIASES``),
the grace/expiry windows, and the ``FREE`` sentinel — plus a thin
``resolve_license`` that delegates to the registered enforcement resolver.

``resolve_license`` never raises: with the pro package installed it verifies
the key and returns the effective tier (Free on any failure, with a warning
naming the failing SOURCE — never the key material); without it, the Free
default resolver returns Free with a loud "package not installed" notice.
Tier ENFORCEMENT itself lives in features.py — this module only answers the
resolution question. ``ResolvedLicense`` must stay a
``dataclasses.replace``-able frozen dataclass (proxy.py's daily refresh).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

KEY_PREFIX = "llmr1"
ENV_KEY = "LLM_REDACT_LICENSE_KEY"
ENV_ALLOW_DEV = "LLM_REDACT_LICENSE_ALLOW_DEV"

TIERS = ("free", "pro", "team", "unlimited", "managed")
TIER_ORDER = {name: rank for rank, name in enumerate(TIERS)}
CLOUDS = ("aws", "azure", "gcp")

# The pre-3.11 tier names. A signed key still carrying them is accepted and
# normalized to the current name, so outstanding licenses never need re-issue.
LEGACY_TIER_ALIASES = {"gold": "team", "platinum": "unlimited"}

# Tiers whose entitlements include every cloud implicitly ("all features").
_ALL_CLOUD_TIERS = frozenset({"unlimited", "managed"})

# Default named-user ceilings per tier; None = unlimited. A signed key may
# carry a lower max_users but issuance never exceeds these.
TIER_USER_CAPS: dict[str, int | None] = {
    "free": 1,
    "pro": 1,
    "team": 25,
    "unlimited": None,
    "managed": None,
}

GRACE_DAYS = 14
# A valid license this close to expiry starts warning (startup log, /status,
# doctor, dashboard — everything that reads ResolvedLicense.warnings), so
# renewal is prompted before the grace window is ever needed.
EXPIRY_WARN_DAYS = 30


class LicenseError(ValueError):
    """A supplied license key failed validation. The message never echoes
    the key material — only which check failed."""


@dataclass(frozen=True)
class License:
    tier: str
    org: str
    email: str
    max_users: int | None
    clouds: tuple[str, ...]
    issued: date
    expires: date
    license_id: str
    kid: str


@dataclass(frozen=True)
class ResolvedLicense:
    """The effective licensing state the rest of the proxy consumes."""

    tier: str
    license: License | None  # None on the keyless Free tier
    source: str  # "env" | "config" | "key_file" | "absent"
    warnings: tuple[str, ...]
    in_grace: bool = False

    @property
    def max_users(self) -> int | None:
        if self.license is not None and self.tier == self.license.tier:
            return self.license.max_users
        return TIER_USER_CAPS[self.tier]

    @property
    def clouds(self) -> tuple[str, ...]:
        if self.license is None or self.tier != self.license.tier:
            return ()
        # Unlimited/Managed carry every cloud entitlement by definition, no
        # matter what the payload listed.
        if self.tier in _ALL_CLOUD_TIERS:
            return CLOUDS
        return self.license.clouds


FREE = ResolvedLicense(tier="free", license=None, source="absent", warnings=())


def resolve_license(
    *,
    env: dict[str, str],
    config_key: str | None = None,
    config_key_file: str | None = None,
    public_keys: dict[str, bytes] | None = None,
    today: date | None = None,
) -> ResolvedLicense:
    """Resolve the effective tier by delegating to the registered enforcement
    resolver (registry-dispatched).

    With llm-redact-pro installed the pro resolver verifies the key and applies
    the grace/expiry policy; without it the Free default returns the Free tier
    (with a notice when a key was configured). Never raises — every failure
    resolves to Free. The public signature is stable so callers
    (proxy/doctor/CLIs) are unaffected by where the resolver lives.
    """
    from .registry import get_registry

    return get_registry().resolve_license(
        env=env,
        config_key=config_key,
        config_key_file=config_key_file,
        public_keys=public_keys,
        today=today,
    )
