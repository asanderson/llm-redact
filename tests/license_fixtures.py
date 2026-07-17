"""Build ResolvedLicense inputs directly, for testing the Free check_license gate.

R4 open-core split: signing a key and RESOLVING it to a paid tier is the pro
enforcement resolver's job, which lives in the separate llm-redact-pro package.
The public Free repo tests `check_license` — a pure policy function over a
ResolvedLicense — so it constructs that input as a dataclass here, no signing
and no resolver. (The resolver itself is tested in the pro repo.)
"""

from __future__ import annotations

from datetime import date

from llm_redact.licensing import FREE, TIER_USER_CAPS, License, ResolvedLicense

__all__ = ["FREE", "resolved"]


def resolved(
    tier: str,
    clouds: list[str] | None = None,
    *,
    max_users: int | None = None,
    in_grace: bool = False,
) -> ResolvedLicense:
    """A ResolvedLicense at `tier` (with `clouds`), built directly as data.

    The inner License carries the same tier so ResolvedLicense.max_users /
    .clouds resolve from it (Unlimited/Managed still imply every cloud). Pass
    tier="free" for the keyless Free result (`FREE`).
    """
    if tier == "free":
        return FREE
    lic = License(
        tier=tier,
        org="Test Org",
        email="test@corp.example",
        max_users=max_users if max_users is not None else TIER_USER_CAPS[tier],
        clouds=tuple(clouds or ()),
        issued=date(2026, 1, 1),
        expires=date(2099, 1, 1),
        license_id="test-license",
        kid="dev-1",
    )
    return ResolvedLicense(tier=tier, license=lic, source="env", warnings=(), in_grace=in_grace)
