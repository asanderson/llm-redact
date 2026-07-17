"""Free-tier licensing data shapes and the resolve_license delegator.

The enforcement core — signature verification, payload parsing, real-key
resolution, grace/expiry policy — moved to llm_redact_pro in R3 and is covered
by tests/pro/test_licensing_enforcement.py. Here we pin the parts that stay in
the public core: the ``ResolvedLicense`` data shapes, the tier map, and the
delegator's keyless behavior (a deployment with no key resolves to Free,
whichever resolver is registered).
"""

from __future__ import annotations

from datetime import date

from llm_redact.licensing import (
    CLOUDS,
    FREE,
    TIER_ORDER,
    TIER_USER_CAPS,
    License,
    ResolvedLicense,
    resolve_license,
)


def test_free_sentinel() -> None:
    assert FREE.tier == "free"
    assert FREE.license is None
    assert FREE.source == "absent"
    assert FREE.max_users == 1
    assert FREE.clouds == ()


def test_tier_map_and_caps() -> None:
    assert list(TIER_ORDER) == ["free", "pro", "team", "unlimited", "managed"]
    assert TIER_ORDER["team"] > TIER_ORDER["pro"]
    assert TIER_USER_CAPS == {
        "free": 1,
        "pro": 1,
        "team": 25,
        "unlimited": None,
        "managed": None,
    }


def _license(tier: str, *, max_users: int | None, clouds: list[str]) -> License:
    return License(
        tier=tier,
        org="Org",
        email="a@corp.example",
        max_users=max_users,
        clouds=tuple(clouds),
        issued=date(2026, 1, 1),
        expires=date(2027, 1, 1),
        license_id="x",
        kid="dev-1",
    )


def test_resolved_max_users_prefers_license_then_cap() -> None:
    lic = _license("team", max_users=10, clouds=["aws"])
    assert ResolvedLicense(tier="team", license=lic, source="env", warnings=()).max_users == 10
    # A tier mismatch (e.g. a post-grace downgrade to free) falls back to the
    # tier's default cap, never the lapsed license's number.
    downgraded = ResolvedLicense(tier="free", license=lic, source="env", warnings=())
    assert downgraded.max_users == 1


def test_resolved_clouds_unlimited_implies_all() -> None:
    unlimited = _license("unlimited", max_users=None, clouds=[])
    assert ResolvedLicense(
        tier="unlimited", license=unlimited, source="env", warnings=()
    ).clouds == (CLOUDS)
    team = _license("team", max_users=25, clouds=["aws"])
    assert ResolvedLicense(tier="team", license=team, source="env", warnings=()).clouds == ("aws",)
    # No license (or a tier mismatch) carries no cloud entitlements.
    assert ResolvedLicense(tier="free", license=None, source="absent", warnings=()).clouds == ()


def test_resolve_license_no_key_is_free() -> None:
    # The delegator with no configured key resolves to Free regardless of which
    # resolver (the pro one or the Free default) is registered.
    resolved = resolve_license(env={})
    assert resolved.tier == "free"
    assert resolved.source == "absent"
    assert resolved.max_users == 1
