"""`llm-redact license show|verify` — inspect the configured license key.

Reads the key from --key, the LLM_REDACT_LICENSE_KEY env var, or the
[license] section of the config file (same order the proxy resolves at
startup). Output never echoes the key material itself — only the signed
payload fields and the verification verdict.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime

from .licensing import ResolvedLicense, resolve_license


def _resolve(args: argparse.Namespace) -> ResolvedLicense:
    env = dict(os.environ)
    if args.key is not None:
        env["LLM_REDACT_LICENSE_KEY"] = args.key
    from .config import apply_env_overrides, load_config

    config = apply_env_overrides(load_config(args.config))
    return resolve_license(
        env=env,
        config_key=config.license.key,
        config_key_file=config.license.key_file,
    )


def run_license_show(args: argparse.Namespace) -> int:
    resolved = _resolve(args)
    lic = resolved.license
    if args.json:
        payload = {
            "tier": resolved.tier,
            "source": resolved.source,
            "in_grace": resolved.in_grace,
            "warnings": list(resolved.warnings),
            "max_users": resolved.max_users,
            "clouds": list(resolved.clouds),
            "license": None
            if lic is None
            else {
                "tier": lic.tier,
                "org": lic.org,
                "email": lic.email,
                "max_users": lic.max_users,
                "clouds": list(lic.clouds),
                "issued": lic.issued.isoformat(),
                "expires": lic.expires.isoformat(),
                "license_id": lic.license_id,
                "kid": lic.kid,
            },
        }
        print(json.dumps(payload, indent=2))
        return 0
    if lic is None and resolved.source == "absent":
        print("no license key configured — Free tier (1 user, core features)")
        return 0
    print(f"effective tier: {resolved.tier}  (key source: {resolved.source})")
    if lic is not None:
        users = "unlimited" if lic.max_users is None else str(lic.max_users)
        clouds = ", ".join(lic.clouds) or "none"
        print(f"licensee: {lic.org} <{lic.email}>  license_id: {lic.license_id}")
        print(f"signed tier: {lic.tier}  users: {users}  clouds: {clouds}")
        print(
            f"issued: {lic.issued.isoformat()}  expires: {lic.expires.isoformat()}  kid: {lic.kid}"
        )
    for warning in resolved.warnings:
        print(f"⚠ {warning}")
    return 0


def run_license_verify(args: argparse.Namespace) -> int:
    """Exit 0 = valid and current; 1 = absent/invalid; 2 = signature-valid
    but expired (in or past grace)."""
    resolved = _resolve(args)
    for warning in resolved.warnings:
        print(f"⚠ {warning}")
    if resolved.license is None:
        if resolved.source == "absent":
            print("no license key configured (Free tier needs none)")
        return 1
    today = datetime.now(UTC).date()
    if today > resolved.license.expires:
        print(f"expired {resolved.license.expires.isoformat()}")
        return 2
    if resolved.tier != resolved.license.tier:
        return 1
    print(f"valid: {resolved.license.tier} until {resolved.license.expires.isoformat()}")
    return 0
