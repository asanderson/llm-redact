"""Fail-closed Free-tier defaults for subsystems the paid package implements.

The open-core split (llm-redact-pro docs/licensing.md) moves the paid subsystem
implementations to the separately-distributed ``llm-redact-pro`` package. When
that package is installed, its plugin overrides these factories through the
registry. When it is not, a config that requests a paid feature must fail
loudly — never a silent downgrade. This is the load-bearing rule: paid config
present but the pro package absent is a ``ConfigError``, reusing the existing
"enabled-without-extra is a ConfigError" pattern.

Only subsystems whose implementation lives ENTIRELY in pro have their default
here. Subsystems the Free core still implements in part (the vault, session
routing) keep their factories in their own modules and raise there only for
the paid sub-cases.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from .config import ConfigError
from .licensing import ENV_KEY, FREE, ResolvedLicense

if TYPE_CHECKING:
    from .config import OtelConfig
    from .plugin_api import Telemetry

_PRO_HINT = "install the llm-redact-pro package to enable it"


def build_telemetry(config: OtelConfig) -> Telemetry | None:
    """None when ``[otel]`` is disabled; otherwise the pro package is required.

    OpenTelemetry export is a paid feature whose implementation lives in
    llm-redact-pro. With the pro package installed this factory is replaced by
    the real builder; without it, enabling ``[otel]`` fails closed here.
    """
    if not config.enabled:
        return None
    raise ConfigError(f"[otel] enabled = true requires OpenTelemetry export; {_PRO_HINT}")


def resolve_license(
    *,
    env: dict[str, str],
    config_key: str | None = None,
    config_key_file: str | None = None,
    public_keys: dict[str, bytes] | None = None,
    today: date | None = None,
) -> ResolvedLicense:
    """The Free default license resolver: no enforcement package, so no key
    can be verified — run the Free tier.

    License verification (the "what did the vendor sign" core) lives in
    llm-redact-pro (R3). Without it a configured key cannot be checked, so we
    resolve to Free — but say so loudly when a key WAS supplied (never a silent
    ignore, mirroring the real resolver's reject-to-Free posture). ``public_keys``
    and ``today`` are part of the resolver contract but only matter to actual
    verification, so they are unused here.
    """
    del public_keys, today  # interface parity with the pro resolver; unused here
    source: str | None = None
    if env.get(ENV_KEY, "").strip():
        source = "env"
    elif config_key:
        source = "config"
    elif config_key_file:
        source = "key_file"
    if source is None:
        return FREE
    return ResolvedLicense(
        tier="free",
        license=None,
        source=source,
        warnings=(
            "a license key is configured but the licensed-features package "
            "(llm-redact-pro) is not installed; running on the Free tier",
        ),
    )
