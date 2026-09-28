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
    from .config import Config, OtelConfig, ProviderConfig, VaultConfig
    from .plugin_api import (
        AccessGate,
        Dashboard,
        DbPasswordProvider,
        Router,
        Telemetry,
        UpstreamAuth,
    )

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


def build_db_password(config: VaultConfig) -> DbPasswordProvider | None:
    """None for the static password (``[vault.rdbms] auth = "password"``);
    identity auth requires the pro package.

    Minting a database token from the proxy's cloud identity (RDS IAM auth,
    Cloud SQL IAM database auth, Entra ID) is credential fetching, which the
    core never does. Without llm-redact-pro, ``auth = "identity"`` fails
    closed here — never a silent fallback to a static password that the
    config deliberately did not name.
    """
    if config.rdbms.auth != "identity":
        return None
    raise ConfigError(
        '[vault.rdbms] auth = "identity" requires cloud-identity database'
        f" authentication; {_PRO_HINT}"
    )


def build_router(config: Config, tier: str) -> Router | None:
    """None when ``[routing]`` is not enabled; otherwise the pro package is required.

    Rule-based upstream routing (named upstreams, credential modes, fallback
    chains with cooldowns and plan-limit detection, monthly budgets) is a paid
    feature whose implementation lives in llm-redact-pro. With that package
    installed this factory is replaced by the real builder (which also
    honors the key's tier); without it, enabling ``[routing]`` fails closed
    here — never a silent "one upstream per protocol" downgrade that would
    quietly ignore the operator's rules, credentials and budgets. ``tier``
    is part of the factory contract and unused here: the Free core enforces
    no tier.
    """
    del tier
    if not config.routing.enabled:
        return None
    raise ConfigError(
        "[routing] enabled = true requires the llm-redact-pro package (0.3+): rule-based"
        " upstream routing, fallback chains and budgets are pro subsystems;"
        f" {_PRO_HINT}"
    )


def build_upstream_auth(name: str, provider: ProviderConfig) -> UpstreamAuth | None:
    """None for ``auth = "passthrough"`` (the client's credential is
    forwarded, as always); otherwise the pro package is required.

    Authorizing upstream requests with the proxy's own cloud identity (AWS
    SigV4, Google/Entra ID bearer tokens) is a paid feature implemented in
    llm-redact-pro. Without it, ``auth = "identity"`` fails closed here —
    never a silent fall back to forwarding the client's credential, which
    the operator configured the proxy to replace.
    """
    if provider.auth == "passthrough":
        return None
    raise ConfigError(
        f'[providers.{name}] auth = "{provider.auth}" (the proxy\'s own cloud identity)'
        " requires the llm-redact-pro package with provider identity support;"
        f" {_PRO_HINT}"
    )


def build_dashboard(tier: str) -> Dashboard | None:
    """Always None: the browser dashboard (status view, config editor,
    redaction preview) is implemented in llm-redact-pro.

    Unlike a paid CONFIG section there is nothing to fail closed on — the
    dashboard is never requested by config — so the core simply answers
    the dashboard paths with a 404 naming the package, while the machine
    APIs (/status, /metrics, …) and the CLI twins (``llm-redact status``,
    ``llm-redact preview``) keep working keyless. ``tier`` is part of the
    factory contract (the pro builder honors it) and unused here.
    """
    del tier
    return None


def build_access_gate(config: Config, license: ResolvedLicense) -> AccessGate | None:
    """None on the Free tier with no named-user config; otherwise fail closed.

    Client admission (named users, per-user keys, seats, and every future
    authentication method) is a paid subsystem implemented entirely in
    llm-redact-pro — the core holds no credential logic. This default runs
    only when no plugin replaced it, so a paid tier or an explicit
    ``[users]`` section here means the operator expects access control the
    proxy cannot provide: refuse to start rather than serve an open proxy.
    A paid tier can only resolve when a pro license resolver is registered,
    so reaching that branch means an llm-redact-pro that predates this seam.
    """
    if license.tier != "free":
        raise ConfigError(
            f"the {license.tier} license includes named-user access control, but the"
            " installed llm-redact-pro does not provide it (no access gate registered);"
            " upgrade llm-redact-pro to 0.8 or later"
        )
    from .registry import pro_package_installed

    if license.source != "absent" and pro_package_installed():
        # A license key is configured and llm-redact-pro is installed, yet its
        # plugin did not register (an llm-redact-pro that no longer loads on
        # this core, e.g. one older than 0.8): the key resolved to Free only
        # because the package failed, and serving on would silently drop the
        # access control the deployment was licensed for.
        raise ConfigError(
            "a license key is configured and llm-redact-pro is installed, but its plugin"
            " did not load, so its access control is unavailable; upgrade llm-redact-pro"
            " to 0.8 or later (or remove the key to run the Free tier)"
        )
    if config.users.path is not None:
        raise ConfigError(f"[users] configures named users, a paid feature; {_PRO_HINT}")
    return None


def tool_base_url(base_url: str) -> str:
    """The base URL ``llm-redact run`` exports to wrapped tools, unchanged.

    A plugin may decorate it (for instance with a client identity); the
    Free core adds nothing.
    """
    return base_url


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
