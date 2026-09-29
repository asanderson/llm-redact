"""Configuration: TOML file over frozen dataclasses, stdlib only."""

import dataclasses
import ipaddress
import json
import math
import os
import re
import tomllib
import urllib.parse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_redact.detection.deny import DenyEntry
from llm_redact.detection.engine import CustomRule, DetectionConfig, NerConfig

if TYPE_CHECKING:
    from llm_redact.plugin_api import ConfigSection

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


@dataclass(frozen=True)
class ProviderConfig:
    upstream_base_url: str
    # Disabling a provider fails closed: matched routes AND pass-through
    # traffic inferred to it are answered by the proxy with an error, never
    # forwarded (forwarding pass-through would send unredacted bodies).
    enabled: bool = True
    # detection = false is a deliberate off-switch (e.g. a local Ollama):
    # requests to this provider are forwarded WITHOUT detection/redaction
    # — values go upstream as-is, like warn mode but provider-wide.
    # Rehydration stays active so placeholders from history still restore.
    detection: bool = True
    # How the proxy authenticates to this upstream. "passthrough" (the
    # default) forwards the client's own credential untouched. "identity"
    # (bedrock, vertex, azure only; implemented in llm-redact-pro) strips
    # every client credential and authorizes each request with the proxy's
    # OWN cloud identity AFTER redaction: AWS SigV4 for bedrock, an OAuth
    # bearer token for vertex/azure. Anyone who can reach the proxy then
    # spends that identity — pair it with the access gate or a loopback bind.
    auth: str = "passthrough"
    # bedrock + auth = "identity" only: the SigV4 signing region when the
    # upstream host does not name one (a VPC endpoint, a proxy in front).
    region: str | None = None


# Providers whose upstream the proxy can authorize with its own cloud
# identity ([providers.NAME] auth = "identity"), and how.
IDENTITY_AUTH_PROVIDERS: dict[str, str] = {
    "bedrock": "AWS SigV4 (service bedrock)",
    "vertex": "Google OAuth bearer token (cloud-platform scope)",
    "azure": "Microsoft Entra ID bearer token (cognitiveservices scope)",
}
PROVIDER_AUTH_MODES = ("passthrough", "identity")
_REGION_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}\Z")


DEFAULT_PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(upstream_base_url="https://api.anthropic.com"),
    "openai": ProviderConfig(upstream_base_url="https://api.openai.com"),
    "gemini": ProviderConfig(upstream_base_url="https://generativelanguage.googleapis.com"),
    "cohere": ProviderConfig(upstream_base_url="https://api.cohere.com"),
    # The native Ollama API of a local daemon; its OpenAI-compatible /v1
    # endpoints are covered by pointing [providers.openai] at it instead.
    "ollama": ProviderConfig(upstream_base_url="http://127.0.0.1:11434"),
    # Azure, Vertex, and Bedrock have no sane defaults: the upstream is the
    # customer's own resource/region URL (Bedrock:
    # https://bedrock-runtime.<region>.amazonaws.com). The proxy answers 502
    # on their routes until configured.
    "azure": ProviderConfig(upstream_base_url=""),
    "vertex": ProviderConfig(upstream_base_url=""),
    "bedrock": ProviderConfig(upstream_base_url=""),
}


@dataclass(frozen=True)
class S3AuditConfig:
    # Off-machine copy of audit ROWS (metadata only — same never-values
    # contract as the audit DB). Default off: shipping anywhere is an
    # explicit trust decision, like [otel]. Credentials come from the
    # standard AWS env vars ONLY — never from this file. Restart-only.
    enabled: bool = False
    provider: str = "aws"  # "aws" | "minio" | "ceph" | "gcs"
    bucket: str = ""
    prefix: str = "llm-redact/"
    region: str = "us-east-1"
    # MinIO/Ceph RGW endpoint (path-style addressing); AWS and GCS derive
    # their host from the bucket and take no endpoint. GCS uses its
    # S3-compatible XML API (storage.googleapis.com) with HMAC interop keys.
    endpoint_url: str | None = None
    flush_seconds: float = 60.0
    # "fernet" encrypts each uploaded batch client-side (crypto extra +
    # LLM_REDACT_AUDIT_ENC_KEY, env only). Enabled-without-key refuses
    # startup; a key that disappears at runtime DROPS batches — the sink
    # never falls back to plaintext. `llm-redact audit decrypt` reads
    # downloaded objects back.
    encryption: str = "none"
    # "keys" = the static credential env vars above (the historical form).
    # "identity" = the workload's cloud identity, resolved at runtime by
    # llm-redact-pro (aws: the standard AWS chain incl. IRSA/ECS/IMDSv2
    # temporary credentials, still SigV4; gcs: Application Default
    # Credentials as an OAuth bearer token on the XML API). minio/ceph have
    # no cloud identity, so identity is refused there.
    auth: str = "keys"


@dataclass(frozen=True)
class AzureAuditConfig:
    # Off-machine copy of audit ROWS to Azure Blob Storage (SharedKey auth,
    # NOT an S3 API — its own section). Same metadata-only / never-values
    # contract and off-machine trust decision as [audit.s3]. The account key
    # comes from AZURE_STORAGE_KEY (env only, never this file). Restart-only.
    enabled: bool = False
    account: str = ""
    container: str = ""
    prefix: str = "llm-redact/"
    # Custom blob endpoint (e.g. Azurite); default derives from the account
    # (https://{account}.blob.core.windows.net).
    endpoint_url: str | None = None
    flush_seconds: float = 60.0
    # Client-side batch encryption — same semantics and key as [audit.s3].
    encryption: str = "none"
    # "key" = SharedKey with AZURE_STORAGE_KEY (the historical form);
    # "sas" = a SAS token from AZURE_STORAGE_SAS_TOKEN appended to the blob
    # URL (no signing); "identity" = a Microsoft Entra ID bearer token for
    # the workload identity, resolved at runtime by llm-redact-pro. All
    # secrets are env-only or runtime-resolved, never this file.
    auth: str = "key"


@dataclass(frozen=True)
class AuditConfig:
    # Default off: a privacy tool does not write request history to disk
    # unless asked. In-memory /status totals are always available.
    enabled: bool = False
    path: str | None = None  # default: $XDG_DATA_HOME/llm-redact/audit.db
    max_rows: int = 10000
    # A per-row HMAC hash-chain making deletion/alteration of any row
    # detectable. Off by default; the HMAC key is env-only
    # (LLM_REDACT_AUDIT_HMAC_KEY), never the config file — enabled without it
    # fails closed at startup (a keyless chain an attacker could recompute is
    # worse than none).
    tamper_evident: bool = False
    # Zero-loss mode ("no audit row, no service"): a write-ahead START row is
    # durably committed BEFORE any upstream contact and a request that cannot
    # be recorded is refused 503. Inverts the default fail-open stance —
    # audit storage joins the availability path — so it is an explicit opt-in.
    # Requires enabled = true and a llm-redact-pro version with write-ahead
    # support (checked fail-closed at startup).
    required: bool = False
    s3: S3AuditConfig = field(default_factory=S3AuditConfig)
    azure: AzureAuditConfig = field(default_factory=AzureAuditConfig)


@dataclass(frozen=True)
class RehydrationConfig:
    # Fuzzy matching restores LLM-mangled placeholders («email_001»,
    # «EMAIL-1»). Default on: every mangle is gated on a vault lookup, so
    # the added false-restore surface is limited to prose that canonicalizes
    # to an actually-issued token, while leaked mangled placeholders are the
    # common user-visible failure without it.
    fuzzy: bool = True


# The server-RDBMS vault backends (Pro tier), all spoken through DB-API 2.0
# in vault_rdbms.py. "dbapi" is the any-existing-RDBMS escape hatch: the
# operator names the driver module and the store uses its portable SQL subset.
RDBMS_BACKENDS = ("postgresql", "mysql", "oracle", "dbapi")

# Config sections a hot reload (SIGHUP / the pro dashboard's editor) pins to
# their running values: changing them requires a restart. The single source
# of truth for apply_config and the editor's read-only set (README.md and
# docs/deployment.md enumerate it — pinned by test_restart_only_docs.py).
RESTART_ONLY_KEYS = (
    "vault",
    "audit",
    "host",
    "port",
    "allowed_hosts",
    "allowed_origins",
    "log",
    "tls",
    "otel",
    "users",
    "email",
)

# Every top-level key the core itself parses. Anything else must be claimed
# by a registered plugin section (plugin_api.ConfigSection) or it is a
# ConfigError.
CORE_SECTION_KEYS = frozenset(
    {
        "host",
        "port",
        "allowed_hosts",
        "allowed_origins",
        "inject_system_note",
        "max_body_bytes",
        "max_body_strings",
        "providers",
        "detection",
        "vault",
        "rehydration",
        "audit",
        "log",
        "tls",
        "otel",
        "license",
        "users",
        "email",
        "upstreams",
        "routing",
        "prices",
    }
)


@dataclass(frozen=True)
class RdbmsConfig:
    # URL-form DSN: postgresql://user@host:5432/db, mysql://user@host:3306/db,
    # oracle://user@host:1521/service. backend = "dbapi" passes the string to
    # module.connect() verbatim. LLM_REDACT_VAULT_DSN overrides at startup so
    # credentials never need to live in this file, and the password
    # additionally resolves from the env var named by `password_env`. DSNs
    # are never logged or echoed in errors (they may embed credentials —
    # the ?key= discipline).
    dsn: str = ""
    password_env: str = "LLM_REDACT_VAULT_DB_PASSWORD"
    # backend = "dbapi" only: the DB-API 2.0 module to import.
    module: str = ""
    # Declared managed-DBMS placement ("aws" | "azure" | "gcp"): pointing the
    # vault at a cloud-managed DBMS (RDS/Aurora, Azure Database, Cloud SQL) is
    # fully included in the Pro tier as of 3.11.0 — no separate cloud
    # entitlement. This declaration is informational/posture only; the off-box
    # fernet rule (a non-local DSN must be encrypted) is enforced separately by
    # hostname, independent of this field.
    cloud: str = ""
    # "password" (default): the static password from `password_env` or the
    # DSN. "identity": a short-lived token minted from the proxy's cloud
    # identity (RDS IAM auth, Cloud SQL IAM database auth, Entra ID) at EVERY
    # connect — llm-redact-pro supplies it through the registry's
    # build_db_password seam. Requires `cloud`, postgresql/mysql, TLS, and no
    # static password anywhere (rdbms_identity_error).
    auth: str = "password"
    # auth = "identity" with cloud = "aws" only: the region the RDS token is
    # signed for. Empty = derived from the RDS hostname.
    region: str = ""


# [vault.rdbms] auth = "identity": the backends whose drivers take a token as
# the password, the clouds that mint one, and the libpq sslmodes that would
# let the token cross the network unencrypted.
RDBMS_AUTH_MODES = ("password", "identity")
IDENTITY_AUTH_BACKENDS = ("postgresql", "mysql")
IDENTITY_AUTH_CLOUDS = ("aws", "gcp", "azure")
PG_TLS_SSLMODES = ("require", "verify-ca", "verify-full")
# A cloud region name as the database token is scoped to it (fullmatch only).
_RDBMS_REGION_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def rdbms_identity_error(backend: str, rdbms: RdbmsConfig, dsn: str) -> str | None:
    """Why ``auth = "identity"`` cannot run with this backend and DSN, or None.

    Run at parse time on the file's DSN and again at connect-build time on
    the effective one (``LLM_REDACT_VAULT_DSN`` wins). Messages name the
    rule, never the DSN (it may embed credentials).
    """
    if rdbms.auth != "identity":
        return None
    if backend not in IDENTITY_AUTH_BACKENDS:
        return (
            f'[vault.rdbms] auth = "identity" supports backend = "postgresql" or "mysql",'
            f" not {backend!r}"
        )
    if rdbms.cloud not in IDENTITY_AUTH_CLOUDS:
        return (
            '[vault.rdbms] auth = "identity" requires cloud = "aws", "gcp" or "azure"'
            " (the identity that mints the database token)"
        )
    if rdbms.password_env != RdbmsConfig().password_env:
        return '[vault.rdbms] password_env cannot be combined with auth = "identity"'
    if rdbms.region and rdbms.cloud != "aws":
        return '[vault.rdbms] region applies only to cloud = "aws"'
    if not dsn:
        return None  # the missing-DSN error is raised where the DSN is resolved
    from urllib.parse import parse_qs, urlsplit

    parts = urlsplit(dsn)
    query = parse_qs(parts.query, keep_blank_values=True)
    if parts.password or "password" in query:
        return (
            '[vault.rdbms] auth = "identity" forbids a password in the DSN'
            " (the token minted from the cloud identity is the password)"
        )
    if backend == "postgresql":
        modes = [mode.lower() for mode in query.get("sslmode", [])]
        if any(mode not in PG_TLS_SSLMODES for mode in modes):
            return (
                '[vault.rdbms] auth = "identity" requires TLS: the DSN sslmode must be'
                f" one of {PG_TLS_SSLMODES} (or absent — the proxy then sets require)"
            )
    return None


KMS_PROVIDERS = ("aws", "gcp", "azure", "hashicorp")
KMS_HASHICORP_AUTH = ("token", "kubernetes")
DEFAULT_SERVICE_ACCOUNT_TOKEN_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/token"
# Keys only a HashiCorp Vault transit key uses; any other provider refuses them.
_KMS_HASHICORP_KEYS = frozenset(
    {"address", "mount", "auth", "role", "auth_mount", "service_account_token_file"}
)
_KMS_KUBERNETES_KEYS = frozenset({"role", "auth_mount", "service_account_token_file"})
_KMS_KEY_ID_SHAPES = {
    # A key or alias ARN: the region the request goes to comes from it.
    "aws": re.compile(r"arn:aws(?:-[a-z]+)*:kms:[a-z0-9-]+:\d{12}:(?:key|alias)/[A-Za-z0-9/_:.-]+"),
    "gcp": re.compile(r"projects/[^/\s]+/locations/[^/\s]+/keyRings/[^/\s]+/cryptoKeys/[^/\s]+"),
    "azure": re.compile(
        r"https://[A-Za-z0-9-]+\.vault\.azure\.net/keys/[A-Za-z0-9-]+(?:/[A-Za-z0-9]+)?",
        re.IGNORECASE,
    ),
    "hashicorp": re.compile(r"[A-Za-z0-9_.-]+"),
}
_KMS_KEY_ID_HINTS = {
    "aws": "a key or alias ARN (arn:aws:kms:<region>:<account>:key/<id> or …:alias/<name>)",
    "gcp": "projects/<p>/locations/<l>/keyRings/<r>/cryptoKeys/<k>",
    "azure": "a key URL https://<vault>.vault.azure.net/keys/<name>[/<version>]",
    "hashicorp": "a transit key name (letters, digits, '_', '.', '-')",
}
_ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MOUNT_RE = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")
# The plaintext key sources [vault.kms] replaces (vault_crypto.ENV_KEY /
# CMD_ENV_KEY, spelled out here: vault_crypto imports the vault module).
_LOCAL_VAULT_KEY_ENVS = ("LLM_REDACT_VAULT_KEY", "LLM_REDACT_VAULT_KEY_CMD")


@dataclass(frozen=True)
class VaultKmsConfig:
    """``[vault.kms]``: the fernet vault key is stored WRAPPED by a cloud KMS
    and unwrapped at startup with the proxy's own identity (llm-redact-pro).

    With this table present the key comes ONLY from the KMS — never from
    LLM_REDACT_VAULT_KEY, the key command or the keychain. The wrapped value
    is ciphertext; it still never lives in this file (an env var or a file
    path names where it is)."""

    provider: str  # one of KMS_PROVIDERS
    # aws: key/alias ARN; gcp: projects/…/cryptoKeys/…; azure: key URL;
    # hashicorp: transit key name. Not secret.
    key_id: str
    # Exactly one: the env var holding the base64 wrapped key, or a file.
    wrapped_key_env: str = ""
    wrapped_key_file: str = ""
    # --- hashicorp only ---
    # Vault server; empty = the VAULT_ADDR env var at startup. https unless
    # the host is loopback (the same rule either way).
    address: str = ""
    mount: str = "transit"  # the transit secrets-engine mount path
    auth: str = "token"  # "token" (VAULT_TOKEN / VAULT_TOKEN_FILE) | "kubernetes"
    role: str = ""  # kubernetes auth role (required with auth = "kubernetes")
    auth_mount: str = "kubernetes"  # the kubernetes auth method's mount path
    service_account_token_file: str = DEFAULT_SERVICE_ACCOUNT_TOKEN_FILE


def plugin_capabilities_required(config: "Config") -> list[str]:
    """The config SHAPES only a recent llm-redact-pro implements and that have
    no factory seam of their own to fail closed in: an older plugin would
    parse past them and silently fall back to the static credentials. Each
    name must appear in ``Registry.config_capabilities`` (advertised by the
    plugin) — checked by ProxyState and doctor when a plugin is loaded."""
    required: list[str] = []
    s3, azure, email = config.audit.s3, config.audit.azure, config.email
    if s3.enabled and s3.auth != S3AuditConfig().auth:
        required.append(f"audit.s3.auth={s3.auth}")
    if azure.enabled and azure.auth != AzureAuditConfig().auth:
        required.append(f"audit.azure.auth={azure.auth}")
    if email.auth != EmailConfig().auth:
        required.append(f"email.auth={email.auth}")
    if email.implicit_tls:
        required.append("email.implicit_tls")
    return required


def unsupported_plugin_capabilities(config: "Config", advertised: Iterable[str]) -> str | None:
    """A ConfigError message naming the configured capabilities the loaded
    plugin does not advertise, else None."""
    missing = sorted(set(plugin_capabilities_required(config)) - set(advertised))
    if not missing:
        return None
    return (
        "the installed llm-redact-pro does not support " + ", ".join(missing) + ";"
        " upgrade llm-redact-pro (an older version would silently use the static"
        " credentials instead)"
    )


def sink_endpoint_problem(url: str, *, allow_path: bool, require_https: bool) -> str | None:
    """Why an audit sink ``endpoint_url`` is unusable, else None: an http(s)
    URL with a host, no userinfo, query or fragment, a path only where the
    sink signs one, and https (unless the host is loopback) when a bearer
    secret rides the request."""
    try:
        parts = urllib.parse.urlsplit(url)
        _ = parts.port  # a malformed port raises
    except ValueError:
        return "must look like https://host:port"
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "must start with http:// or https:// and name a host"
    if "@" in parts.netloc or parts.query or parts.fragment:
        return "must not carry userinfo, a query string or a fragment"
    if not allow_path and parts.path not in ("", "/"):
        return "must not carry a path (requests are signed over /bucket/key)"
    if require_https and parts.scheme == "http" and not _is_loopback_host(parts.hostname):
        return "must be https unless the host is loopback (a bearer credential rides it)"
    return None


def kms_address_problem(address: str) -> str | None:
    """Why a HashiCorp Vault address is unusable, else None: an http(s) URL
    with a host and no query/fragment, https unless the host is loopback (a
    token and the vault key would otherwise cross the network in the clear).
    Shared by the parser and llm-redact-pro's runtime VAULT_ADDR check."""
    from urllib.parse import urlsplit

    parts = urlsplit(address)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "must look like https://vault.example:8200"
    if parts.query or parts.fragment:
        return "must not carry a query string or fragment"
    if parts.scheme == "http" and not _is_loopback_host(parts.hostname):
        return "must be https unless the host is loopback"
    return None


def parse_vault_kms(
    raw: Any, *, where: str = "[vault.kms]", require_wrapped: bool = True
) -> VaultKmsConfig:
    """Validate a ``[vault.kms]`` table. ``require_wrapped=False`` is the
    wrapping CLI's form (it produces the wrapped value, so it names none).
    Error messages name keys and expected shapes, never a value."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where} must be a table")
    _require_keys(
        dict(raw),
        {"provider", "key_id", "wrapped_key_env", "wrapped_key_file", *_KMS_HASHICORP_KEYS},
        where,
    )
    for key, value in raw.items():
        if not isinstance(value, str):
            raise ConfigError(f"{where} {key} must be a string")
    provider = raw.get("provider", "")
    if provider not in KMS_PROVIDERS:
        raise ConfigError(f"{where} provider must be one of {KMS_PROVIDERS}")
    key_id = raw.get("key_id", "")
    if _KMS_KEY_ID_SHAPES[provider].fullmatch(key_id) is None:
        raise ConfigError(
            f"{where} key_id for provider {provider!r} must be {_KMS_KEY_ID_HINTS[provider]}"
        )
    wrapped_env = raw.get("wrapped_key_env", "")
    wrapped_file = raw.get("wrapped_key_file", "")
    if require_wrapped and bool(wrapped_env) == bool(wrapped_file):
        raise ConfigError(f"{where} needs exactly one of wrapped_key_env or wrapped_key_file")
    if wrapped_env and _ENV_NAME_RE.fullmatch(wrapped_env) is None:
        raise ConfigError(f"{where} wrapped_key_env must be an environment variable name")
    if wrapped_env in _LOCAL_VAULT_KEY_ENVS:
        raise ConfigError(
            f"{where} wrapped_key_env must not be {wrapped_env}: that variable is the"
            " plaintext key source [vault.kms] replaces (use e.g. LLM_REDACT_VAULT_KEY_WRAPPED)"
        )
    if provider == "hashicorp":
        return _parse_kms_hashicorp(raw, where, key_id, wrapped_env, wrapped_file)
    hashicorp_keys = sorted(_KMS_HASHICORP_KEYS & set(raw))
    if hashicorp_keys:
        raise ConfigError(f"{where} {hashicorp_keys} apply only to provider = 'hashicorp'")
    return VaultKmsConfig(
        provider=provider,
        key_id=key_id,
        wrapped_key_env=wrapped_env,
        wrapped_key_file=wrapped_file,
    )


def _parse_kms_hashicorp(
    raw: Mapping[str, Any], where: str, key_id: str, wrapped_env: str, wrapped_file: str
) -> VaultKmsConfig:
    defaults = VaultKmsConfig(provider="hashicorp", key_id=key_id)
    address = str(raw.get("address", "")).rstrip("/")
    if address:
        problem = kms_address_problem(address)
        if problem is not None:
            raise ConfigError(f"{where} address {problem}")
    auth = str(raw.get("auth", defaults.auth))
    if auth not in KMS_HASHICORP_AUTH:
        raise ConfigError(f"{where} auth must be one of {KMS_HASHICORP_AUTH}")
    mount = str(raw.get("mount", defaults.mount))
    auth_mount = str(raw.get("auth_mount", defaults.auth_mount))
    for key, value in (("mount", mount), ("auth_mount", auth_mount)):
        if _MOUNT_RE.fullmatch(value) is None:
            raise ConfigError(f"{where} {key} must be a mount path like 'transit' (no leading '/')")
    role = str(raw.get("role", ""))
    token_file = str(raw.get("service_account_token_file", defaults.service_account_token_file))
    if auth == "kubernetes":
        if not role:
            raise ConfigError(f"{where} auth = 'kubernetes' requires role")
        if not token_file:
            raise ConfigError(f"{where} service_account_token_file must not be empty")
    else:
        kube_only = sorted(_KMS_KUBERNETES_KEYS & set(raw))
        if kube_only:
            raise ConfigError(f"{where} {kube_only} apply only to auth = 'kubernetes'")
    return VaultKmsConfig(
        provider="hashicorp",
        key_id=key_id,
        wrapped_key_env=wrapped_env,
        wrapped_key_file=wrapped_file,
        address=address,
        mount=mount,
        auth=auth,
        role=role,
        auth_mount=auth_mount,
        service_account_token_file=token_file,
    )


@dataclass(frozen=True)
class VaultConfig:
    # "memory" is the deliberate default: persisting real secrets to disk
    # must be an explicit opt-in for a privacy tool.
    backend: str = "memory"
    path: str | None = None  # default: $XDG_DATA_HOME/llm-redact/vault.db
    session: str = "default"
    # "static": one namespace named by `session` (default). "per-conversation":
    # each conversation gets its own namespace derived from its first user
    # message; `session` then names only the fallback for anchor-less traffic.
    session_mode: str = "static"
    # "fernet" encrypts stored originals (requires the crypto extra and
    # LLM_REDACT_VAULT_KEY): at rest for the sqlite backend, in RAM for the
    # memory backend. Sqlite migration to encrypted is one-way.
    encryption: str = "none"
    # Retention: whole sessions idle longer than this many days are pruned by
    # a background task (bounds unbounded per-conversation growth). 0 (default)
    # disables auto-prune. Whole-session-only deletion, never the active
    # session — the same never-wrong-value discipline as the CLI prune.
    session_ttl_days: int = 0
    # Connection settings for the RDBMS_BACKENDS (ignored otherwise).
    rdbms: RdbmsConfig = field(default_factory=RdbmsConfig)
    # [vault.kms]: the fernet key is KMS-wrapped (llm-redact-pro unwraps it
    # at startup). None = the local sources (env / key command / keychain).
    kms: VaultKmsConfig | None = None


@dataclass(frozen=True)
class LogConfig:
    # "text" (human-readable, the default) or "json" (one object per line,
    # for log shippers). Log content is identical either way: paths,
    # statuses, and detection counts — never values or headers.
    format: str = "text"


@dataclass(frozen=True)
class OtelConfig:
    # OpenTelemetry export (requires the `otel` extra). Emitted telemetry is
    # metadata-only, the same contract as audit/recent rows: paths, statuses,
    # durations, and detection counts — never values or headers. Restart-only.
    enabled: bool = False
    # OTLP/HTTP base endpoint (e.g. "http://127.0.0.1:4318"); when unset the
    # SDK's OTEL_EXPORTER_OTLP_* environment variables apply.
    endpoint: str | None = None
    service_name: str = "llm-redact"


# [users] unrecorded_objects: the shape of llm-redact-pro's unknown-owner
# policy (docs/authentication.md there). "refuse" (default): under a
# credential the proxy holds, a named user's reference to a stored object no
# user is recorded creating is refused. "allow": forwarded, and read sealed.
UNRECORDED_OBJECT_POLICIES = ("refuse", "allow")


@dataclass(frozen=True)
class UsersConfig:
    # Named-user registry database (Pro+ tiers; llm-redact-pro docs/licensing.md).
    # Default: $XDG_DATA_HOME/llm-redact/users.db. Restart-only.
    path: str | None = None
    # One of UNRECORDED_OBJECT_POLICIES; implemented by llm-redact-pro only.
    unrecorded_objects: str = "refuse"


@dataclass(frozen=True)
class EmailConfig:
    # Operator-run SMTP for user-verification emails (stdlib smtplib —
    # nothing leaves the operator's own infrastructure, no vendor service).
    # The SMTP password comes from the env var named by password_env,
    # NEVER this file. Restart-only.
    smtp_host: str | None = None
    smtp_port: int = 587
    starttls: bool = True
    username: str | None = None
    password_env: str = "LLM_REDACT_SMTP_PASSWORD"
    from_address: str | None = None
    # implicit_tls: TLS from the first byte (SMTP_SSL, usually port 465)
    # instead of STARTTLS; the two are mutually exclusive.
    implicit_tls: bool = False
    # auth: "password" (SMTP AUTH LOGIN/PLAIN with password_env) or "oauth"
    # (SASL XOAUTH2 bearer token, llm-redact-pro docs/email-oauth.md; TLS
    # required). oauth_provider picks the token source: "azure" (Entra ID
    # workload identity, Microsoft 365), "google" (service account with
    # domain-wide delegation; oauth_subject = the mailbox, default
    # username/from_address) or "refresh_token" (a generic RFC 6749
    # refresh-token grant against oauth_token_url, https only). Secrets are
    # named by *_env keys and read from the environment, never this file.
    auth: str = "password"
    oauth_provider: str | None = None
    oauth_subject: str | None = None
    oauth_token_url: str | None = None
    oauth_client_id: str | None = None
    oauth_client_secret_env: str | None = None
    oauth_refresh_token_env: str | None = None
    oauth_scope: str | None = None

    @property
    def configured(self) -> bool:
        return self.smtp_host is not None and self.from_address is not None

    @property
    def tls(self) -> str:
        """The transport-security mode: "implicit", "starttls" or "none"."""
        if self.implicit_tls:
            return "implicit"
        return "starttls" if self.starttls else "none"


@dataclass(frozen=True)
class LicenseConfig:
    # [license]: the signed feature-tier key (llm-redact-pro docs/licensing.md). Resolution
    # order: LLM_REDACT_LICENSE_KEY env var > key > key_file. Absent or
    # invalid resolves to the FREE tier (loud warning, never silent) and any
    # configured above-tier feature then refuses startup by name.
    key: str | None = None
    key_file: str | None = None


@dataclass(frozen=True)
class TlsConfig:
    # Server TLS (certfile + keyfile, always together) and — required for
    # any non-loopback bind — mutual TLS (client_ca: connecting clients
    # must present a certificate this CA signed). Restart-only.
    certfile: str | None = None
    keyfile: str | None = None
    client_ca: str | None = None

    @property
    def enabled(self) -> bool:
        return self.certfile is not None

    @property
    def mutual(self) -> bool:
        return self.client_ca is not None


DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MiB: covers 200k-token bodies
# plus image blocks while bounding the memory the proxy buffers per request.
# Redaction costs per STRING as well as per byte, and runs on the event loop:
# the strings (JSON string values, form fields, file names, uploaded JSONL
# lines) and multipart parts one redactable body may carry. A 1M-token Claude
# Code history is ~12k strings; 10 MiB of tiny strings would be ~300k-2M.
DEFAULT_MAX_BODY_STRINGS = 100_000


# --- [upstreams] / [routing] / [prices] shapes -------------------------------
#
# Rule-based upstream routing is a paid subsystem implemented in the
# llm-redact-pro package; the core keeps only the config SHAPES (these
# constants and dataclasses, the parser below and the emitter in
# config_write.py) so `serve --check`, `config show`, doctor and the
# (llm-redact-pro) config editor validate the same schema with or without
# the package. Every
# routing DECISION — rule selection, credentials, chains, cooldowns,
# plan-limit detection, budgets, prices — lives in that package.

# The four wire formats a rule can match. Every other adapter/provider (azure,
# vertex, bedrock, cohere, custom:*) keeps the legacy one-upstream-per-provider
# path untouched.
PROTOCOLS: tuple[str, ...] = ("anthropic", "openai", "gemini", "ollama")

# The anthropic-beta capability token Claude Code sends on subscription (OAuth)
# traffic; its presence together with a Bearer authorization classifies the
# request as `auth = "oauth"` (R-6). Overridable via routing.oauth_beta_marker.
OAUTH_BETA_MARKER = "oauth-2025-04-20"

# Anthropic's unified rate-limit response headers whose listed values mark an
# exhausted subscription window (plan-limit) rather than a throttle (R-20).
# A config override replaces this table entirely.
DEFAULT_PLAN_LIMIT_HEADERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic-ratelimit-unified-status", ("rejected",)),
    ("anthropic-ratelimit-unified-5h-status", ("rejected",)),
    ("anthropic-ratelimit-unified-7d-status", ("rejected",)),
)

# Inbound headers that carry the client's own credential. They are forwarded
# byte-exact to a passthrough upstream and DROPPED for none/env upstreams
# (I-3: a server-held key never travels beside the client's).
CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key"})

SPECIAL_STATUS_KEYS = ("plan_limit_429", "throttle_429")
CLASS_KEYS = ("4xx", "5xx")
RETRY_SAME = "retry-same"
AUTH_KINDS = ("oauth", "gateway-key", "none", "any")
CREDENTIAL_MODES = ("passthrough", "none", "env")
COSTS = ("metered", "zero")
REISSUE_POLICIES = ("stateless-only", "always", "never")
PLAN_LIMIT_DETECTION = ("headers", "off")


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens. The single definition; pricing.py imports it."""

    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(frozen=True)
class UpstreamConfig:
    name: str
    protocol: str  # one of PROTOCOLS
    base_url: str  # rstripped "/"
    credential: str = "passthrough"  # "passthrough" | "none" | "env:VAR"
    cost: str = "metered"  # "metered" | "zero"
    # Resolved at parse: False for passthrough (I-4), else the top-level
    # inject_system_note. The note decision is made once, for the FIRST
    # upstream of a request (the redacted body is reused on later hops).
    inject_system_note: bool = False
    # False for upstreams without /v1/messages/count_tokens (Ollama hangs on
    # it): the proxy answers 404 locally, no fallback (decision 7).
    count_tokens: bool = True
    monthly_budget_usd: float | None = None
    monthly_budget_tokens: int | None = None
    cooldown_seconds: float = 60.0
    extra_headers: tuple[tuple[str, str], ...] = ()  # (lowercased name, value), sorted
    body_defaults_json: str = "{}"  # canonical json.dumps(sort_keys=True) of the table
    # Auto-registered from [providers.NAME] when a [routing] table exists
    # (R-1); never emitted by the config writer — it re-registers.
    legacy: bool = False

    @property
    def credential_mode(self) -> str:
        if self.credential.startswith("env:"):
            return "env"
        return self.credential

    @property
    def env_var(self) -> str | None:
        if self.credential.startswith("env:"):
            return self.credential[len("env:") :]
        return None

    @property
    def is_passthrough(self) -> bool:
        return self.credential == "passthrough"

    @property
    def zero_cost(self) -> bool:
        return self.cost == "zero"

    @property
    def has_budget(self) -> bool:
        # Zero-cost upstreams ignore budgets even when one is configured
        # (parse rejects the combination on passthrough; zero-cost + budget
        # is allowed but inert).
        if self.zero_cost:
            return False
        return self.monthly_budget_usd is not None or self.monthly_budget_tokens is not None

    def body_defaults(self) -> dict[str, Any]:
        # A fresh object per call: callers merge it into request bodies and
        # must never share nested containers between requests.
        loaded = json.loads(self.body_defaults_json)
        return dict(loaded) if isinstance(loaded, dict) else {}


@dataclass(frozen=True)
class RuleMatch:
    protocol: str
    models: tuple[str, ...] = ()  # fnmatchcase globs; () = any
    headers: tuple[tuple[str, str], ...] = ()  # (lowercased name, glob) all must match
    path: str | None = None  # glob on the inbound path
    auth: str = "any"


@dataclass(frozen=True)
class RouteRule:
    id: str
    match: RuleMatch
    upstream: str
    model_rewrite: str | None = None
    # key -> chain | RETRY_SAME, in file order. Keys: an exact status
    # ("529"), a class ("4xx"/"5xx"), or a special key (SPECIAL_STATUS_KEYS).
    # The status -> chain precedence ladder that reads this table is routing
    # POLICY and lives in llm-redact-pro (resolve_action / chain_for).
    on_status: tuple[tuple[str, tuple[str, ...] | str], ...] = ()
    reissue_policy: str = "stateless-only"  # resolved default per R-8
    on_budget_exhausted: tuple[str, ...] = ()


@dataclass(frozen=True)
class PricesConfig:
    table: str = "builtin"  # or a file path
    overrides: tuple[tuple[str, ModelPrice], ...] = ()  # sorted by model id


@dataclass(frozen=True)
class RoutingConfig:
    enabled: bool = False
    default_upstreams: tuple[tuple[str, str], ...] = ()  # (protocol, upstream) sorted
    max_hops: int = 3
    request_deadline_seconds: float = 600.0
    plan_limit_detection: str = "headers"  # "headers" | "off"
    plan_limit_headers: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_PLAN_LIMIT_HEADERS
    oauth_beta_marker: str = OAUTH_BETA_MARKER
    throttle_retry_max_seconds: float = 30.0
    budget_reset_day: int = 1
    debug_headers: bool = False
    expose_models: bool = False
    model_catalog: tuple[str, ...] = ()
    upstreams: tuple[UpstreamConfig, ...] = ()  # sorted by name
    rules: tuple[RouteRule, ...] = ()  # file order
    warnings: tuple[str, ...] = ()  # I-6 etc., surfaced not raised
    # True iff the TOML had a [routing] table (legacy upstreams register only
    # then; `enabled = false` is still "present").
    present: bool = False

    def upstream(self, name: str) -> UpstreamConfig:
        for upstream in self.upstreams:
            if upstream.name == name:
                return upstream
        raise KeyError(name)

    def upstream_names(self) -> tuple[str, ...]:
        return tuple(upstream.name for upstream in self.upstreams)

    def default_for(self, protocol: str) -> str | None:
        for proto, name in self.default_upstreams:
            if proto == protocol:
                return name
        return None


@dataclass(frozen=True)
class Config:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    # More host names the proxy answers to besides 127.0.0.1, localhost, ::1
    # and `host` — a compose service or Kubernetes Service name its clients
    # use. A request that would spend a credential the proxy holds (or that a
    # browser sent) must be addressed to one of these names: the DNS-rebinding
    # defense. Lowercased and sorted, no ports; restart-only like the bind.
    allowed_hosts: tuple[str, ...] = ()
    # Browser origins (scheme://host[:port], serialized) whose pages the proxy
    # serves although they are cross-origin — a web chat UI the operator
    # trusts. OPT-IN and default empty: a listed origin can read restored
    # values back through the proxy and spend any credential it holds.
    # Their Host must still be a name the proxy answers to; restart-only.
    allowed_origins: tuple[str, ...] = ()
    inject_system_note: bool = True
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    max_body_strings: int = DEFAULT_MAX_BODY_STRINGS
    providers: dict[str, ProviderConfig] = field(default_factory=lambda: dict(DEFAULT_PROVIDERS))
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    vault: VaultConfig = field(default_factory=VaultConfig)
    rehydration: RehydrationConfig = field(default_factory=RehydrationConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    log: LogConfig = field(default_factory=LogConfig)
    tls: TlsConfig = field(default_factory=TlsConfig)
    otel: OtelConfig = field(default_factory=OtelConfig)
    license: LicenseConfig = field(default_factory=LicenseConfig)
    users: UsersConfig = field(default_factory=UsersConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    # [upstreams] + [routing] (rule-based upstream selection, fallback chains,
    # budgets — the llm-redact-pro routing guide) and [prices] (the budget price
    # table). All three reload on SIGHUP; a config without them behaves exactly
    # as before.
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    prices: PricesConfig = field(default_factory=PricesConfig)
    # Plugin-owned top-level tables (plugin_api.ConfigSection), keyed by
    # section name and holding the plugin's parsed value; only sections
    # present in the file appear. Restart-only as a whole.
    extensions: dict[str, Any] = field(default_factory=dict)


class ConfigError(ValueError):
    pass


_CUSTOM_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,31}\Z")


def _add_custom_provider(
    providers: "dict[str, ProviderConfig]", name: str, section: object
) -> None:
    """Validate one custom OpenAI-compatible upstream into the map.

    Names are lowercase [a-z0-9-] (they become URL path segments, TOML
    keys, and metrics labels); the upstream URL is REQUIRED — there is no
    sane default for a user-named backend.
    """
    if not _CUSTOM_NAME_RE.fullmatch(name):
        raise ConfigError(f"[providers.custom] name {name!r} must match [a-z0-9][a-z0-9-]{{0,31}}")
    if not isinstance(section, dict):
        raise ConfigError(f"[providers.custom.{name}] must be a table")
    _require_keys(
        section, {"upstream_base_url", "enabled", "detection"}, f"[providers.custom.{name}]"
    )
    url = str(section.get("upstream_base_url", "")).rstrip("/")
    if not url:
        raise ConfigError(f"[providers.custom.{name}] upstream_base_url is required")
    providers[f"custom:{name}"] = ProviderConfig(
        upstream_base_url=url,
        enabled=_bool_key(section, "enabled", True, f"[providers.custom.{name}]"),
        detection=_bool_key(section, "detection", True, f"[providers.custom.{name}]"),
    )


def _provider_auth(name: str, section: Mapping[str, Any]) -> tuple[str, str | None]:
    """``auth`` and ``region`` of one built-in ``[providers.NAME]`` table.

    ``identity`` is only meaningful where the proxy knows the cloud's
    authorization scheme (IDENTITY_AUTH_PROVIDERS); anywhere else it would
    strip the client's key and send nothing — refused. ``region`` feeds
    the SigV4 credential scope, so it is bedrock-with-identity only (an
    inert region elsewhere would read as if it did something).
    """
    where = f"[providers.{name}]"
    auth = _str_key(section, "auth", "passthrough", where)
    if auth not in PROVIDER_AUTH_MODES:
        raise ConfigError(f'{where} auth must be "passthrough" or "identity", got {auth!r}')
    if auth == "identity" and name not in IDENTITY_AUTH_PROVIDERS:
        raise ConfigError(
            f'{where} auth = "identity" is supported only for'
            f" {', '.join(sorted(IDENTITY_AUTH_PROVIDERS))} (the proxy signs or authorizes"
            " those requests with its own cloud identity)"
        )
    if "region" not in section:
        return auth, None
    region = _str_key(section, "region", "", where)
    if name != "bedrock" or auth != "identity":
        raise ConfigError(
            f'{where} region is only used with [providers.bedrock] auth = "identity"'
            " (the SigV4 signing region)"
        )
    if not _REGION_RE.fullmatch(region):
        raise ConfigError(f"{where} region must look like an AWS region (e.g. us-east-1)")
    return auth, region


def identity_upstream_problem(url: str) -> str | None:
    """Why ``url`` cannot carry the proxy's own cloud credential, or None.

    Under ``auth = "identity"`` every forwarded request carries the proxy's
    SigV4 signature or OAuth bearer token, so the upstream must be https
    (wss for realtime, derived by scheme swap) unless it is loopback — a
    cleartext hop would hand the credential to the network. An unset URL
    (the provider answers 502 until configured) is not a problem here.
    """
    if not url:
        return None
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" or (
        parts.scheme == "http" and _is_loopback_host(parts.hostname or "")
    ):
        return None
    return (
        'upstream_base_url must be https with auth = "identity" unless the host is loopback'
        " (every request carries the proxy's own cloud credential)"
    )


def default_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg) / "llm-redact" / "config.toml"


# Standard system path; also the documented container config mount point.
ETC_CONFIG_PATH = Path("/etc/llm-redact/config.toml")


def apply_env_overrides(config: Config) -> Config:
    """Apply LLM_REDACT_HOST / LLM_REDACT_PORT on top of file/default config.

    Precedence overall: CLI flag > environment variable > config file >
    built-in default. (LLM_REDACT_CONFIG is handled in load_config.)
    """
    host = os.environ.get("LLM_REDACT_HOST")
    if host:
        config = dataclasses.replace(config, host=host)
    port_raw = os.environ.get("LLM_REDACT_PORT")
    if port_raw:
        try:
            config = dataclasses.replace(config, port=int(port_raw))
        except ValueError as exc:
            raise ConfigError(f"LLM_REDACT_PORT is not an integer: {port_raw!r}") from exc
    return config


def _require_keys(section: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} in {where}")


def _bool_key(section: Mapping[str, Any], key: str, default: bool, where: str) -> bool:
    """A boolean config key, rejected unless it is a real TOML boolean.

    `bool(raw.get(...))` was the 3.2.0 behavior and it coerced by truthiness:
    `enabled = "false"` (a quoted string) meant TRUE. For a privacy tool that
    can flip audit-to-disk or a detection toggle to the opposite of what the
    user wrote, so a wrong type is a hard error, never a coercion.
    """
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{where} {key} must be a boolean (true/false, unquoted), got {value!r}")
    return value


def _str_list(
    section: Mapping[str, Any], key: str, default: tuple[str, ...], where: str
) -> tuple[str, ...]:
    """A list-of-strings config key, rejected unless it is exactly that.

    `tuple("^192.168")` iterates a string into single characters, and the
    one-character allowlist patterns `.` / `^` match EVERY value — a config
    typo that silently disabled all redaction in 3.2.0 while posture reported
    full protection. The value is never echoed (allowlist/deny-adjacent keys
    can hold sensitive strings); only its type is named.
    """
    if key not in section:
        return default
    value = section[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(
            f'{where} {key} must be an array of strings — e.g. {key} = ["..."]'
            f" (got {type(value).__name__})"
        )
    if any(not item for item in value):
        raise ConfigError(f"{where} {key} entries must be non-empty strings")
    return tuple(value)


# A host name in allowed_hosts: a DNS or container name (letters, digits,
# "-", "_", inner "."), checked after lowercasing — never a scheme, port,
# path, userinfo or wildcard. IP literals are accepted separately.
_ALLOWED_HOST_RE = re.compile(r"[a-z0-9_](?:[a-z0-9_.-]*[a-z0-9_])?\Z")


def _parse_allowed_hosts(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """``allowed_hosts``: more host names the proxy answers to, compared
    with a request's Host the way ``_host_allowed`` compares its own names
    (lowercased, no port; an IPv6 literal in its compressed form, brackets
    optional). A malformed entry is named by its position, never echoed."""
    if "allowed_hosts" not in raw:
        return ()
    value = raw["allowed_hosts"]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(
            'allowed_hosts must be an array of host names — e.g. allowed_hosts = ["llm-redact"]'
        )
    names: set[str] = set()
    for position, item in enumerate(value, 1):
        name = item.strip().lower()
        literal = name[1:-1] if name.startswith("[") and name.endswith("]") else name
        try:
            names.add(str(ipaddress.ip_address(literal)))
            continue
        except ValueError:
            pass
        if not _ALLOWED_HOST_RE.match(name) or ".." in name:
            raise ConfigError(
                f"allowed_hosts entry {position} is not a host name: list each name clients"
                " use exactly, without a scheme, port, path or wildcard"
            )
        names.add(name)
    return tuple(sorted(names))


def normalize_origin(value: str) -> str | None:
    """``value`` as a serialized web origin — ``scheme://host[:port]``: http
    or https, the host lowercased (an IPv6 literal bracketed and compressed),
    the scheme's default port left out — or None when it is not one: the
    opaque ``null`` origin, another scheme, a path (a lone ``/`` aside),
    query, fragment, user info, wildcard, non-ASCII host or bad port. The
    form browsers send in ``Origin``, so a listed origin and a request's
    ``Origin`` compare equal exactly when they name the same origin."""
    text = value.strip()
    if not text.isascii():
        return None  # a browser sends the ASCII (punycode) form
    try:
        parsed = urllib.parse.urlsplit(text)
        port = parsed.port
    except ValueError:  # a malformed port or IPv6 literal
        return None
    host = parsed.hostname
    if (
        parsed.scheme not in ("http", "https")
        or not host
        or port == 0
        or "@" in parsed.netloc
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if not _ALLOWED_HOST_RE.match(host) or ".." in host:
            return None  # a wildcard, a stray character, an empty label
    else:
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    default_port = 443 if parsed.scheme == "https" else 80
    return f"{parsed.scheme}://{host}" + ("" if port in (None, default_port) else f":{port}")


def _loopback_origin_host(origin: str) -> bool:
    """Whether a serialized origin's host is this machine: ``localhost``, a
    name under ``.localhost`` (browsers resolve those to loopback
    themselves), or a loopback address."""
    host = urllib.parse.urlsplit(origin).hostname or ""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _parse_allowed_origins(raw: Mapping[str, Any]) -> tuple[str, ...]:
    """``allowed_origins``: the browser origins the proxy serves across
    origins, in their serialized form (``normalize_origin``). A plain-http
    origin must be this machine: any host on the network path could serve a
    page as a remote http origin. A malformed entry is named by its
    position, never echoed."""
    if "allowed_origins" not in raw:
        return ()
    value = raw["allowed_origins"]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(
            "allowed_origins must be an array of web origins — e.g."
            ' allowed_origins = ["https://chat.example.com"]'
        )
    origins: set[str] = set()
    for position, item in enumerate(value, 1):
        origin = normalize_origin(item)
        if origin is None:
            raise ConfigError(
                f"allowed_origins entry {position} is not a web origin: write scheme://host[:port]"
                " with http or https and an ASCII host — no path, query, user info or wildcard"
                ' (the opaque "null" origin can never be listed)'
            )
        if origin.startswith("http:") and not _loopback_origin_host(origin):
            raise ConfigError(
                f"allowed_origins entry {position} is a plain-http origin on another machine:"
                " anyone on the network path can serve a page as that origin — use https"
                " (http is accepted for localhost and loopback addresses only)"
            )
        origins.add(origin)
    return tuple(sorted(origins))


# Object-key prefixes stay in the URI-unreserved set so the SigV4 canonical
# URI never needs percent-encoding (a wrong encoding is a signature failure,
# which the sink can only WARN-and-drop on).
_S3_PREFIX_RE = re.compile(r"[A-Za-z0-9._/-]*\Z")

# How each off-machine audit sink authenticates (credentials are env-only or
# resolved at runtime from the workload identity — never this file).
S3_AUTH_MODES = ("keys", "identity")
AZURE_AUTH_MODES = ("key", "sas", "identity")


def _parse_audit_s3(s3_raw: object) -> S3AuditConfig:
    if not isinstance(s3_raw, dict):
        raise ConfigError("[audit.s3] must be a table")
    _require_keys(
        s3_raw,
        {
            "enabled",
            "provider",
            "bucket",
            "prefix",
            "region",
            "endpoint_url",
            "flush_seconds",
            "encryption",
            "auth",
        },
        "[audit.s3]",
    )
    s3_encryption = str(s3_raw.get("encryption", "none"))
    if s3_encryption not in ("none", "fernet"):
        raise ConfigError(
            f"[audit.s3] encryption must be 'none' or 'fernet', got {s3_encryption!r}"
        )
    default = S3AuditConfig()
    provider = str(s3_raw.get("provider", default.provider))
    if provider not in ("aws", "minio", "ceph", "gcs"):
        # Azure Blob (SharedKey, not an S3 API) lives in its own [audit.azure]
        # section, not here.
        raise ConfigError(
            f"[audit.s3] provider must be 'aws', 'minio', 'ceph' or 'gcs', got {provider!r}"
        )
    enabled = _bool_key(s3_raw, "enabled", False, "[audit.s3]")
    bucket = str(s3_raw.get("bucket", ""))
    prefix = str(s3_raw.get("prefix", default.prefix))
    if not _S3_PREFIX_RE.fullmatch(prefix):
        raise ConfigError("[audit.s3] prefix must contain only [A-Za-z0-9._/-] characters")
    endpoint_url = str(s3_raw["endpoint_url"]).rstrip("/") if "endpoint_url" in s3_raw else None
    flush_seconds = float(s3_raw.get("flush_seconds", default.flush_seconds))
    if flush_seconds <= 0:
        raise ConfigError("[audit.s3] flush_seconds must be positive")
    if enabled and not bucket:
        raise ConfigError("[audit.s3] bucket is required when enabled")
    if provider in ("minio", "ceph"):
        if enabled and not endpoint_url:
            raise ConfigError(f"[audit.s3] endpoint_url is required for provider {provider!r}")
        if endpoint_url is not None:
            # The request is signed over /bucket/key and the endpoint's host:
            # a path, userinfo or query would be sent but never signed.
            problem = sink_endpoint_problem(endpoint_url, allow_path=False, require_https=False)
            if problem is not None:
                raise ConfigError(f"[audit.s3] endpoint_url {problem}")
    elif endpoint_url is not None:
        raise ConfigError(
            "[audit.s3] endpoint_url applies to minio/ceph only; aws and gcs derive"
            " the host from the bucket"
        )
    auth = str(s3_raw.get("auth", default.auth))
    if auth not in S3_AUTH_MODES:
        raise ConfigError(f"[audit.s3] auth must be 'keys' or 'identity', got {auth!r}")
    if auth == "identity" and provider not in ("aws", "gcs"):
        raise ConfigError(
            f"[audit.s3] auth = 'identity' applies to provider 'aws' or 'gcs' only;"
            f" {provider!r} has no cloud workload identity (use auth = 'keys')"
        )
    return S3AuditConfig(
        enabled=enabled,
        provider=provider,
        bucket=bucket,
        prefix=prefix,
        region=str(s3_raw.get("region", default.region)),
        endpoint_url=endpoint_url,
        flush_seconds=flush_seconds,
        encryption=s3_encryption,
        auth=auth,
    )


def _parse_audit_azure(raw: object) -> AzureAuditConfig:
    if not isinstance(raw, dict):
        raise ConfigError("[audit.azure] must be a table")
    _require_keys(
        raw,
        {
            "enabled",
            "account",
            "container",
            "prefix",
            "endpoint_url",
            "flush_seconds",
            "encryption",
            "auth",
        },
        "[audit.azure]",
    )
    azure_encryption = str(raw.get("encryption", "none"))
    if azure_encryption not in ("none", "fernet"):
        raise ConfigError(
            f"[audit.azure] encryption must be 'none' or 'fernet', got {azure_encryption!r}"
        )
    default = AzureAuditConfig()
    enabled = _bool_key(raw, "enabled", False, "[audit.azure]")
    account = str(raw.get("account", ""))
    container = str(raw.get("container", ""))
    prefix = str(raw.get("prefix", default.prefix))
    if not _S3_PREFIX_RE.fullmatch(prefix):
        raise ConfigError("[audit.azure] prefix must contain only [A-Za-z0-9._/-] characters")
    endpoint_url = str(raw["endpoint_url"]).rstrip("/") if "endpoint_url" in raw else None
    flush_seconds = float(raw.get("flush_seconds", default.flush_seconds))
    if flush_seconds <= 0:
        raise ConfigError("[audit.azure] flush_seconds must be positive")
    if enabled and (not account or not container):
        raise ConfigError("[audit.azure] account and container are required when enabled")
    azure_auth = str(raw.get("auth", default.auth))
    if azure_auth not in AZURE_AUTH_MODES:
        raise ConfigError(
            f"[audit.azure] auth must be 'key', 'sas' or 'identity', got {azure_auth!r}"
        )
    if endpoint_url is not None:
        # Azurite's endpoint carries the account as a path, so a path is
        # fine. A SAS or an Entra token is a bearer secret (SharedKey sends
        # only a signature): those never cross the network in the clear.
        problem = sink_endpoint_problem(
            endpoint_url, allow_path=True, require_https=azure_auth != "key"
        )
        if problem is not None:
            raise ConfigError(f"[audit.azure] endpoint_url {problem}")
    return AzureAuditConfig(
        enabled=enabled,
        account=account,
        container=container,
        prefix=prefix,
        endpoint_url=endpoint_url,
        flush_seconds=flush_seconds,
        encryption=azure_encryption,
        auth=azure_auth,
    )


# Deny placeholder types must stay inside the token grammar and leave room
# for the numeric suffix within MAX_PLACEHOLDER_LEN.
_DENY_TYPE_RE = re.compile(r"[A-Z][A-Z0-9_]{0,19}\Z")


def _parse_deny(detection_raw: dict[str, Any]) -> tuple[DenyEntry, ...]:
    """Both deny surfaces -> one canonical, sorted DenyEntry tuple.

    `deny = ["value"]` is sugar for a case-insensitive DENY-typed entry;
    `[[detection.deny_strings]]` is the full per-entry form. Values are the
    user's own secrets, so error messages name positions, never values.
    """
    entries: list[DenyEntry] = []
    deny_raw = detection_raw.get("deny", [])
    if not isinstance(deny_raw, list):
        raise ConfigError("[detection] deny must be an array of strings")
    for position, value in enumerate(deny_raw):
        entries.append(_deny_entry(str(value), False, "DENY", f"deny[{position}]"))
    strings_raw = detection_raw.get("deny_strings", [])
    if not isinstance(strings_raw, list):
        raise ConfigError("[detection.deny_strings] must be an array of tables")
    for position, entry_raw in enumerate(strings_raw):
        where = f"[[detection.deny_strings]] #{position + 1}"
        if not isinstance(entry_raw, dict) or "value" not in entry_raw:
            raise ConfigError(f"{where} must be a table with a `value` key")
        _require_keys(entry_raw, {"value", "case_sensitive", "type"}, where)
        entries.append(
            _deny_entry(
                str(entry_raw["value"]),
                _bool_key(entry_raw, "case_sensitive", False, where),
                str(entry_raw.get("type", "DENY")),
                where,
            )
        )
    if len({(e.value, e.case_sensitive) for e in entries}) != len(entries):
        raise ConfigError("duplicate deny string (same value and case_sensitive)")
    # Sorted for canonical equality (reload's detector-reuse check).
    return tuple(sorted(entries, key=lambda e: (e.value, e.case_sensitive, e.detector_type)))


def _deny_entry(value: str, case_sensitive: bool, detector_type: str, where: str) -> "DenyEntry":
    if not value:
        raise ConfigError(f"{where}: deny value must be a non-empty string")
    if "«" in value or "»" in value:
        raise ConfigError(f"{where}: deny values may not contain guillemets («»)")
    if not _DENY_TYPE_RE.match(detector_type):
        raise ConfigError(
            f"{where}: type must match [A-Z][A-Z0-9_]* and be at most 20 characters,"
            f" got {detector_type!r}"
        )
    return DenyEntry(value=value, case_sensitive=case_sensitive, detector_type=detector_type)


# --- [upstreams] / [routing] / [prices] parser (shapes above) -----------------
#
# Upstream names and rule ids become metrics labels, response header values
# (x-llm-redact-upstream) and log fields, so they stay in a plain token
# alphabet. Env var names follow the POSIX portable set; header names are
# RFC 7230 tokens (lowercased at parse — names compare case-insensitively).
_ROUTE_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,63}\Z")
_ENV_VAR_RE = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_UPSTREAM_KEYS = {
    "protocol",
    "base_url",
    "credential",
    "cost",
    "inject_system_note",
    "count_tokens",
    "monthly_budget_usd",
    "monthly_budget_tokens",
    "cooldown_seconds",
    "extra_headers",
    "body_defaults",
}
_ROUTING_KEYS = {
    "enabled",
    "default_upstream",
    "max_hops",
    "request_deadline_seconds",
    "plan_limit_detection",
    "plan_limit_headers",
    "oauth_beta_marker",
    "throttle_retry_max_seconds",
    "budget_reset_day",
    "debug_headers",
    "expose_models",
    "model_catalog",
    "rule",
}
_RULE_KEYS = {
    "id",
    "match",
    "upstream",
    "model_rewrite",
    "on_status",
    "reissue_policy",
    "on_budget_exhausted",
}
_MATCH_KEYS = {"protocol", "model", "headers", "path", "auth"}


def _finite(value: int | float, key: str, where: str) -> float:
    """``value`` as a float, or a ConfigError naming ``where``/``key`` (never
    the value) when it is not FINITE: TOML spells ``nan``/``inf`` natively and
    an int beyond ``float`` range overflows on conversion. Either would pass
    ``serve --check`` and then break every consumer downstream — ``nan`` is
    unequal to itself (the editor's reparse-equality guard answers 500 on
    every later edit), ``inf``/``nan`` reach /status through the budget
    snapshot and Starlette serializes JSON with ``allow_nan=False``, and a
    ``nan`` budget/rate can never trip a threshold. One helper for every
    numeric routing/price key so the gap cannot reopen per call site."""
    try:
        as_float = float(value)
    except OverflowError:  # an int too large for a float is as unusable as inf
        as_float = math.inf
    if not math.isfinite(as_float):
        raise ConfigError(f"{where} {key} must be a finite number (nan/inf are not accepted)")
    return as_float


def _number_key(section: Mapping[str, Any], key: str, default: float, where: str) -> float:
    """An int-or-float config key; booleans and strings are rejected, never
    coerced (the _bool_key discipline), and the value must be finite."""
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where} {key} must be a number, got {type(value).__name__}")
    return _finite(value, key, where)


def _int_key(section: Mapping[str, Any], key: str, default: int, where: str) -> int:
    """An integer config key (bool is not an int here). The finite check
    rejects ints beyond float range: ``monthly_budget_tokens`` and friends are
    compared against float arithmetic and rendered by JavaScript, where such
    an int is Infinity."""
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} {key} must be an integer, got {type(value).__name__}")
    _finite(value, key, where)
    return value


def _required_str(section: Mapping[str, Any], key: str, where: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} {key} is required and must be a non-empty string")
    return value


def _optional_str(section: Mapping[str, Any], key: str, where: str) -> str | None:
    if key not in section:
        return None
    value = section[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where} {key} must be a non-empty string")
    return value


def _str_key(section: Mapping[str, Any], key: str, default: str, where: str) -> str:
    """A string config key with a default; any other type is a hard error
    naming the TYPE (the _bool_key discipline — `str(3)` would hand a path
    named "3" to the price table or a marker no request carries to the
    OAuth classifier). The value itself is never echoed: credential-shaped
    keys go through here."""
    value = section.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{where} {key} must be a string, got {type(value).__name__}")
    return value


def _header_table(section: Mapping[str, Any], key: str, where: str) -> tuple[tuple[str, str], ...]:
    """A `NAME = "value"` table of headers -> (lowercased name, value), sorted.

    Names are validated as HTTP tokens and lowercased (two spellings of one
    header are a duplicate); values are opaque strings (globs in rule
    matches, literal values in extra_headers) and never echoed.
    """
    raw = section.get(key, {})
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} {key} must be a table of header = string")
    out: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(value, str):
            raise ConfigError(f"{where} {key}: header {name!r} must map to a string")
        if not _HEADER_NAME_RE.fullmatch(name):
            raise ConfigError(f"{where} {key}: {name!r} is not a valid header name")
        lowered = name.lower()
        if lowered in out:
            raise ConfigError(f"{where} {key}: header {lowered!r} is listed twice")
        out[lowered] = value
    return tuple(sorted(out.items()))


def _parse_upstream(name: str, section: Mapping[str, Any], *, inject_note: bool) -> UpstreamConfig:
    where = f"[upstreams.{name}]"
    _require_keys(dict(section), _UPSTREAM_KEYS, where)
    protocol = _required_str(section, "protocol", where)
    if protocol not in PROTOCOLS:
        raise ConfigError(f"{where} protocol must be one of {PROTOCOLS}, got {protocol!r}")
    base_url = _required_str(section, "base_url", where).rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError(f"{where} base_url must start with http:// or https://")
    credential = _str_key(section, "credential", "passthrough", where)
    if credential.startswith("env:"):
        if not _ENV_VAR_RE.fullmatch(credential[len("env:") :]):
            raise ConfigError(
                f"{where} credential env:VAR needs a variable name matching [A-Z_][A-Z0-9_]*"
            )
    elif credential not in ("passthrough", "none"):
        # NEVER echoed: the realistic mistake is pasting the key itself where
        # `env:VAR` belongs, and this message reaches serve --check, doctor,
        # the startup/SIGHUP log and the editor's 400 body.
        raise ConfigError(
            f"{where} credential must be 'passthrough', 'none' or 'env:VAR'"
            " (the value is not shown: it may be a key)"
        )
    passthrough = credential == "passthrough"
    cost = _str_key(section, "cost", "metered", where)
    if cost not in COSTS:
        raise ConfigError(f"{where} cost must be one of {COSTS}, got {cost!r}")
    # I-4: the note is a `system` mutation, which a passthrough upstream must
    # never see; the default for every other upstream is the top-level switch.
    inject = _bool_key(section, "inject_system_note", False if passthrough else inject_note, where)
    if passthrough and inject:
        raise ConfigError(
            f"{where} inject_system_note = true is not allowed on a passthrough upstream"
            " (its system array is forwarded unchanged)"
        )
    budget_usd: float | None = None
    if "monthly_budget_usd" in section:
        budget_usd = _number_key(section, "monthly_budget_usd", 0.0, where)
        if budget_usd <= 0:
            raise ConfigError(f"{where} monthly_budget_usd must be positive")
    budget_tokens: int | None = None
    if "monthly_budget_tokens" in section:
        budget_tokens = _int_key(section, "monthly_budget_tokens", 0, where)
        if budget_tokens <= 0:
            raise ConfigError(f"{where} monthly_budget_tokens must be a positive integer")
    if passthrough and (budget_usd is not None or budget_tokens is not None):
        raise ConfigError(
            f"{where} monthly_budget_* is not allowed on a passthrough upstream"
            " (subscription usage is the provider's to meter)"
        )
    cooldown = _number_key(section, "cooldown_seconds", 60.0, where)
    if cooldown < 0:
        raise ConfigError(f"{where} cooldown_seconds must be >= 0")
    extra_headers = _header_table(section, "extra_headers", where)
    for header_name, _value in extra_headers:
        if header_name in CREDENTIAL_HEADERS or header_name == "x-goog-api-key":
            raise ConfigError(
                f"{where} extra_headers may not set {header_name!r}: credentials come"
                ' from credential = "env:VAR", never the config file'
            )
    body_raw = section.get("body_defaults", {})
    if not isinstance(body_raw, dict):
        raise ConfigError(f"{where} body_defaults must be a table")
    try:
        body_json = json.dumps(body_raw, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        # TOML datetimes / inf / nan (or a JSON null via the editor path) have
        # no JSON body representation.
        raise ConfigError(
            f"{where} body_defaults must contain only JSON-representable values"
        ) from exc
    if passthrough and (extra_headers or body_raw):
        raise ConfigError(
            f"{where} extra_headers/body_defaults apply only to none/env upstreams"
            " (a passthrough request is forwarded byte-exact)"
        )
    return UpstreamConfig(
        name=name,
        protocol=protocol,
        base_url=base_url,
        credential=credential,
        cost=cost,
        inject_system_note=inject,
        count_tokens=_bool_key(section, "count_tokens", True, where),
        monthly_budget_usd=budget_usd,
        monthly_budget_tokens=budget_tokens,
        cooldown_seconds=cooldown,
        extra_headers=extra_headers,
        body_defaults_json=body_json,
    )


def _parse_upstreams(
    raw: object,
    *,
    providers: Mapping[str, ProviderConfig],
    routing_present: bool,
    inject_note: bool,
) -> tuple[UpstreamConfig, ...]:
    """[upstreams.NAME] tables plus, when a [routing] table exists, the legacy
    auto-registration of [providers.{anthropic,openai,gemini,ollama}] as
    passthrough upstreams of the same name (R-1; explicit wins)."""
    if not isinstance(raw, dict):
        raise ConfigError("[upstreams] must contain named subtables")
    upstreams: dict[str, UpstreamConfig] = {}
    for name, section in raw.items():
        if not _ROUTE_NAME_RE.fullmatch(name):
            raise ConfigError(
                f"[upstreams] name {name!r} must match [A-Za-z0-9_][A-Za-z0-9_-]{{0,63}}"
            )
        if not isinstance(section, dict):
            raise ConfigError(f"[upstreams.{name}] must be a table")
        upstreams[name] = _parse_upstream(name, section, inject_note=inject_note)
    if routing_present:
        for legacy in PROTOCOLS:
            provider = providers.get(legacy)
            if (
                provider is None
                or legacy in upstreams
                or not provider.enabled
                or not provider.upstream_base_url
            ):
                continue
            upstreams[legacy] = UpstreamConfig(
                name=legacy,
                protocol=legacy,
                base_url=provider.upstream_base_url.rstrip("/"),
                credential="passthrough",
                inject_system_note=False,
                legacy=True,
            )
    return tuple(upstreams[name] for name in sorted(upstreams))


def _chain_members(
    section: Mapping[str, Any],
    key: str,
    *,
    where: str,
    protocol: str,
    own: str,
    by_name: Mapping[str, UpstreamConfig],
) -> tuple[str, ...]:
    """An ordered list of fallback upstream names, validated for I-5 (same
    protocol) and decision 3 (never a passthrough member — I-1/I-2). A chain
    continues to OTHER upstreams: naming the rule's own (`own` — the one that
    just failed or is exhausted; `retry-same` is the retry form) or the same
    member twice (a wasted hop against max_hops) is rejected."""
    raw = section.get(key)
    if not isinstance(raw, list) or not raw or not all(isinstance(n, str) for n in raw):
        raise ConfigError(f"{where} {key} must be a non-empty array of upstream names")
    for position, member in enumerate(raw):
        upstream = by_name.get(member)
        if upstream is None:
            raise ConfigError(f"{where} {key} names unknown upstream {member!r}")
        if member == own:
            raise ConfigError(
                f"{where} {key} names the rule's own upstream {member!r}: a chain continues"
                ' to other upstreams (use "retry-same" to retry this one)'
            )
        if member in raw[:position]:
            raise ConfigError(f"{where} {key} lists {member!r} twice")
        if upstream.protocol != protocol:
            raise ConfigError(
                f"{where} {key} member {member!r} speaks {upstream.protocol!r},"
                f" not the rule's protocol {protocol!r} (no protocol translation)"
            )
        if upstream.is_passthrough:
            raise ConfigError(
                f"{where} {key} member {member!r} is a passthrough upstream: a chain may"
                " only continue to env:/none upstreams (no credential pooling)"
            )
    return tuple(raw)


def _parse_match(section: object, where: str) -> RuleMatch:
    if not isinstance(section, dict):
        raise ConfigError(f"{where} match must be a table")
    _require_keys(section, _MATCH_KEYS, f"{where} match")
    protocol = _required_str(section, "protocol", f"{where} match")
    if protocol not in PROTOCOLS:
        raise ConfigError(f"{where} match protocol must be one of {PROTOCOLS}, got {protocol!r}")
    models: tuple[str, ...] = ()
    if "model" in section:
        model_raw = section["model"]
        if isinstance(model_raw, str):
            model_raw = [model_raw]
        if (
            not isinstance(model_raw, list)
            or not model_raw
            or not all(isinstance(m, str) and m for m in model_raw)
        ):
            raise ConfigError(
                f"{where} match model must be a glob string or a non-empty array of globs"
            )
        models = tuple(model_raw)
    auth = _str_key(section, "auth", "any", f"{where} match")
    if auth not in AUTH_KINDS:
        raise ConfigError(f"{where} match auth must be one of {AUTH_KINDS}, got {auth!r}")
    return RuleMatch(
        protocol=protocol,
        models=models,
        headers=_header_table(section, "headers", f"{where} match"),
        path=_optional_str(section, "path", f"{where} match"),
        auth=auth,
    )


def _status_key(key: object, where: str) -> str:
    text = str(key)
    if text in SPECIAL_STATUS_KEYS or text in CLASS_KEYS:
        return text
    if text.isdigit() and 100 <= int(text) <= 599:
        return str(int(text))
    raise ConfigError(
        f"{where} on_status key {text!r} must be an HTTP status (100-599), a class"
        f" ({', '.join(CLASS_KEYS)}) or one of {SPECIAL_STATUS_KEYS}"
    )


def _parse_rule(
    section: object,
    position: int,
    *,
    by_name: Mapping[str, UpstreamConfig],
    warnings: list[str],
) -> RouteRule:
    where = f"[[routing.rule]] #{position + 1}"
    if not isinstance(section, dict):
        raise ConfigError(f"{where} must be a table")
    _require_keys(section, _RULE_KEYS, where)
    rule_id = _required_str(section, "id", where)
    if not _ROUTE_NAME_RE.fullmatch(rule_id):
        raise ConfigError(f"{where} id {rule_id!r} must match [A-Za-z0-9_][A-Za-z0-9_-]{{0,63}}")
    where = f"[[routing.rule]] {rule_id!r}"
    if "match" not in section:
        raise ConfigError(f"{where} match is required")
    match = _parse_match(section["match"], where)
    upstream_name = _required_str(section, "upstream", where)
    upstream = by_name.get(upstream_name)
    if upstream is None:
        raise ConfigError(f"{where} names unknown upstream {upstream_name!r}")
    if upstream.protocol != match.protocol:
        raise ConfigError(
            f"{where} upstream {upstream_name!r} speaks {upstream.protocol!r}, not the"
            f" matched protocol {match.protocol!r} (no protocol translation)"
        )
    # A Gemini model id lives in the URL (models/{m}:generateContent), never
    # in the body: the proxy derives `model` from that path segment for rule
    # matching and applies `model_rewrite` to the SAME segment (decision 15b),
    # so gemini rules take model globs and rewrites like every other protocol.
    model_rewrite = _optional_str(section, "model_rewrite", where)
    policy = _str_key(
        section,
        "reissue_policy",
        "stateless-only" if upstream.is_passthrough else "always",
        where,
    )
    if policy not in REISSUE_POLICIES:
        raise ConfigError(
            f"{where} reissue_policy must be one of {REISSUE_POLICIES}, got {policy!r}"
        )
    on_status_raw = section.get("on_status", {})
    if not isinstance(on_status_raw, dict):
        raise ConfigError(f"{where} on_status must be a table of status = chain")
    on_status: list[tuple[str, tuple[str, ...] | str]] = []
    for raw_key, value in on_status_raw.items():
        key = _status_key(raw_key, where)
        if any(seen == key for seen, _ in on_status):
            raise ConfigError(f"{where} on_status key {key!r} is listed twice")
        if key in SPECIAL_STATUS_KEYS and match.protocol != "anthropic":
            warnings.append(
                f"{where} on_status key {key!r} never fires for protocol"
                f" {match.protocol!r} (plan-limit classification is Anthropic-only)"
            )
        if value == RETRY_SAME:
            on_status.append((key, RETRY_SAME))
            continue
        chain = _chain_members(
            {key: value},
            key,
            where=f"{where} on_status",
            protocol=match.protocol,
            own=upstream_name,
            by_name=by_name,
        )
        if policy == "never":
            # R-8: "never" re-issues nothing after a failed hop, so a chain
            # (unlike retry-same, which stays on the same upstream) is dead.
            warnings.append(
                f'{where} on_status {key} never applies: reissue_policy = "never"'
                " forbids re-issuing to another upstream"
            )
        on_status.append((key, chain))
    on_budget: tuple[str, ...] = ()
    if "on_budget_exhausted" in section:
        on_budget = _chain_members(
            section,
            "on_budget_exhausted",
            where=where,
            protocol=match.protocol,
            own=upstream_name,
            by_name=by_name,
        )
        if not upstream.has_budget:
            warnings.append(
                f"{where} on_budget_exhausted never applies: upstream {upstream_name!r}"
                " has no monthly budget"
            )
        elif policy == "never":
            # The proxy consults reissue_policy before the budget chain too
            # (an exhausted primary re-issues like a failed one).
            warnings.append(
                f'{where} on_budget_exhausted never applies: reissue_policy = "never"'
                " forbids re-issuing to another upstream"
            )
    return RouteRule(
        id=rule_id,
        match=match,
        upstream=upstream_name,
        model_rewrite=model_rewrite,
        on_status=tuple(on_status),
        reissue_policy=policy,
        on_budget_exhausted=on_budget,
    )


def _parse_defaults(
    raw: object, by_name: Mapping[str, UpstreamConfig]
) -> tuple[tuple[str, str], ...]:
    """`default_upstream = "name"` (that upstream's protocol only) or a
    protocol-keyed table -> sorted (protocol, name) pairs."""
    where = "[routing] default_upstream"
    if isinstance(raw, str):
        upstream = by_name.get(raw)
        if upstream is None:
            raise ConfigError(f"{where} names unknown upstream {raw!r}")
        return ((upstream.protocol, raw),)
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be an upstream name or a table of protocol = name")
    pairs: list[tuple[str, str]] = []
    for protocol, name in raw.items():
        if protocol not in PROTOCOLS:
            raise ConfigError(f"{where} key {protocol!r} must be one of {PROTOCOLS}")
        if not isinstance(name, str):
            raise ConfigError(f"{where} {protocol} must be an upstream name")
        upstream = by_name.get(name)
        if upstream is None:
            raise ConfigError(f"{where} {protocol} names unknown upstream {name!r}")
        if upstream.protocol != protocol:
            raise ConfigError(
                f"{where} {protocol} = {name!r}: that upstream speaks {upstream.protocol!r}"
            )
        pairs.append((protocol, name))
    return tuple(sorted(pairs))


def _parse_plan_limit_headers(
    raw: object,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    where = "[routing] plan_limit_headers"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a table of header = [values]")
    if not raw:
        raise ConfigError(
            f'{where} must not be empty; set plan_limit_detection = "off" to disable'
            " classification instead"
        )
    out: list[tuple[str, tuple[str, ...]]] = []
    for name, values in raw.items():
        if not _HEADER_NAME_RE.fullmatch(name):
            raise ConfigError(f"{where}: {name!r} is not a valid header name")
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(v, str) and v for v in values)
        ):
            raise ConfigError(f"{where}: {name!r} must map to a non-empty array of strings")
        lowered = name.lower()
        if any(seen == lowered for seen, _ in out):
            # Two spellings of one header would collapse to the last one in
            # the emitter (a lossy round trip), the _header_table rule.
            raise ConfigError(f"{where}: header {lowered!r} is listed twice")
        out.append((lowered, tuple(values)))
    return tuple(out)


def _parse_routing(
    raw: object, *, upstreams: tuple[UpstreamConfig, ...], present: bool
) -> RoutingConfig:
    if not present:
        # [upstreams] alone is parsed and kept but drives nothing: say so
        # rather than let a half-configured feature look active.
        warnings = (
            ("[upstreams] is configured but there is no [routing] table: upstreams are inert",)
            if upstreams
            else ()
        )
        return RoutingConfig(upstreams=upstreams, warnings=warnings)
    if not isinstance(raw, dict):
        raise ConfigError("[routing] must be a table")
    where = "[routing]"
    _require_keys(raw, _ROUTING_KEYS, where)
    by_name = {upstream.name: upstream for upstream in upstreams}
    enabled = _bool_key(raw, "enabled", False, where)
    defaults = (
        _parse_defaults(raw["default_upstream"], by_name) if "default_upstream" in raw else ()
    )
    if enabled and not defaults:
        # I-6 / R-11: the fail-closed destination for unmatched traffic must
        # be explicit whenever rules are live.
        raise ConfigError(
            "[routing] default_upstream is required when enabled = true (the fail-closed"
            " destination for requests no rule matches)"
        )
    max_hops = _int_key(raw, "max_hops", 3, where)
    if max_hops < 1:
        raise ConfigError("[routing] max_hops must be >= 1 (the original attempt counts)")
    deadline = _number_key(raw, "request_deadline_seconds", 600.0, where)
    if deadline <= 0:
        raise ConfigError("[routing] request_deadline_seconds must be positive")
    plan_limit_detection = _str_key(raw, "plan_limit_detection", "headers", where)
    if plan_limit_detection not in PLAN_LIMIT_DETECTION:
        raise ConfigError(
            f"[routing] plan_limit_detection must be one of {PLAN_LIMIT_DETECTION},"
            f" got {plan_limit_detection!r}"
        )
    default_routing = RoutingConfig()
    plan_limit_headers = default_routing.plan_limit_headers
    if "plan_limit_headers" in raw:
        plan_limit_headers = _parse_plan_limit_headers(raw["plan_limit_headers"])
    marker = _str_key(raw, "oauth_beta_marker", default_routing.oauth_beta_marker, where).strip()
    if not marker:
        raise ConfigError("[routing] oauth_beta_marker must be a non-empty string")
    throttle_max = _number_key(raw, "throttle_retry_max_seconds", 30.0, where)
    if throttle_max < 0:
        raise ConfigError("[routing] throttle_retry_max_seconds must be >= 0")
    reset_day = _int_key(raw, "budget_reset_day", 1, where)
    if not 1 <= reset_day <= 28:
        raise ConfigError("[routing] budget_reset_day must be between 1 and 28")
    rules_raw = raw.get("rule", [])
    if not isinstance(rules_raw, list):
        raise ConfigError("[[routing.rule]] must be an array of tables")
    warnings_list: list[str] = []
    rules: list[RouteRule] = []
    for position, section in enumerate(rules_raw):
        rule = _parse_rule(section, position, by_name=by_name, warnings=warnings_list)
        if any(existing.id == rule.id for existing in rules):
            raise ConfigError(f"[[routing.rule]] id {rule.id!r} is used twice")
        rules.append(rule)
    if enabled:
        for protocol, name in defaults:
            if not by_name[name].zero_cost:
                warnings_list.append(
                    f"[routing] default_upstream for {protocol} is {name!r}, which is not"
                    ' cost = "zero": unmatched traffic will spend there'
                )
    return RoutingConfig(
        enabled=enabled,
        default_upstreams=defaults,
        max_hops=max_hops,
        request_deadline_seconds=deadline,
        plan_limit_detection=plan_limit_detection,
        plan_limit_headers=plan_limit_headers,
        oauth_beta_marker=marker,
        throttle_retry_max_seconds=throttle_max,
        budget_reset_day=reset_day,
        debug_headers=_bool_key(raw, "debug_headers", False, where),
        expose_models=_bool_key(raw, "expose_models", False, where),
        model_catalog=_str_list(raw, "model_catalog", (), where),
        upstreams=upstreams,
        rules=tuple(rules),
        warnings=tuple(warnings_list),
        present=True,
    )


def _parse_prices(raw: object) -> PricesConfig:
    if not isinstance(raw, dict):
        raise ConfigError("[prices] must be a table")
    _require_keys(raw, {"table", "override"}, "[prices]")
    table = _str_key(raw, "table", "builtin", "[prices]")
    if not table:
        raise ConfigError('[prices] table must be "builtin" or a file path')
    overrides_raw = raw.get("override", {})
    if not isinstance(overrides_raw, dict):
        raise ConfigError(
            '[prices.override] must contain per-model tables ([prices.override."id"])'
        )
    overrides: list[tuple[str, ModelPrice]] = []
    for model_id, entry in overrides_raw.items():
        where = f"[prices.override.{model_id!r}]"
        if not model_id or not isinstance(entry, dict):
            raise ConfigError(f"{where} must be a table with input and output prices")
        _require_keys(entry, {"input", "output", "cache_read", "cache_write"}, where)
        missing = {"input", "output"} - entry.keys()
        if missing:
            raise ConfigError(f"{where} is missing required key(s) {sorted(missing)}")
        price = ModelPrice(
            input=_number_key(entry, "input", 0.0, where),
            output=_number_key(entry, "output", 0.0, where),
            cache_read=_number_key(entry, "cache_read", 0.0, where),
            cache_write=_number_key(entry, "cache_write", 0.0, where),
        )
        if min(price.input, price.output, price.cache_read, price.cache_write) < 0:
            raise ConfigError(f"{where} prices must be >= 0 (USD per 1M tokens)")
        overrides.append((model_id, price))
    return PricesConfig(table=table, overrides=tuple(sorted(overrides)))


def resolve_credentials(routing: RoutingConfig, environ: Mapping[str, str]) -> None:
    """R-3: every `env:VAR` credential of an ENABLED routing config must be
    set and non-empty. Raises ConfigError naming the variable(s) and their
    upstreams — never a value. Called by the proxy at build, serve --check
    and doctor (parse_config deliberately does not read the environment)."""
    if not routing.enabled:
        return
    missing = [
        f"{upstream.env_var} (upstream {upstream.name!r})"
        for upstream in routing.upstreams
        if upstream.credential_mode == "env" and not environ.get(upstream.env_var or "")
    ]
    if missing:
        raise ConfigError(
            "[upstreams] credential environment variable(s) unset or empty: " + ", ".join(missing)
        )


def load_config(path: Path | None = None) -> Config:
    """Load configuration.

    Search order when no explicit path is given: $LLM_REDACT_CONFIG (set but
    missing is a hard error, never a silent fall-through), then
    $XDG_CONFIG_HOME/llm-redact/config.toml, then /etc/llm-redact/config.toml,
    then built-in defaults.
    """
    if path is None:
        path = resolve_config_path()
        if path is None:
            return Config()
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return parse_config(raw, str(path))


def resolve_config_path() -> Path | None:
    """The effective config file per the search order, or None if none exists.

    Also used by the (llm-redact-pro) config editor to decide which file an
    edit rewrites, so
    an env-selected or /etc config is edited in place rather than shadowed by
    a freshly created XDG file.
    """
    env_path = os.environ.get("LLM_REDACT_CONFIG")
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise ConfigError(f"LLM_REDACT_CONFIG points to a missing file: {env_path}")
        return path
    if default_config_path().exists():
        return default_config_path()
    if ETC_CONFIG_PATH.exists():
        return ETC_CONFIG_PATH
    return None


def parse_config(raw: dict[str, Any], where: str) -> Config:
    """Validate an already-parsed TOML document into a Config.

    The single validation path: load_config and the llm-redact-pro
    dashboard's /__llm-redact/config editor endpoint both go through here,
    so a value the editor accepts is exactly a value the file would accept.
    """
    sections = _plugin_sections()
    unknown = set(raw) - CORE_SECTION_KEYS - set(sections)
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in {where} (a section supplied by"
            " llm-redact-pro, such as [auth], needs that package installed)"
        )
    extensions: dict[str, Any] = {}
    for name, section in sections.items():
        if name in raw:
            extensions[name] = section.parse(raw[name], f"[{name}] in {where}")
    max_body_bytes = int(raw.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))
    if max_body_bytes <= 0:
        raise ConfigError("max_body_bytes must be a positive integer")
    max_body_strings = int(raw.get("max_body_strings", DEFAULT_MAX_BODY_STRINGS))
    if max_body_strings <= 0:
        raise ConfigError("max_body_strings must be a positive integer")

    providers = dict(DEFAULT_PROVIDERS)
    for name, section in raw.get("providers", {}).items():
        if name == "custom":
            # [providers.custom.NAME]: named OpenAI-compatible upstreams,
            # routed under /custom/NAME/ and stored as "custom:NAME".
            if not isinstance(section, dict) or not all(
                isinstance(sub, dict) for sub in section.values()
            ):
                raise ConfigError("[providers.custom] must contain named subtables")
            for custom_name, custom_section in section.items():
                _add_custom_provider(providers, custom_name, custom_section)
            continue
        if name.startswith("custom:"):
            # Flat spelling of the same thing — the (pro) config editor posts
            # this form; the emitter always writes the nested canonical.
            _add_custom_provider(providers, name.removeprefix("custom:"), section)
            continue
        if name not in providers:
            raise ConfigError(
                f"unknown provider {name!r}; known: {sorted(providers)} — for your"
                f" own OpenAI-compatible upstream use [providers.custom.{name}]"
                f" (served under /custom/{name}/)"
            )
        _require_keys(
            section,
            {"upstream_base_url", "enabled", "detection", "auth", "region"},
            f"[providers.{name}]",
        )
        # upstream_base_url may be omitted for an enabled-only edit; the
        # provider then keeps its default upstream.
        url = section.get("upstream_base_url", providers[name].upstream_base_url)
        auth, region = _provider_auth(name, section)
        if auth == "identity":
            problem = identity_upstream_problem(str(url))
            if problem is not None:
                raise ConfigError(f"[providers.{name}] {problem}")
        providers[name] = ProviderConfig(
            upstream_base_url=str(url).rstrip("/"),
            enabled=_bool_key(section, "enabled", True, f"[providers.{name}]"),
            detection=_bool_key(section, "detection", True, f"[providers.{name}]"),
            auth=auth,
            region=region,
        )

    detection_raw = raw.get("detection", {})
    _require_keys(
        detection_raw,
        {
            "enabled",
            "languages",
            "allowlist",
            "allowlist_patterns",
            "allowlist_by_type",
            "custom_rules",
            "ner",
            "modes",
            "deny",
            "deny_strings",
            "mcp",
        },
        "[detection]",
    )
    languages: tuple[str, ...] | None = None
    if "languages" in detection_raw:
        languages_raw = detection_raw["languages"]
        if not isinstance(languages_raw, list) or not all(
            isinstance(lang, str) and lang.strip() for lang in languages_raw
        ):
            raise ConfigError("[detection] languages must be an array of non-empty strings")
        # Lowercased ISO 639-1 codes, sorted + deduplicated (canonical
        # equality, like modes); "EN" must scope the same rules as "en".
        languages = tuple(sorted({lang.strip().lower() for lang in languages_raw}))
        if not languages:
            raise ConfigError(
                "[detection] languages must not be empty when set; omit it to run all rules"
            )
    mcp_raw = detection_raw.get("mcp", {})
    if not isinstance(mcp_raw, dict):
        raise ConfigError("[detection.mcp] must be a table")
    _require_keys(mcp_raw, {"exempt_servers"}, "[detection.mcp]")
    exempt_raw = mcp_raw.get("exempt_servers", [])
    if not isinstance(exempt_raw, list) or not all(
        isinstance(server, str) and server for server in exempt_raw
    ):
        raise ConfigError("[detection.mcp] exempt_servers must be an array of non-empty strings")
    mcp_exempt_servers = tuple(sorted(set(exempt_raw)))
    by_type_raw = detection_raw.get("allowlist_by_type", {})
    if not isinstance(by_type_raw, dict):
        raise ConfigError("[detection.allowlist_by_type] must be a table of TYPE = [values]")
    by_type_entries = []
    for detector_type, values in by_type_raw.items():
        if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
            raise ConfigError(
                f"[detection.allowlist_by_type] {detector_type} must be a list of non-empty strings"
            )
        by_type_entries.append((str(detector_type), tuple(sorted(values))))
    allowlist_by_type = tuple(sorted(by_type_entries))
    modes_raw = detection_raw.get("modes", {})
    if not isinstance(modes_raw, dict):
        # TOML can't produce a non-table here, but the (pro) /config editor feeds
        # arbitrary JSON through this same path: 400, not a 500.
        raise ConfigError("[detection.modes] must be a table of rule_name = mode")
    for rule_name, mode in modes_raw.items():
        if mode not in ("redact", "warn", "block"):
            raise ConfigError(
                f"[detection.modes] {rule_name} must be 'redact', 'warn' or 'block', got {mode!r}"
            )
    # Sorted for canonical equality; unknown rule names are caught by
    # build_modes (the same split as enabled/build_detectors).
    modes = tuple(sorted((str(name), str(mode)) for name, mode in modes_raw.items()))
    ner_raw = detection_raw.get("ner", {})
    _require_keys(
        ner_raw,
        {
            "enabled",
            "backend",
            "backends",
            "entities",
            "max_chars",
            "score_threshold",
            "language",
            "model",
            "models",
        },
        "[detection.ner]",
    )
    known_backends = ("spacy", "gliner", "presidio", "stanza", "hf")
    backend_name = str(ner_raw.get("backend", "spacy"))
    if backend_name not in known_backends:
        raise ConfigError(
            f"[detection.ner] backend must be one of {known_backends}, got {backend_name!r}"
        )
    backends: tuple[str, ...] | None = None
    if "backends" in ner_raw:
        backends = _str_list(ner_raw, "backends", (), "[detection.ner]")
        bad = [b for b in backends if b not in known_backends]
        if bad:
            raise ConfigError(f"[detection.ner] backends: unknown backend(s) {bad!r}")
        if len(set(backends)) != len(backends):
            raise ConfigError("[detection.ner] backends must not repeat")
        if not backends:
            raise ConfigError("[detection.ner] backends must not be empty when set")
    models_raw = ner_raw.get("models", {})
    _require_keys(models_raw, set(known_backends), "[detection.ner.models]")
    models = tuple(sorted((str(k), str(v)) for k, v in models_raw.items()))
    active = backends if backends is not None else (backend_name,)
    _confidence_backends = ("gliner", "presidio", "hf")
    if "score_threshold" in ner_raw and not any(b in _confidence_backends for b in active):
        # spaCy and stanza emit no per-entity confidences; a silently ignored
        # knob would violate the reject-unknown-keys spirit.
        raise ConfigError(
            f"[detection.ner] score_threshold requires one of {_confidence_backends} backends"
        )
    default_ner = NerConfig()
    ner = NerConfig(
        enabled=_bool_key(ner_raw, "enabled", False, "[detection.ner]"),
        backend=backend_name,
        backends=backends,
        entities=_str_list(ner_raw, "entities", default_ner.entities, "[detection.ner]"),
        max_chars=int(ner_raw.get("max_chars", default_ner.max_chars)),
        score_threshold=float(ner_raw.get("score_threshold", default_ner.score_threshold)),
        language=str(ner_raw.get("language", default_ner.language)),
        model=str(ner_raw["model"]) if "model" in ner_raw else None,
        models=models,
    )
    if ner.enabled and languages is not None and ner.language.lower() not in languages:
        # Loud, not silent: an NER model scanning a language the deployment
        # says it never sees is a misconfiguration, not a preference.
        raise ConfigError(
            f"[detection.ner] language {ner.language!r} is not in [detection] languages"
            f" {list(languages)!r}; add it to the list or change the NER model"
        )
    custom_rules = []
    custom_raw = detection_raw.get("custom_rules", [])
    if not isinstance(custom_raw, list):
        raise ConfigError("[[detection.custom_rules]] must be an array of tables")
    for position, rule in enumerate(custom_raw):
        where = f"[[detection.custom_rules]] #{position + 1}"
        if not isinstance(rule, dict):
            raise ConfigError(f"{where} must be a table")
        missing = {"name", "type", "pattern"} - rule.keys()
        if missing:
            # A KeyError here was a raw traceback in serve --check and doctor
            # (3.2.0) — the two tools the docs point users at first.
            raise ConfigError(
                f"{where} is missing required key(s) {sorted(missing)}"
                " (name, type, and pattern are required)"
            )
        _require_keys(
            rule,
            {"name", "type", "pattern", "priority", "validator", "required", "anchors"},
            where,
        )
        validator = rule.get("validator")
        if validator is not None and not isinstance(validator, str):
            raise ConfigError(f"{where} validator must be a string")
        custom_rules.append(
            CustomRule(
                name=str(rule["name"]),
                detector_type=str(rule["type"]),
                pattern=str(rule["pattern"]),
                priority=int(rule.get("priority", 100)),
                validator=str(validator) if validator is not None else None,
                required=_str_list(rule, "required", (), where),
                anchors=_str_list(rule, "anchors", (), where),
            )
        )
    deny_strings = _parse_deny(detection_raw)
    default_detection = DetectionConfig()
    detection = DetectionConfig(
        enabled=_str_list(detection_raw, "enabled", default_detection.enabled, "[detection]"),
        languages=languages,
        allowlist=_str_list(detection_raw, "allowlist", (), "[detection]"),
        allowlist_patterns=_str_list(detection_raw, "allowlist_patterns", (), "[detection]"),
        allowlist_by_type=allowlist_by_type,
        custom_rules=tuple(custom_rules),
        ner=ner,
        modes=modes,
        deny_strings=deny_strings,
        mcp_exempt_servers=mcp_exempt_servers,
    )

    rehydration_raw = raw.get("rehydration", {})
    _require_keys(rehydration_raw, {"fuzzy"}, "[rehydration]")
    rehydration = RehydrationConfig(
        fuzzy=_bool_key(rehydration_raw, "fuzzy", True, "[rehydration]")
    )

    vault_raw = raw.get("vault", {})
    _require_keys(
        vault_raw,
        {
            "backend",
            "path",
            "session",
            "session_mode",
            "encryption",
            "session_ttl_days",
            "rdbms",
            "kms",
        },
        "[vault]",
    )
    backend = str(vault_raw.get("backend", "memory"))
    known_vault_backends = ("memory", "sqlite", *RDBMS_BACKENDS)
    if backend not in known_vault_backends:
        raise ConfigError(f"[vault] backend must be one of {known_vault_backends}, got {backend!r}")
    rdbms_raw = vault_raw.get("rdbms", {})
    _require_keys(
        rdbms_raw,
        {"dsn", "password_env", "module", "cloud", "auth", "region"},
        "[vault.rdbms]",
    )
    rdbms = RdbmsConfig(
        dsn=str(rdbms_raw.get("dsn", "")),
        password_env=str(rdbms_raw.get("password_env", "LLM_REDACT_VAULT_DB_PASSWORD")),
        module=str(rdbms_raw.get("module", "")),
        cloud=str(rdbms_raw.get("cloud", "")),
        auth=str(rdbms_raw.get("auth", "password")),
        region=str(rdbms_raw.get("region", "")),
    )
    if rdbms.cloud not in ("", "aws", "azure", "gcp"):
        raise ConfigError(
            f"[vault.rdbms] cloud must be 'aws', 'azure', or 'gcp', got {rdbms.cloud!r}"
        )
    if rdbms.auth not in RDBMS_AUTH_MODES:
        raise ConfigError(
            f"[vault.rdbms] auth must be one of {RDBMS_AUTH_MODES}, got {rdbms.auth!r}"
        )
    if rdbms.region and not _RDBMS_REGION_RE.fullmatch(rdbms.region):
        raise ConfigError(f"[vault.rdbms] region {rdbms.region!r} is not a region name")
    if rdbms.region and rdbms.auth != "identity":
        raise ConfigError('[vault.rdbms] region applies only to auth = "identity"')
    if rdbms.auth == "identity" and "password_env" in rdbms_raw:
        # Explicitly naming the default env var is still a static password.
        raise ConfigError('[vault.rdbms] password_env cannot be combined with auth = "identity"')
    identity_error = (
        rdbms_identity_error(backend, rdbms, rdbms.dsn) if backend in RDBMS_BACKENDS else None
    )
    if identity_error is not None:
        raise ConfigError(identity_error)
    if rdbms != RdbmsConfig() and backend not in RDBMS_BACKENDS:
        raise ConfigError(
            f"[vault.rdbms] applies only to the RDBMS backends {RDBMS_BACKENDS},"
            f" not backend = {backend!r}"
        )
    if backend == "dbapi" and not rdbms.module:
        raise ConfigError(
            '[vault.rdbms] module (a DB-API 2.0 module name) is required for backend = "dbapi"'
        )
    if rdbms.module and backend != "dbapi":
        raise ConfigError('[vault.rdbms] module applies only to backend = "dbapi"')
    session_mode = str(vault_raw.get("session_mode", "static"))
    if session_mode not in ("static", "per-conversation"):
        raise ConfigError(
            f"[vault] session_mode must be 'static' or 'per-conversation', got {session_mode!r}"
        )
    encryption = str(vault_raw.get("encryption", "none"))
    if encryption not in ("none", "fernet"):
        raise ConfigError(f"[vault] encryption must be 'none' or 'fernet', got {encryption!r}")
    session_ttl_days = int(vault_raw.get("session_ttl_days", 0))
    if session_ttl_days < 0:
        raise ConfigError("[vault] session_ttl_days must be >= 0 (0 disables auto-prune)")
    kms: VaultKmsConfig | None = None
    if "kms" in vault_raw:
        kms = parse_vault_kms(vault_raw["kms"])
        if encryption != "fernet":
            raise ConfigError(
                '[vault.kms] wraps the fernet vault key: it requires [vault] encryption = "fernet"'
            )
    vault = VaultConfig(
        backend=backend,
        path=str(vault_raw["path"]) if "path" in vault_raw else None,
        session=str(vault_raw.get("session", "default")),
        session_mode=session_mode,
        encryption=encryption,
        session_ttl_days=session_ttl_days,
        rdbms=rdbms,
        kms=kms,
    )

    audit_raw = raw.get("audit", {})
    _require_keys(
        audit_raw,
        {"enabled", "path", "max_rows", "tamper_evident", "required", "s3", "azure"},
        "[audit]",
    )
    max_rows = int(audit_raw.get("max_rows", 10000))
    if max_rows <= 0:
        raise ConfigError("[audit] max_rows must be a positive integer")
    audit = AuditConfig(
        enabled=_bool_key(audit_raw, "enabled", False, "[audit]"),
        path=str(audit_raw["path"]) if "path" in audit_raw else None,
        max_rows=max_rows,
        tamper_evident=_bool_key(audit_raw, "tamper_evident", False, "[audit]"),
        required=_bool_key(audit_raw, "required", False, "[audit]"),
        s3=_parse_audit_s3(audit_raw.get("s3", {})),
        azure=_parse_audit_azure(audit_raw.get("azure", {})),
    )
    if audit.required and not audit.enabled:
        raise ConfigError("[audit] required = true needs [audit] enabled = true")

    log_raw = raw.get("log", {})
    _require_keys(log_raw, {"format"}, "[log]")
    log_format = str(log_raw.get("format", "text"))
    if log_format not in ("text", "json"):
        raise ConfigError(f'[log] format must be "text" or "json", not {log_format!r}')
    log = LogConfig(format=log_format)

    tls_raw = raw.get("tls", {})
    _require_keys(tls_raw, {"certfile", "keyfile", "client_ca"}, "[tls]")
    tls = TlsConfig(
        certfile=str(tls_raw["certfile"]) if "certfile" in tls_raw else None,
        keyfile=str(tls_raw["keyfile"]) if "keyfile" in tls_raw else None,
        client_ca=str(tls_raw["client_ca"]) if "client_ca" in tls_raw else None,
    )
    if (tls.certfile is None) != (tls.keyfile is None):
        raise ConfigError("[tls] certfile and keyfile must be set together")
    if tls.client_ca is not None and tls.certfile is None:
        raise ConfigError("[tls] client_ca requires certfile and keyfile (mutual TLS)")

    otel_raw = raw.get("otel", {})
    _require_keys(otel_raw, {"enabled", "endpoint", "service_name"}, "[otel]")
    service_name = str(otel_raw.get("service_name", "llm-redact"))
    if not service_name:
        raise ConfigError("[otel] service_name must be a non-empty string")
    otel = OtelConfig(
        enabled=_bool_key(otel_raw, "enabled", False, "[otel]"),
        endpoint=str(otel_raw["endpoint"]).rstrip("/") if "endpoint" in otel_raw else None,
        service_name=service_name,
    )

    license_raw = raw.get("license", {})
    _require_keys(license_raw, {"key", "key_file"}, "[license]")
    license_cfg = LicenseConfig(
        key=str(license_raw["key"]) if "key" in license_raw else None,
        key_file=str(license_raw["key_file"]) if "key_file" in license_raw else None,
    )

    users_raw = raw.get("users", {})
    _require_keys(users_raw, {"path", "unrecorded_objects"}, "[users]")
    unrecorded = users_raw.get("unrecorded_objects", UsersConfig().unrecorded_objects)
    if not isinstance(unrecorded, str) or unrecorded not in UNRECORDED_OBJECT_POLICIES:
        raise ConfigError(
            '[users] unrecorded_objects must be "refuse" (the default) or "allow"'
            " (llm-redact-pro docs/authentication.md)"
        )
    users_cfg = UsersConfig(
        path=str(users_raw["path"]) if "path" in users_raw else None,
        unrecorded_objects=unrecorded,
    )

    email_cfg = _parse_email(raw.get("email", {}))

    # Routing last: it depends on the resolved providers (legacy
    # auto-registration) and the top-level note switch (per-upstream default),
    # and keeping it after every pre-existing check leaves the error order of
    # older configs untouched.
    inject_system_note = _bool_key(raw, "inject_system_note", True, "top-level")
    routing_present = "routing" in raw
    upstreams = _parse_upstreams(
        raw.get("upstreams", {}),
        providers=providers,
        routing_present=routing_present,
        inject_note=inject_system_note,
    )
    routing = _parse_routing(raw.get("routing"), upstreams=upstreams, present=routing_present)
    prices = _parse_prices(raw.get("prices", {}))

    return Config(
        host=str(raw.get("host", DEFAULT_HOST)),
        port=int(raw.get("port", DEFAULT_PORT)),
        allowed_hosts=_parse_allowed_hosts(raw),
        allowed_origins=_parse_allowed_origins(raw),
        inject_system_note=inject_system_note,
        max_body_bytes=max_body_bytes,
        max_body_strings=max_body_strings,
        providers=providers,
        detection=detection,
        vault=vault,
        rehydration=rehydration,
        audit=audit,
        log=log,
        tls=tls,
        otel=otel,
        license=license_cfg,
        users=users_cfg,
        email=email_cfg,
        routing=routing,
        prices=prices,
        extensions=extensions,
    )


_EMAIL_AUTH_MODES = ("password", "oauth")
_EMAIL_OAUTH_PROVIDERS = ("azure", "google", "refresh_token")
# The generic refresh-token grant's own keys; azure/google resolve their
# credentials from the workload identity and must not carry them.
_EMAIL_REFRESH_KEYS = (
    "oauth_token_url",
    "oauth_client_id",
    "oauth_client_secret_env",
    "oauth_refresh_token_env",
    "oauth_scope",
)
_EMAIL_OAUTH_KEYS = ("oauth_provider", "oauth_subject", *_EMAIL_REFRESH_KEYS)


def _email_env_name(section: Mapping[str, Any], key: str) -> str | None:
    """An env-var NAME key: the secret lives in that variable, never in the
    file. The value is never echoed — the realistic mistake is pasting the
    secret itself here."""
    value = _optional_str(section, key, "[email]")
    if value is not None and not _ENV_VAR_RE.fullmatch(value):
        raise ConfigError(
            f"[email] {key} must name an environment variable matching [A-Z_][A-Z0-9_]*"
            " (the secret itself never lives in the config file)"
        )
    return value


def _parse_email(email_raw: dict[str, Any]) -> EmailConfig:
    _require_keys(
        email_raw,
        {
            "smtp_host",
            "smtp_port",
            "starttls",
            "username",
            "password_env",
            "from_address",
            "implicit_tls",
            "auth",
            *_EMAIL_OAUTH_KEYS,
        },
        "[email]",
    )
    implicit_tls = _bool_key(email_raw, "implicit_tls", False, "[email]")
    # STARTTLS defaults on, except under implicit TLS where it cannot apply.
    starttls = _bool_key(email_raw, "starttls", not implicit_tls, "[email]")
    if implicit_tls and starttls:
        raise ConfigError(
            "[email] implicit_tls = true and starttls = true are mutually exclusive"
            " (implicit TLS is already encrypted; set starttls = false)"
        )
    auth = _str_key(email_raw, "auth", "password", "[email]")
    if auth not in _EMAIL_AUTH_MODES:
        raise ConfigError(f'[email] auth must be "password" or "oauth", got {auth!r}')
    email_cfg = EmailConfig(
        smtp_host=str(email_raw["smtp_host"]) if "smtp_host" in email_raw else None,
        smtp_port=int(email_raw.get("smtp_port", 587)),
        starttls=starttls,
        username=str(email_raw["username"]) if "username" in email_raw else None,
        password_env=str(email_raw.get("password_env", "LLM_REDACT_SMTP_PASSWORD")),
        from_address=str(email_raw["from_address"]) if "from_address" in email_raw else None,
        implicit_tls=implicit_tls,
        auth=auth,
        oauth_provider=_optional_str(email_raw, "oauth_provider", "[email]"),
        oauth_subject=_optional_str(email_raw, "oauth_subject", "[email]"),
        oauth_token_url=_optional_str(email_raw, "oauth_token_url", "[email]"),
        oauth_client_id=_optional_str(email_raw, "oauth_client_id", "[email]"),
        oauth_client_secret_env=_email_env_name(email_raw, "oauth_client_secret_env"),
        oauth_refresh_token_env=_email_env_name(email_raw, "oauth_refresh_token_env"),
        oauth_scope=_optional_str(email_raw, "oauth_scope", "[email]"),
    )
    if (email_cfg.smtp_host is None) != (email_cfg.from_address is None):
        raise ConfigError("[email] smtp_host and from_address must be set together")
    if auth == "oauth":
        _check_email_oauth(email_cfg, email_raw)
    else:
        stray = [key for key in _EMAIL_OAUTH_KEYS if key in email_raw]
        if stray:
            raise ConfigError(f'[email] {", ".join(stray)} require auth = "oauth"')
    return email_cfg


def _check_email_oauth(email_cfg: EmailConfig, email_raw: Mapping[str, Any]) -> None:
    """auth = "oauth" invariants: a configured sender, TLS on the wire (a
    bearer token is a password-equivalent), a known provider, and exactly
    the keys that provider reads."""
    if not email_cfg.configured:
        raise ConfigError('[email] auth = "oauth" requires smtp_host and from_address')
    if email_cfg.tls == "none":
        raise ConfigError(
            '[email] auth = "oauth" requires TLS: set starttls = true or implicit_tls = true'
            " (a bearer token is never sent over a cleartext connection)"
        )
    provider = email_cfg.oauth_provider
    if provider not in _EMAIL_OAUTH_PROVIDERS:
        raise ConfigError(
            '[email] auth = "oauth" requires oauth_provider = "azure", "google" or'
            f' "refresh_token" (got {provider!r})'
        )
    if email_cfg.oauth_subject is not None and provider != "google":
        raise ConfigError('[email] oauth_subject applies only to oauth_provider = "google"')
    if provider != "refresh_token":
        stray = [key for key in _EMAIL_REFRESH_KEYS if key in email_raw]
        if stray:
            raise ConfigError(
                f'[email] {", ".join(stray)} apply only to oauth_provider = "refresh_token"'
                f" ({provider} credentials come from the workload identity)"
            )
        return
    for key in ("oauth_token_url", "oauth_client_id", "oauth_refresh_token_env"):
        if key not in email_raw:
            raise ConfigError(f'[email] oauth_provider = "refresh_token" requires {key}')
    if not _is_https_url(email_cfg.oauth_token_url or ""):
        raise ConfigError(
            "[email] oauth_token_url must be an https:// URL with a host and no"
            " userinfo or fragment"
        )


def _is_https_url(url: str) -> bool:
    try:
        parts = urllib.parse.urlsplit(url)
        _ = parts.port  # a malformed port raises
    except ValueError:
        return False
    return (
        parts.scheme == "https"
        and bool(parts.hostname)
        and "@" not in parts.netloc
        and not parts.fragment
        and not any(char.isspace() for char in url)
    )


def _plugin_sections() -> "dict[str, ConfigSection]":
    """The registered plugin sections by name (a name that collides with a
    core key is ignored — a plugin can never reshape a core section)."""
    from .registry import get_registry  # lazy: registry imports this module

    sections: dict[str, ConfigSection] = {}
    for section in get_registry().config_sections:
        if section.name not in CORE_SECTION_KEYS and section.name not in sections:
            sections[section.name] = section
    return sections


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A hostname we cannot prove is loopback is treated as non-loopback:
        # the policy fails closed.
        return False


def validate_bind_security(host: str, tls: TlsConfig, environ: Mapping[str, str]) -> None:
    """Fail-closed bind policy, checked by `serve` before the socket opens.

    A non-loopback bind exposes the vault's rehydrated values, the ops
    surface (and the llm-redact-pro config editor), and detection metadata
    to the network, so it requires FULL
    mutual TLS — server certfile+keyfile AND client_ca. Server-only TLS is
    allowed on loopback (encrypting local traffic is harmless). The
    LLM_REDACT_INSECURE_BIND=1 hatch exists solely for confined wider
    binds — the container image sets it because 0.0.0.0 inside the
    container's network namespace is still only reachable through the
    published port (documented as 127.0.0.1-only).
    """
    if _is_loopback_host(host):
        return
    if environ.get("LLM_REDACT_INSECURE_BIND") == "1":
        return
    if not (tls.certfile and tls.keyfile and tls.client_ca):
        raise ConfigError(
            f"refusing to bind {host!r} without mutual TLS: set [tls] certfile,"
            " keyfile, and client_ca (clients must present a certificate), or"
            " keep host on 127.0.0.1. LLM_REDACT_INSECURE_BIND=1 overrides only"
            " for binds that are confined some other way (container netns with"
            " a loopback-only publish)."
        )


def url_host(host: str) -> str:
    """``host`` as the host part of a URL: an IPv6 literal in brackets
    (``::1`` -> ``[::1]``; ``http://::1:8787`` is no URL), anything else as
    written."""
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def dial_host(host: str) -> str:
    """The URL host a client on THIS machine dials to reach a proxy bound to
    ``host`` (the ``host`` config key). A wildcard bind (``0.0.0.0``, ``::``
    or empty: every interface) is dialed at loopback — ``127.0.0.1``, or
    ``[::1]`` for an IPv6 one (an IPv6-only listener never answers on
    127.0.0.1); an IPv6 literal is bracketed (``url_host``); a host name or
    any other address as written."""
    bare = host.strip()
    if bare.startswith("[") and bare.endswith("]"):
        bare = bare[1:-1]
    if not bare:
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(bare)
    except ValueError:
        return bare  # a host name
    if address.is_unspecified:
        return "[::1]" if address.version == 6 else "127.0.0.1"
    return url_host(bare)


def dial_url(host: str, port: int, *, scheme: str = "http") -> str:
    """``scheme://host:port`` for a client on this machine reaching a proxy
    bound to ``host`` (``dial_host``) — the base of the ``status``,
    ``doctor``, ``run``, ``plugin install`` and ``vault rotate-key``
    probes and of what ``run`` exports to tools. No path: a log line may
    print it whole."""
    return f"{scheme}://{dial_host(host)}:{port}"
