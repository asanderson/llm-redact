"""Configuration: TOML file over frozen dataclasses, stdlib only."""

import dataclasses
import ipaddress
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_redact.detection.deny import DenyEntry
from llm_redact.detection.engine import CustomRule, DetectionConfig, NerConfig

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


@dataclass(frozen=True)
class UsersConfig:
    # Named-user registry database (Pro+ tiers; llm-redact-pro docs/licensing.md).
    # Default: $XDG_DATA_HOME/llm-redact/users.db. Restart-only.
    path: str | None = None


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

    @property
    def configured(self) -> bool:
        return self.smtp_host is not None and self.from_address is not None


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


@dataclass(frozen=True)
class Config:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    inject_system_note: bool = True
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
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


# Object-key prefixes stay in the URI-unreserved set so the SigV4 canonical
# URI never needs percent-encoding (a wrong encoding is a signature failure,
# which the sink can only WARN-and-drop on).
_S3_PREFIX_RE = re.compile(r"[A-Za-z0-9._/-]*\Z")


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
        if endpoint_url is not None and not endpoint_url.startswith(("http://", "https://")):
            raise ConfigError("[audit.s3] endpoint_url must start with http:// or https://")
    elif endpoint_url is not None:
        raise ConfigError(
            "[audit.s3] endpoint_url applies to minio/ceph only; aws and gcs derive"
            " the host from the bucket"
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
    if endpoint_url is not None and not endpoint_url.startswith(("http://", "https://")):
        raise ConfigError("[audit.azure] endpoint_url must start with http:// or https://")
    return AzureAuditConfig(
        enabled=enabled,
        account=account,
        container=container,
        prefix=prefix,
        endpoint_url=endpoint_url,
        flush_seconds=flush_seconds,
        encryption=azure_encryption,
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

    Also used by the config editor to decide which file an edit rewrites, so
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

    The single validation path: load_config and the /__llm-redact/config
    editor endpoint both go through here, so a value the editor accepts is
    exactly a value the file would accept.
    """
    _require_keys(
        raw,
        {
            "host",
            "port",
            "inject_system_note",
            "max_body_bytes",
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
        },
        where,
    )
    max_body_bytes = int(raw.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))
    if max_body_bytes <= 0:
        raise ConfigError("max_body_bytes must be a positive integer")

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
            # Flat spelling of the same thing — the config editor posts
            # this form; the emitter always writes the nested canonical.
            _add_custom_provider(providers, name.removeprefix("custom:"), section)
            continue
        if name not in providers:
            raise ConfigError(
                f"unknown provider {name!r}; known: {sorted(providers)} — for your"
                f" own OpenAI-compatible upstream use [providers.custom.{name}]"
                f" (served under /custom/{name}/)"
            )
        _require_keys(section, {"upstream_base_url", "enabled", "detection"}, f"[providers.{name}]")
        # upstream_base_url may be omitted for an enabled-only edit; the
        # provider then keeps its default upstream.
        url = section.get("upstream_base_url", providers[name].upstream_base_url)
        providers[name] = ProviderConfig(
            upstream_base_url=str(url).rstrip("/"),
            enabled=_bool_key(section, "enabled", True, f"[providers.{name}]"),
            detection=_bool_key(section, "detection", True, f"[providers.{name}]"),
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
        # TOML can't produce a non-table here, but the /config editor feeds
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
        {"backend", "path", "session", "session_mode", "encryption", "session_ttl_days", "rdbms"},
        "[vault]",
    )
    backend = str(vault_raw.get("backend", "memory"))
    known_vault_backends = ("memory", "sqlite", *RDBMS_BACKENDS)
    if backend not in known_vault_backends:
        raise ConfigError(f"[vault] backend must be one of {known_vault_backends}, got {backend!r}")
    rdbms_raw = vault_raw.get("rdbms", {})
    _require_keys(rdbms_raw, {"dsn", "password_env", "module", "cloud"}, "[vault.rdbms]")
    rdbms = RdbmsConfig(
        dsn=str(rdbms_raw.get("dsn", "")),
        password_env=str(rdbms_raw.get("password_env", "LLM_REDACT_VAULT_DB_PASSWORD")),
        module=str(rdbms_raw.get("module", "")),
        cloud=str(rdbms_raw.get("cloud", "")),
    )
    if rdbms.cloud not in ("", "aws", "azure", "gcp"):
        raise ConfigError(
            f"[vault.rdbms] cloud must be 'aws', 'azure', or 'gcp', got {rdbms.cloud!r}"
        )
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
    vault = VaultConfig(
        backend=backend,
        path=str(vault_raw["path"]) if "path" in vault_raw else None,
        session=str(vault_raw.get("session", "default")),
        session_mode=session_mode,
        encryption=encryption,
        session_ttl_days=session_ttl_days,
        rdbms=rdbms,
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
    _require_keys(users_raw, {"path"}, "[users]")
    users_cfg = UsersConfig(path=str(users_raw["path"]) if "path" in users_raw else None)

    email_raw = raw.get("email", {})
    _require_keys(
        email_raw,
        {"smtp_host", "smtp_port", "starttls", "username", "password_env", "from_address"},
        "[email]",
    )
    email_cfg = EmailConfig(
        smtp_host=str(email_raw["smtp_host"]) if "smtp_host" in email_raw else None,
        smtp_port=int(email_raw.get("smtp_port", 587)),
        starttls=_bool_key(email_raw, "starttls", True, "[email]"),
        username=str(email_raw["username"]) if "username" in email_raw else None,
        password_env=str(email_raw.get("password_env", "LLM_REDACT_SMTP_PASSWORD")),
        from_address=str(email_raw["from_address"]) if "from_address" in email_raw else None,
    )
    if (email_cfg.smtp_host is None) != (email_cfg.from_address is None):
        raise ConfigError("[email] smtp_host and from_address must be set together")

    return Config(
        host=str(raw.get("host", DEFAULT_HOST)),
        port=int(raw.get("port", DEFAULT_PORT)),
        inject_system_note=_bool_key(raw, "inject_system_note", True, "top-level"),
        max_body_bytes=max_body_bytes,
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
    )


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

    A non-loopback bind exposes the vault's rehydrated values, the config
    editor, and detection metadata to the network, so it requires FULL
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
