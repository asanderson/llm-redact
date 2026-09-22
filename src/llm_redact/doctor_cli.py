"""`llm-redact doctor`: read-only environment diagnostics.

Prints one PASS/WARN/FAIL line per check and exits non-zero if anything
FAILs. Never prints secret values — paths, modes, and versions only. A
WARN is something worth knowing (proxy not running, no config file); a
FAIL is something that will break serving or silently weaken protection
(unreadable TLS files, loose vault permissions, an extra the config
requires but that is not installed).
"""

import argparse
import importlib.util
import os
import re
import socket
import stat
import sys
from pathlib import Path

from llm_redact import __version__
from llm_redact.config import (
    Config,
    ConfigError,
    apply_env_overrides,
    load_config,
    resolve_config_path,
    validate_bind_security,
)

_NER_MODULES = {
    "spacy": "spacy",
    "gliner": "gliner",
    "presidio": "presidio_analyzer",
    "stanza": "stanza",
    "hf": "transformers",
}
_NER_EXTRAS = {
    "spacy": "ner",
    "gliner": "gliner",
    "presidio": "presidio",
    "stanza": "stanza",
    "hf": "hf",
}
_ENV_OVERRIDES = ("LLM_REDACT_HOST", "LLM_REDACT_PORT", "LLM_REDACT_CONFIG")


class _Report:
    def __init__(self, json_mode: bool = False) -> None:
        self.failed = False
        self.json_mode = json_mode
        self.rows: list[dict[str, str]] = []

    def line(self, level: str, area: str, message: str) -> None:
        if level == "FAIL":
            self.failed = True
        if self.json_mode:
            # The same value-free text as the human lines — messages carry
            # paths/modes/versions only, never secrets, so JSON framing
            # changes nothing about what leaves the machine.
            self.rows.append({"level": level, "area": area, "message": message})
        else:
            print(f"{level:<4}  {area}: {message}")

    def finish(self) -> None:
        if self.json_mode:
            import json

            from llm_redact import __version__ as version

            print(
                json.dumps(
                    {"version": version, "failed": self.failed, "checks": self.rows},
                    ensure_ascii=False,
                )
            )


def _check_platform(report: _Report) -> None:
    """Windows-specific posture (silent elsewhere — no noise on the
    platforms where nothing differs). The supported Windows scope is the
    Free tier plus the agent plugins; SIGHUP does not exist there, so the
    reload path is the dashboard config editor (hot-apply) or a restart."""
    if sys.platform != "win32":
        return
    report.line(
        "PASS",
        "platform",
        "Windows: supported scope is the Free tier + agent plugins; reload via"
        " the config editor or a restart (SIGHUP is unavailable), and run the"
        " proxy as a foreground/logon task (`llm-redact service install`"
        " prints a Task Scheduler command)",
    )


def _check_private(report: _Report, area: str, path: Path, want_dir_private: bool) -> None:
    if sys.platform == "win32":
        # POSIX mode bits are synthetic on Windows (regular files commonly
        # report 666) — checking them would only raise false alarms. NTFS
        # ACLs are the real control, and the user-profile defaults are
        # private to the user; say so instead of pretending to verify.
        report.line(
            "WARN",
            area,
            f"{path}: POSIX permission checks do not apply on Windows — keep"
            " this file under your user profile, where default NTFS ACLs are"
            " private to your account",
        )
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        report.line("FAIL", area, f"{path} is group/world accessible (mode {mode:03o})")
    else:
        report.line("PASS", area, f"{path} permissions ok (mode {mode:03o})")
    if want_dir_private:
        parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
        if parent_mode & 0o077:
            report.line(
                "FAIL", area, f"{path.parent} is group/world accessible (mode {parent_mode:03o})"
            )


def _check_config(report: _Report, args: argparse.Namespace) -> Config | None:
    try:
        path = args.config if args.config is not None else resolve_config_path()
    except ConfigError as problem:  # LLM_REDACT_CONFIG points nowhere
        report.line("FAIL", "config", str(problem))
        return None
    if path is None or not path.exists():
        report.line("WARN", "config", "no config file found; built-in defaults apply")
        source = "defaults"
    else:
        source = str(path)
    try:
        config = apply_env_overrides(load_config(args.config))
    except ConfigError as problem:
        report.line("FAIL", "config", f"{source}: {problem}")
        return None
    active = [name for name in _ENV_OVERRIDES if os.environ.get(name)]
    suffix = f" (env overrides: {', '.join(active)})" if active else ""
    if source != "defaults":
        report.line("PASS", "config", f"{source} parses{suffix}")
    return config


def _check_build(report: _Report, config: Config) -> None:
    """Parse is not enough: unknown rule names, unknown custom-rule
    validators, and conflicting mode targets are deliberately deferred past
    parse_config to the detector BUILD — serve refuses them at startup and a
    SIGHUP reload rejects them with only a log line. Dry-running the build
    here means a green doctor actually predicts a working serve/reload."""
    from dataclasses import replace

    from llm_redact.detection.engine import build_allowlist, build_detectors, build_modes

    detection = config.detection
    ner_note = ""
    if detection.ner.enabled:
        # doctor is read-only: NER backends load (or download) models at
        # build time, so they are checked for importability only (the ner
        # check below) and swapped out of the dry-run build.
        detection = replace(detection, ner=replace(detection.ner, enabled=False))
        ner_note = " (NER backends not built — doctor never loads models; see the ner check)"
    try:
        detectors = build_detectors(detection)
        build_modes(config.detection)
        build_allowlist(config.detection)
    except (ValueError, ConfigError, re.error) as problem:
        report.line(
            "FAIL",
            "build",
            f"config parses but does not BUILD: {problem} — serve would refuse"
            " this config and a SIGHUP reload would keep the current one",
        )
        return
    report.line("PASS", "build", f"{len(detectors)} detectors build{ner_note}")


def _check_body_cap(report: _Report, config: Config) -> None:
    # Informational: batch/file uploads bigger than the cap are rejected
    # 413 fail-closed (never forwarded unredacted) — batch workflows often
    # need a bigger cap than chat traffic does.
    mib = config.max_body_bytes / (1024 * 1024)
    report.line(
        "PASS",
        "body cap",
        f"max_body_bytes {config.max_body_bytes} (~{mib:.0f} MiB); redactable"
        " requests above this answer 413 — raise it for large batch/file"
        " uploads",
    )


def _check_proxy(report: _Report, config: Config) -> None:
    import httpx

    from llm_redact.proxy import RESERVED_PREFIX

    scheme = "https" if config.tls.enabled else "http"
    url = f"{scheme}://{config.host}:{config.port}{RESERVED_PREFIX}/status"
    try:
        response = httpx.get(url, timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError as problem:
        if config.tls.mutual:
            report.line(
                "WARN",
                "proxy",
                f"not reachable at {config.host}:{config.port} — mutual TLS is on, so this"
                " may just mean doctor has no client certificate",
            )
        else:
            report.line("WARN", "proxy", f"not running at {config.host}:{config.port} ({problem})")
        _check_port_free(report, config)
        return
    running = str(response.json().get("version", "?"))
    if running != __version__:
        report.line(
            "WARN",
            "proxy",
            f"running version {running} differs from installed {__version__} — restart to update",
        )
    else:
        report.line("PASS", "proxy", f"running {running} at {config.host}:{config.port}")


def _check_port_free(report: _Report, config: Config) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((config.host, config.port))
    except OSError:
        report.line(
            "WARN",
            "proxy",
            f"port {config.port} is in use by something that does not answer"
            " /__llm-redact/status — is another service bound there?",
        )
    else:
        report.line("PASS", "proxy", f"port {config.port} is free")
    finally:
        probe.close()


def _check_vault(report: _Report, config: Config) -> None:
    from llm_redact.config import RDBMS_BACKENDS
    from llm_redact.vault import default_vault_path

    if config.vault.backend == "memory":
        report.line("PASS", "vault", "in-memory backend (nothing on disk)")
    elif config.vault.backend in RDBMS_BACKENDS:
        _check_vault_rdbms(report, config)
    else:
        path = Path(config.vault.path).expanduser() if config.vault.path else default_vault_path()
        if path.exists():
            _check_private(report, "vault", path, want_dir_private=True)
        else:
            report.line("WARN", "vault", f"{path} not created yet (first request creates it)")

    if config.vault.encryption == "fernet":
        if importlib.util.find_spec("cryptography") is None:
            report.line(
                "FAIL",
                "vault",
                'encryption = "fernet" but the crypto extra is not installed;'
                " install it: pip install 'llm-redact-proxy[crypto]'",
            )
        if not os.environ.get("LLM_REDACT_VAULT_KEY"):
            report.line(
                "FAIL",
                "vault",
                'encryption = "fernet" but LLM_REDACT_VAULT_KEY is not set'
                " (generate one: llm-redact vault gen-key)",
            )
        elif importlib.util.find_spec("cryptography") is not None:
            _check_vault_key_matches(report, config)


def _check_vault_rdbms(report: _Report, config: Config) -> None:
    """Read-only RDBMS vault posture: driver importable and DSN shape valid
    (no connection is attempted), managed-DBMS recognition, and the off-box
    plaintext rule — mirroring exactly what serve will enforce."""
    from llm_redact import vault_rdbms
    from llm_redact.config import ConfigError

    backend = config.vault.backend
    try:
        vault_rdbms.validate_connector(config.vault)
    except ConfigError as problem:
        report.line("FAIL", "vault", str(problem))
        return
    report.line("PASS", "vault", f"{backend} driver importable, DSN shape valid (not probed)")

    cloud = vault_rdbms.managed_dbms_cloud(config.vault)
    if cloud is not None:
        report.line(
            "PASS",
            "vault",
            f"managed-DBMS host recognized ({cloud}): included with llm-redact-pro"
            " (persistent vault), no separate cloud entitlement required",
        )
    violation = vault_rdbms.offbox_violation(config.vault)
    if violation is not None:
        report.line("FAIL", "vault", f"{violation} — the proxy will refuse to start")
    elif config.vault.encryption != "fernet":
        if os.environ.get(vault_rdbms.ENV_REMOTE_PLAINTEXT) == "1":
            report.line(
                "WARN",
                "vault",
                "LLM_REDACT_VAULT_REMOTE_PLAINTEXT=1: plaintext vault rows may"
                " leave this machine (the off-box rule is bypassed)",
            )
        elif backend == "dbapi":
            report.line(
                "WARN",
                "vault",
                'backend "dbapi" DSNs are opaque — locality cannot be verified;'
                ' keep the database local or set [vault] encryption = "fernet"',
            )


def _check_vault_key_matches(report: _Report, config: Config) -> None:
    """Turn 'key present' into 'key MATCHES this vault' — a wrong key otherwise
    reports PASS at doctor time and only fails at the first request (open)."""
    import sqlite3

    from llm_redact.vault import VaultKeyError, default_vault_path

    path = Path(config.vault.path).expanduser() if config.vault.path else default_vault_path()
    if config.vault.backend != "sqlite" or not path.exists():
        # Memory/RDBMS backends verify the key at open (key_check row);
        # nothing on local disk to compare against here.
        report.line("PASS", "vault", "fernet configured and key present")
        return
    from llm_redact.config import ConfigError
    from llm_redact.registry import get_registry

    try:
        cipher = get_registry().build_cipher(config.vault)
    except (VaultKeyError, ConfigError) as problem:
        report.line("FAIL", "vault", str(problem))
        return
    assert cipher is not None  # caller gates on encryption == "fernet"
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        if int(conn.execute("PRAGMA user_version").fetchone()[0]) < 3:
            report.line("WARN", "vault", f"{path} is not encrypted yet (migrates on next serve)")
            return
        row = conn.execute("SELECT value FROM vault_meta WHERE key = 'key_check'").fetchone()
    finally:
        conn.close()
    if row is not None and str(row[0]) != cipher.key_check():
        report.line("FAIL", "vault", f"LLM_REDACT_VAULT_KEY does not match the vault at {path}")
    else:
        report.line("PASS", "vault", "fernet key matches the vault")


def _check_extras(report: _Report, config: Config) -> None:
    if config.detection.ner.enabled:
        # EVERY active backend, not just the legacy single one — a
        # multi-backend config with one missing extra fails serve at startup.
        for backend in config.detection.ner.active_backends():
            if importlib.util.find_spec(_NER_MODULES[backend]) is None:
                report.line(
                    "FAIL",
                    "ner",
                    f'backend "{backend}" but its extra is not installed;'
                    f" install it: uv sync --extra {_NER_EXTRAS[backend]}",
                )
            else:
                report.line("PASS", "ner", f"{backend} backend importable")
    if config.otel.enabled:
        if importlib.util.find_spec("opentelemetry") is None:
            report.line(
                "FAIL",
                "otel",
                "enabled but the otel extra is not installed;"
                " install it: pip install 'llm-redact-proxy[otel]'",
            )
        else:
            report.line("PASS", "otel", "sdk importable")
    if importlib.util.find_spec("websockets") is None:
        # WARN, not FAIL: HTTP proxying is fully functional without it.
        report.line(
            "WARN",
            "realtime",
            "websockets not installed — WebSocket APIs (OpenAI Realtime, Gemini"
            " Live) will be refused; install: pip install 'llm-redact-proxy[realtime]'",
        )
    else:
        report.line("PASS", "realtime", "websockets importable (WS relay active)")


def _check_tls_and_bind(report: _Report, config: Config) -> None:
    for label, value in (
        ("certfile", config.tls.certfile),
        ("keyfile", config.tls.keyfile),
        ("client_ca", config.tls.client_ca),
    ):
        if value is None:
            continue
        path = Path(value).expanduser()
        if path.exists() and os.access(path, os.R_OK):
            report.line("PASS", "tls", f"{label} {path} readable")
        else:
            report.line("FAIL", "tls", f"{label} {path} missing or unreadable")
    try:
        validate_bind_security(config.host, config.tls, os.environ)
    except ConfigError as problem:
        report.line("FAIL", "bind", str(problem))
    else:
        if config.host not in ("127.0.0.1", "localhost", "::1"):
            report.line("PASS", "bind", f"non-loopback {config.host} allowed (mTLS or env hatch)")
        else:
            report.line("PASS", "bind", f"loopback bind ({config.host})")


def _check_license(report: _Report, config: Config) -> None:
    """Resolve the license exactly as serve would and report it —
    informational only. The FOSS (AGPL-3.0) core has no tier gates: every
    subsystem in this repository works keyless, and a config that requests
    an llm-redact-pro-only subsystem without that package fails closed in
    the build dry-run with the feature and package named."""
    from llm_redact.licensing import resolve_license

    resolved = resolve_license(
        env=dict(os.environ),
        config_key=config.license.key,
        config_key_file=config.license.key_file,
    )
    for warning in resolved.warnings:
        report.line("WARN", "license", warning)
    if resolved.license is None:
        report.line("PASS", "license", "no key configured (FOSS core: nothing is gated)")
    else:
        report.line(
            "PASS",
            "license",
            f"{resolved.tier} tier ({resolved.license.org}),"
            f" expires {resolved.license.expires.isoformat()}",
        )


def _check_licensed_features(report: _Report) -> None:
    """The honest open-core signal (llm-redact-pro LICENSING.md): is the paid
    ``llm-redact-pro`` package installed? Never a FAIL — the FOSS core is
    fully functional alone, and any pro-only *config* without the package
    already fails closed in the build dry-run. But an operator should never
    have to guess whether the paid subsystems are even present, and a
    package that is installed yet whose plugin failed to register (paid
    features silently off) is exactly the kind of quiet downgrade this
    project surfaces."""
    from llm_redact.registry import get_registry, loaded_plugins, pro_package_installed

    if not pro_package_installed():
        report.line(
            "PASS",
            "license",
            "licensed-features package not installed (FOSS core is complete;"
            " pro-only config fails closed)",
        )
        return
    get_registry()  # ensure the entry-point scan ran so loaded_plugins() is authoritative
    plugins = loaded_plugins()
    if plugins:
        report.line(
            "PASS",
            "license",
            f"licensed-features package installed ({', '.join(sorted(plugins))} active)",
        )
    else:
        report.line(
            "WARN",
            "license",
            "licensed-features package present but its plugin did not register — paid"
            " features stay OFF (reinstall llm-redact-pro or check the startup log)",
        )


def _check_posture(report: _Report, config: Config) -> None:
    """Loud reminders for every configured coverage opt-out. Each is a
    deliberate feature, so these are WARN (never FAIL) — but an operator
    reading `doctor` should never be surprised that some traffic is
    forwarded unredacted. When nothing is opted out, one PASS says so."""
    from llm_redact.detection.engine import active_rule_names

    opted_out = False

    warn_rules = sorted(name for name, mode in config.detection.modes if mode == "warn")
    if warn_rules:
        opted_out = True
        report.line(
            "WARN",
            "posture",
            f"warn mode on {', '.join(warn_rules)} — matched values are FORWARDED"
            " upstream (observation only, not protection)",
        )

    detection_off = sorted(
        name for name, provider in config.providers.items() if not provider.detection
    )
    if detection_off:
        opted_out = True
        report.line(
            "WARN",
            "posture",
            f"detection disabled for provider(s) {', '.join(detection_off)} — ALL"
            " their requests are forwarded unredacted (rehydration still runs)",
        )

    exempt = config.detection.mcp_exempt_servers
    if exempt:
        opted_out = True
        report.line(
            "WARN",
            "posture",
            f"{len(exempt)} MCP server(s) exempt from detection — their content"
            " blocks are forwarded unredacted",
        )

    if config.detection.languages is not None:
        inactive = sorted(set(config.detection.enabled) - set(active_rule_names(config.detection)))
        if inactive:
            opted_out = True
            report.line(
                "WARN",
                "posture",
                f"language scope {list(config.detection.languages)} leaves"
                f" {', '.join(inactive)} unbuilt — those IDs are not detected",
            )

    if not opted_out:
        report.line("PASS", "posture", "no coverage opt-outs configured (all traffic redacted)")


def run_doctor(args: argparse.Namespace) -> int:
    report = _Report(json_mode=getattr(args, "json", False))
    config = _check_config(report, args)
    if config is None:
        report.finish()
        return 1
    _check_platform(report)
    _check_license(report, config)
    _check_licensed_features(report)
    _check_build(report, config)
    _check_tls_and_bind(report, config)
    _check_body_cap(report, config)
    _check_proxy(report, config)
    _check_vault(report, config)
    _check_extras(report, config)
    _check_posture(report, config)
    if config.audit.enabled:
        from llm_redact.audit import default_audit_path

        audit_path = (
            Path(config.audit.path).expanduser() if config.audit.path else default_audit_path()
        )
        if audit_path.exists():
            _check_private(report, "audit", audit_path, want_dir_private=False)
        else:
            report.line("WARN", "audit", f"{audit_path} not created yet")
        if config.audit.tamper_evident:
            from llm_redact.audit import AUDIT_HMAC_ENV

            if os.environ.get(AUDIT_HMAC_ENV):
                report.line("PASS", "audit", "tamper-evident chain enabled with a key present")
            else:
                report.line(
                    "FAIL",
                    "audit",
                    f"tamper_evident = true but {AUDIT_HMAC_ENV} not set — the proxy will"
                    " refuse to start (the HMAC key comes from the environment, never the"
                    " config file)",
                )
    _check_audit_s3(report, config)
    _check_audit_azure(report, config)
    report.finish()
    return 1 if report.failed else 0


def run_config_show(args: argparse.Namespace) -> int:
    """`llm-redact config show`: the effective configuration and where each
    layer came from — file truth re-emitted as TOML, with active env
    overrides named separately (CLI > env > file > defaults). Safe to print
    by construction: credentials never live in the config file (they are
    env-only across the board), so the emitted TOML carries no secrets."""
    from llm_redact.config_write import emit_config_toml

    try:
        path = args.config if args.config is not None else resolve_config_path()
    except ConfigError as problem:
        print(f"config: {problem}")
        return 1
    if args.path:
        print(str(path) if path is not None and path.exists() else "(defaults; no config file)")
        return 0
    try:
        config = apply_env_overrides(load_config(args.config))
    except ConfigError as problem:
        print(f"config: {problem}")
        return 1
    source = str(path) if path is not None and path.exists() else "(defaults; no config file)"
    print(f"# source: {source}")
    overrides = [name for name in _ENV_OVERRIDES if os.environ.get(name)]
    if overrides:
        print(f"# env overrides active: {', '.join(overrides)} (already applied below)")
    print()
    print(emit_config_toml(config, banner=False), end="")
    return 0


def _check_audit_azure(report: _Report, config: Config) -> None:
    from llm_redact.audit_s3 import AZURE_STORAGE_KEY_ENV

    az = config.audit.azure
    if not az.enabled:
        return
    if not os.environ.get(AZURE_STORAGE_KEY_ENV):
        report.line(
            "FAIL",
            "audit.azure",
            f"enabled but {AZURE_STORAGE_KEY_ENV} not set — batches will be dropped"
            " (the account key comes from the environment, never the config file)",
        )
    else:
        host = az.endpoint_url or f"{az.account}.blob.core.windows.net"
        report.line(
            "PASS",
            "audit.azure",
            f"SharedKey sink configured (container {az.container} via {host});"
            " metadata rows leave this machine",
        )
    _check_audit_encryption(report, "audit.azure", az.encryption)


def _check_audit_encryption(report: _Report, area: str, encryption: str) -> None:
    """Batch-encryption posture for one enabled sink: key + extra present
    (the same fail-closed pair serve enforces), or plaintext noted."""
    from llm_redact.audit_s3 import AUDIT_ENC_KEY_ENV, audit_enc_key_from_env

    if encryption != "fernet":
        return
    if importlib.util.find_spec("cryptography") is None:
        report.line(
            "FAIL",
            area,
            'encryption = "fernet" but the crypto extra is not installed —'
            " the proxy will refuse to start",
        )
    elif audit_enc_key_from_env() is None:
        report.line(
            "FAIL",
            area,
            f'encryption = "fernet" but {AUDIT_ENC_KEY_ENV} not set — the proxy'
            " will refuse to start (the key comes from the environment, never"
            " the config file)",
        )
    else:
        report.line("PASS", area, "batches are Fernet-encrypted client-side before upload")


def _check_audit_s3(report: _Report, config: Config) -> None:
    from llm_redact.audit_s3 import credential_env_names

    s3 = config.audit.s3
    if not s3.enabled:
        return
    # Presence only — never a byte of the values themselves. The credential
    # env vars vary by provider (GCS uses its own HMAC interop keys).
    access_env, secret_env, _ = credential_env_names(s3.provider)
    missing = [name for name in (access_env, secret_env) if not os.environ.get(name)]
    if missing:
        report.line(
            "FAIL",
            "audit.s3",
            f"enabled but {' and '.join(missing)} not set — batches will be"
            " dropped (credentials come from the environment, never the config file)",
        )
    else:
        target = {
            "aws": f"s3.{s3.region}.amazonaws.com",
            "gcs": "storage.googleapis.com",
        }.get(s3.provider, s3.endpoint_url or "?")
        report.line(
            "PASS",
            "audit.s3",
            f"{s3.provider} sink configured (bucket {s3.bucket} via {target});"
            " metadata rows leave this machine",
        )
    _check_audit_encryption(report, "audit.s3", s3.encryption)
