"""`llm-redact doctor`: read-only environment diagnostics.

Prints one PASS/WARN/FAIL line per check and exits non-zero if anything
FAILs. Never prints secret values — paths, modes, and versions only. A
WARN is something worth knowing (proxy not running, no config file); a
FAIL is something that will break serving or silently weaken protection
(unreadable TLS files, loose vault permissions, an extra the config
requires but that is not installed).
"""

import argparse
import dataclasses
import importlib.metadata
import importlib.util
import os
import re
import socket
import stat
import sys
import urllib.parse
from pathlib import Path
from typing import Any

from llm_redact import __version__
from llm_redact.config import (
    Config,
    ConfigError,
    VaultKmsConfig,
    apply_env_overrides,
    dial_url,
    load_config,
    resolve_config_path,
    validate_bind_security,
)

_NER_MODULES = {
    "spacy": "spacy",
    "gliner": "gliner",
    "gliner2": "gliner2",
    "presidio": "presidio_analyzer",
    "stanza": "stanza",
    "hf": "transformers",
}
# Modules a backend's library imports only when it loads a model, which
# the library's own import does not need: gliner2 imports peft in its
# extraction runtime, so without it `serve` fails at the model load.
_NER_LOAD_MODULES = {"gliner2": ("peft",)}
_NER_EXTRAS = {
    "spacy": "ner",
    "gliner": "gliner",
    "gliner2": "gliner2",
    "presidio": "presidio",
    "stanza": "stanza",
    "hf": "hf",
}
# Backends whose library runs on torch. Their extras pin torch>=2.6: 2.6
# fixed CVE-2025-32434, a torch.load(weights_only=True) bypass, and GLiNER
# loads pytorch_model.bin checkpoints that way. An environment built without
# the extra can hold the library without torch (transformers imports fine
# and fails only when a model loads) or with an older torch.
_TORCH_BACKENDS = frozenset({"gliner", "gliner2", "stanza", "hf"})
_TORCH_FLOOR = (2, 6)
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
            from llm_redact import __version__ as version
            from llm_redact.jsonwalk import json_text

            print(json_text({"version": version, "failed": self.failed, "checks": self.rows}))


def _check_platform(report: _Report) -> None:
    """Windows-specific posture (silent elsewhere — no noise on the
    platforms where nothing differs). The supported Windows scope is the
    Free tier plus the agent plugins; SIGHUP does not exist there, so the
    reload path is a restart (or, with llm-redact-pro, the dashboard config
    editor's hot-apply)."""
    if sys.platform != "win32":
        return
    report.line(
        "PASS",
        "platform",
        "Windows: supported scope is the Free tier + agent plugins; reload via"
        " a restart or the llm-redact-pro config editor (SIGHUP is unavailable), and run the"
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
    # Informational: batch/file uploads bigger than the caps are rejected
    # 413 fail-closed (never forwarded unredacted) — batch workflows often
    # need bigger caps than chat traffic does.
    mib = config.max_body_bytes / (1024 * 1024)
    report.line(
        "PASS",
        "body cap",
        f"max_body_bytes {config.max_body_bytes} (~{mib:.0f} MiB), max_body_strings"
        f" {config.max_body_strings}; redactable requests above either answer 413"
        " — raise them for large batch/file uploads",
    )


def _check_proxy(report: _Report, config: Config) -> None:
    import httpx

    from llm_redact.proxy import RESERVED_PREFIX

    scheme = "https" if config.tls.enabled else "http"
    # Where the configured bind is reached from this machine: loopback for
    # a wildcard bind, an IPv6 literal in brackets.
    base = dial_url(config.host, config.port, scheme=scheme)
    where = urllib.parse.urlsplit(base).netloc
    try:
        response = httpx.get(f"{base}{RESERVED_PREFIX}/status", timeout=2.0)
        response.raise_for_status()
    except (httpx.HTTPError, httpx.InvalidURL) as problem:
        if config.tls.mutual:
            report.line(
                "WARN",
                "proxy",
                f"not reachable at {where} — mutual TLS is on, so this"
                " may just mean doctor has no client certificate",
            )
        else:
            report.line("WARN", "proxy", f"not running at {where} ({problem})")
        _check_port_free(report, config)
        return
    status = response.json()
    running = str(status.get("version", "?"))
    if running != __version__:
        report.line(
            "WARN",
            "proxy",
            f"running version {running} differs from installed {__version__} — restart to update",
        )
    else:
        report.line("PASS", "proxy", f"running {running} at {where}")
    _check_live_vault(report, status)


def _check_live_vault(report: _Report, status: Any) -> None:
    """What only the running proxy knows about its vault (doctor never
    connects to a vault database): an RDBMS response map its database user
    could not ALTER to add the kind column, so stored objects' owner
    records share the Responses rows' bound — heavy Responses traffic can
    then push them out."""
    vault = status.get("vault") if isinstance(status, dict) else None
    if isinstance(vault, dict) and vault.get("owner_bound_shared") is True:
        report.line(
            "WARN",
            "vault",
            "the running proxy's database user could not add the kind column to"
            " llm_redact_response_sessions: stored objects' owner records share the"
            " Responses bound (ALTER TABLE llm_redact_response_sessions ADD kind VARCHAR(8)"
            " DEFAULT 'response' NOT NULL, then restart)",
        )


def _check_port_free(report: _Report, config: Config) -> None:
    # An IPv6 bind needs an IPv6 socket: an IPv4 one fails to bind "::1"
    # whether or not the port is free.
    family = socket.AF_INET6 if ":" in config.host else socket.AF_INET
    try:
        probe = socket.socket(family, socket.SOCK_STREAM)
    except OSError as exc:
        kind = "IPv6" if family == socket.AF_INET6 else "IPv4"
        report.line(
            "WARN",
            "proxy",
            f"cannot probe port {config.port} on {config.host}: this machine cannot open"
            f" an {kind} socket ({type(exc).__name__})",
        )
        return
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

    if config.vault.backend != "memory":
        _check_map_writes(report, config)

    if config.vault.encryption == "fernet":
        if importlib.util.find_spec("cryptography") is None:
            report.line(
                "FAIL",
                "vault",
                'encryption = "fernet" but the crypto extra is not installed;'
                " install it: pip install 'llm-redact-proxy[crypto]'",
            )
        if config.vault.kms is not None:
            _check_vault_kms(report, config)
        elif not os.environ.get("LLM_REDACT_VAULT_KEY"):
            report.line(
                "FAIL",
                "vault",
                'encryption = "fernet" but LLM_REDACT_VAULT_KEY is not set'
                " (generate one: llm-redact vault gen-key)",
            )
        elif importlib.util.find_spec("cryptography") is not None:
            _check_vault_key_matches(report, config)


def _check_map_writes(report: _Report, config: Config) -> None:
    """The effective ``[vault] map_writes`` of a persistent vault (the
    in-memory one keeps no durable map): informational, never a WARN — both
    modes are safe (a record another replica cannot read yet is refused or
    sealed there, never a wrong value); they differ in what a follow-up
    reaching another replica meets."""
    from llm_redact.config import map_writes_mode

    mode = map_writes_mode(config.vault)
    origin = "set" if config.vault.map_writes is not None else "default"
    if mode == "before_answer":
        detail = (
            f"an answer waits (at most {config.vault.map_write_wait_seconds:g} s,"
            " map_write_wait_seconds) for its durable map writes, so a follow-up"
            " reaching any replica sharing the vault finds the record"
        )
    else:
        detail = (
            "durable map writes land after the answer: this process answers at once,"
            " another replica sharing the vault reads the record once its write landed"
        )
    report.line("PASS", "vault", f"map_writes = {mode} ({origin}): {detail}")


def _check_vault_kms(report: _Report, config: Config) -> None:
    """[vault.kms] posture — read-only and OFFLINE: doctor never calls the
    KMS (no unwrap, no credential fetch), so a key that does not match the
    vault shows up only in `serve --check` / `vault verify`. Checks what
    serve would refuse on sight: an ambiguous local key, no plugin able to
    unwrap, a missing wrapped value; for hashicorp, the server address."""
    from llm_redact.registry import get_registry
    from llm_redact.vault_crypto import ambiguous_local_key, resolve_vault_key

    kms = config.vault.kms
    assert kms is not None  # caller gates on it
    ambiguous = ambiguous_local_key(config.vault)
    if ambiguous is not None:
        report.line(
            "FAIL",
            "vault",
            f"[vault.kms] is configured and {ambiguous} is also set — the proxy will refuse"
            f" to start (unset {ambiguous}: the key comes only from the KMS)",
        )
    if get_registry().resolve_vault_key is resolve_vault_key:
        report.line(
            "FAIL",
            "vault",
            "[vault.kms] needs llm-redact-pro (a version that supports it) to unwrap the"
            " key — the proxy will refuse to start",
        )
    if kms.wrapped_key_env and not os.environ.get(kms.wrapped_key_env, "").strip():
        report.line(
            "FAIL", "vault", f"[vault.kms] wrapped_key_env {kms.wrapped_key_env} is not set"
        )
    elif kms.wrapped_key_file and not Path(kms.wrapped_key_file).expanduser().is_file():
        report.line(
            "FAIL", "vault", f"[vault.kms] wrapped_key_file {kms.wrapped_key_file} does not exist"
        )
    if kms.provider == "hashicorp":
        _check_vault_kms_hashicorp(report, kms)
    report.line(
        "PASS",
        "vault",
        f"key source kms:{kms.provider} — unwrapped at startup with the proxy's identity"
        " (not probed: doctor never calls the KMS; `serve --check` does)",
    )


def _check_vault_kms_hashicorp(report: _Report, kms: VaultKmsConfig) -> None:
    from llm_redact.config import kms_address_problem

    address = kms.address or os.environ.get("VAULT_ADDR", "").strip()
    if not address:
        report.line("FAIL", "vault", "[vault.kms] hashicorp: no address and VAULT_ADDR is not set")
    elif (problem := kms_address_problem(address)) is not None:
        report.line("FAIL", "vault", f"[vault.kms] hashicorp: the Vault address {problem}")
    if kms.auth == "token" and not (
        os.environ.get("VAULT_TOKEN") or os.environ.get("VAULT_TOKEN_FILE")
    ):
        report.line(
            "FAIL",
            "vault",
            "[vault.kms] hashicorp auth = token: set VAULT_TOKEN or VAULT_TOKEN_FILE",
        )
    elif kms.auth == "kubernetes" and not Path(kms.service_account_token_file).is_file():
        report.line(
            "WARN",
            "vault",
            f"[vault.kms] hashicorp auth = kubernetes: {kms.service_account_token_file}"
            " does not exist here (fine if doctor runs outside the pod)",
        )


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
    if config.vault.rdbms.auth == "identity":
        _check_vault_identity(report, config)

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


def _check_vault_identity(report: _Report, config: Config) -> None:
    """``[vault.rdbms] auth = "identity"``: where the password comes from.
    No credential source is built and nothing is fetched (no network)."""
    from llm_redact.registry import pro_package_installed

    if not pro_package_installed():
        report.line(
            "FAIL",
            "vault",
            'auth = "identity" requires the llm-redact-pro package — the proxy will refuse'
            " to start",
        )
        return
    from llm_redact.vault_rdbms import ENV_TLS_UNVERIFIED, identity_tls_unverified

    if identity_tls_unverified(config.vault):
        report.line(
            "WARN",
            "vault",
            f"{ENV_TLS_UNVERIFIED}=1: the database token goes over TLS that does not"
            " verify the server certificate (whoever answers the handshake gets it)",
        )
    report.line(
        "PASS",
        "vault",
        f'auth = "identity": the database password is a short-lived {config.vault.rdbms.cloud}'
        " token minted from the proxy's cloud identity at every connect, over TLS"
        " (identity not probed)",
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


def _torch_problem() -> str | None:
    """Why the installed torch cannot run a torch backend, or None.

    Reads the distribution metadata only: doctor never imports torch.
    """
    if importlib.util.find_spec("torch") is None:
        return "needs torch, which is not installed"
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None  # importable without metadata (a source tree): nothing to compare
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is not None and (int(match[1]), int(match[2])) < _TORCH_FLOOR:
        return f"needs torch >= 2.6 (CVE-2025-32434), but torch {version} is installed"
    return None


def _transformers_problem() -> str | None:
    """Why the installed transformers cannot load an hf model, or None
    (distribution metadata only, as the build reads it)."""
    from llm_redact.detection.hf_ner import transformers_problem

    return transformers_problem()


def _check_extras(report: _Report, config: Config) -> None:
    if config.detection.ner.enabled:
        # EVERY active backend, not just the legacy single one — a
        # multi-backend config with one missing extra fails serve at startup.
        for backend in config.detection.ner.active_backends():
            hint = f"install it: uv sync --extra {_NER_EXTRAS[backend]}"
            modules = (_NER_MODULES[backend], *_NER_LOAD_MODULES.get(backend, ()))
            if config.detection.ner.onnx_for(backend) is not None:
                # [detection.ner.onnx]: the gliner extra brings onnxruntime
                # (gliner >= 0.2.29 no longer depends on it).
                modules = (*modules, "onnxruntime")
            if any(importlib.util.find_spec(module) is None for module in modules):
                report.line(
                    "FAIL", "ner", f'backend "{backend}" but its extra is not installed; {hint}'
                )
            elif backend in _TORCH_BACKENDS and (problem := _torch_problem()) is not None:
                report.line("FAIL", "ner", f'backend "{backend}" {problem}; {hint}')
            elif backend == "hf" and (problem := _transformers_problem()) is not None:
                # What the hf build refuses too (hf_ner.TRANSFORMERS_MINIMUM).
                report.line(
                    "FAIL",
                    "ner",
                    f'backend "hf" {problem}; upgrade it:'
                    " uv sync --extra hf --upgrade-package transformers",
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
        from llm_redact.registry import pro_package_installed

        # Keyless the FOSS core gates nothing; with llm-redact-pro installed
        # its factories run the Free tier and refuse its paid subsystems, so
        # "nothing is gated" would contradict the FAIL rows those produce.
        message = (
            "no key configured (Free tier: llm-redact-pro features refuse to start"
            " without a key that selects a paid tier)"
            if pro_package_installed()
            else "no key configured (FOSS core: nothing is gated)"
        )
        report.line("PASS", "license", message)
    else:
        report.line(
            "PASS",
            "license",
            f"{resolved.tier} tier ({resolved.license.org}),"
            f" expires {resolved.license.expires.isoformat()}",
        )


def _check_licensed_features(report: _Report, config: Config) -> None:
    """The honest open-core signal (llm-redact-pro LICENSING.md): is the paid
    ``llm-redact-pro`` package installed? Never a FAIL — the FOSS core is
    fully functional alone, and any pro-only *config* without the package
    already fails closed in the build dry-run. But an operator should never
    have to guess whether the paid subsystems are even present, and a
    package that is installed yet whose plugin failed to register (paid
    features silently off) is exactly the kind of quiet downgrade this
    project surfaces."""
    from llm_redact.config import unsupported_plugin_capabilities
    from llm_redact.registry import get_registry, loaded_plugins, pro_package_installed

    if not pro_package_installed():
        report.line(
            "PASS",
            "license",
            "licensed-features package not installed (FOSS core is complete;"
            " pro-only config fails closed)",
        )
        return
    registry = get_registry()  # the entry-point scan runs: loaded_plugins() is authoritative
    plugins = loaded_plugins()
    if plugins:
        report.line(
            "PASS",
            "license",
            f"licensed-features package installed ({', '.join(sorted(plugins))} active)",
        )
        unsupported = unsupported_plugin_capabilities(config, registry.config_capabilities)
        if unsupported is not None:
            report.line("FAIL", "license", f"{unsupported} — the proxy will refuse to start")
    else:
        report.line(
            "WARN",
            "license",
            "licensed-features package present but its plugin did not register — paid"
            " features stay OFF (reinstall llm-redact-pro or check the startup log)",
        )


def _check_ner_labels(report: _Report, config: Config) -> None:
    """The NER label policy's config-only warnings (doctor never loads a
    model): configured raw entities whose placeholder type changes in
    2.0.0 — the same lines serve logs at startup."""
    from llm_redact.detection.engine import ner_warnings

    for warning in ner_warnings(config.detection):
        report.line("WARN", "ner", warning)


def _check_models(report: _Report, config: Config) -> None:
    """Where the NER models of the Hugging Face Hub backends (gliner, gliner2,
    hf)
    come from: the download and pickle switches, each model's pin, and
    whether its files are in the local Hugging Face cache (or its folder)
    at that pin. Never loads a model and never touches the network — file
    lookups in the local cache only. Silent while NER is off."""
    from llm_redact.detection.model_sources import hub_sources

    ner = config.detection.ner
    sources = hub_sources(ner) if ner.enabled else []
    if not sources:
        return
    if ner.allow_download:
        report.line(
            "WARN",
            "models",
            "allow_download = true: the proxy's startup may download model weights from"
            " huggingface.co (the model id and revision, never request content; a reload"
            " never downloads)",
        )
    else:
        report.line(
            "PASS",
            "models",
            "downloads off (allow_download = false): models load from the local Hugging"
            " Face cache or local folders only",
        )
    if ner.allow_pickle_weights and "hf" in ner.active_backends():
        report.line(
            "WARN",
            "models",
            "allow_pickle_weights = true: an hf model without safetensors weights loads"
            " pytorch_model.bin, a pickle (loading a pickle can run code)",
        )
    try:
        import huggingface_hub  # noqa: F401  (doctor only asks the local cache)

        cache = True
    except ImportError:
        cache = False
    for source in sources:
        _check_model_pin(report, source)
        _check_model_threshold(report, ner, source)
        _check_model_files(report, ner, source, cache=cache)


def _check_model_threshold(report: _Report, ner: Any, source: Any) -> None:
    """A row naming the confidence threshold a Hub backend runs at when the
    model catalog has a default for its model (``NerConfig.score_threshold_
    for``): PASS when the catalog's default applies (the configuration does
    not show it), WARN when a configured ``score_threshold`` overrides a
    different catalog default — files written before 1.12.0 by ``config
    show`` or the config editor carry 0.5, which would otherwise look
    deliberate. Silent for a model without a catalog default, and for a
    folder whose sidecar cannot be read (its pin row FAILs already)."""
    if source.sidecar_problem is not None:
        return
    value, origin = ner.score_threshold_for(source.backend)
    if origin == "catalog":
        report.line(
            "PASS",
            "models",
            f"{source.backend}: score_threshold {value} is the model catalog's default for"
            f" {source.model_id} ([detection.ner] score_threshold overrides it)",
        )
        return
    default, default_origin = dataclasses.replace(ner, score_threshold=None).score_threshold_for(
        source.backend
    )
    if origin == "config" and default_origin == "catalog" and default != value:
        report.line(
            "WARN",
            "models",
            f"{source.backend}: [detection.ner] score_threshold {value} overrides the model"
            f" catalog's default {default} for {source.model_id}; delete the key to run at"
            " the default (files written before 1.12.0 by `config show` or the config"
            " editor carry 0.5)",
        )


def _catalog_note(source: Any) -> str:
    """What the model catalog says, as a row suffix."""
    entry = source.entry
    if entry is None:
        return " (not in the model catalog)" if source.model_id is not None else ""
    note = f" (model catalog: {entry.status}, {entry.license})"
    return f"{note}: {entry.describe()}" if entry.status == "caution" else note


def _check_model_pin(report: _Report, source: Any) -> None:
    """Which model and commit one Hub backend loads, and what the model
    catalog says about it: a WARN for a model nothing pins and for a model
    the catalog lists as restricted (the startup warning's text), a FAIL
    when an installed library is older than the model needs."""
    from llm_redact.detection.model_catalog import SIDECAR_NAME

    where = f"{source.backend}: {source.model}"
    note = _catalog_note(source)
    if source.sidecar_problem is not None:
        report.line("FAIL", "models", f"{source.backend}: {source.sidecar_problem}")
    elif source.local and source.model_id is None:
        report.line(
            "PASS",
            "models",
            f"{where} is a local folder without {SIDECAR_NAME} (it loads as it is;"
            " nothing identifies the model)",
        )
    elif source.local:
        revision = source.revision or "an unrecorded revision"
        report.line(
            "PASS",
            "models",
            f"{where} is a local folder holding {source.model_id} at {revision}{note}",
        )
    elif source.pinned:
        by = "[detection.ner.revisions]" if source.pinned_by == "config" else "the model catalog"
        report.line("PASS", "models", f"{where} pinned at {source.revision} by {by}{note}")
    else:
        report.line(
            "WARN",
            "models",
            f"{where} has no pin{note}: the newest cached revision of its default branch"
            f" loads; pin a commit in [detection.ner.revisions] {source.backend}"
            " (`llm-redact models pull` prints the one it fetches)",
        )
    restricted = source.restricted_warning()
    if restricted is not None:
        report.line("WARN", "models", restricted)
    for problem in _version_problems(source):
        report.line("FAIL", "models", problem)


def _version_problems(source: Any) -> list[str]:
    """The libraries installed older than the model catalog says ``source``'s
    model needs (distribution metadata only: nothing is imported). A library
    that is not installed is the ``ner`` check's FAIL."""
    from llm_redact.detection.model_sources import version_problems

    return version_problems(source.entry, source.backend, source.model)


def _check_model_files(report: _Report, ner: Any, source: Any, *, cache: bool) -> None:
    """Whether one Hub backend's model files are where its load reads them,
    complete: the local cache at its pin (base model included), or its
    folder. A model missing from the cache FAILs while downloads are off —
    after an upgrade the likeliest startup failure."""
    from llm_redact.config import ConfigError
    from llm_redact.detection.model_files import ModelNotCached, is_local
    from llm_redact.detection.model_sources import local_files

    if not cache and not source.local:
        report.line(
            "WARN",
            "models",
            f"{source.backend}: {source.model}: the local Hugging Face cache was not checked:"
            f" huggingface_hub is not installed (the {source.backend} extra installs it)",
        )
        return
    try:
        files = local_files(ner, source)
    except ModelNotCached as missing:
        if ner.allow_download:
            report.line(
                "WARN",
                "models",
                f"{source.backend}: {missing.what} {missing.model} is not (completely) in the"
                " local Hugging Face cache; the proxy's startup will fetch it"
                " (allow_download = true)",
            )
        else:
            report.line("FAIL", "models", str(missing))
        return
    except ConfigError as problem:
        report.line("FAIL", "models", str(problem))
        return
    place = "its folder" if source.local else "the local Hugging Face cache"
    report.line(
        "PASS",
        "models",
        f"{source.backend}: {source.model}: every file the loader reads is in {place}",
    )
    backbone = files.backbone
    if backbone is not None and files.backbone_revision is None and not is_local(backbone):
        report.line(
            "WARN",
            "models",
            f"{source.backend}: {source.model} takes its tokenizer and encoder"
            f" configuration from its base model {backbone}, whose revision the model"
            " catalog does not pin: the newest cached revision loads",
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

    # Stated either way, as a PASS: a binary file cannot be redacted at all
    # (like base64 media in a chat body, the documented non-goal), so
    # forwarding it with the client's own key is the default, not an
    # opt-out — but the operator must know those bytes leave unread.
    if config.detection.binary_uploads == "forward":
        report.line(
            "PASS",
            "posture",
            'binary_uploads = "forward": binary file uploads (PDF, images, archives) sent'
            " with the client's own key are forwarded UNSCANNED (counted in /status);"
            " text files are redacted, and a credential the proxy holds never carries one",
        )
    else:
        report.line(
            "PASS",
            "posture",
            'binary_uploads = "refuse": binary file uploads are refused (400); text files'
            " are redacted",
        )

    opted_out = _check_overrides(report, config) or opted_out

    if not opted_out:
        report.line("PASS", "posture", "no coverage opt-outs configured (all traffic redacted)")


def _check_overrides(report: _Report, config: Config) -> bool:
    """Refusal overrides (overrides.py): the store read offline (never
    created). A WARN with the counts when approved overrides exist —
    requests that pass on one forward the refused values (or bodies) as
    sent; True then (an opt-out). Off (the default): an informational line,
    or a WARN when the store still holds approvals — inert now, applied
    again the moment overrides are turned on. Counts only, never a value or
    a code."""
    from llm_redact.overrides import ENABLE_SETTING, OverrideStore, default_overrides_path

    path = (
        Path(config.overrides.path).expanduser()
        if config.overrides.path
        else default_overrides_path()
    )
    store = OverrideStore(path, read_only=True)
    try:
        counts = store.counts()
    except Exception as exc:  # noqa: BLE001 — doctor reports, never raises
        problem = type(exc).__name__
        report.line("WARN", "posture", f"the refusal override store could not be read ({problem})")
        return False
    finally:
        store.close()
    if not config.overrides.enabled:
        return _check_overrides_off(report, counts, path, ENABLE_SETTING)
    if counts["always"] or counts["once"]:
        report.line(
            "WARN",
            "posture",
            f"refusal overrides: {counts['always']} every-time rule(s), {counts['once']}"
            " one-time grant(s) — requests they pass FORWARD the refused values (or bodies)"
            " as sent (`llm-redact override list`)",
        )
        return True
    report.line(
        "PASS",
        "posture",
        "refusal overrides enabled, none approved: a refusal carries a code a person"
        " may approve on the terminal",
    )
    return False


def _check_overrides_off(report: _Report, counts: dict[str, int], path: Path, enable: str) -> bool:
    """Overrides off (the default): every refusal is final. Approvals the
    store kept from when they were on are inert, yet the moment ``enable``
    is set they apply again — a WARN naming the counts and the file. Never
    an opt-out (False)."""
    if counts["always"] or counts["once"]:
        report.line(
            "WARN",
            "posture",
            f"refusal overrides are off, but their store ({path}) still holds"
            f" {counts['always']} every-time rule(s) and {counts['once']} one-time"
            f" grant(s): inert now, applied again if {enable} — review them with"
            " `llm-redact override list` and drop them with `llm-redact override revoke ID`"
            " (it works while off), or delete the file to drop them all",
        )
        return False
    report.line(
        "PASS",
        "posture",
        f"refusal overrides off (the default): every refusal is final, no code; {enable} opts in",
    )
    return False


def _access_gate_requires_identity(config: Config) -> bool:
    """Whether llm-redact-pro's access gate is configured to require a client
    identity on API requests (``[auth] require`` or brokered mode). Read
    generically: the ``[auth]`` shape belongs to the package."""
    auth = config.extensions.get("auth")
    return bool(getattr(auth, "require", False) or getattr(auth, "broker", False))


def _check_upstream_auth(report: _Report, config: Config) -> None:
    """[providers.NAME] auth = "identity": the proxy's own cloud identity.
    Build-and-close each authorizer through the registry (construction does
    no network I/O — credentials are fetched per request, so doctor never
    contacts a cloud), then WARN when a non-loopback proxy lets any client
    that reaches it spend that identity without an access gate. Silent when
    every provider forwards the client's own credential."""
    from llm_redact.registry import get_registry

    identity = sorted(name for name, p in config.providers.items() if p.auth == "identity")
    if not identity:
        return
    registry = get_registry()
    for name in identity:
        try:
            auth = registry.build_upstream_auth(name, config.providers[name])
        except ConfigError as problem:
            report.line("FAIL", "upstream-auth", f"{problem} — serve would refuse this config")
            continue
        if auth is not None:
            auth.close()
        report.line(
            "PASS",
            "upstream-auth",
            f"[providers.{name}] requests are authorized with the proxy's own cloud identity"
            " (credentials are resolved per request; not contacted here)",
        )
    if config.host in ("127.0.0.1", "localhost", "::1") or _access_gate_requires_identity(config):
        return
    report.line(
        "WARN",
        "upstream-auth",
        f"non-loopback bind {config.host} with auth = identity for {', '.join(identity)} and"
        " no access gate requiring an identity: any client that reaches the proxy spends its"
        " cloud identity — set [auth] require = true (llm-redact-pro) or bind 127.0.0.1",
    )


def _credential_lenders(config: Config) -> list[str]:
    """The config sections that make the proxy spend a credential it holds:
    identity-authorized providers, and — with routing on — every upstream
    that does not forward the client's own (an operator key, or none)."""
    lenders = [
        f"[providers.{name}]"
        for name, provider in sorted(config.providers.items())
        if provider.auth != "passthrough"
    ]
    if config.routing.enabled:
        lenders += [
            f"[upstreams.{upstream.name}]"
            for upstream in config.routing.upstreams
            if upstream.credential != "passthrough"
        ]
    return lenders


def _check_allowed_hosts(report: _Report, config: Config) -> None:
    """allowed_hosts: the extra host names the proxy answers to. A request
    that spends a credential the proxy holds must name one (or a loopback
    name, or the bind host) unless it arrives over TLS — so WARN where a
    non-loopback plain-HTTP bind lends one with no names listed: a client
    reaching it as a compose service or a Kubernetes Service would get 403s.
    Silent when nothing is lent."""
    if config.allowed_hosts:
        report.line(
            "PASS",
            "hosts",
            f"allowed_hosts: {len(config.allowed_hosts)} more host name(s) this proxy answers"
            " to (checked on browser requests and on requests that spend its credential)",
        )
        return
    lenders = _credential_lenders(config)
    if not lenders or config.tls.enabled or config.host in ("127.0.0.1", "localhost", "::1"):
        return
    report.line(
        "WARN",
        "hosts",
        f"{', '.join(lenders)} spend a credential the proxy holds, and this plain-HTTP proxy"
        f" binds {config.host}: a client addressing it by any name other than 127.0.0.1,"
        f" localhost, ::1 or {config.host} is refused (403) on those requests — list the"
        " names clients use (a compose service, a Kubernetes Service) in allowed_hosts",
    )


def _check_allowed_origins(report: _Report, config: Config) -> None:
    """allowed_origins: browser origins the operator opted in. A page on a
    listed origin can read restored values back through the proxy (every
    rehydrating route answers it) and spend any credential the proxy
    holds, so the opt-in is always a WARN naming them. Silent when none."""
    if not config.allowed_origins:
        return
    lenders = _credential_lenders(config)
    spends = (
        f", and spend the credentials the proxy holds for {', '.join(lenders)}" if lenders else ""
    )
    report.line(
        "WARN",
        "origins",
        f"allowed_origins: {', '.join(config.allowed_origins)} — pages on these origins can"
        f" read your redacted values back through the proxy{spends}; list only origins you"
        " trust with them",
    )


def _check_routing(report: _Report, config: Config, offline: bool) -> None:
    """R-31 lives in llm-redact-pro (routing is a pro subsystem); this shell
    reports the config/package posture and hands the checks to the package."""
    routing = config.routing
    if not routing.enabled:
        report.line(
            "PASS",
            "routing",
            "disabled ([routing] enabled = false)"
            if routing.present
            else "not configured (protocol → provider upstream, no fallback, no budgets)",
        )
        return
    try:
        from llm_redact_pro.routing_doctor import routing_checks
    except ImportError:
        report.line(
            "FAIL",
            "routing",
            "[routing] enabled = true requires the llm-redact-pro package 0.3 or newer — the"
            " proxy will refuse to start (rule-based upstream routing, fallback chains and"
            " budgets are pro subsystems; see docs/editions.md)",
        )
        return
    for level, message in routing_checks(config, offline=offline, environ=os.environ):
        report.line(level, "routing", message)


def _check_extraction(report: _Report, config: Config) -> None:
    """[extraction] (docs/extraction.md), offline and value-free: the
    ``extract`` extra, each credential variable by NAME, a key file, which
    services see files off this machine, convert mode, and what a clean
    scan lets through."""
    from llm_redact.extraction import extraction_checks

    if config.extraction.enabled:
        _check_inspector_factory(report, config)
    for level, message in extraction_checks(config.extraction, os.environ):
        report.line(level, "extraction", message)


def _check_inspector_factory(report: _Report, config: Config) -> None:
    """When a plugin replaces the upload inspector factory, build it as
    serve would (then close it): one that builds none for an enabled
    [extraction] — a plugin predating the core's extraction, which reads
    the section from its own config — makes the proxy refuse to start."""
    import asyncio

    from llm_redact import free_defaults
    from llm_redact.licensing import resolve_license
    from llm_redact.registry import get_registry

    factory = get_registry().build_upload_inspector
    if factory is free_defaults.build_upload_inspector:
        return
    tier = resolve_license(
        env=dict(os.environ),
        config_key=config.license.key,
        config_key_file=config.license.key_file,
    ).tier
    refusal = "the proxy will refuse to start: upgrade the plugin (llm-redact-pro)"
    try:
        inspector = factory(config, tier)
    except ConfigError as exc:
        report.line("FAIL", "extraction", f"{exc} — {refusal}")
        return
    if inspector is None:
        report.line(
            "FAIL",
            "extraction",
            f"enabled, but the plugin's upload inspector factory builds none — {refusal}"
            " to a version that leaves [extraction] to the core",
        )
        return
    report.line(
        "PASS",
        "extraction",
        f"a plugin's upload inspector ({type(inspector).__name__}) replaces the core's extractors",
    )
    asyncio.run(inspector.aclose())


def _check_access(report: _Report, config: Config) -> None:
    """Access control lives in llm-redact-pro; this shell hands the checks to
    the package (read-only: its registry is never written) and stays silent
    without it or on a version that has no access checks."""
    try:
        from llm_redact_pro.access_doctor import access_checks
    except ImportError:
        return
    for level, message in access_checks(config, environ=os.environ):
        report.line(level, "access", message)


def _check_email(report: _Report, config: Config) -> None:
    """[email] delivery posture for llm-redact-pro's invites: the SMTP auth
    mode, the transport security and whether the env-held secrets are
    PRESENT (names only, never values; no network — a token is fetched only
    when an invite is sent). Silent when [email] is not configured."""
    email = config.email
    if not email.configured:
        return
    tls = {"implicit": "implicit TLS", "starttls": "STARTTLS", "none": "no TLS"}[email.tls]
    if email.auth == "oauth":
        _check_email_oauth(report, config, tls)
        return
    if email.username is None:
        report.line("PASS", "email", f"SMTP without authentication ({tls})")
    elif not os.environ.get(email.password_env):
        report.line(
            "WARN",
            "email",
            f"username is set but {email.password_env} is not — invites will fail to send"
            " (the SMTP password comes from the environment, never the config file)",
        )
    elif email.tls == "none":
        report.line(
            "WARN",
            "email",
            "SMTP password auth without TLS — the password crosses the network in cleartext"
            " (set starttls = true or implicit_tls = true)",
        )
    else:
        report.line("PASS", "email", f"SMTP password auth ({tls}; {email.password_env} set)")


def _check_email_oauth(report: _Report, config: Config, tls: str) -> None:
    email = config.email
    if email.oauth_provider != "refresh_token":
        source = {
            "azure": "Microsoft Entra ID workload credentials",
            "google": "a Google service account with domain-wide delegation",
        }[str(email.oauth_provider)]
        report.line(
            "PASS",
            "email",
            f"SMTP OAuth 2.0 (XOAUTH2, {tls}) with tokens from {source}, fetched at send time",
        )
        return
    missing = [
        name
        for name in (email.oauth_refresh_token_env, email.oauth_client_secret_env)
        if name is not None and not os.environ.get(name)
    ]
    if missing:
        report.line(
            "WARN",
            "email",
            f"SMTP OAuth 2.0 refresh-token grant but {' and '.join(missing)} not set —"
            " invites will fail to send (secrets come from the environment, never the"
            " config file)",
        )
        return
    host = urllib.parse.urlsplit(email.oauth_token_url or "").hostname
    report.line(
        "PASS",
        "email",
        f"SMTP OAuth 2.0 (XOAUTH2, {tls}) via a refresh-token grant at {host}",
    )


def run_doctor(args: argparse.Namespace) -> int:
    report = _Report(json_mode=getattr(args, "json", False))
    config = _check_config(report, args)
    if config is None:
        report.finish()
        return 1
    _check_platform(report)
    _check_license(report, config)
    _check_licensed_features(report, config)
    _check_build(report, config)
    _check_tls_and_bind(report, config)
    _check_body_cap(report, config)
    _check_proxy(report, config)
    _check_vault(report, config)
    _check_extras(report, config)
    _check_posture(report, config)
    _check_ner_labels(report, config)
    _check_models(report, config)
    _check_extraction(report, config)
    _check_upstream_auth(report, config)
    _check_allowed_hosts(report, config)
    _check_allowed_origins(report, config)
    _check_routing(report, config, bool(getattr(args, "offline", False)))
    _check_access(report, config)
    _check_email(report, config)
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
    from llm_redact.audit_s3 import required_credential_env

    az = config.audit.azure
    if not az.enabled:
        return
    host = az.endpoint_url or f"{az.account}.blob.core.windows.net"
    mode = {"key": "SharedKey", "sas": "SAS-token", "identity": "Entra ID"}.get(az.auth, az.auth)
    # Presence only — never a byte of the values themselves.
    missing = [n for n in required_credential_env("azure", auth=az.auth) if not os.environ.get(n)]
    if missing:
        report.line(
            "FAIL",
            "audit.azure",
            f"enabled (auth = {az.auth!r}) but {' and '.join(missing)} not set — batches"
            " will be dropped (credentials come from the environment, never the config file)",
        )
    elif az.auth == "identity":
        report.line(
            "PASS",
            "audit.azure",
            f"{mode} sink configured (container {az.container} via {host}); the token"
            " resolves at runtime from the workload identity (not checked offline);"
            " metadata rows leave this machine",
        )
    else:
        report.line(
            "PASS",
            "audit.azure",
            f"{mode} sink configured (container {az.container} via {host});"
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
    from llm_redact.audit_s3 import required_credential_env

    s3 = config.audit.s3
    if not s3.enabled:
        return
    # Presence only — never a byte of the values themselves. The credential
    # env vars vary by provider (GCS uses its own HMAC interop keys) and by
    # auth mode (identity needs none: it resolves at runtime).
    required = required_credential_env("s3", s3.provider, s3.auth)
    missing = [name for name in required if not os.environ.get(name)]
    target = {
        "aws": f"s3.{s3.region}.amazonaws.com",
        "gcs": "storage.googleapis.com",
    }.get(s3.provider, s3.endpoint_url or "?")
    if missing:
        report.line(
            "FAIL",
            "audit.s3",
            f"enabled but {' and '.join(missing)} not set — batches will be"
            " dropped (credentials come from the environment, never the config file)",
        )
    elif s3.auth == "identity":
        report.line(
            "PASS",
            "audit.s3",
            f"{s3.provider} sink configured (bucket {s3.bucket} via {target}) with"
            " workload identity; credentials resolve at runtime (not checked offline);"
            " metadata rows leave this machine",
        )
    else:
        report.line(
            "PASS",
            "audit.s3",
            f"{s3.provider} sink configured (bucket {s3.bucket} via {target});"
            " metadata rows leave this machine",
        )
    _check_audit_encryption(report, "audit.s3", s3.encryption)
