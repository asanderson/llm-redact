import argparse
import dataclasses
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from llm_redact.config import (
    Config,
    ConfigError,
    apply_env_overrides,
    load_config,
    validate_bind_security,
)


def build_parser() -> argparse.ArgumentParser:
    from llm_redact import __version__

    parser = argparse.ArgumentParser(prog="llm-redact")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the redaction proxy")
    serve.add_argument("--config", type=Path, default=None, help="path to config.toml")
    serve.add_argument("--port", type=int, default=None, help="override listen port")
    serve.add_argument(
        "--session",
        default=None,
        help="vault session name (sqlite backend); same session => same tokens across restarts",
    )
    serve.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=None,
        help="log line framing (overrides [log] format; json = one object per line)",
    )
    serve.add_argument(
        "--check",
        action="store_true",
        help="run serve's full startup build (config, detectors, vault open + key"
        " verify, bind policy) and exit without binding — the deploy/reload gate",
    )

    status = subparsers.add_parser("status", help="query a running proxy's status endpoint")
    status.add_argument("--config", type=Path, default=None, help="path to config.toml")
    status.add_argument("--port", type=int, default=None, help="override proxy port")
    status.add_argument("--json", action="store_true", help="print raw JSON")
    status.add_argument("--ca", type=Path, default=None, help="CA bundle to verify a TLS proxy")
    status.add_argument("--cert", type=Path, default=None, help="client certificate (mutual TLS)")
    status.add_argument("--key", type=Path, default=None, help="client private key (mutual TLS)")

    def _db_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", type=Path, default=None, help="path to config.toml")
        sub.add_argument("--db", type=Path, default=None, help="vault database path")

    sessions = subparsers.add_parser("sessions", help="inspect or prune vault sessions")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    sessions_list = sessions_sub.add_parser("list", help="list sessions with entry counts")
    _db_args(sessions_list)
    sessions_list.add_argument("--json", action="store_true", help="print raw JSON")
    sessions_prune = sessions_sub.add_parser(
        "prune", help="delete whole sessions idle longer than a cutoff"
    )
    _db_args(sessions_prune)
    sessions_prune.add_argument(
        "--older-than", required=True, help="idle cutoff in whole days, e.g. 90d"
    )
    sessions_prune.add_argument("--yes", action="store_true", help="skip confirmation")

    lookup = subparsers.add_parser(
        "lookup", help="resolve a placeholder to its original value (prints the secret)"
    )
    _db_args(lookup)
    lookup.add_argument("token", nargs="?", default=None, help="placeholder, e.g. «EMAIL_001»")
    lookup.add_argument("--value", default=None, help="reverse: find the placeholder for a value")
    lookup.add_argument("--session", default=None, help="restrict to one session id")

    vault = subparsers.add_parser("vault", help="vault utilities")
    vault_sub = vault.add_subparsers(dest="vault_command", required=True)
    vault_sub.add_parser("gen-key", help="generate a LLM_REDACT_VAULT_KEY value")
    vault_sub.add_parser("set-key", help="store the vault key in the OS keychain (keyring extra)")
    vault_verify = vault_sub.add_parser("verify", help="read-only integrity sweep of the vault")
    _db_args(vault_verify)
    vault_rotate = vault_sub.add_parser(
        "rotate-key", help="re-encrypt the vault under a new key (offline; stop the proxy first)"
    )
    _db_args(vault_rotate)
    vault_rotate.add_argument("--yes", action="store_true", help="skip confirmation")
    vault_backup = vault_sub.add_parser(
        "backup", help="write a consistent single-file snapshot of the vault"
    )
    _db_args(vault_backup)
    vault_backup.add_argument("dest", help="destination path for the snapshot")
    vault_backup.add_argument("--force", action="store_true", help="overwrite an existing dest")

    audit = subparsers.add_parser("audit", help="audit-log utilities")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    audit_verify = audit_sub.add_parser("verify", help="verify the tamper-evident audit hash-chain")
    audit_verify.add_argument(
        "--config", type=Path, default=None, help="config file (default: search XDG/etc)"
    )
    audit_verify.add_argument("--json", action="store_true", help="print raw JSON")
    audit_decrypt = audit_sub.add_parser(
        "decrypt", help="decrypt a downloaded .ndjson.fernet audit backup object"
    )
    audit_decrypt.add_argument("file", type=Path, help="the downloaded object")
    audit_decrypt.add_argument(
        "--out", type=Path, default=None, help="write NDJSON here instead of stdout"
    )

    subparsers.add_parser("guide", help="print the packaged user guide (web UIs + plugin commands)")

    init = subparsers.add_parser("init", help="interactive setup: write a config, print exports")
    init.add_argument("--tools", default=None, help="comma list: claude,codex,gemini,ollama")
    init.add_argument(
        "--vault",
        choices=("memory", "sqlite"),
        default=None,
        help="where token-to-value mappings live; sqlite (default) survives restarts",
    )
    init.add_argument(
        "--encryption",
        choices=("none", "fernet"),
        default=None,
        help="encrypt the sqlite vault at rest (Pro; needs the crypto extra + key)",
    )
    init.add_argument("--port", type=int, default=None, help="proxy port (default 8787)")
    init.add_argument(
        "--yes", action="store_true", help="non-interactive; defaults for unset flags"
    )
    init.add_argument("--force", action="store_true", help="overwrite an existing config file")

    service = subparsers.add_parser("service", help="run the proxy at login (launchd/systemd)")
    service_sub = service.add_subparsers(dest="service_command", required=True)
    service_install = service_sub.add_parser("install", help="write and load the user unit")
    service_install.add_argument(
        "--print-only", action="store_true", help="print the unit file; change nothing"
    )
    service_sub.add_parser("uninstall", help="unload and remove the user unit")
    service_sub.add_parser("status", help="unit file + loader status")

    plugin = subparsers.add_parser(
        "plugin", help="install slash-command plugins for Claude Code / Codex / OpenCode / Cursor"
    )
    plugin_sub = plugin.add_subparsers(dest="plugin_command", required=True)
    plugin_install = plugin_sub.add_parser("install", help="write the tool's command files")
    plugin_install.add_argument("tool", choices=("claude", "codex", "opencode", "cursor"))
    plugin_install.add_argument(
        "--print-only", action="store_true", help="print target paths; change nothing"
    )
    plugin_install.add_argument(
        "--force", action="store_true", help="overwrite files whose content was modified"
    )
    plugin_install.add_argument(
        "--proxy-url",
        default=None,
        help="point the commands at an EXISTING llm-redact proxy"
        " (https required off-loopback; sets up LLM_REDACT_PROXY_URL usage)",
    )
    plugin_install.add_argument(
        "--install-proxy",
        action="store_true",
        help="also set up a local proxy: init --yes (if no config) + service install",
    )
    plugin_uninstall = plugin_sub.add_parser("uninstall", help="remove the tool's command files")
    plugin_uninstall.add_argument("tool", choices=("claude", "codex", "opencode", "cursor"))
    plugin_sub.add_parser("status", help="per-tool install state")

    run = subparsers.add_parser(
        "run", help="run a tool through the proxy (env injected; proxy auto-started if absent)"
    )
    run.add_argument("--config", type=Path, default=None, help="path to config.toml")
    run.add_argument("--port", type=int, default=None, help="override proxy port")
    run.add_argument(
        "--proxy-url",
        default=None,
        help="use an EXISTING llm-redact proxy at this URL instead of the"
        " local config one (overrides LLM_REDACT_PROXY_URL; https required"
        " off-loopback; never auto-starts a proxy)",
    )
    run.add_argument(
        "--tools",
        default=None,
        help="comma list of tools to route (claude,codex,gemini,ollama); default all",
    )
    run.add_argument(
        "--set-env",
        action="append",
        default=None,
        metavar="VAR",
        help="also point this environment variable at the proxy (repeatable;"
        " escape hatch for tools not on the --tools list)",
    )
    run.add_argument(
        "tool_command",
        nargs=argparse.REMAINDER,
        help="command to run, after -- (e.g. -- claude)",
    )

    doctor = subparsers.add_parser(
        "doctor", help="diagnose the local setup (read-only; non-zero exit on failures)"
    )
    doctor.add_argument("--config", type=Path, default=None, help="path to config.toml")
    doctor.add_argument("--json", action="store_true", help="machine-readable check rows")

    config_cmd = subparsers.add_parser("config", help="inspect the effective configuration")
    config_sub = config_cmd.add_subparsers(dest="config_command", required=True)
    config_show = config_sub.add_parser(
        "show", help="print the effective config as TOML (env overrides applied and named)"
    )
    config_show.add_argument("--config", type=Path, default=None, help="path to config.toml")
    config_show.add_argument(
        "--path", action="store_true", help="print only the resolved config file path"
    )

    preview = subparsers.add_parser(
        "preview",
        help="dry-run detection over text (stdin or --text); no proxy, no upstream",
    )
    preview.add_argument("--config", type=Path, default=None, help="path to config.toml")
    preview.add_argument("--text", default=None, help="text to scan (default: read stdin)")
    preview.add_argument("--json", action="store_true", help="machine-readable output")

    users = subparsers.add_parser(
        "users", help="named-user management (Pro+ tiers; email-verified seats)"
    )
    users_sub = users.add_subparsers(dest="users_command", required=True)

    def _users_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", type=Path, default=None, help="path to config.toml")
        sub.add_argument("--db", type=Path, default=None, help="users database path")

    users_invite = users_sub.add_parser("invite", help="invite a named user (sends a code)")
    _users_args(users_invite)
    users_invite.add_argument("name", help="display name")
    users_invite.add_argument("email", help="the user's email address")
    users_invite.add_argument(
        "--print-code",
        action="store_true",
        help="print the verification code instead of emailing it",
    )
    users_verify = users_sub.add_parser("verify", help="redeem a code; prints the per-user key")
    _users_args(users_verify)
    users_verify.add_argument("email")
    users_verify.add_argument("code")
    users_list = users_sub.add_parser("list", help="seats used vs the license cap")
    _users_args(users_list)
    users_list.add_argument("--json", action="store_true", help="machine-readable output")
    users_revoke = users_sub.add_parser("revoke", help="revoke a user (key stops working)")
    _users_args(users_revoke)
    users_revoke.add_argument("email")
    users_revoke.add_argument("--yes", action="store_true", help="skip confirmation")
    users_revoke.add_argument(
        "--purge",
        action="store_true",
        help="delete the row entirely (frees the email for re-invite)",
    )

    license_cmd = subparsers.add_parser("license", help="inspect the configured license key")
    license_sub = license_cmd.add_subparsers(dest="license_command", required=True)
    license_show = license_sub.add_parser(
        "show", help="decoded license payload + verification status"
    )
    license_show.add_argument("--config", type=Path, default=None, help="path to config.toml")
    license_show.add_argument(
        "--key", default=None, help="verify this key instead of the configured one"
    )
    license_show.add_argument("--json", action="store_true", help="machine-readable output")
    license_verify = license_sub.add_parser(
        "verify", help="exit 0 valid / 1 absent or invalid / 2 expired"
    )
    license_verify.add_argument("--config", type=Path, default=None, help="path to config.toml")
    license_verify.add_argument(
        "--key", default=None, help="verify this key instead of the configured one"
    )

    completions = subparsers.add_parser("completions", help="print a shell completion script")
    completions.add_argument("shell", choices=("bash", "zsh", "fish"))

    subparsers.add_parser(
        "fips-check",
        help="report the host's FIPS posture (kernel, OpenSSL, python hashlib)",
    )

    return parser


def _serve_config(args: argparse.Namespace) -> Config:
    """Load the config and apply serve's CLI overrides — shared verbatim by
    `serve` and `serve --check` so the check exercises exactly what serve
    would run."""
    from llm_redact.log import setup_logging

    config = apply_env_overrides(load_config(args.config))
    if args.log_format is not None:
        config = dataclasses.replace(
            config, log=dataclasses.replace(config.log, format=args.log_format)
        )
    setup_logging(config.log.format)
    if args.session is not None:
        if config.vault.backend == "memory":
            logging.getLogger("llm_redact").warning(
                "--session has no effect with the in-memory vault backend; "
                'set [vault] backend = "sqlite" to persist sessions'
            )
        if config.vault.session_mode == "per-conversation":
            logging.getLogger("llm_redact").warning(
                "--session names only the fallback session in "
                "per-conversation mode; conversations derive their own"
            )
        config = dataclasses.replace(
            config, vault=dataclasses.replace(config.vault, session=args.session)
        )
    if args.port is not None:
        # Bake the override into the config so /status and the config
        # editor report the port actually being served.
        config = dataclasses.replace(config, port=args.port)
    return config


def _run_serve_check(args: argparse.Namespace) -> int:
    """serve's full startup, minus the socket: config load + CLI overrides,
    bind policy, and the complete app build (detectors incl. NER, vault open
    with key verification, audit setup). If this exits 0, serve would start
    — the gate to run before a deploy or a `kill -HUP` reload (which
    rejects a bad config with only a log line). Creates the same state
    files serve's own startup would (vault/audit databases)."""
    from llm_redact.proxy import create_app
    from llm_redact.vault import VaultKeyError

    try:
        config = _serve_config(args)
        validate_bind_security(config.host, config.tls, os.environ)
        create_app(config, config_path=args.config)
    except (ConfigError, ValueError, VaultKeyError, re.error) as problem:
        # re.error is belt-and-braces: build_detectors/build_allowlist wrap
        # user-pattern compiles into named ValueErrors, but the deploy gate
        # must never print a traceback for a config problem.
        print(f"serve --check: FAIL: {problem}", file=sys.stderr)
        return 1
    print("serve --check: OK — config loads, builds, and passes bind policy")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        from llm_redact.proxy import create_app

        if args.check:
            raise SystemExit(_run_serve_check(args))
        config = _serve_config(args)
        # Fail-closed bind policy: refuse a non-loopback bind without full
        # mutual TLS, before any socket opens.
        validate_bind_security(config.host, config.tls, os.environ)
        run_kwargs: dict[str, object] = {}
        if config.log.format == "json":
            # Disable uvicorn's own logging config so its records propagate
            # to the root handler and come out as JSON lines too — its
            # default handlers would interleave plain text into the stream.
            run_kwargs["log_config"] = None
        if config.tls.enabled:
            run_kwargs["ssl_certfile"] = config.tls.certfile
            run_kwargs["ssl_keyfile"] = config.tls.keyfile
            if config.tls.client_ca is not None:
                import ssl

                run_kwargs["ssl_ca_certs"] = config.tls.client_ca
                run_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
            logging.getLogger("llm_redact").info(
                "TLS enabled (%s)",
                "mutual: clients must present a certificate"
                if config.tls.mutual
                else "server-only",
            )
        # One startup line: uvicorn's own "running on ..." banner is
        # suppressed by log_level="warning", so without this the proxy
        # started in TOTAL silence — no confirmation, no port, no pointer
        # to the dashboard for a first-time user.
        from llm_redact import __version__

        scheme = "https" if config.tls.enabled else "http"
        logging.getLogger("llm_redact").info(
            "llm-redact %s serving on %s://%s:%d — dashboard %s://%s:%d/__llm-redact/",
            __version__,
            scheme,
            config.host,
            config.port,
            scheme,
            "127.0.0.1" if config.host == "0.0.0.0" else config.host,
            config.port,
        )
        uvicorn.run(
            create_app(config, config_path=args.config),
            host=config.host,
            port=config.port,
            log_level="warning",
            # Access-log lines include the full request line with query
            # strings, which can carry provider API keys (Gemini ?key=).
            access_log=False,
            **run_kwargs,  # type: ignore[arg-type]
        )
    elif args.command == "status":
        raise SystemExit(run_status(args))
    elif args.command == "sessions":
        from llm_redact.vault_cli import run_sessions_list, run_sessions_prune

        if args.sessions_command == "list":
            raise SystemExit(run_sessions_list(args))
        raise SystemExit(run_sessions_prune(args))
    elif args.command == "lookup":
        from llm_redact.vault_cli import run_lookup

        if (args.token is None) == (args.value is None):
            print("lookup needs exactly one of: a placeholder token, or --value")
            raise SystemExit(2)
        raise SystemExit(run_lookup(args))
    elif args.command == "vault":
        from llm_redact.vault_cli import (
            run_vault_backup,
            run_vault_gen_key,
            run_vault_rotate_key,
            run_vault_set_key,
            run_vault_verify,
        )

        if args.vault_command == "set-key":
            raise SystemExit(run_vault_set_key(args))
        if args.vault_command == "verify":
            raise SystemExit(run_vault_verify(args))
        if args.vault_command == "rotate-key":
            raise SystemExit(run_vault_rotate_key(args))
        if args.vault_command == "backup":
            raise SystemExit(run_vault_backup(args))
        raise SystemExit(run_vault_gen_key(args))
    elif args.command == "audit":
        try:
            from llm_redact_pro.audit_cli import run_audit_decrypt, run_audit_verify
        except ImportError:
            print(
                "audit tooling requires the llm-redact-pro package (the audit log,"
                " tamper-evident chain, and encrypted backups are pro subsystems);"
                " see docs/editions.md"
            )
            raise SystemExit(1) from None

        if args.audit_command == "decrypt":
            raise SystemExit(run_audit_decrypt(args))
        raise SystemExit(run_audit_verify(args))
    elif args.command == "guide":
        raise SystemExit(run_guide(args))
    elif args.command == "init":
        from llm_redact.init_cli import run_init

        raise SystemExit(run_init(args))
    elif args.command == "service":
        from llm_redact.service_cli import run_service

        raise SystemExit(run_service(args))
    elif args.command == "plugin":
        from llm_redact.plugin_cli import run_plugin

        raise SystemExit(run_plugin(args))
    elif args.command == "run":
        from llm_redact.run_cli import run_run

        raise SystemExit(run_run(args))
    elif args.command == "doctor":
        from llm_redact.doctor_cli import run_doctor

        raise SystemExit(run_doctor(args))
    elif args.command == "config":
        from llm_redact.doctor_cli import run_config_show

        raise SystemExit(run_config_show(args))
    elif args.command == "fips-check":
        from llm_redact.fips import run_fips_check

        raise SystemExit(run_fips_check())
    elif args.command == "license":
        from llm_redact.license_cli import run_license_show, run_license_verify

        if args.license_command == "verify":
            raise SystemExit(run_license_verify(args))
        raise SystemExit(run_license_show(args))
    elif args.command == "users":
        try:
            from llm_redact_pro.users_cli import (
                run_users_invite,
                run_users_list,
                run_users_revoke,
                run_users_verify,
            )
        except ImportError:
            print(
                "user management requires the llm-redact-pro package (without it"
                " the proxy serves the implicit single local user); see"
                " docs/editions.md"
            )
            raise SystemExit(1) from None

        if args.users_command == "invite":
            raise SystemExit(run_users_invite(args))
        if args.users_command == "verify":
            raise SystemExit(run_users_verify(args))
        if args.users_command == "revoke":
            raise SystemExit(run_users_revoke(args))
        raise SystemExit(run_users_list(args))
    elif args.command == "preview":
        raise SystemExit(run_preview(args))
    elif args.command == "completions":
        from llm_redact.completions import script_for

        print(script_for(args.shell), end="")
        raise SystemExit(0)


def run_preview(args: argparse.Namespace) -> int:
    """Dry-run the detection pipeline over text, entirely locally (no proxy,
    no upstream, no vault write). Reads the config from disk, builds the live
    detectors/allowlist/modes, and reports what WOULD be redacted / warned /
    blocked. Text stays on the machine; a throwaway vault issues placeholders."""
    import json
    import sys

    from llm_redact.config import load_config
    from llm_redact.detection.engine import build_allowlist, build_detectors, build_modes
    from llm_redact.redactor import BlockedRequest, Redactor
    from llm_redact.vault import InMemoryVault

    config = apply_env_overrides(load_config(args.config))
    text = args.text if args.text is not None else sys.stdin.read()
    redactor = Redactor(
        build_detectors(config.detection),
        InMemoryVault(),
        build_allowlist(config.detection),
        modes=build_modes(config.detection),
    )
    blocked: str | None = None
    redacted: str | None = None
    try:
        redacted = redactor.redact_text(text)
    except BlockedRequest as exc:
        blocked = exc.detector_type

    if args.json:
        print(
            json.dumps(
                {
                    "redacted": redacted,
                    "detections": dict(redactor.counts),
                    "warnings": dict(redactor.warn_counts),
                    "blocked": blocked,
                },
                indent=2,
            )
        )
        return 0

    if blocked is not None:
        print(f"BLOCKED: a {blocked} value matched a block-mode rule (a real request would 400)")
        return 0
    assert redacted is not None  # only None when blocked, handled above
    for detector_type, count in sorted(redactor.counts.items()):
        print(f"redact  {detector_type} x{count}")
    for detector_type, count in sorted(redactor.warn_counts.items()):
        print(f"warn    {detector_type} x{count}  (VALUE FORWARDED — observation, not protection)")
    if not redactor.counts and not redactor.warn_counts:
        print("nothing detected")
    print("---")
    print(redacted, end="" if redacted.endswith("\n") else "\n")
    return 0


def run_guide(_args: argparse.Namespace) -> int:
    """Print the packaged user guide as markdown. A running proxy serves
    the same content formatted at /__llm-redact/guide."""
    import importlib.resources

    print(importlib.resources.files("llm_redact").joinpath("user_guide.md").read_text("utf-8"))
    return 0


def run_status(args: argparse.Namespace) -> int:
    import json
    from urllib.parse import urlsplit

    import httpx

    from llm_redact.config import load_config
    from llm_redact.proxy import RESERVED_PREFIX
    from llm_redact.run_cli import resolve_proxy_url, validate_proxy_url

    config = apply_env_overrides(load_config(args.config))
    pointed_at = resolve_proxy_url(None)
    if pointed_at is not None:
        # LLM_REDACT_PROXY_URL: query the pointed-at proxy (possibly remote,
        # https + the same --ca/--cert/--key flags for mTLS deployments).
        problem = validate_proxy_url(pointed_at)
        if problem is not None:
            print(f"LLM_REDACT_PROXY_URL: {problem}")
            return 2
        base = pointed_at
    else:
        port = args.port if args.port is not None else config.port
        scheme = "https" if config.tls.enabled else "http"
        base = f"{scheme}://{config.host}:{port}"
    url = f"{base}{RESERVED_PREFIX}/status"
    request_kwargs: dict[str, Any] = {"timeout": 5.0}
    if base.startswith("https://"):
        import ssl

        context = ssl.create_default_context(cafile=str(args.ca) if args.ca else None)
        if args.cert is not None:
            # Mutual TLS: the proxy rejects clients without a certificate.
            context.load_cert_chain(str(args.cert), str(args.key) if args.key else None)
        request_kwargs["verify"] = context
    try:
        response = httpx.get(url, **request_kwargs)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        # Never echo the raw exception verbatim: httpx.HTTPStatusError embeds
        # the full request URL, which may carry a /u/<key> identity credential
        # from LLM_REDACT_PROXY_URL. Report the netloc + a URL-free reason
        # (the `run` command's scheme://netloc discipline). Fail closed if the
        # message somehow still contains the URL path.
        if isinstance(exc, httpx.HTTPStatusError):
            detail = f"HTTP {exc.response.status_code}"
        else:
            detail = type(exc).__name__
        path = urlsplit(base).path
        if path and path in detail:
            detail = type(exc).__name__
        print(
            f"proxy not reachable at {urlsplit(base).netloc} ({detail})"
            " — start it: llm-redact serve (or: llm-redact run -- <tool>)"
        )
        return 1
    payload = response.json()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"llm-redact {payload['version']} — up {payload['uptime_seconds']}s")
    license_info = payload.get("license") or {}
    if license_info:
        # The FOSS core enforces nothing — keyless, the caps a key would
        # carry are meaningless, so show the honest "nothing gated" line
        # instead of a phantom "users: 1".
        if license_info.get("tier", "free") == "free" and not license_info.get("expires"):
            print("license: none (FOSS core — nothing gated)")
        else:
            users = license_info.get("max_users")
            users_text = "unlimited" if users is None else str(users)
            clouds = ", ".join(license_info.get("clouds") or []) or "none"
            line = (
                f"license: {license_info.get('tier', 'free')}  users: {users_text}"
                f"  clouds: {clouds}"
            )
            if license_info.get("expires"):
                line += f"  expires: {license_info['expires']}"
            print(line)
        # Open-core honesty (llm-redact-pro docs/licensing.md): whether the paid
        # llm-redact-pro package is present. Older proxies omit the field, so
        # only print when it is reported (never invent a state).
        if "package_installed" in license_info:
            if license_info["package_installed"]:
                plugins = license_info.get("plugins") or []
                active = f" ({', '.join(plugins)} active)" if plugins else " (no plugin registered)"
                print(f"licensed-features package: installed{active}")
            else:
                print("licensed-features package: not installed (FOSS core is complete)")
    print(
        f"session: {payload['session']}  vault: {payload['vault']['backend']}"
        f" ({payload['vault']['entries']} entries)"
    )
    detections = payload["detections_total"] or {}
    rehydrations = payload["rehydrations_total"] or {}
    print("detections:  " + (" ".join(f"{k}×{v}" for k, v in sorted(detections.items())) or "none"))
    print(
        "rehydrations: " + (" ".join(f"{k}×{v}" for k, v in sorted(rehydrations.items())) or "none")
    )
    audit = payload["audit"]
    print(
        f"audit: {'enabled, ' + str(audit['rows']) + ' rows' if audit['enabled'] else 'disabled'}"
    )
    print(f"fuzzy: {payload['rehydration']['fuzzy']}  ner: {payload['detection']['ner_enabled']}")
    _print_posture(payload)
    for name, upstream in payload["providers"].items():
        if upstream:
            print(f"upstream[{name}]: {upstream}")
        else:
            print(f"upstream[{name}]: (not configured — its routes answer 502)")
    return 0


def _print_posture(payload: dict[str, Any]) -> None:
    """Loud, honest reminders of every runtime coverage opt-out. Silent when
    nothing reduces protection — the same contract as `doctor`'s posture
    check, but reporting live counts rather than static config."""
    lines: list[str] = []
    license_info = payload.get("license") or {}
    for license_warning in license_info.get("warnings") or []:
        lines.append(f"license: {license_warning}")
    # Installed-but-not-registered: the paid package is present yet its plugin
    # did not load, so paid features are silently OFF — a quiet downgrade this
    # posture block exists to surface.
    if license_info.get("package_installed") and not (license_info.get("plugins") or []):
        lines.append(
            "licensed-features package present but no plugin registered — paid features OFF"
        )
    warnings = payload.get("warnings_total") or {}
    if warnings:
        seen = " ".join(f"{k}×{v}" for k, v in sorted(warnings.items()))
        lines.append(f"warn mode: {seen} FORWARDED upstream (observation, not protection)")
    detection_off = payload.get("providers_detection_off") or []
    if detection_off:
        lines.append(f"detection OFF for: {', '.join(detection_off)} (forwarded unredacted)")
    exempt = payload.get("mcp_exempt_servers") or 0
    if exempt:
        lines.append(f"MCP exempt servers: {exempt} (their blocks forwarded unredacted)")
    inactive = payload.get("detection", {}).get("language_inactive_rules") or []
    if inactive:
        lines.append(f"language-inactive rules: {', '.join(inactive)} (not detected)")
    forks = payload.get("compaction_forks") or 0
    if forks:
        lines.append(f"compaction forks: {forks} (history rewrites that forked a fresh session)")
    for sink in ("s3", "azure"):
        dropped = (payload.get("audit", {}).get(sink) or {}).get("rows_dropped") or 0
        if dropped:
            lines.append(f"audit.{sink}: {dropped} rows dropped (upload failures)")
    vault_block = payload.get("vault") or {}
    if vault_block.get("remote_plaintext"):
        lines.append(
            "vault: PLAINTEXT rows may leave this machine"
            " (LLM_REDACT_VAULT_REMOTE_PLAINTEXT hatch active)"
        )
    disabled = payload.get("providers_disabled") or []
    if disabled:
        # Fail-closed, so protection is intact — noted for completeness.
        lines.append(f"providers disabled (fail closed): {', '.join(disabled)}")
    if lines:
        print("posture:")
        for line in lines:
            print(f"  ⚠ {line}")
    else:
        print("posture: all traffic redacted (no coverage opt-outs)")
