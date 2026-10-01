"""``llm-redact override``: approve, list and revoke refusal overrides.

    llm-redact override CODE --once | --always   approve one refusal
    llm-redact override list [--json]           pending codes and live rules
    llm-redact override revoke ID               drop a rule (r12) or a code (p7)

Approving asks for a PERSON: the refusal is shown and a confirmation is
read from the controlling terminal (``/dev/tty``) — never from stdin, so
piping ``allow`` into the command approves nothing, and without a terminal
the command refuses. This guards against an accidental approval, not
against local software running as the operator: an agent with a shell can
open a pseudo-terminal and type the answer, or edit the store file
(docs/overrides.md). The CLI works on the override store
file directly (like ``lookup``), as the local operator: it approves only
refusals of requests the proxy admitted without a named user (llm-redact-pro
named users approve their own in the dashboard); it lists and revokes any.
Nothing printed is a value, a digest or a code.

Refusal overrides are off by default. While the config (the proxy's: the
same search as ``serve``, or ``--config``) leaves them off, every form exits
1 naming ``[overrides] enabled``: approve and revoke touch nothing, and list
still prints what the store holds (value-free, read-only) with a line saying
those records are inert — the proxy applies none of them."""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from llm_redact.config import Config, ConfigError, apply_env_overrides, load_config
from llm_redact.overrides import (
    DISABLED_REASON,
    TO_ENABLE,
    OverrideEntry,
    OverrideError,
    OverrideStore,
    default_overrides_path,
    normalize_code,
)

CONFIRM_WORD = "allow"


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    override = subparsers.add_parser(
        "override",
        help="approve (on the terminal), list or revoke refusal overrides",
    )
    override.add_argument("target", help="the refusal's CODE, or list, or revoke")
    override.add_argument("entry", nargs="?", default=None, help="the id to revoke (r12, p7)")
    scope = override.add_mutually_exclusive_group()
    scope.add_argument("--once", action="store_true", help="let the next such request through")
    scope.add_argument(
        "--always", action="store_true", help="let these values (or this route) through every time"
    )
    override.add_argument("--json", action="store_true", help="list: machine-readable output")
    override.add_argument("--config", type=Path, default=None, help="path to config.toml")
    override.add_argument("--db", type=Path, default=None, help="override store path")


def _store(args: argparse.Namespace, config: Config, *, read_only: bool = False) -> OverrideStore:
    if args.db is not None:
        path = Path(args.db).expanduser()
    elif config.overrides.path:
        path = Path(config.overrides.path).expanduser()
    else:
        path = default_overrides_path()
    return OverrideStore(path, ttl_seconds=config.overrides.ttl_minutes * 60, read_only=read_only)


def _open_tty() -> IO[str]:
    """The controlling terminal, for reading and writing (OSError without
    one). Unbuffered bytes under a text layer: ``open("/dev/tty", "r+")``
    wraps a terminal in a seekable BufferedRandom, which a terminal is not
    (io.UnsupportedOperation on every real one). Module-level so tests
    substitute a fake terminal; tests/test_override_cli.py also runs this
    one under a real pseudo-terminal."""
    raw = io.FileIO("/dev/tty", "r+")
    try:
        return io.TextIOWrapper(raw, encoding="utf-8", write_through=True)
    except BaseException:
        raw.close()
        raise


def _when(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%SZ")


def _describe(entry: OverrideEntry) -> str:
    return (
        f"kind {entry.kind}, types {', '.join(entry.types) or '-'},"
        f" route {entry.route}, requester {entry.subject or 'operator'}"
    )


def run_override(args: argparse.Namespace) -> int:
    target = args.target
    try:
        config = apply_env_overrides(load_config(args.config))
    except ConfigError as exc:
        print(f"llm-redact override: {exc}", file=sys.stderr)
        return 2
    try:
        if target == "list":
            return _list(args, config)
        if not config.overrides.enabled:
            # Off (the default): approving or revoking a record the proxy
            # never applies would only look like it did something.
            print(f"llm-redact override: {DISABLED_REASON}; {TO_ENABLE}", file=sys.stderr)
            return 1
        if target == "revoke":
            if args.entry is None:
                print("usage: llm-redact override revoke ID (as `override list` shows it)")
                return 2
            _store(args, config).revoke(args.entry)
            print(f"revoked {args.entry}")
            return 0
        return _approve(args, config)
    except OverrideError as exc:
        print(f"llm-redact override: {exc}", file=sys.stderr)
        return 1


def _approve(args: argparse.Namespace, config: Config) -> int:
    code = normalize_code(args.target)
    if code is None:
        print(
            "llm-redact override: that is not a refusal code (12 characters, as the refusal"
            " message shows it); or use: list, revoke ID",
            file=sys.stderr,
        )
        return 2
    if not (args.once or args.always):
        print("llm-redact override: choose --once or --always", file=sys.stderr)
        return 2
    scope = "always" if args.always else "once"
    store = _store(args, config)
    entry = store.describe(code)
    if entry.subject:
        raise OverrideError(
            "that refusal belongs to a named user; they approve it themselves in the"
            " llm-redact-pro dashboard"
        )
    if scope == "always":
        effect = (
            "EVERY request of this requester will forward these exact values UNREDACTED"
            if entry.kind in ("block", "binary_values", "verbatim_field")
            else "EVERY plain-text body on this route will be forwarded UNSCANNED"
        )
    else:
        effect = "the next such request (within the code's lifetime) is let through once"
    try:
        tty = _open_tty()
    except OSError:
        raise OverrideError(
            "approving an override needs a terminal: a person types the confirmation"
            " (it is never read from stdin)"
        ) from None
    with tty:
        tty.write(
            f"Refused request: {_describe(entry)}.\n"
            f"Approve {scope} for kind {entry.kind},"
            f" types {', '.join(entry.types) or '-'}: {effect}.\n"
            f"Type '{CONFIRM_WORD}' to confirm: "
        )
        tty.flush()
        answer = tty.readline()
    if answer.strip().lower() != CONFIRM_WORD:
        print("not approved", file=sys.stderr)
        return 1
    approved = store.approve(scope, approver=None, code=code)
    print(f"approved {scope}: {_describe(approved)}")
    return 0


def _list(args: argparse.Namespace, config: Config) -> int:
    """The store's records (value-free, read-only). While overrides are off
    they are still listed — so an operator sees what would apply again on
    enabling them — followed by a line saying they are inert; exit 1."""
    entries = _store(args, config, read_only=True).entries()
    if args.json:
        print(json.dumps([entry.as_dict() for entry in entries], indent=2))
    elif not entries:
        print("no pending refusals or overrides")
    else:
        header = f"{'ID':<6} {'STATE':<8} {'KIND':<15} {'TYPES':<24} {'USES':>4}"
        print(f"{header}  CREATED / REQUESTER / ROUTE")
        for entry in entries:
            print(
                f"{entry.id:<6} {entry.state:<8} {entry.kind:<15}"
                f" {','.join(entry.types) or '-':<24} {entry.uses:>4}  {_when(entry.created)}"
                f"  {entry.subject or 'operator'}  {entry.route}"
            )
    if config.overrides.enabled:
        return 0
    print(
        f"llm-redact override: {DISABLED_REASON}; the records listed are inert (the proxy"
        f" applies none of them); {TO_ENABLE}",
        file=sys.stderr,
    )
    return 1
