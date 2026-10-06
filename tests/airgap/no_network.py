"""Run an ``llm-redact`` command and fail if it tried to reach the network.

Used by the CI ``airgap`` job inside a network namespace with no route
(.github/workflows/ci.yml): there every connection fails anyway, and a
library that swallows the failure (or falls back after it) would leave the
command green. A Python audit hook records every attempt to resolve a host
name or connect an IP socket, whoever made it — llm-redact, a model
library, or anything they import — and the exit status is 3 when there was
one (the hosts are printed: names and addresses only, never data).

    python tests/airgap/no_network.py serve --check --config tests/airgap/config.toml
"""

import socket
import sys

_IP_FAMILIES = (socket.AF_INET, socket.AF_INET6)
_attempts: list[str] = []


def _audit(event: str, args: tuple[object, ...]) -> None:
    if event == "socket.connect":
        sock, address = args[0], args[1]
        if getattr(sock, "family", None) in _IP_FAMILIES:
            _attempts.append(f"connect {address!r}")
    elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex"):
        _attempts.append(f"{event.split('.', 1)[1]} {args[0]!r}")


def main(argv: list[str]) -> int:
    sys.addaudithook(_audit)
    from llm_redact.cli import main as cli

    status = 0
    try:
        cli(argv)
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else 1
    if _attempts:
        print(f"no_network: {len(_attempts)} network attempt(s):", file=sys.stderr)
        for attempt in _attempts:
            print(f"  {attempt}", file=sys.stderr)
        return 3
    print(f"no_network: no network attempt (llm-redact exit {status})", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
