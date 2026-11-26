"""The server ``llm-redact serve`` runs: uvicorn's, with one shutdown step.

uvicorn's shutdown stops accepting, asks each connection to finish, then
waits — with no limit, since ``serve`` sets no graceful-shutdown timeout, so
a streamed answer still running is never cut — for every open response
before it runs the app's lifespan shutdown (docs/resilience.md, "Shutdown
order": the audit sinks' final flush, then the audit database, then the
vault). The dashboard's ``/__llm-redact/events`` stream never ends on its
own, so an open dashboard kept a SIGTERM from ever reaching that shutdown:
the supervisor killed the process instead. ``ProxyServer`` ends those
streams first, then shuts down as uvicorn does.

On Windows a wildcard listener (``host = "0.0.0.0"``, ``"::"`` or empty) is
bound here, with SO_EXCLUSIVEADDRUSE, and handed to uvicorn
(``listening_sockets``): another program could otherwise bind the port on
127.0.0.1 beside it and receive what local clients send there (``ports``).
uvicorn binds a specific address (the default 127.0.0.1) as everywhere, and
Linux and macOS always keep uvicorn's bind.
"""

from __future__ import annotations

import logging
import socket
import sys
from contextlib import suppress
from typing import Any

import uvicorn

from llm_redact import ports

# uvicorn's own exit status when the server never started (uvicorn.run).
STARTUP_FAILURE = 3
# uvicorn's exit status when it cannot bind (Server.startup).
BIND_FAILURE = 1
# What Windows answers for a port it will not bind (WSAEACCES).
_WSAEACCES = 10013

logger = logging.getLogger("llm_redact")


def listening_sockets(host: str, port: int) -> list[socket.socket] | None:
    """On Windows, for a host that resolves to a wildcard address, the
    proxy's listening sockets (``ports.bind_exclusive``); None otherwise,
    where uvicorn binds. When the host cannot be resolved or a socket bound,
    serve stops as uvicorn's own bind failure stops it (BIND_FAILURE),
    saying why."""
    if sys.platform != "win32" or ports.exclusive_option() is None:
        return None
    try:
        infos = ports.addresses(host, port)
        if not any(ports.is_wildcard(info) for info in infos):
            return None
        return ports.bind_exclusive(host, port, infos)
    except OSError as problem:
        reserved = _WSAEACCES in (getattr(problem, "winerror", None), problem.errno)
        logger.error(
            "cannot listen on %s (%s)%s",
            ports.where((host, port)),
            problem.strerror or problem,
            "; Windows may reserve the port for Hyper-V, WSL or Docker (netsh int ipv4 show"
            " excludedportrange protocol=tcp), or another program holds it"
            if reserved
            else "",
        )
        raise SystemExit(BIND_FAILURE) from None


class ProxyServer(uvicorn.Server):
    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        state = getattr(getattr(self.config.app, "state", None), "proxy", None)
        if state is not None:
            state.connections.end_event_streams()
        await super().shutdown(sockets)


def run_server(app: Any, **kwargs: Any) -> None:
    """``uvicorn.run(app, **kwargs)`` for an app object (no reload, one
    worker — what ``serve`` runs), with ``ProxyServer``; on Windows a
    wildcard host's port is bound exclusively first (``listening_sockets``)
    and closed when the server ends."""
    server = ProxyServer(uvicorn.Config(app, **kwargs))
    sockets = listening_sockets(server.config.host, server.config.port)
    try:
        with suppress(KeyboardInterrupt):
            server.run(sockets=sockets)
    finally:
        for sock in sockets or ():
            sock.close()
    if not server.started:
        raise SystemExit(STARTUP_FAILURE)
