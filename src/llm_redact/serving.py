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
"""

from __future__ import annotations

import socket
from contextlib import suppress
from typing import Any

import uvicorn

# uvicorn's own exit status when the server never started (uvicorn.run).
STARTUP_FAILURE = 3


class ProxyServer(uvicorn.Server):
    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        state = getattr(getattr(self.config.app, "state", None), "proxy", None)
        if state is not None:
            state.connections.end_event_streams()
        await super().shutdown(sockets)


def run_server(app: Any, **kwargs: Any) -> None:
    """``uvicorn.run(app, **kwargs)`` for an app object (no reload, one
    worker — what ``serve`` runs), with ``ProxyServer``."""
    server = ProxyServer(uvicorn.Config(app, **kwargs))
    with suppress(KeyboardInterrupt):
        server.run()
    if not server.started:
        raise SystemExit(STARTUP_FAILURE)
