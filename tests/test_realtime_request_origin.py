"""Cross-site WebSocket hijacking: a web page never drives a realtime route.

Browsers apply no CORS to a WebSocket handshake, so any page can open
``ws://127.0.0.1:PORT/<realtime path>`` and read every frame back: with an
identity-authorized route (Azure Realtime, Vertex Live) it converses as the
proxy's cloud identity; on any route it can send its own key and a token like
«EMAIL_001», and the relay would restore the operator's value into the
frames it reads. A browser always sends ``Origin`` on the handshake, so the
relay refuses a foreign one (and a host name the proxy does not answer to)
with accept-then-close 1008 before any credential fetch or upstream dial —
recorded like its HTTP twin. Real sockets end to end (uvicorn on port 0 + a
fake ``websockets`` upstream), the tests/test_realtime_relay.py pattern.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
import websockets

from llm_redact.config import Config
from test_realtime_identity import AZURE_GA, PROXY_TOKEN, FakeAuth, _install, _recent
from test_realtime_identity import _status as _proxy_status
from test_realtime_relay import FakeUpstream, _config, _proxy

pytestmark = pytest.mark.asyncio

EMAIL = "jane.doe@corp.example"
EVIL = "https://evil.example"


def _connect(proxy_host: str, path: str, *, host: str | None = None, **kwargs: Any) -> Any:
    """``websockets.connect`` to the proxy; ``host`` names a different Host
    header while the TCP connection still goes to the proxy (DNS rebinding,
    or a compose service / Kubernetes Service name)."""
    address, port = proxy_host.rsplit(":", 1)
    uri = f"ws://{host or proxy_host}{path}"
    return websockets.connect(uri, host=address, port=int(port), proxy=None, **kwargs)


async def _closed(client: Any) -> Any:
    """The close frame the proxy sent (a bounded wait: an open relay would
    otherwise block the test forever)."""
    with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
        await asyncio.wait_for(client.recv(), timeout=10)
    assert closed.value.rcvd is not None
    return closed.value.rcvd


def _hello(text: str) -> str:
    content = [{"type": "input_text", "text": text}]
    item = {"type": "message", "role": "user", "content": content}
    return json.dumps({"type": "conversation.item.created", "item": item})


@pytest.mark.parametrize(
    ("kwargs", "kind"),
    [
        ({"origin": EVIL}, "origin"),
        ({"origin": "null"}, "origin"),
        ({"host": "rebind.example:8787", "origin": "http://rebind.example:8787"}, "host"),
        ({"host": "rebind.example:8787"}, "host"),  # identity: no browser marker needed
        ({"additional_headers": {"sec-fetch-site": "cross-site"}}, "fetch_site"),
    ],
)
async def test_a_web_page_never_converses_as_the_proxys_identity(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], kind: str
) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, "azure", auth="identity")) as proxy_host:
            async with _connect(proxy_host, f"{AZURE_GA}?model=d", **kwargs) as client:
                closed = await _closed(client)
            row = await _recent(proxy_host, lambda r: r["method"] == "WS")
            status = await _proxy_status(proxy_host)
    assert closed.code == 1008
    assert "evil.example" not in closed.reason and "rebind.example" not in closed.reason
    assert auth.calls == [] and fake.paths == []  # no credential fetched, never dialled
    assert (row["status"], row["provider"], row["path"]) == (403, "azure", AZURE_GA)
    assert status["request_origin_refusals_total"] == {kind: 1}


async def test_a_web_page_learns_nothing_about_the_routes_behind_the_proxy() -> None:
    # Refused first: an unknown path, a disabled provider or a missing route
    # would otherwise each answer a page with its own reason.
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, enabled=False)) as proxy_host:
            for path in ("/v1/unknown", "/v1/realtime"):
                async with _connect(proxy_host, path, origin=EVIL) as client:
                    closed = await _closed(client)
                assert (closed.code, path) == (1008, path)
                assert "disabled" not in closed.reason and "route" not in closed.reason
            status = await _proxy_status(proxy_host)
    assert status["request_origin_refusals_total"] == {"origin": 2}
    assert fake.paths == []


async def test_the_proxys_own_origin_and_tools_still_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port, "azure", auth="identity")) as proxy_host:
            # A tool (no Origin) and a page served by the proxy itself.
            for origin in (None, f"http://{proxy_host}"):
                async with _connect(proxy_host, f"{AZURE_GA}?model=d", origin=origin) as client:
                    await client.send(_hello(f"mail {EMAIL}"))
                    echoed = json.loads(await client.recv())
                assert echoed["item"]["content"][0]["text"] == f"mail {EMAIL}"
    assert len(auth.calls) == 2
    assert [headers["authorization"] for headers in fake.headers] == [PROXY_TOKEN] * 2


async def test_a_web_page_cannot_read_the_vault_back_over_a_realtime_route() -> None:
    # The page brings its OWN key (the insecure subprotocol browsers use);
    # the upstream echoes whatever it is sent, so a relayed «EMAIL_001»
    # would come back restored to the operator's value.
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port)) as proxy_host:
            async with _connect(proxy_host, "/v1/realtime?model=m") as tool:
                await tool.send(_hello(EMAIL))  # the operator's tool fills the vault
                assert EMAIL in await tool.recv()
            async with _connect(
                proxy_host,
                "/v1/realtime?model=m",
                origin=EVIL,
                subprotocols=[
                    websockets.Subprotocol("realtime"),
                    websockets.Subprotocol("openai-insecure-api-key.sk-attacker"),
                ],
            ) as page:
                closed = await _closed(page)
    assert closed.code == 1008
    assert fake.paths == ["/v1/realtime?model=m"]  # only the tool's connection was dialled


async def test_an_alias_host_without_browser_markers_keeps_its_own_key_path() -> None:
    async with FakeUpstream() as fake:
        with _proxy(_config(fake.port)) as proxy_host:
            async with _connect(
                proxy_host,
                "/v1/realtime",
                host="llm-redact:8787",
                additional_headers={"Authorization": "Bearer sk-tool"},
            ) as client:
                await client.send('{"type":"noop"}')
                assert json.loads(await client.recv()) == {"type": "noop"}
    assert fake.headers[0]["authorization"] == "Bearer sk-tool"


async def test_allowed_hosts_lets_an_alias_host_spend_the_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = FakeAuth()
    _install(monkeypatch, auth)
    async with FakeUpstream() as fake:
        base = _config(fake.port, "azure", auth="identity")
        config = Config(providers=base.providers, allowed_hosts=("llm-redact",))
        with _proxy(config) as proxy_host:
            path = f"{AZURE_GA}?model=d"
            async with _connect(proxy_host, path, host="llm-redact:8787") as client:
                await client.send(_hello("hi"))
                await client.recv()
    assert len(auth.calls) == 1 and len(fake.paths) == 1


# --- allowed_origins: the operator's opt-in browser origins -----------------------------

APP = "https://chat.example.com"


async def test_a_listed_origin_may_open_a_realtime_connection() -> None:
    # A browser realtime app the operator listed: served like a tool, its
    # Origin forwarded to the provider (which decides for itself), its
    # frames redacted and restored.
    async with FakeUpstream() as fake:
        config = Config(providers=_config(fake.port).providers, allowed_origins=(APP,))
        with _proxy(config) as proxy_host:
            async with _connect(proxy_host, "/v1/realtime?model=m", origin=APP) as client:
                await client.send(_hello(f"mail {EMAIL}"))
                echoed = json.loads(await client.recv())
            status = await _proxy_status(proxy_host)
    assert echoed["item"]["content"][0]["text"] == f"mail {EMAIL}"
    assert fake.headers[0]["origin"] == APP
    [frame] = fake.received
    assert isinstance(frame, str) and EMAIL not in frame
    assert status["request_origin_refusals_total"] == {}
    assert status["allowed_origins"] == 1


@pytest.mark.parametrize(
    ("kwargs", "kind"),
    [
        ({"origin": EVIL}, "origin"),  # listing one origin admits no other
        ({"origin": "https://chat.example.com:8443"}, "origin"),
        ({"origin": APP, "host": "rebind.example:8787"}, "host"),  # still a named Host
    ],
)
async def test_only_the_listed_origin_on_an_answered_host(
    kwargs: dict[str, Any], kind: str
) -> None:
    async with FakeUpstream() as fake:
        config = Config(providers=_config(fake.port).providers, allowed_origins=(APP,))
        with _proxy(config) as proxy_host:
            async with _connect(proxy_host, "/v1/realtime", **kwargs) as client:
                closed = await _closed(client)
            status = await _proxy_status(proxy_host)
    assert closed.code == 1008
    assert fake.paths == []
    assert status["request_origin_refusals_total"] == {kind: 1}
