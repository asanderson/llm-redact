"""A hot reload never mixes two configurations inside one request.

``handle()`` decides whether a provider is authorized with the proxy's own
identity BEFORE it reads the request body (the unrecognized-route 403, the
identity body rule, the ownership check), and signs AFTER. A reload applied
while a body is still arriving (SIGHUP, the pro editor) used to be seen by
the second half only: a request admitted as pass-through was then stripped
and signed with the proxy's identity — an unrecognized route (the whole
cloud API as the proxy's principal) or a ``detection = false`` body
included. The request now keeps the decision it was admitted under, as the
reload promises in-flight requests: its authorizer and its upstream are
read once, with the provider's config.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import create_app
from test_upstream_auth import BEDROCK, VERTEX, _config, _identity, _install, _Upstream


async def _call(
    app: Any, method: str, path: str, body: bytes, during_body: Callable[[], None]
) -> int:
    """Drive the app with raw ASGI; ``during_body`` runs while the body is
    still arriving (between its first and last chunk)."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"127.0.0.1:8787"),
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer client-key"),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8787),
    }
    half = len(body) // 2
    chunks = [body[:half], body[half:]]
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        if not chunks:
            return {"type": "http.disconnect"}
        if len(chunks) == 1:
            during_body()
        chunk = chunks.pop(0)
        return {"type": "http.request", "body": chunk, "more_body": bool(chunks)}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async with app.router.lifespan_context(app):
        await app(scope, receive, send)
    return int(next(m for m in sent if m["type"] == "http.response.start")["status"])


@pytest.mark.parametrize(
    ("path", "body", "passthrough"),
    [
        # An unrecognized Vertex route: never signed with the identity.
        (
            "/v1/projects/p/locations/us-east5/endpoints/e:deployModel",
            b'{"deployedModel": {"model": "m"}}',
            ProviderConfig(VERTEX),
        ),
        # A recognized route of a detection = false provider: its body is
        # forwarded unredacted, so it must not be signed either.
        (
            "/v1/projects/p/locations/us-east5/publishers/google/models/g:generateContent",
            b"not json at all, jane.doe@corp.example",
            ProviderConfig(VERTEX, detection=False),
        ),
    ],
    ids=["unrecognized-route", "detection-off"],
)
async def test_reload_to_identity_mid_body_never_signs_a_passthrough_request(
    monkeypatch: pytest.MonkeyPatch, path: str, body: bytes, passthrough: ProviderConfig
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(_config(vertex=passthrough), upstream_transport=httpx.MockTransport(upstream))
    state = app.state.proxy
    identity = _identity(VERTEX, detection=passthrough.detection)

    status = await _call(
        app, "POST", path, body, lambda: state.apply_config(_config(vertex=identity))
    )

    assert built and built[0].calls == []  # the new authorizer never signed it
    assert status == 200
    (sent,) = upstream.requests
    assert "x-fake-signed" not in sent.headers
    assert sent.headers["authorization"] == "Bearer client-key"  # its own credential
    assert sent.content == body
    # A request that starts after the reload gets the identity's rules.
    assert "vertex" in state.upstream_auth


async def test_reload_away_from_identity_mid_body_keeps_the_admitted_authorizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The mirror image: a request admitted under identity is still signed
    # (credentials stripped) and sent to the upstream it was admitted for —
    # never forwarded with the client's credential to a newly configured host.
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(bedrock=_identity(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    state = app.state.proxy
    fresh = _config(bedrock=ProviderConfig("https://elsewhere.example"))

    status = await _call(
        app,
        "POST",
        "/model/anthropic.claude-3-5-sonnet-20240620-v1%3A0/invoke",
        b'{"anthropic_version": "bedrock-2023-05-31", "messages": []}',
        lambda: state.apply_config(fresh),
    )

    assert status == 200
    assert len(built[0].calls) == 1
    (sent,) = upstream.requests
    assert sent.url.host == "bedrock-runtime.us-east-1.amazonaws.com"
    assert sent.headers["x-fake-signed"] == "1"
    assert state.upstream_auth == {}


async def test_passthrough_upstream_is_the_one_the_request_was_admitted_for() -> None:
    upstream = _Upstream(b"{}")
    app = create_app(
        Config(providers={**Config().providers, "openai": ProviderConfig("https://one.example")}),
        upstream_transport=httpx.MockTransport(upstream),
    )
    state = app.state.proxy
    fresh = Config(
        providers={**Config().providers, "openai": ProviderConfig("https://two.example")}
    )
    status = await _call(
        app,
        "POST",
        "/v1/chat/completions",
        b'{"model": "m", "messages": []}',
        lambda: state.apply_config(fresh),
    )
    assert status == 200
    assert upstream.requests[0].url.host == "one.example"


async def test_first_configuration_mid_body_answers_the_admitted_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Azure had no upstream when the request was admitted; a reload that
    # configures it (under identity) while the body arrives must not turn
    # the request into a signed one — it keeps the "configure" 502.
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    state = app.state.proxy
    status = await _call(
        app,
        "POST",
        "/openai/deployments/gpt/chat/completions",
        b'{"messages": [{"role": "user", "content": "hi"}]}',
        lambda: state.apply_config(_config(azure=_identity("https://res.openai.azure.com"))),
    )
    assert status == 502
    assert upstream.requests == [] and built[0].calls == []
    assert state.recent[-1]["status"] == 502
