"""Web pages never drive the proxy's API routes (CSRF, DNS rebinding).

Any page the operator's browser visits can send requests to
``http://127.0.0.1:8787``: a "simple" cross-origin POST (``text/plain``, no
preflight), a CORS request whose preflight the proxy used to forward to a
CORS-friendly upstream, or — after DNS rebinding — a same-origin request
that can also READ the answer. Two things the proxy lends make that
dangerous: a credential it holds (``[providers.NAME] auth = "identity"``, a
routed plan's operator key) and its vault — every rehydrating route restores
the operator's values into whatever the upstream echoes, so a page with its
OWN provider key could read the vault back token by token.

The rule, applied before any credential fetch or upstream contact:

- a request carrying browser markers (``Origin`` or any ``Sec-Fetch-*``
  header — names page script can neither set nor remove) must be addressed
  to a host name the proxy answers to, carry only its own origin, and a
  ``Sec-Fetch-Site`` of ``same-origin`` or ``none``;
- a request that would spend a credential the PROXY holds must be addressed
  to such a host name even without browser markers (a browser without Fetch
  Metadata sends none on a same-origin GET), unless it arrived over TLS —
  a browser verifies the proxy's certificate against the name it resolved,
  so a rebound page cannot reach a TLS listener.

CLI tools and SDKs send no browser headers: an alias host name (a compose
service, a Kubernetes Service) keeps working for them on the client's own
credential, and is listed in ``allowed_hosts`` where the proxy lends its own.
Driven through ``create_app`` + ``httpx.ASGITransport`` with fake upstreams
and a fake authorizer / router on a bare Registry — keyless and pro-free.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import tomllib
from collections.abc import Callable
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from license_fixtures import resolved
from llm_redact.config import (
    Config,
    ConfigError,
    ProviderConfig,
    RoutingConfig,
    TlsConfig,
    UpstreamConfig,
    parse_config,
)
from llm_redact.config_write import emit_config_toml
from llm_redact.doctor_cli import _check_allowed_hosts, _Report
from llm_redact.proxy import REQUEST_ORIGIN_REFUSALS, ProxyState, create_app
from test_access_seam import DashboardGate
from test_upstream_auth import (
    AZURE,
    AZURE_PATH,
    BEDROCK,
    BEDROCK_PATH,
    CLAUDE_VERTEX_PATH,
    VERTEX,
    VERTEX_PATH,
    FakeAuth,
    _claude_body,
    _identity,
    _install,
)

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"
EVIL = "https://evil.example"
# What a browser adds to a cross-site "simple" request (no preflight needed).
CROSS_SITE = {
    "origin": EVIL,
    "sec-fetch-site": "cross-site",
    "sec-fetch-mode": "no-cors",
    "sec-fetch-dest": "empty",
}

IDENTITY_ROUTES = [
    ("bedrock", BEDROCK, BEDROCK_PATH, _claude_body(f"mail {EMAIL}")),
    (
        "vertex",
        VERTEX,
        VERTEX_PATH,
        {"contents": [{"role": "user", "parts": [{"text": f"mail {EMAIL}"}]}]},
    ),
    (
        "vertex",
        VERTEX,
        CLAUDE_VERTEX_PATH,
        {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": f"mail {EMAIL}"}],
        },
    ),
    (
        "azure",
        AZURE,
        AZURE_PATH + "?api-version=2024-10-21",
        {"messages": [{"role": "user", "content": f"mail {EMAIL}"}]},
    ),
]


class _Echo:
    """A fake provider: records every request and answers with the first
    text it finds in the body (an upstream model asked to repeat a string),
    or a stored Response carrying a placeholder for a GET."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET":
            stored = {
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": f"stored {TOKEN}"}],
                    }
                ],
            }
            return httpx.Response(200, json=stored)
        text = _first_text(json.loads(request.content)) if request.content else ""
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "message": {"role": "assistant", "content": text},
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            },
        )


def _first_text(body: Any) -> str:
    """The first user-supplied string in a request body (messages, contents
    or Ollama/OpenAI chat shapes)."""
    if isinstance(body, dict):
        for key in ("messages", "contents", "content", "parts", "text"):
            if key in body:
                found = _first_text(body[key])
                if found:
                    return found
        return ""
    if isinstance(body, list):
        for item in body:
            found = _first_text(item)
            if found:
                return found
        return ""
    return body if isinstance(body, str) else ""


def _app(config: Config, upstream: Callable[[httpx.Request], httpx.Response]) -> Any:
    return create_app(config, upstream_transport=httpx.MockTransport(upstream))


def _client(app: Any, base_url: str = "http://127.0.0.1") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


def _state(app: Any) -> ProxyState:
    state: ProxyState = app.state.proxy
    return state


def _refusals(app: Any) -> dict[str, int]:
    return dict(_state(app).request_origin_refusals)


def _config(**providers: ProviderConfig) -> Config:
    return Config(providers={**Config().providers, **providers})


# --- the proxy's own cloud identity -------------------------------------------------------


@pytest.mark.parametrize(("provider", "base", "path", "body"), IDENTITY_ROUTES)
async def test_a_tool_on_loopback_still_spends_the_identity(
    monkeypatch: pytest.MonkeyPatch, provider: str, base: str, path: str, body: dict[str, Any]
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Echo()
    app = _app(_config(**{provider: _identity(base)}), upstream)
    for base_url in ("http://127.0.0.1", "http://localhost:8787", "http://[::1]:8787"):
        async with _client(app, base_url) as client:
            response = await client.post(path, json=body)
        assert response.status_code == 200, (base_url, response.text)
    assert len(built[0].calls) == 3 and len(upstream.requests) == 3
    assert _refusals(app) == {}


@pytest.mark.parametrize(("provider", "base", "path", "body"), IDENTITY_ROUTES)
async def test_a_cross_site_simple_post_never_spends_the_identity(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    base: str,
    path: str,
    body: dict[str, Any],
) -> None:
    # The CSRF shape: a text/plain POST needs no preflight, and the proxy
    # would sign and send it (spend, async jobs, stored objects).
    caplog.set_level(logging.DEBUG, logger="llm_redact")
    _, built = _install(monkeypatch)
    upstream = _Echo()
    app = _app(_config(**{provider: _identity(base)}), upstream)
    async with _client(app) as client:
        response = await client.post(
            path,
            content=json.dumps(body),
            headers={**CROSS_SITE, "content-type": "text/plain;charset=UTF-8"},
        )
    assert response.status_code == 403
    assert built[0].calls == [] and upstream.requests == []
    row = _state(app).recent[-1]
    assert (row["status"], row["provider"], row["method"]) == (403, provider, "POST")
    assert _refusals(app) == {"origin": 1}
    assert "evil.example" not in response.text and "evil.example" not in caplog.text
    assert EMAIL not in caplog.text


@pytest.mark.parametrize(
    ("headers", "kind"),
    [
        ({"origin": EVIL}, "origin"),
        ({"origin": "null"}, "origin"),
        ({"origin": "http://localhost"}, "origin"),  # another origin of this machine
        ({"origin": "http://127.0.0.1:3000"}, "origin"),  # same host, another port
        ({"origin": "https://127.0.0.1"}, "origin"),  # another scheme
        ({"sec-fetch-site": "cross-site"}, "fetch_site"),
        ({"sec-fetch-site": "same-site"}, "fetch_site"),
        ({"origin": "http://127.0.0.1", "sec-fetch-site": "same-site"}, "fetch_site"),
    ],
)
async def test_browser_requests_from_other_origins_are_refused(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], kind: str
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Echo()
    app = _app(_config(bedrock=_identity(BEDROCK)), upstream)
    async with _client(app) as client:
        response = await client.post(BEDROCK_PATH, json=_claude_body("hi"), headers=headers)
    assert response.status_code == 403
    assert set(response.json()) == {"message"}  # Bedrock's error shape
    assert built[0].calls == [] and upstream.requests == []
    assert _refusals(app) == {kind: 1}


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "http://127.0.0.1", "sec-fetch-site": "same-origin"},
        {"sec-fetch-site": "same-origin", "sec-fetch-mode": "cors"},
        {"sec-fetch-site": "none", "sec-fetch-mode": "navigate"},  # typed in the address bar
        {"sec-fetch-site": " Same-Origin "},
    ],
)
async def test_the_proxys_own_origin_is_served(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Echo()
    app = _app(_config(bedrock=_identity(BEDROCK)), upstream)
    async with _client(app) as client:
        response = await client.post(BEDROCK_PATH, json=_claude_body("hi"), headers=headers)
    assert response.status_code == 200
    assert len(built[0].calls) == 1


async def test_origin_must_match_the_port_the_request_was_sent_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    app = _app(_config(bedrock=_identity(BEDROCK)), _Echo())
    async with _client(app, "http://localhost:8787") as client:
        same = await client.post(
            BEDROCK_PATH, json=_claude_body("hi"), headers={"origin": "http://LOCALHOST:8787"}
        )
        other = await client.post(
            BEDROCK_PATH, json=_claude_body("hi"), headers={"origin": "http://localhost:8788"}
        )
        malformed = await client.post(
            BEDROCK_PATH, json=_claude_body("hi"), headers={"origin": "http://localhost:x"}
        )
        with_path = await client.post(
            BEDROCK_PATH, json=_claude_body("hi"), headers={"origin": "http://localhost:8787/p"}
        )
    assert [r.status_code for r in (same, other, malformed, with_path)] == [200, 403, 403, 403]


@pytest.mark.parametrize(
    "headers",
    [
        {},  # no browser markers: an old browser's same-origin GET looks like this
        {"sec-fetch-site": "same-origin"},  # a rebound page is "same-origin" to the browser
        {"origin": "http://rebind.example:8787", "sec-fetch-site": "same-origin"},
    ],
)
async def test_a_host_the_proxy_does_not_answer_to_never_spends_the_identity(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, headers: dict[str, str]
) -> None:
    # DNS rebinding: the attacker's name now resolves to 127.0.0.1, so the
    # page is same-origin with the proxy and could read the answer too.
    caplog.set_level(logging.DEBUG, logger="llm_redact")
    _, built = _install(monkeypatch)
    upstream = _Echo()
    app = _app(_config(bedrock=_identity(BEDROCK)), upstream)
    async with _client(app, "http://rebind.example:8787") as client:
        response = await client.post(BEDROCK_PATH, json=_claude_body("hi"), headers=headers)
        listing = await client.get("/async-invoke", headers=headers)
    assert (response.status_code, listing.status_code) == (403, 403)
    assert "allowed_hosts" in response.json()["message"]
    assert built[0].calls == [] and upstream.requests == []
    assert _refusals(app) == {"host": 2}
    assert "rebind.example" not in response.text and "rebind.example" not in caplog.text


async def test_allowed_hosts_names_an_alias_the_proxy_answers_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A compose service / Kubernetes Service name, declared by the operator.
    _, built = _install(monkeypatch)
    upstream = _Echo()
    config = Config(
        providers={**Config().providers, "bedrock": _identity(BEDROCK)},
        allowed_hosts=("llm-redact", "llm-redact.team.svc.cluster.local"),
    )
    app = _app(config, upstream)
    async with _client(app, "http://LLM-Redact:8787") as client:
        plain = await client.post(BEDROCK_PATH, json=_claude_body("hi"))
        same_origin = await client.post(
            BEDROCK_PATH,
            json=_claude_body("hi"),
            headers={"origin": "http://llm-redact:8787", "sec-fetch-site": "same-origin"},
        )
    async with _client(app, "http://llm-redact.team.svc.cluster.local") as client:
        fqdn = await client.post(BEDROCK_PATH, json=_claude_body("hi"))
    async with _client(app, "http://other-service:8787") as client:
        other = await client.post(BEDROCK_PATH, json=_claude_body("hi"))
    assert [plain.status_code, same_origin.status_code, fqdn.status_code] == [200, 200, 200]
    assert other.status_code == 403
    assert len(built[0].calls) == 3
    # The same names answer the reserved endpoints' Host check.
    async with _client(app, "http://llm-redact:8787") as client:
        assert (await client.get("/__llm-redact/recent")).status_code == 200


async def test_over_tls_a_tool_may_address_the_proxy_by_any_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A browser verifies the proxy's certificate against the name it
    # resolved, so a rebound page cannot reach a TLS listener: a team
    # server's clients use its DNS names without listing them. A request
    # with browser markers keeps the Host check (defense in depth).
    _, built = _install(monkeypatch)
    app = _app(_config(bedrock=_identity(BEDROCK)), _Echo())
    async with _client(app, "https://redact.corp.example:8787") as client:
        tool = await client.post(BEDROCK_PATH, json=_claude_body("hi"))
        browser = await client.post(
            BEDROCK_PATH, json=_claude_body("hi"), headers={"sec-fetch-site": "same-origin"}
        )
    assert (tool.status_code, browser.status_code) == (200, 403)
    assert len(built[0].calls) == 1


async def test_the_gates_public_origin_is_the_proxys_own(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = DashboardGate(origin="https://proxy.team.example")
    auth = FakeAuth("bedrock")
    reg = registry_mod.Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_access_gate = lambda config, license: gate
    reg.build_upstream_auth = lambda name, provider: auth if provider.auth == "identity" else None
    monkeypatch.setattr(registry_mod, "_registry", reg)
    upstream = _Echo()
    app = _app(_config(bedrock=_identity(BEDROCK)), upstream)
    async with _client(app, "http://proxy.team.example") as client:
        served = await client.post(
            BEDROCK_PATH,
            json=_claude_body("hi"),
            headers={"origin": "https://proxy.team.example", "sec-fetch-site": "same-origin"},
        )
        foreign = await client.post(BEDROCK_PATH, json=_claude_body("hi"), headers=CROSS_SITE)
    assert (served.status_code, foreign.status_code) == (200, 403)
    assert len(auth.calls) == 1


# --- a routed plan that spends a key the proxy holds ---------------------------------------


def _routed(monkeypatch: pytest.MonkeyPatch, **plan_kwargs: Any) -> tuple[Any, FakeRouter, _Echo]:
    script = [Hop("a", "http://a.example/v1/messages"), Stop()]
    router = FakeRouter({"go": script}, plan_kwargs={"go": plan_kwargs})
    install(monkeypatch, router)
    upstream = _Echo()
    return _app(routed_config(), upstream), router, upstream


@pytest.mark.parametrize("lends", [{}, {"proxy_credential": True}])
async def test_a_routed_proxy_key_is_never_spent_for_a_foreign_host(
    monkeypatch: pytest.MonkeyPatch, lends: dict[str, Any]
) -> None:
    # A plan without the optional member counts as the proxy's (fail closed).
    app, router, upstream = _routed(monkeypatch, **lends)
    body = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    async with _client(app, "http://rebind.example:8787") as client:
        response = await client.post("/v1/messages", json=body, headers={ROUTE_HEADER: "go"})
    assert response.status_code == 403
    assert response.json()["error"]["type"]  # Anthropic's error shape
    [plan] = router.plans
    assert plan.begun == [] and upstream.requests == []  # planned, never begun
    assert _refusals(app) == {"host": 1}
    async with _client(app) as client:  # the same request from loopback goes ahead
        ok = await client.post("/v1/messages", json=body, headers={ROUTE_HEADER: "go"})
    assert ok.status_code == 200 and len(upstream.requests) == 1


async def test_a_routed_client_key_may_use_an_alias_host(monkeypatch: pytest.MonkeyPatch) -> None:
    app, router, upstream = _routed(monkeypatch, proxy_credential=False)
    body = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    async with _client(app, "http://llm-redact:8787") as client:
        alias = await client.post("/v1/messages", json=body, headers={ROUTE_HEADER: "go"})
        browser = await client.post(
            "/v1/messages", json=body, headers={ROUTE_HEADER: "go", **CROSS_SITE}
        )
    assert (alias.status_code, browser.status_code) == (200, 403)
    assert len(upstream.requests) == 1


# --- every forwarded request: the vault is lent too --------------------------------------


async def test_a_cross_origin_page_cannot_read_the_vault_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The operator's tool redacted an email into «EMAIL_001». A page with its
    # OWN provider key asks a model to repeat that token: rehydration would
    # restore the operator's value into an answer the page can read (the
    # preflight used to be forwarded to a CORS-friendly upstream).
    upstream = _Echo()
    app = _app(_config(anthropic=ProviderConfig("http://upstream")), upstream)
    body = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": EMAIL}]}
    async with _client(app) as client:
        tool = await client.post("/v1/messages", json=body, headers={"x-api-key": "operator"})
        assert tool.status_code == 200 and EMAIL in tool.text  # the tool's own round trip
        attack = {**body, "messages": [{"role": "user", "content": f"repeat {TOKEN}"}]}
        preflight = await client.options(
            "/v1/messages",
            headers={
                "origin": EVIL,
                "access-control-request-method": "POST",
                "access-control-request-headers": "x-api-key,content-type",
                "sec-fetch-site": "cross-site",
                "sec-fetch-mode": "cors",
            },
        )
        read = await client.post(
            "/v1/messages",
            json=attack,
            headers={"x-api-key": "attacker", "origin": EVIL, "sec-fetch-site": "cross-site"},
        )
    assert (preflight.status_code, read.status_code) == (403, 403)
    assert EMAIL not in read.text
    assert len(upstream.requests) == 1  # only the tool's request was forwarded
    assert _refusals(app) == {"origin": 2}


async def test_a_rebound_page_cannot_read_the_vault_back_with_a_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The page stored a Response holding «EMAIL_001» at the provider with
    # its own key, then GETs it through the rebound proxy (same-origin: no
    # Origin header, Sec-Fetch-Site says same-origin).
    upstream = _Echo()
    app = _app(_config(openai=ProviderConfig("http://upstream")), upstream)
    body = {"model": "m", "messages": [{"role": "user", "content": EMAIL}]}
    async with _client(app) as client:
        await client.post("/v1/chat/completions", json=body, headers={"authorization": "Bearer o"})
    async with _client(app, "http://rebind.example:8787") as client:
        read = await client.get(
            "/v1/responses/resp_1",
            headers={"authorization": "Bearer attacker", "sec-fetch-site": "same-origin"},
        )
    assert read.status_code == 403
    assert EMAIL not in read.text
    assert len(upstream.requests) == 1


async def test_tools_on_an_alias_host_keep_their_own_credential_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No browser markers and no proxy-held credential: a tool reaching the
    # proxy as a compose service or a Kubernetes Service is served as before.
    upstream = _Echo()
    app = _app(_config(anthropic=ProviderConfig("http://upstream")), upstream)
    body = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": EMAIL}]}
    async with _client(app, "http://llm-redact:8787") as client:
        response = await client.post("/v1/messages", json=body, headers={"x-api-key": "k"})
        passthrough = await client.get("/v1/models", headers={"x-api-key": "k"})
    assert response.status_code == 200 and EMAIL in response.text
    assert passthrough.status_code == 200
    assert len(upstream.requests) == 2
    assert _refusals(app) == {}


@pytest.mark.parametrize(
    ("provider", "path", "body"),
    [
        ("ollama", "/api/chat", {"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        ("ollama", "/api/tags", None),  # pass-through on a keyless upstream
        (
            "custom:vllm",
            "/custom/vllm/v1/chat/completions",
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        ),
    ],
)
async def test_keyless_upstreams_are_not_lent_to_web_pages(
    provider: str, path: str, body: dict[str, Any] | None
) -> None:
    # Ollama / a local vLLM need no key: the proxy lends ACCESS (and it
    # rewrites Host, which defeats Ollama's own DNS-rebinding check).
    upstream = _Echo()
    app = _app(_config(**{provider: ProviderConfig("http://127.0.0.1:11434")}), upstream)
    method = "POST" if body is not None else "GET"
    async with _client(app) as client:
        refused = await client.request(method, path, json=body, headers=CROSS_SITE)
    rebinding = {"sec-fetch-site": "same-origin"}
    async with _client(app, "http://rebind.example:8787") as client:
        rebound = await client.request(method, path, json=body, headers=rebinding)
    async with _client(app, "http://host.docker.internal:8787") as client:
        tool = await client.request(method, path, json=body)
    assert (refused.status_code, rebound.status_code, tool.status_code) == (403, 403, 200)
    assert len(upstream.requests) == 1  # the tool's
    assert _refusals(app) == {"origin": 1, "host": 1}


async def test_refusals_are_counted_in_status_and_recorded() -> None:
    app = _app(Config(), _Echo())
    async with _client(app) as client:
        await client.post("/v1/messages", json={}, headers={"origin": EVIL})
        await client.post("/v1/messages", json={}, headers={"sec-fetch-site": "cross-site"})
        status = (await client.get("/__llm-redact/status")).json()
    async with _client(app, "http://rebind.example") as client:
        await client.post("/v1/messages", json={}, headers={"sec-fetch-dest": "empty"})
    assert status["request_origin_refusals_total"] == {"origin": 1, "fetch_site": 1}
    assert _refusals(app) == {"origin": 1, "fetch_site": 1, "host": 1}
    rows = [row for row in _state(app).recent if row["path"] == "/v1/messages"]
    assert [row["status"] for row in rows] == [403, 403, 403]


async def test_a_malformed_origin_is_refused_not_a_server_error() -> None:
    # An unterminated IPv6 literal made urlsplit raise: the reserved guard
    # chain answered 500, and an API route must not either.
    app = create_app(Config(), upstream_transport=httpx.MockTransport(_Echo()))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        reserved = await client.get("/__llm-redact/sessions", headers={"origin": "http://[::1"})
        api = await client.post("/v1/messages", json={}, headers={"origin": "http://[::1"})
    assert reserved.status_code == 403 and reserved.json() == {"error": "origin not allowed"}
    assert api.status_code == 403
    assert _refusals(app) == {"origin": 1}


def test_refusal_reasons_fit_a_websocket_close_frame() -> None:
    assert set(REQUEST_ORIGIN_REFUSALS) == {"host", "origin", "fetch_site"}
    for reason in REQUEST_ORIGIN_REFUSALS.values():
        assert len(reason.encode("utf-8")) <= 123  # RFC 6455: never cut short


# --- the allowed_hosts config key -------------------------------------------------------


def test_allowed_hosts_parses_and_normalizes() -> None:
    config = parse_config(
        {"allowed_hosts": ["LLM-Redact", "host.docker.internal", "10.0.0.5", "[FE80::1]", "a_b"]},
        "<t>",
    )
    assert config.allowed_hosts == (
        "10.0.0.5",
        "a_b",
        "fe80::1",
        "host.docker.internal",
        "llm-redact",
    )
    assert parse_config({}, "<t>").allowed_hosts == ()


@pytest.mark.parametrize(
    "value",
    [
        "http://llm-redact",
        "llm-redact:8787",
        "*.svc.cluster.local",
        "*",
        "",
        "  ",
        "llm redact",
        "llm-redact/v1",
        "user@llm-redact",
        ".svc",
        "a..b",
    ],
)
def test_allowed_hosts_refuses_anything_but_a_host_name(value: str) -> None:
    with pytest.raises(ConfigError, match="allowed_hosts entry 2") as refused:
        parse_config({"allowed_hosts": ["ok", value]}, "<t>")
    if value.strip():
        assert value not in str(refused.value)


@pytest.mark.parametrize("value", ["llm-redact", ["x", 3], {"a": 1}])
def test_allowed_hosts_must_be_a_list_of_strings(value: Any) -> None:
    with pytest.raises(ConfigError, match="allowed_hosts must be an array"):
        parse_config({"allowed_hosts": value}, "<t>")


def test_allowed_hosts_round_trips_through_the_emitter() -> None:
    config = parse_config({"allowed_hosts": ["llm-redact", "10.0.0.5"]}, "<t>")
    text = emit_config_toml(config)
    assert 'allowed_hosts = [\n    "10.0.0.5",\n    "llm-redact",\n]\n' in text
    assert text.index("allowed_hosts") < text.index("[providers.")  # a top-level key
    assert parse_config(tomllib.loads(text), "<emitted>") == config
    assert "allowed_hosts" not in emit_config_toml(Config())  # omitted at the default


def test_allowed_hosts_is_restart_only() -> None:
    app = _app(Config(allowed_hosts=("a",)), _Echo())
    state = _state(app)
    restart = state.apply_config(Config(allowed_hosts=("a", "b")))
    assert restart == ["allowed_hosts"]
    assert state.config.allowed_hosts == ("a",)  # kept until a restart


# --- doctor -------------------------------------------------------------------------------


def _doctor_rows(config: Config) -> list[dict[str, str]]:
    report = _Report(json_mode=True)
    _check_allowed_hosts(report, config)
    return report.rows


def test_doctor_lists_allowed_hosts_and_warns_where_aliases_would_be_refused() -> None:
    assert _doctor_rows(Config()) == []  # nothing lends a credential: silent
    identity = _config(bedrock=_identity(BEDROCK))
    assert _doctor_rows(identity) == []  # loopback: tools use 127.0.0.1
    wide = dataclasses.replace(identity, host="0.0.0.0")
    [row] = _doctor_rows(wide)
    assert row["level"] == "WARN" and row["area"] == "hosts"
    assert "[providers.bedrock]" in row["message"] and "allowed_hosts" in row["message"]
    # Over TLS a browser cannot be rebound onto the listener: no warning.
    tls = TlsConfig(certfile="c.pem", keyfile="k.pem", client_ca="ca.pem")
    assert _doctor_rows(dataclasses.replace(wide, tls=tls)) == []
    listed = dataclasses.replace(wide, allowed_hosts=("llm-redact", "llm-redact.ns.svc"))
    [row] = _doctor_rows(listed)
    assert row["level"] == "PASS" and "2 more host name(s)" in row["message"]
    # A routed upstream spending the operator's key (or none) lends too.
    routing = RoutingConfig(
        enabled=True,
        present=True,
        upstreams=(
            UpstreamConfig(name="own", protocol="openai", base_url="http://o"),
            UpstreamConfig(name="op", protocol="openai", base_url="http://p", credential="env:K"),
        ),
    )
    [row] = _doctor_rows(dataclasses.replace(Config(), host="0.0.0.0", routing=routing))
    assert row["level"] == "WARN" and "[upstreams.op]" in row["message"]
    assert "[upstreams.own]" not in row["message"]
    disabled = dataclasses.replace(routing, enabled=False)
    assert _doctor_rows(dataclasses.replace(Config(), host="0.0.0.0", routing=disabled)) == []
