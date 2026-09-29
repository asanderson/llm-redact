"""``allowed_origins``: the operator's opt-in list of browser origins.

By default the proxy refuses every request a web page on another origin
sends (tests/test_request_origin.py): any page could otherwise read the
operator's restored values back through rehydration, or spend a credential
the proxy holds. A browser app the operator trusts — a web chat UI on
``https://chat.example.com``, a local dev server on ``http://localhost:3000``
— is listed in ``allowed_origins``; a browser-marked request whose Origin
matches a listed origin exactly (after normalization) is then served even
though it is cross-site. Everything else holds: its Host must still be a
name the proxy answers to, the lent-credential and identity rules are
unchanged, and the reserved ``/__llm-redact/*`` paths never consult the list.

The proxy stays transparent to CORS: the preflight and the request are
forwarded, and the provider's own CORS answer reaches the browser. Listing
an origin hands it the vault: doctor WARNs with the list, ``/status`` counts
it, and ``llm-redact status`` prints it in the posture block.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from typing import Any

import httpx
import pytest

from llm_redact.config import (
    Config,
    ConfigError,
    ProviderConfig,
    normalize_origin,
    parse_config,
)
from llm_redact.config_write import emit_config_toml
from llm_redact.doctor_cli import _check_allowed_origins, _Report
from llm_redact.proxy import CSRF_HEADER, ProxyState, create_app
from test_upstream_auth import BEDROCK, BEDROCK_PATH, _claude_body, _identity, _install

EMAIL = "jane.doe@corp.example"
APP = "https://chat.example.com"
LOCAL_APP = "http://localhost:3000"
EVIL = "https://evil.example"
# What a browser adds to a cross-site fetch() from the app's page.
FROM_APP = {"origin": APP, "sec-fetch-site": "cross-site", "sec-fetch-mode": "cors"}


class _CorsUpstream:
    """A provider that supports browser CORS the way the big APIs do: it
    answers a preflight and reflects the request's Origin in
    ``Access-Control-Allow-Origin``; a POST echoes the first message."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        origin = request.headers.get("origin")
        cors = {"access-control-allow-origin": origin, "vary": "Origin"} if origin else {}
        if request.method == "OPTIONS":
            allowed = request.headers.get("access-control-request-headers", "")
            return httpx.Response(
                204,
                headers={
                    **cors,
                    "access-control-allow-methods": "POST",
                    "access-control-allow-headers": allowed,
                    "access-control-max-age": "600",
                },
            )
        text = json.loads(request.content)["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"role": "assistant", "content": [{"type": "text", "text": text}]},
            headers=cors,
        )


def _app(config: Config, upstream: _CorsUpstream) -> Any:
    return create_app(config, upstream_transport=httpx.MockTransport(upstream))


def _client(app: Any, base_url: str = "http://127.0.0.1:8787") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


def _state(app: Any) -> ProxyState:
    state: ProxyState = app.state.proxy
    return state


def _listed(*origins: str, **providers: ProviderConfig) -> Config:
    return Config(
        providers={**Config().providers, "anthropic": ProviderConfig("http://up"), **providers},
        allowed_origins=origins,
    )


def _body(text: str) -> dict[str, Any]:
    return {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": text}]}


# --- the request path ---------------------------------------------------------------------


async def test_a_listed_origin_is_served_with_the_providers_cors_answer() -> None:
    upstream = _CorsUpstream()
    app = _app(_listed(APP), upstream)
    browser_headers = {"x-api-key": "the-apps-own-key", "anthropic-version": "2023-06-01"}
    async with _client(app) as client:
        preflight = await client.options(
            "/v1/messages",
            headers={
                **FROM_APP,
                "access-control-request-method": "POST",
                "access-control-request-headers": "anthropic-version,content-type,x-api-key",
            },
        )
        response = await client.post(
            "/v1/messages", json=_body(f"mail {EMAIL}"), headers={**FROM_APP, **browser_headers}
        )
    # The preflight reached the provider and its CORS answer came back as is.
    assert preflight.status_code == 204
    assert preflight.headers["access-control-allow-origin"] == APP
    assert preflight.headers["access-control-allow-headers"] == (
        "anthropic-version,content-type,x-api-key"
    )
    # The request was redacted, forwarded with the page's Origin (the
    # provider decides CORS), and its answer restored.
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == APP
    assert response.json()["content"][0]["text"] == f"mail {EMAIL}"
    options, post = upstream.requests
    assert options.method == "OPTIONS" and options.headers["origin"] == APP
    assert post.headers["origin"] == APP and EMAIL.encode() not in post.content
    assert dict(_state(app).request_origin_refusals) == {}


@pytest.mark.parametrize(
    ("origin", "served"),
    [
        (APP, True),
        ("HTTPS://Chat.Example.COM", True),  # browsers never send it so; still the same origin
        ("https://chat.example.com:443", True),  # the default port, spelled out
        ("https://chat.example.com:8443", False),  # another port: another origin
        ("http://chat.example.com", False),  # another scheme
        ("https://evil.chat.example.com", False),  # a subdomain is another origin
        ("https://chat.example.com.evil.example", False),
        ("https://chat.example.com/path", False),  # not a serialized origin
        ("null", False),
    ],
)
async def test_only_an_exactly_listed_origin_is_served(origin: str, served: bool) -> None:
    upstream = _CorsUpstream()
    app = _app(_listed(APP), upstream)
    async with _client(app) as client:
        response = await client.post(
            "/v1/messages",
            json=_body("hi"),
            headers={"origin": origin, "sec-fetch-site": "cross-site"},
        )
    assert (response.status_code == 200) is served
    assert len(upstream.requests) == (1 if served else 0)
    assert dict(_state(app).request_origin_refusals) == ({} if served else {"origin": 1})


async def test_a_listed_origin_still_needs_a_host_the_proxy_answers_to() -> None:
    # A rebound name (or any alias not in allowed_hosts) is refused even
    # when the page's origin is listed.
    upstream = _CorsUpstream()
    app = _app(_listed(APP), upstream)
    async with _client(app, "http://rebind.example:8787") as client:
        response = await client.post("/v1/messages", json=_body("hi"), headers=FROM_APP)
    assert response.status_code == 403
    assert upstream.requests == []
    assert dict(_state(app).request_origin_refusals) == {"host": 1}


async def test_a_cross_site_request_without_an_origin_is_not_a_listed_one() -> None:
    # A no-cors fetch or an <img> carries no Origin: there is nothing to
    # check against the list, so Sec-Fetch-Site still refuses it.
    upstream = _CorsUpstream()
    app = _app(_listed(APP), upstream)
    async with _client(app) as client:
        response = await client.get("/v1/models", headers={"sec-fetch-site": "cross-site"})
    assert response.status_code == 403
    # GET /v1/models is a recognized (redact-only) OpenAI route: the
    # refusal is OpenAI-shaped.
    assert "allowed_origins" in response.json()["error"]["message"]
    assert upstream.requests == []
    assert dict(_state(app).request_origin_refusals) == {"fetch_site": 1}


async def test_the_refusal_names_the_opt_in() -> None:
    app = _app(_listed(APP), _CorsUpstream())
    async with _client(app) as client:
        response = await client.post(
            "/v1/messages",
            json=_body("hi"),
            headers={"origin": EVIL, "sec-fetch-site": "cross-site"},
        )
    assert response.status_code == 403
    message = response.json()["error"]["message"]
    assert "allowed_origins" in message and "evil.example" not in message


async def test_reserved_paths_never_consult_the_list() -> None:
    # The dashboard surface stays same-origin + CSRF token: a listed origin
    # is as foreign there as any other (even holding the token), and no
    # reserved reply grants CORS.
    app = _app(_listed(APP, LOCAL_APP), _CorsUpstream())
    token = _state(app).csrf_token
    async with _client(app) as client:
        sessions = await client.get("/__llm-redact/sessions", headers={"origin": APP})
        prune = await client.post(
            "/__llm-redact/sessions/prune",
            json={"older_than_days": 1},
            headers={"origin": APP, CSRF_HEADER: token},
        )
        status = await client.get("/__llm-redact/status", headers=FROM_APP)
    assert (sessions.status_code, prune.status_code) == (403, 403)
    assert sessions.json() == {"error": "origin not allowed"}
    assert status.status_code == 200
    assert not any(name.startswith("access-control-") for name in status.headers)


async def test_every_other_rule_still_holds_for_a_listed_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Identity auth: a listed page on a name the proxy answers to spends the
    # identity like a local tool (what listing it means); the identity rules
    # are untouched — a preflight is an unrecognized route there, refused
    # 403 and never forwarded, and a foreign Host is refused.
    _, built = _install(monkeypatch)
    upstream = _CorsUpstream()
    app = _app(_listed(APP, bedrock=_identity(BEDROCK)), upstream)
    async with _client(app) as client:
        signed = await client.post(BEDROCK_PATH, json=_claude_body("hi"), headers=FROM_APP)
        preflight = await client.options(
            BEDROCK_PATH, headers={**FROM_APP, "access-control-request-method": "POST"}
        )
    async with _client(app, "http://rebind.example:8787") as client:
        rebound = await client.post(BEDROCK_PATH, json=_claude_body("hi"), headers=FROM_APP)
    assert (signed.status_code, preflight.status_code, rebound.status_code) == (200, 403, 403)
    assert 'auth = "identity"' in preflight.json()["error"]
    assert len(built[0].calls) == 1 and len(upstream.requests) == 1


async def test_status_counts_the_listed_origins() -> None:
    app = _app(_listed(APP, LOCAL_APP), _CorsUpstream())
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["allowed_origins"] == 2  # the count, never the list
    assert APP not in json.dumps(status)
    assert status["request_origin_refusals_total"] == {}


def test_status_posture_names_the_opt_in(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.cli import _print_posture

    _print_posture({"warnings_total": {}, "detection": {}, "audit": {}, "allowed_origins": 2})
    out = capsys.readouterr().out
    assert "2 browser origin(s)" in out and "allowed_origins" in out
    _print_posture({"warnings_total": {}, "detection": {}, "audit": {}, "allowed_origins": 0})
    assert "allowed_origins" not in capsys.readouterr().out


# --- the config key -----------------------------------------------------------------------


def test_allowed_origins_parse_to_their_serialized_form() -> None:
    config = parse_config(
        {
            "allowed_origins": [
                "https://Chat.Example.com/",
                "https://chat.example.com:443",
                "http://LOCALHOST:3000",
                "http://127.0.0.1:5173",
                "http://[::1]:8080",
                "http://app.localhost",
                "https://[2001:DB8::1]:443",
                "https://10.0.0.5:8443",
            ]
        },
        "<t>",
    )
    assert config.allowed_origins == (
        "http://127.0.0.1:5173",
        "http://[::1]:8080",
        "http://app.localhost",
        "http://localhost:3000",
        "https://10.0.0.5:8443",
        "https://[2001:db8::1]",
        "https://chat.example.com",
    )
    assert parse_config({}, "<t>").allowed_origins == ()


@pytest.mark.parametrize(
    "value",
    [
        "null",
        "chat.example.com",  # no scheme
        "ftp://chat.example.com",
        "chrome-extension://abcdefghijklmnop",
        "file:///home/user/app.html",
        "https://chat.example.com/app",
        "https://chat.example.com?x=1",
        "https://chat.example.com#top",
        "https://user@chat.example.com",
        "https://*.example.com",
        "*",
        "https://",
        "https://chat.example.com:0",
        "https://chat.example.com:99999",
        "https://bücher.example",
        "https://a..b",
        "",
    ],
)
def test_allowed_origins_refuse_anything_but_a_web_origin(value: str) -> None:
    with pytest.raises(ConfigError, match="allowed_origins entry 2") as refused:
        parse_config({"allowed_origins": [APP, value]}, "<t>")
    if value.strip() and value != "null":  # the message names the null origin itself
        assert value not in str(refused.value)


@pytest.mark.parametrize(
    "value",
    ["http://chat.example.com", "http://10.0.0.5:3000", "http://[2001:db8::1]", "http://proxy"],
)
def test_a_plain_http_origin_must_be_loopback(value: str) -> None:
    # Anyone on the network path can serve a plain-HTTP page as that origin.
    with pytest.raises(ConfigError, match="allowed_origins entry 1 .*https") as refused:
        parse_config({"allowed_origins": [value]}, "<t>")
    assert value not in str(refused.value)


@pytest.mark.parametrize("value", ["https://chat.example.com", [APP, 3], {"a": APP}])
def test_allowed_origins_must_be_a_list_of_strings(value: Any) -> None:
    with pytest.raises(ConfigError, match="allowed_origins must be an array"):
        parse_config({"allowed_origins": value}, "<t>")


def test_normalize_origin_is_the_request_time_comparison() -> None:
    assert normalize_origin(" HTTPS://Chat.Example.COM:443 ") == APP
    assert normalize_origin("http://[0:0::1]:3000") == "http://[::1]:3000"
    for bad in ("null", "https://a/b", "ws://chat.example.com", "http://[::1", "http://h:x"):
        assert normalize_origin(bad) is None


def test_allowed_origins_round_trip_through_the_emitter() -> None:
    config = parse_config({"allowed_origins": [LOCAL_APP, APP]}, "<t>")
    text = emit_config_toml(config)
    assert f'allowed_origins = [\n    "{LOCAL_APP}",\n    "{APP}",\n]\n' in text
    assert text.index("allowed_origins") < text.index("[providers.")  # a top-level key
    assert parse_config(tomllib.loads(text), "<emitted>") == config
    assert "allowed_origins" not in emit_config_toml(Config())  # omitted at the default


def test_allowed_origins_is_restart_only() -> None:
    app = _app(Config(allowed_origins=(APP,)), _CorsUpstream())
    state = _state(app)
    assert state.apply_config(Config(allowed_origins=(APP, LOCAL_APP))) == ["allowed_origins"]
    assert state.config.allowed_origins == (APP,)  # kept until a restart


# --- doctor ------------------------------------------------------------------------------


def _doctor_rows(config: Config) -> list[dict[str, str]]:
    report = _Report(json_mode=True)
    _check_allowed_origins(report, config)
    return report.rows


def test_doctor_warns_with_the_listed_origins() -> None:
    assert _doctor_rows(Config()) == []
    [row] = _doctor_rows(Config(allowed_origins=(LOCAL_APP, APP)))
    assert (row["level"], row["area"]) == ("WARN", "origins")
    assert APP in row["message"] and LOCAL_APP in row["message"]
    assert "read your redacted values back" in row["message"]
    assert "credential" not in row["message"]  # the proxy lends none here
    lending = dataclasses.replace(
        Config(allowed_origins=(APP,)),
        providers={**Config().providers, "bedrock": _identity(BEDROCK)},
    )
    [row] = _doctor_rows(lending)
    assert "[providers.bedrock]" in row["message"] and "credential" in row["message"]
