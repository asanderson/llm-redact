"""The upstream-authorization SEAM: ``[providers.NAME] auth = "identity"``.

The proxy's own cloud identity (AWS SigV4 for Bedrock, OAuth bearer tokens
for Vertex AI and Azure OpenAI) is llm-redact-pro code; the core parses the
config shape, fails closed without the package, strips every client
credential channel, hands the registered ``plugin_api.UpstreamAuth`` the
FINAL request (after redaction and note injection) and sends exactly what it
authorized. Driven here with a fake authorizer on a bare Registry —
keyless and pro-free.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from fake_router import ROUTE_HEADER, FakeRouter, Hop, install, routed_config
from llm_redact.config import Config, ConfigError, ProviderConfig, parse_config
from llm_redact.config_write import emit_config_toml
from llm_redact.plugin_api import UpstreamAuthError
from llm_redact.proxy import (
    ProxyState,
    _strip_credential_query,
    create_app,
    strip_client_credentials,
)
from llm_redact.registry import Registry

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"
BEDROCK = "https://bedrock-runtime.us-east-1.amazonaws.com"
VERTEX = "https://us-east5-aiplatform.googleapis.com"
AZURE = "https://res.openai.azure.com"
BEDROCK_PATH = "/model/anthropic.claude-3-5-sonnet-20240620-v1%3A0/invoke"
VERTEX_PATH = (
    "/v1/projects/p/locations/us-east5/publishers/google/models/gemini-2.0:generateContent"
)
CLAUDE_VERTEX_PATH = (
    "/v1/projects/p/locations/us-east5/publishers/anthropic/models/claude-x:rawPredict"
)
AZURE_PATH = "/openai/deployments/gpt/chat/completions"

# Every client credential channel a tool might fill in.
CLIENT_CREDENTIALS = {
    "authorization": "Bearer client-secret",
    "x-api-key": "client-key",
    "api-key": "client-azure-key",
    "x-goog-api-key": "client-goog-key",
    "anthropic-api-key": "client-anthropic",
    "x-amz-date": "20240101T000000Z",
    "x-amz-security-token": "client-session",
    "x-amz-content-sha256": "deadbeef",
    "x-goog-iam-authorization-token": "client-iam",
    "x-goog-iam-authority-selector": "client-selector",
    "x-goog-user-project": "someone-elses-billing-project",
    "x-ms-authorization-auxiliary": "Bearer aux",
    "proxy-authorization": "Basic cHJveHk=",
    "ocp-apim-subscription-key": "apim",
    "cookie": "session=abc",
}


class FakeAuth:
    """A scripted ``plugin_api.UpstreamAuth``: records what it saw and adds
    its own credential, or raises."""

    def __init__(self, name: str, *, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.calls: list[tuple[str, str, list[tuple[str, str]], bytes]] = []
        self.closed = 0

    async def authorize(
        self, method: str, url: str, headers: list[tuple[str, str]], body: bytes
    ) -> list[tuple[str, str]]:
        self.calls.append((method, url, list(headers), body))
        if self.error is not None:
            raise self.error
        return [*headers, ("authorization", f"Proxy {self.name}"), ("x-fake-signed", "1")]

    def close(self) -> None:
        self.closed += 1


def _install(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[str, ProviderConfig], FakeAuth | None] | None = None,
    registry: Registry | None = None,
) -> tuple[Registry, list[FakeAuth]]:
    reg = registry or Registry()
    built: list[FakeAuth] = []

    def build(name: str, provider: ProviderConfig) -> FakeAuth | None:
        if provider.auth == "passthrough":
            return None
        auth = factory(name, provider) if factory is not None else FakeAuth(name)
        if auth is not None:
            built.append(auth)
        return auth

    reg.build_upstream_auth = build
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg, built


class _Upstream:
    def __init__(
        self, content: bytes = b'{"content":[{"type":"text","text":"ok ' + TOKEN.encode() + b'"}]}'
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.content = content
        self.content_type = "application/json"

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(
            200, content=self.content, headers={"content-type": self.content_type}
        )


def _config(**providers: ProviderConfig) -> Config:
    return Config(providers={**Config().providers, **providers})


def _identity(url: str, **extra: Any) -> ProviderConfig:
    return ProviderConfig(url, auth="identity", **extra)


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def _claude_body(text: str) -> dict[str, Any]:
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": text}],
    }


# --- config shape ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["bedrock", "vertex", "azure"])
def test_identity_parses_for_the_three_cloud_providers(name: str) -> None:
    config = parse_config({"providers": {name: {"auth": "identity"}}}, "<t>")
    assert config.providers[name].auth == "identity"
    assert config.providers[name].region is None
    assert Config().providers[name].auth == "passthrough"


@pytest.mark.parametrize("name", ["anthropic", "openai", "gemini", "cohere", "ollama"])
def test_identity_elsewhere_is_a_config_error(name: str) -> None:
    with pytest.raises(ConfigError, match="supported only for azure, bedrock, vertex"):
        parse_config({"providers": {name: {"auth": "identity"}}}, "<t>")
    # passthrough is accepted everywhere (the explicit default).
    assert parse_config({"providers": {name: {"auth": "passthrough"}}}, "<t>")


def test_custom_providers_have_no_auth_key() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(
            {"providers": {"custom": {"v": {"upstream_base_url": "http://x", "auth": "identity"}}}},
            "<t>",
        )


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"auth": "sigv4"}, 'auth must be "passthrough" or "identity"'),
        ({"auth": True}, "auth must be a string"),
        ({"auth": "identity", "region": 3}, "region must be a string"),
        ({"auth": "identity", "region": "US East"}, "region must look like an AWS region"),
        ({"auth": "identity", "region": ""}, "region must look like an AWS region"),
        # The 32-character cap (a later same-named pattern once shadowed it).
        ({"auth": "identity", "region": "a" * 40}, "region must look like an AWS region"),
        ({"region": "us-east-1"}, "region is only used with"),
    ],
)
def test_bedrock_auth_and_region_validation(section: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config({"providers": {"bedrock": section}}, "<t>")


def test_region_is_bedrock_only() -> None:
    with pytest.raises(ConfigError, match="region is only used with"):
        parse_config({"providers": {"vertex": {"auth": "identity", "region": "us-east5"}}}, "<t>")


def test_emitter_round_trips_auth_and_region() -> None:
    config = parse_config(
        {
            "providers": {
                "bedrock": {
                    "upstream_base_url": BEDROCK,
                    "auth": "identity",
                    "region": "eu-west-1",
                },
                "azure": {"upstream_base_url": AZURE, "auth": "identity"},
                "vertex": {"upstream_base_url": VERTEX, "auth": "passthrough"},
            }
        },
        "<t>",
    )
    text = emit_config_toml(config)
    assert 'auth = "identity" # the proxy authorizes with its own cloud identity' in text
    assert 'region = "eu-west-1"' in text
    assert text.count("auth = ") == 2  # passthrough (the default) is omitted
    assert parse_config(tomllib.loads(text), "<emitted>") == config


# --- Free default: fail closed -------------------------------------------------------


def test_free_default_is_none_for_passthrough_and_refuses_identity() -> None:
    reg = Registry()
    assert reg.build_upstream_auth("bedrock", ProviderConfig(BEDROCK)) is None
    with pytest.raises(ConfigError, match="llm-redact-pro") as refused:
        reg.build_upstream_auth("bedrock", _identity(BEDROCK))
    assert "[providers.bedrock]" in str(refused.value)


def test_free_core_refuses_to_start_with_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        create_app(_config(vertex=_identity(VERTEX)))
    # Passthrough everywhere builds nothing.
    state: ProxyState = create_app(Config()).state.proxy
    assert state.upstream_auth == {}


# --- credential stripping ------------------------------------------------------------


def test_strip_credential_query_keeps_everything_else_verbatim() -> None:
    assert (
        _strip_credential_query(
            "api-version=2024-10-21&key=k&alt=sse&API%2DKEY=x&access_token=t&api_key=a"
            "&X-Amz-Signature=s&x-amz-credential=c&keyring=1&q=a%20b"
        )
        == "api-version=2024-10-21&alt=sse&keyring=1&q=a%20b"
    )
    assert _strip_credential_query("key=only") == ""


def test_strip_client_credentials_url_and_headers() -> None:
    url, headers = strip_client_credentials(
        f"{BEDROCK}/model/a%3A0/invoke?key=secret",
        [*CLIENT_CREDENTIALS.items(), ("content-type", "application/json"), ("x-idempotency", "1")],
    )
    assert url == f"{BEDROCK}/model/a%3A0/invoke"  # encoded path preserved, no "?"
    assert headers == [("content-type", "application/json"), ("x-idempotency", "1")]
    # httpx's on-the-wire form: what authorize() sees is what is sent.
    url, _ = strip_client_credentials(f"{AZURE}/openai/x y?api-version=1&key=k", [])
    assert url == f"{AZURE}/openai/x%20y?api-version=1"


def test_an_identity_provider_without_an_authorizer_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A plugin answering None for an identity provider would otherwise let
    # the client's own credential through in the proxy's place.
    _install(monkeypatch, lambda name, provider: None)
    with pytest.raises(ConfigError, match="built no authorizer"):
        create_app(_config(vertex=_identity(VERTEX)))


# --- the request path -----------------------------------------------------------------


async def test_bedrock_identity_signs_the_final_redacted_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream()
    app = create_app(
        _config(bedrock=_identity(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        response = await client.post(
            f"{BEDROCK_PATH}?trace=1&key=client-query-key",
            json=_claude_body(f"mail {EMAIL}"),
            headers=CLIENT_CREDENTIALS,
        )
    assert response.status_code == 200
    assert EMAIL in response.text  # rehydrated as usual
    (auth,) = built
    ((method, url, seen_headers, body),) = auth.calls
    assert method == "POST"
    assert url == f"{BEDROCK}{BEDROCK_PATH}?trace=1"
    seen = {name for name, _ in seen_headers}
    assert not seen & set(CLIENT_CREDENTIALS)
    # The FINAL body: redacted, note injected — and exactly what was sent.
    assert EMAIL.encode() not in body
    assert TOKEN.encode() in body
    assert b"character for character" in body  # the system note rides in the signed bytes
    (sent,) = upstream.requests
    assert sent.content == body
    assert str(sent.url) == url
    assert sent.headers["authorization"] == "Proxy bedrock"
    assert sent.headers["x-fake-signed"] == "1"
    for name in CLIENT_CREDENTIALS:
        if name != "authorization":
            assert name not in sent.headers
    assert "client" not in json.dumps(dict(sent.headers))


@pytest.mark.parametrize(
    ("provider", "base", "path", "body"),
    [
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
    ],
)
async def test_vertex_and_azure_identity(
    monkeypatch: pytest.MonkeyPatch, provider: str, base: str, path: str, body: dict[str, Any]
) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(**{provider: _identity(base)}), upstream_transport=httpx.MockTransport(upstream)
    )
    async with _client(app) as client:
        response = await client.post(path, json=body, headers=CLIENT_CREDENTIALS)
    assert response.status_code == 200
    (auth,) = built
    assert auth.name == provider
    (sent,) = upstream.requests
    assert sent.headers["authorization"] == f"Proxy {provider}"
    assert "api-key" not in sent.headers and "x-goog-api-key" not in sent.headers
    assert EMAIL.encode() not in sent.content
    assert auth.calls[0][3] == sent.content


async def test_streaming_response_path_is_signed_too(monkeypatch: pytest.MonkeyPatch) -> None:
    _, built = _install(monkeypatch)
    chunk = {"candidates": [{"content": {"parts": [{"text": f"hi {TOKEN}"}]}, "index": 0}]}
    upstream = _Upstream(f"data: {json.dumps(chunk)}\n\n".encode())
    upstream.content_type = "text/event-stream"
    app = create_app(
        _config(vertex=_identity(VERTEX)), upstream_transport=httpx.MockTransport(upstream)
    )
    body = {"contents": [{"role": "user", "parts": [{"text": f"mail {EMAIL}"}]}]}
    stream_path = VERTEX_PATH.replace(":generateContent", ":streamGenerateContent") + "?alt=sse"
    async with _client(app) as client:
        response = await client.post(stream_path, json=body, headers={"x-goog-api-key": "k"})
    assert response.status_code == 200
    assert EMAIL in response.text
    (sent,) = upstream.requests
    assert sent.headers["authorization"] == "Proxy vertex"
    assert "x-goog-api-key" not in sent.headers
    assert built[0].calls[0][1].endswith("?alt=sse")


async def test_pass_through_traffic_to_an_identity_provider_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The proxy's identity is lent only to routes it recognizes: an
    # unrecognized path would otherwise reach the whole cloud API as the
    # proxy's principal, unredacted.
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(bedrock=_identity(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    state: ProxyState = app.state.proxy
    async with _client(app) as client:
        response = await client.get("/guardrail/g1/version/1", headers={"x-api-key": "k"})
    assert response.status_code == 403
    assert 'auth = "identity"' in response.json()["error"]
    assert built[0].calls == [] and upstream.requests == []
    assert state.recent[-1]["status"] == 403


async def test_passthrough_providers_are_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    _, built = _install(monkeypatch)
    upstream = _Upstream(b"{}")
    app = create_app(
        _config(bedrock=ProviderConfig(BEDROCK), vertex=_identity(VERTEX)),
        upstream_transport=httpx.MockTransport(upstream),
    )
    async with _client(app) as client:
        await client.post(
            BEDROCK_PATH,
            json=_claude_body("hello"),
            headers={"authorization": "Bearer client-bedrock-key"},
        )
    (sent,) = upstream.requests
    assert sent.headers["authorization"] == "Bearer client-bedrock-key"
    assert [auth.calls for auth in built] == [[]]


@pytest.mark.parametrize(
    ("error", "source"),
    [
        (UpstreamAuthError("AWS credential chain"), "AWS credential chain"),
        (RuntimeError("secret-bearing message"), "RuntimeError"),
    ],
)
async def test_credential_failure_is_a_recorded_502_and_nothing_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    source: str,
) -> None:
    _install(monkeypatch, lambda name, provider: FakeAuth(name, error=error))
    upstream = _Upstream()
    app = create_app(
        _config(bedrock=_identity(BEDROCK)), upstream_transport=httpx.MockTransport(upstream)
    )
    state: ProxyState = app.state.proxy
    with caplog.at_level("INFO", logger="llm_redact"):
        async with _client(app) as client:
            response = await client.post(
                BEDROCK_PATH,
                json=_claude_body(f"mail {EMAIL}"),
                headers={"authorization": "Bearer client-secret"},
            )
    assert response.status_code == 502
    message = response.json()["message"]  # Bedrock's provider shape
    assert source in message and "nothing was forwarded" in message
    assert upstream.requests == []
    assert state.upstream_errors["bedrock"] == 1
    row = state.recent[-1]
    assert (row["status"], row["provider"]) == (502, "bedrock")
    assert row["detections"] == {"EMAIL": 1}
    assert "upstream credentials unavailable for bedrock" in caplog.text
    for secret in ("secret-bearing", "client-secret", EMAIL):
        assert secret not in caplog.text
        assert secret not in response.text


async def test_identity_providers_are_never_routed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routing protocols are anthropic/openai/gemini/ollama, and the core does
    not even ASK the router about a provider it authorizes itself: the
    request below would be routed to the fake's upstream if it did."""
    router = FakeRouter({"r": [Hop("routed", "http://routed.example/x")]})
    reg, _ = install(monkeypatch, router)
    _, built = _install(monkeypatch, registry=reg)
    upstream = _Upstream(b"{}")
    config = routed_config(providers={**Config().providers, "azure": _identity(AZURE)})
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    async with _client(app) as client:
        response = await client.post(
            AZURE_PATH + "?api-version=1",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={ROUTE_HEADER: "r"},
        )
    assert response.status_code == 200
    assert router.inbounds == []
    (sent,) = upstream.requests
    assert sent.url.host == "res.openai.azure.com"
    assert sent.headers["authorization"] == "Proxy azure"
    assert built[0].calls


# --- lifecycle: reload, editor dry run, shutdown ------------------------------------------


def test_reload_rebuilds_only_on_auth_changes_and_closes_displaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, built = _install(monkeypatch)
    state: ProxyState = create_app(_config(bedrock=_identity(BEDROCK))).state.proxy
    first = state.upstream_auth["bedrock"]
    # An unrelated change keeps the authorizer (and its cached credentials).
    state.apply_config(_config(bedrock=_identity(BEDROCK), openai=ProviderConfig("http://o")))
    assert state.upstream_auth["bedrock"] is first
    # A region change rebuilds it and closes the old one at swap time.
    state.apply_config(_config(bedrock=_identity(BEDROCK, region="eu-west-1")))
    assert state.upstream_auth["bedrock"] is not first
    assert first.closed == 1
    # Back to passthrough: dropped and closed.
    second = state.upstream_auth["bedrock"]
    state.apply_config(Config())
    assert state.upstream_auth == {}
    assert second.closed == 1
    assert len(built) == 2


def test_reload_refusal_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(name: str, provider: ProviderConfig) -> FakeAuth:
        if name == "vertex":
            raise ConfigError("no region")
        return FakeAuth(name)

    _, built = _install(monkeypatch, factory)
    state: ProxyState = create_app(Config()).state.proxy
    with pytest.raises(ConfigError, match="no region"):
        state.apply_config(_config(bedrock=_identity(BEDROCK), vertex=_identity(VERTEX)))
    assert state.upstream_auth == {}
    assert state.config.providers["bedrock"].auth == "passthrough"
    # The one built before the refusal was closed, never leaked.
    assert [auth.closed for auth in built] == [1]


def test_reload_router_refusal_closes_fresh_authorizers(monkeypatch: pytest.MonkeyPatch) -> None:
    router = FakeRouter(reconfigure_error=ConfigError("bad routing"))
    reg, _ = install(monkeypatch, router)
    _, built = _install(monkeypatch, registry=reg)
    state: ProxyState = create_app(routed_config()).state.proxy
    with pytest.raises(ConfigError, match="bad routing"):
        state.apply_config(
            routed_config(providers={**Config().providers, "azure": _identity(AZURE)})
        )
    assert state.upstream_auth == {}
    assert [auth.closed for auth in built] == [1]


def test_validate_config_probes_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    _, built = _install(monkeypatch)
    state: ProxyState = create_app(Config()).state.proxy
    state.validate_config(_config(azure=_identity(AZURE)))
    assert [auth.closed for auth in built] == [1]
    assert state.upstream_auth == {}  # a dry run swaps nothing
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        state.validate_config(_config(azure=_identity(AZURE)))


async def test_shutdown_closes_authorizers_and_tolerates_close_faults(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class BadClose(FakeAuth):
        def close(self) -> None:
            raise OSError("boom")

    def factory(name: str, provider: ProviderConfig) -> FakeAuth:
        return BadClose(name) if name == "azure" else FakeAuth(name)

    _, built = _install(monkeypatch, factory)
    app = create_app(_config(azure=_identity(AZURE), bedrock=_identity(BEDROCK)))
    with caplog.at_level("WARNING", logger="llm_redact"):
        async with app.router.lifespan_context(app):
            pass
    assert [auth.closed for auth in built if not isinstance(auth, BadClose)] == [1]
    assert "upstream auth for azure failed to close (OSError)" in caplog.text


# --- surfaces ---------------------------------------------------------------------------


async def test_status_reports_modes_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch)
    app = create_app(_config(bedrock=_identity(BEDROCK, region="us-west-2")))
    async with _client(app) as client:
        status = (await client.get("/__llm-redact/status")).json()
    assert status["providers_auth"]["bedrock"] == "identity"
    assert status["providers_auth"]["openai"] == "passthrough"
    assert "us-west-2" not in json.dumps(status["providers_auth"])


def test_status_posture_names_identity_providers(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.cli import _print_posture

    _print_posture(
        {
            "warnings_total": {},
            "detection": {},
            "audit": {},
            "providers_auth": {"bedrock": "identity", "azure": "identity", "openai": "passthrough"},
        }
    )
    out = capsys.readouterr().out
    assert "proxy holds cloud credentials for: azure, bedrock" in out
    _print_posture({"warnings_total": {}, "detection": {}, "audit": {}, "providers_auth": {}})
    assert "all traffic redacted" in capsys.readouterr().out


def _doctor_rows(config: Config) -> list[dict[str, str]]:
    from llm_redact.doctor_cli import _check_upstream_auth, _Report

    report = _Report(json_mode=True)
    _check_upstream_auth(report, config)
    return report.rows


def test_doctor_silent_without_identity() -> None:
    assert _doctor_rows(Config()) == []


def test_doctor_fails_on_the_free_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    rows = _doctor_rows(_config(vertex=_identity(VERTEX)))
    assert [row["level"] for row in rows] == ["FAIL"]
    assert "llm-redact-pro" in rows[0]["message"]


def test_doctor_pass_and_non_loopback_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    _, built = _install(monkeypatch)
    loopback = _config(vertex=_identity(VERTEX))
    rows = _doctor_rows(loopback)
    assert [row["level"] for row in rows] == ["PASS"]
    assert "[providers.vertex]" in rows[0]["message"]
    assert built[0].calls == [] and built[0].closed == 1  # built, closed, never used

    exposed = dataclasses.replace(loopback, host="0.0.0.0")
    rows = _doctor_rows(exposed)
    assert [row["level"] for row in rows] == ["PASS", "WARN"]
    assert "any client that reaches the proxy spends its cloud identity" in rows[1]["message"]

    @dataclasses.dataclass(frozen=True)
    class AuthSection:
        require: bool = False
        broker: bool = False

    gated = dataclasses.replace(exposed, extensions={"auth": AuthSection(require=True)})
    assert [row["level"] for row in _doctor_rows(gated)] == ["PASS"]
    brokered = dataclasses.replace(exposed, extensions={"auth": AuthSection(broker=True)})
    assert [row["level"] for row in _doctor_rows(brokered)] == ["PASS"]
    open_gate = dataclasses.replace(exposed, extensions={"auth": AuthSection()})
    assert [row["level"] for row in _doctor_rows(open_gate)] == ["PASS", "WARN"]


def test_doctor_passes_when_factory_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, lambda name, provider: None)
    rows = _doctor_rows(_config(azure=_identity(AZURE)))
    assert [row["level"] for row in rows] == ["PASS"]


def test_identity_exposure_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_redact.proxy import identity_exposure_warning

    _install(monkeypatch)
    exposed: ProxyState = create_app(_config(vertex=_identity(VERTEX))).state.proxy
    message = identity_exposure_warning(exposed, "0.0.0.0")
    assert message is not None and "vertex" in message and "no access gate" in message
    assert identity_exposure_warning(exposed, "127.0.0.1") is None
    plain: ProxyState = create_app(_config()).state.proxy
    assert identity_exposure_warning(plain, "0.0.0.0") is None
    exposed.access_gate = object()  # type: ignore[assignment]
    assert identity_exposure_warning(exposed, "0.0.0.0") is None


def test_serve_check_prints_the_exposure_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.cli import main

    _install(monkeypatch)
    config = tmp_path / "config.toml"
    config.write_text(
        '[providers.vertex]\nupstream_base_url = "https://us-east5-aiplatform.googleapis.com"\n'
        'auth = "identity"\n'
    )
    monkeypatch.setenv("LLM_REDACT_INSECURE_BIND", "1")
    monkeypatch.setenv("LLM_REDACT_HOST", "0.0.0.0")
    with pytest.raises(SystemExit) as exit_info:
        main(["serve", "--check", "--config", str(config)])
    assert exit_info.value.code == 0
    assert "WARN: providers vertex use the proxy's own cloud identity" in capsys.readouterr().err
