"""An upstream redirect is never relayed where a following client would leak (C-R2-02).

A redirect-following client (httpx ``follow_redirects=True``, fetch,
reqwest) repeats its ORIGINAL request — the unredacted body, and every
credential header httpx/fetch keep across origins (``x-api-key``,
``api-key``, ``x-llm-redact-user``) — at whatever the ``Location`` names.
So the proxy answers a 3xx carrying a ``Location`` with a recorded,
provider-shaped 502 naming the status only (never the Location: it can hold
a presigned credential) on every route it redacts, routes or signs, and on
any request that presented a proxy credential. The client below FOLLOWS
redirects, and every host but the proxy is a capture: nothing may reach it.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import create_app
from llm_redact.registry import Registry

EMAIL = "jane.doe@corp.example"
ELSEWHERE = "https://elsewhere.example/landing?X-Amz-Signature=presigned"


def _chat() -> dict[str, Any]:
    return {"model": "m", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]}


class Upstream:
    """Answers every request with ``status`` + a Location; records requests."""

    def __init__(
        self,
        status: int = 307,
        *,
        location: str | None = ELSEWHERE,
        content_type: str = "application/json",
    ) -> None:
        self.status = status
        self.location = location
        self.content_type = content_type
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        headers = {"content-type": self.content_type}
        if self.location is not None:
            headers["location"] = self.location
        return httpx.Response(self.status, headers=headers, content=b"{}")


class Following:
    """A redirect-following tool: the proxy app for 127.0.0.1, a capture
    for every other host (where a relayed redirect would send it)."""

    def __init__(self, app: Any) -> None:
        self.proxy = httpx.ASGITransport(app=app)
        self.elsewhere: list[httpx.Request] = []

    async def __aenter__(self) -> httpx.AsyncClient:
        async def capture(request: httpx.Request) -> httpx.Response:
            self.elsewhere.append(request)
            return httpx.Response(200, json={"captured": True})

        mock = httpx.MockTransport(capture)
        proxy = self.proxy

        class Split(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                if request.url.host == "127.0.0.1":
                    return await proxy.handle_async_request(request)
                return await mock.handle_async_request(request)

        self.client = httpx.AsyncClient(
            transport=Split(), base_url="http://127.0.0.1", follow_redirects=True
        )
        return self.client

    async def __aexit__(self, *exc: object) -> None:
        await self.client.aclose()


def _providers(**extra: ProviderConfig) -> dict[str, ProviderConfig]:
    providers = dict(Config().providers)
    providers.update(extra)
    return providers


def _assert_refused(response: httpx.Response, following: Following, app: Any, status: int) -> None:
    assert response.status_code == 502, response.text
    assert "location" not in response.headers
    assert f"redirect ({status})" in response.text
    assert "elsewhere" not in response.text and "presigned" not in response.text
    assert following.elsewhere == []
    row = app.state.proxy.recent[0]
    assert row["status"] == 502


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_a_redirect_on_a_redacted_route_is_never_relayed(status: int) -> None:
    upstream = Upstream(status)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    following = Following(app)
    async with following as client:
        response = await client.post(
            "/v1/messages",
            json=_chat(),
            headers={"x-api-key": "client-key", "anthropic-version": "2023-06-01"},
        )
    _assert_refused(response, following, app, status)
    (sent,) = upstream.requests
    assert EMAIL not in sent.content.decode()  # the one hop the proxy made was redacted
    assert app.state.proxy.upstream_errors["anthropic"] == 1


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/v1/chat/completions", "text/event-stream"),
        ("/api/chat", "application/x-ndjson"),
        ("/v1/audio/speech", "audio/mpeg"),  # the buffered non-JSON branch
    ],
)
async def test_the_streaming_and_raw_branches_refuse_it_too(path: str, content_type: str) -> None:
    upstream = Upstream(308, content_type=content_type)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    following = Following(app)
    async with following as client:
        response = await client.post(
            path, json={**_chat(), "input": EMAIL}, headers={"authorization": "Bearer sk-x"}
        )
    _assert_refused(response, following, app, 308)


async def test_a_body_less_read_on_a_recognized_route_may_follow_a_download_redirect() -> None:
    """Nothing to leak: no body to repeat, the client's own credential, an
    unrouted first-party provider — a provider's CDN redirect for a download
    still works (its bytes simply bypass rehydration, placeholders intact)."""
    upstream = Upstream(302, location="https://cdn.example/video.mp4?sig=abc")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get(
            "/v1/videos/video_1/content", headers={"authorization": "Bearer sk-x"}
        )
        refused = await client.get(
            "/v1/videos/video_1/content",
            headers={"authorization": "Bearer sk-x", "x-llm-redact-user": "lrk_demo"},
        )
    assert response.status_code == 302
    assert response.headers["location"] == "https://cdn.example/video.mp4?sig=abc"
    assert refused.status_code == 502 and "location" not in refused.headers


async def test_the_bedrock_eventstream_branch_refuses_it() -> None:
    upstream = Upstream(307, content_type="application/vnd.amazon.eventstream")
    config = Config(
        providers=_providers(
            bedrock=ProviderConfig("https://bedrock-runtime.us-east-1.amazonaws.com")
        )
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    following = Following(app)
    async with following as client:
        response = await client.post(
            "/model/m/converse-stream",
            json={"messages": [{"role": "user", "content": [{"text": EMAIL}]}]},
            headers={"authorization": "Bearer ABSK"},
        )
    _assert_refused(response, following, app, 307)
    assert "message" in response.json()  # Bedrock's error shape


async def test_a_relative_location_from_a_custom_upstream_is_refused() -> None:
    """The client would resolve it against the proxy, dropping the
    /custom/NAME prefix: /v1/chat/completions/ once reached api.openai.com
    unredacted with the custom upstream's key."""
    upstream = Upstream(307, location="/v1/chat/completions")
    config = Config(providers=_providers(**{"custom:vllm": ProviderConfig("http://llm.corp")}))
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    following = Following(app)
    async with following as client:
        response = await client.post(
            "/custom/vllm/v1/some/unknown/route",  # pass-through under the prefix
            json=_chat(),
            headers={"authorization": "Bearer vllm-key"},
        )
    _assert_refused(response, following, app, 307)
    assert len(upstream.requests) == 1  # the proxy's own hop only


async def test_a_request_presenting_a_proxy_credential_is_refused_on_pass_through() -> None:
    """Without a gate the core drops x-llm-redact-* before forwarding — but a
    following client would re-send it to the Location host."""
    upstream = Upstream(302, location="https://elsewhere.example/models")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    following = Following(app)
    async with following as client:
        response = await client.get(
            "/v1/assistants",  # pass-through
            headers={"authorization": "Bearer sk-x", "x-llm-redact-user": "lrk_demo_user_key"},
        )
    _assert_refused(response, following, app, 302)
    assert "x-llm-redact-user" not in upstream.requests[0].headers
    assert app.state.proxy.upstream_errors["passthrough"] == 1


async def test_a_first_party_pass_through_redirect_with_the_clients_own_key_is_relayed() -> None:
    """The documented exception: nothing the proxy protects is at stake —
    the body went to that provider verbatim anyway, the credential is the
    client's own, and the Location is the provider's (a download CDN)."""
    upstream = Upstream(302, location="https://cdn.example/file?sig=abc")
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.get(
            "/download/v1beta/files/abc:download?alt=media",
            headers={"x-goog-api-key": "AIzaFAKE"},
        )
    assert response.status_code == 302
    assert response.headers["location"] == "https://cdn.example/file?sig=abc"


@pytest.mark.parametrize(
    ("status", "location"),
    [(304, ELSEWHERE), (300, None), (302, None)],
)
async def test_not_modified_and_location_less_answers_pass_unchanged(
    status: int, location: str | None
) -> None:
    upstream = Upstream(status, location=location)
    app = create_app(Config(), upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/messages",
            json=_chat(),
            headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
        )
    assert response.status_code == status


async def test_a_routed_hop_that_redirects_is_a_fault_the_router_sees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = Upstream(307)
    router = FakeRouter({"go": [Hop("a", "http://a.example/v1/messages"), Stop()]})
    install(monkeypatch, router)
    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(upstream))
    following = Following(app)
    async with following as client:
        response = await client.post(
            "/v1/messages",
            json=_chat(),
            headers={
                "x-api-key": "lrk_demo_user_key",
                "anthropic-version": "2023-06-01",
                ROUTE_HEADER: "go",
            },
        )
    assert response.status_code == 502
    assert "location" not in response.headers
    assert following.elsewhere == []
    (plan,) = router.plans
    (result,) = plan.results
    assert result.status is None and result.fault == "UpstreamRedirect"
    assert app.state.proxy.upstream_errors["a"] == 1
    assert plan.delivered is not None and plan.delivered.finished == [502]


class FakeAuth:
    async def authorize(
        self, method: str, url: str, headers: list[tuple[str, str]], body: bytes
    ) -> list[tuple[str, str]]:
        return [*headers, ("authorization", "Proxy azure")]

    def close(self) -> None:
        pass


async def test_an_identity_signed_request_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = Registry()
    reg.build_upstream_auth = lambda name, provider: FakeAuth() if name == "azure" else None
    monkeypatch.setattr(registry_mod, "_registry", reg)
    upstream = Upstream(308)
    config = Config(
        providers=_providers(azure=ProviderConfig("https://res.openai.azure.com", auth="identity"))
    )
    app = create_app(config, upstream_transport=httpx.MockTransport(upstream))
    following = Following(app)
    async with following as client:
        response = await client.post(
            "/openai/deployments/d/chat/completions?api-version=2024-10-21",
            content=json.dumps(_chat()).encode(),
            headers={"content-type": "application/json", "api-key": "client-azure-key"},
        )
    _assert_refused(response, following, app, 308)
    (sent,) = upstream.requests
    assert sent.headers["authorization"] == "Proxy azure"
