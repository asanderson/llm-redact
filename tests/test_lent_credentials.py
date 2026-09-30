"""A credential the PROXY holds is lent only to routes llm-redact recognizes (C-R2-04).

Under ``auth = "identity"`` an unrecognized route was already a recorded 403.
A routed plan that spends an operator key (``credential = "env:…"``) or none
at all (``RoutePlan.proxy_credential``, absent = True: fail closed) is the
same principal-lending shape — the one ``[auth] broker`` requires — so an
adapter-less request under it is refused the same way: after the router's
local answer, before the body is read, the audit START row, ``begin`` and any
hop. The id-only and metadata routes tools need (model listings, batch
polls, deletes) are RECOGNIZED (redact-only, a no-op on a body-less request)
and keep working. Keyless: scripted fakes stand in for llm-redact-pro.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from llm_redact.plugin_api import LocalAnswer
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
OPERATOR = "http://op.example"


class Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"id": "x"})


def _app(
    monkeypatch: pytest.MonkeyPatch, path: str, **plan_kwargs: Any
) -> tuple[Any, FakeRouter, Upstream]:
    router = FakeRouter(
        {"r": [Hop("op", OPERATOR + path), Stop()]},
        plan_kwargs={"r": plan_kwargs},
        local={
            "/v1/answered-locally": LocalAnswer(200, {"data": []}, provider="routing", reason="t")
        },
    )
    install(monkeypatch, router)
    upstream = Upstream()
    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(upstream))
    return app, router, upstream


async def _send(app: Any, method: str, path: str, headers: dict[str, str], body: Any = None) -> Any:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        content = None if body is None else json.dumps(body).encode()
        return await client.request(
            method, path, headers={**headers, ROUTE_HEADER: "r"}, content=content
        )


UNRECOGNIZED = [
    ("POST", "/v1/threads/thread_abc/messages", {"authorization": "Bearer sk-client"}),
    ("POST", "/v1/fine_tuning/jobs", {"authorization": "Bearer sk-client"}),
    ("POST", "/v1/moderations", {"authorization": "Bearer sk-client"}),
    ("GET", "/v1/organization/admin_api_keys", {"authorization": "Bearer sk-client"}),
    ("POST", "/v1/files", {"anthropic-version": "2023-06-01", "x-api-key": "lrk_user"}),
    ("POST", "/api/create", {}),
]


@pytest.mark.parametrize("lends", [{}, {"proxy_credential": True}], ids=["member-absent", "true"])
@pytest.mark.parametrize(("method", "path", "headers"), UNRECOGNIZED)
async def test_an_unrecognized_route_is_never_sent_with_a_proxy_held_credential(
    method: str,
    path: str,
    headers: dict[str, str],
    lends: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, router, upstream = _app(monkeypatch, path, **lends)
    body = {"role": "user", "content": f"mail {EMAIL}"} if method == "POST" else None
    response = await _send(app, method, path, headers, body)
    assert response.status_code == 403, response.text
    message = response.json()["error"]
    assert "credential the proxy holds" in message and EMAIL not in message
    assert upstream.requests == []
    (plan,) = router.plans
    assert plan.begun == []  # never begun: no audit START row, no hop
    (row,) = app.state.proxy.recent
    assert row["status"] == 403 and row["detections"] == {}


async def test_the_clients_own_credential_still_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "/v1/threads/thread_abc/messages"
    app, router, upstream = _app(monkeypatch, path, proxy_credential=False)
    raw = {"role": "user", "content": f"mail {EMAIL}"}
    response = await _send(app, "POST", path, {"authorization": "Bearer sk-client"}, raw)
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert json.loads(sent.content) == raw  # pass-through: forwarded verbatim
    (plan,) = router.plans
    assert plan.begun and plan.begun[0][0] == json.dumps(raw).encode()


async def test_the_routers_local_answer_precedes_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, router, upstream = _app(monkeypatch, "/v1/answered-locally")
    response = await _send(app, "GET", "/v1/answered-locally", {"authorization": "Bearer sk-x"})
    assert response.status_code == 200 and response.json() == {"data": []}
    assert router.plans == [] and upstream.requests == []


async def test_a_no_route_refusal_for_an_unrecognized_route_precedes_the_body_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = FakeRouter()
    install(monkeypatch, router)
    upstream = Upstream()
    app = create_app(routed_config(), upstream_transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        response = await client.post(
            "/v1/fine_tuning/jobs",
            content=b"{}",
            headers={"authorization": "Bearer sk-x", ROUTE_HEADER: "refuse"},
        )
    assert response.status_code == 502 and upstream.requests == []


@pytest.mark.parametrize(
    ("method", "path", "headers"),
    [
        ("GET", "/v1/models", {"authorization": "Bearer sk-client"}),
        ("GET", "/v1/models/gpt-x", {"authorization": "Bearer sk-client"}),
        ("GET", "/v1/models", {"anthropic-version": "2023-06-01", "x-api-key": "lrk_user"}),
        (
            "GET",
            "/v1/messages/batches/msgbatch_1",
            {"anthropic-version": "2023-06-01", "x-api-key": "lrk_user"},
        ),
        (
            "POST",
            "/v1/messages/batches/msgbatch_1/cancel",
            {"anthropic-version": "2023-06-01", "x-api-key": "lrk_user"},
        ),
        ("DELETE", "/v1/files/file-1", {"authorization": "Bearer sk-client"}),
        ("DELETE", "/v1/responses/resp_1", {"authorization": "Bearer sk-client"}),
        ("DELETE", "/v1/conversations/conv_1", {"authorization": "Bearer sk-client"}),
        ("GET", "/v1/videos/video_1/content", {"authorization": "Bearer sk-client"}),
        ("GET", "/api/tags", {}),
        ("POST", "/api/show", {}),
        ("GET", "/v1beta/models", {"x-goog-api-key": "AIza"}),
        ("GET", "/v1beta/batches/b1", {"x-goog-api-key": "AIza"}),
        ("GET", "/v1beta/batches", {"x-goog-api-key": "AIza"}),
        ("DELETE", "/v1beta/batches/b1", {"x-goog-api-key": "AIza"}),
    ],
)
async def test_recognized_metadata_routes_keep_working_under_a_proxy_held_credential(
    method: str, path: str, headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, router, upstream = _app(monkeypatch, path)
    body = {"model": "llama3"} if path == "/api/show" else None
    response = await _send(app, method, path, headers, body)
    assert response.status_code == 200, response.text
    assert len(upstream.requests) == 1
    (inbound,) = router.inbounds
    assert inbound.adapter_name is not None
