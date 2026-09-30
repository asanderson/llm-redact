"""Harness for the OpenAI stored-object API tests (fine-tuning jobs, vector
stores, containers): a routed proxy whose plan spends an OPERATOR key (a
credential the proxy holds, ``RoutePlan.proxy_credential``), a session router
recording what the core reports, and a provider fake answering per path.

Keyless: ``fake_router`` stands in for llm-redact-pro's routing layer and
``Recorder`` for its session router.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from fake_router import ROUTE_HEADER, FakeRouter, Hop, Stop, install, routed_config
from llm_redact.proxy import create_app

EMAIL = "jane.doe@corp.example"
TOKEN = "«EMAIL_001»"
OPERATOR = "http://op.example"
CLIENT = {"authorization": "Bearer sk-client"}


class Recorder:
    """A session router (static mode) with the optional stored-object
    members, recording what the core asks and reports."""

    mode = "static"

    def __init__(self) -> None:
        self.objects: list[tuple[str, str]] = []
        self.listed: list[str] = []
        self.checks: list[tuple[str | None, str, str, Any, bool]] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return "default"

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def record_object_id(self, object_id: str, session_id: str) -> bool:
        self.objects.append((object_id, session_id))
        return True

    def object_access_refusal(
        self, adapter_name: str | None, method: str, path: str, body: Any, *, identity: bool
    ) -> str | None:
        self.checks.append((adapter_name, method, path, body, identity))
        return None

    def listing_item_session(self, object_id: str) -> str | None:
        self.listed.append(object_id)
        return None  # as the listing's own session delivers it


Responder = Callable[[httpx.Request], httpx.Response]


class Provider:
    """The upstream: every request recorded, answered by ``respond``."""

    def __init__(self, respond: Responder) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def last_json(self) -> Any:
        return json.loads(self.requests[-1].content)


class Routed:
    """An app whose every request is routed to ``OPERATOR`` + its own path
    with the proxy's operator key (``proxy_credential=True``)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, respond: Responder, paths: list[str]):
        scripts = {path: [Hop("op", OPERATOR + path), Stop()] for path in paths}
        self.router = FakeRouter(
            scripts, plan_kwargs={path: {"proxy_credential": True} for path in paths}
        )
        registry, _ = install(monkeypatch, self.router)
        self.sessions = Recorder()
        registry.build_session_router = lambda config, **kw: self.sessions
        self.provider = Provider(respond)
        self.app = create_app(
            routed_config(), upstream_transport=httpx.MockTransport(self.provider)
        )

    async def send(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        if body is not None:
            content = json.dumps(body).encode()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://127.0.0.1"
        ) as client:
            return await client.request(
                method,
                path,
                content=content,
                headers={**CLIENT, **(headers or {}), ROUTE_HEADER: path},
            )
