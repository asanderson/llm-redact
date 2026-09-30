"""Shared fakes for the stored-object API suites run under a credential the
PROXY holds (tests/test_gemini_batches.py, test_gemini_files.py,
test_anthropic_files.py).

``LentRouter`` is a routing layer (the llm-redact-pro shape, scripted) that
routes EVERY request to one upstream with a plan lending the proxy's own
credential (``RoutePlan.proxy_credential = True``: an operator key), so a
route llm-redact does not recognize is refused 403 before its body is read,
and a recognized one is forwarded — redacted, restored, tracked. ``Owners``
is a session router in llm-redact-pro's named-users shape: every request
resolves to one user's session except the listings it is told about (read
in an EMPTY session, like pro's), each stored object a 2xx answer names is
recorded as that session's, and a listed item is attributed to the reader's
session only when the reader created it.

Keyless: both are registered on a bare Registry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import httpx
import pytest

from fake_router import FakePlan, FakeRouter, Hop, Stop, install, routed_config
from llm_redact.config import Config, ProviderConfig
from llm_redact.plugin_api import RouteInbound
from llm_redact.proxy import create_app

SESSION = "user:n1:main"
EMPTY = "read-empty"


class LentRouter(FakeRouter):
    """Routes every request to ``base`` (its raw path and query kept),
    lending the proxy's credential unless ``proxy_credential`` is False."""

    def __init__(self, base: str, *, proxy_credential: bool = True) -> None:
        super().__init__()
        self.base = base
        self.proxy_credential = proxy_credential

    def plan(self, inbound: RouteInbound) -> FakePlan:
        self.inbounds.append(inbound)
        url = self.base + inbound.raw_path + (f"?{inbound.query}" if inbound.query else "")
        plan = FakePlan([Hop("op", url), Stop()], proxy_credential=self.proxy_credential)
        self.plans.append(plan)
        return plan


class Owners:
    """A per-user session router recording who created what (see module)."""

    def __init__(self, listings: Iterable[str] = ()) -> None:
        self.mode = "per-user"
        self.listings = frozenset(listings)
        self.objects: list[tuple[str, str]] = []

    def resolve(self, adapter_name: str | None, method: str, path: str, body: Any) -> str:
        return EMPTY if method == "GET" and path in self.listings else SESSION

    def record_response_id(self, response_id: str, session_id: str) -> None:
        return None

    def record_object_id(self, object_id: str, session_id: str) -> bool:
        self.objects.append((object_id, session_id))
        return True

    def listing_item_session(self, object_id: str) -> str | None:
        return SESSION if (object_id, SESSION) in self.objects else None


def lent_app(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    upstream_base: str,
    respond: Callable[[httpx.Request], httpx.Response],
    *,
    owners: Owners | None = None,
    proxy_credential: bool = True,
    **config: Any,
) -> tuple[Any, LentRouter]:
    """The real app under a routed plan lending the proxy's credential (or,
    with ``proxy_credential=False``, the client's own), ``provider``
    configured at ``upstream_base``."""
    router = LentRouter(upstream_base, proxy_credential=proxy_credential)
    reg, _ = install(monkeypatch, router)
    if owners is not None:
        reg.build_session_router = lambda _config, **_kw: owners
    providers = {**Config().providers, provider: ProviderConfig(upstream_base)}
    app = create_app(
        routed_config(providers=providers, **config),
        upstream_transport=httpx.MockTransport(respond),
    )
    return app, router


def client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")
