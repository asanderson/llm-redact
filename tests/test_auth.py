"""The client-authentication chain (auth.py): every credential channel is
scrubbed from the scope before anything else reads it, the first presented
credential decides, and the proxy's named-user behavior is unchanged.

The named-user registry itself is llm-redact-pro; these tests stand in a
fake store on a bare Registry so the chain is exercised keyless.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from starlette.requests import Request
from starlette.websockets import WebSocket

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact.auth import (
    USER_KEY_HEADER,
    AuthChain,
    AuthResult,
    Credential,
    Identity,
    Surface,
    UserKeyAuthenticator,
)
from llm_redact.config import Config, ProviderConfig
from llm_redact.proxy import _extract_user_key, create_app
from llm_redact.registry import Registry
from llm_redact.users import UserRow

UPSTREAM = "https://upstream.test"
KEYS = {"lrk_ada": "ada", "lrk_grace": "grace"}


class FakeUsersStore:
    """Two verified users; lookup_key is all the request path needs."""

    def __init__(self, keys: dict[str, str]) -> None:
        self.keys = keys

    def lookup_key(self, presented_key: str) -> str | None:
        return self.keys.get(presented_key)

    def verified_count(self) -> int:
        return len(self.keys)

    def active_count(self) -> int:
        return len(self.keys)

    def list_users(self) -> list[UserRow]:
        return []

    def invite(self, name: str, email: str, *, max_users: int | None) -> str:
        raise NotImplementedError

    def revoke(self, email: str, *, purge: bool = False) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def _scope(path: str, headers: list[tuple[bytes, bytes]] | None = None, **extra: Any) -> dict:
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "headers": headers or [],
        "query_string": b"",
    }
    scope.update(extra)
    return scope


def _chain(keys: dict[str, str] | None = KEYS) -> AuthChain:
    store = FakeUsersStore(keys) if keys is not None else None
    return AuthChain((UserKeyAuthenticator(lambda: store),))


# --- the user_key authenticator ---------------------------------------------


def test_path_key_is_extracted_and_scrubbed() -> None:
    scope = _scope("/u/lrk_ada/v1/messages")
    result = _chain().authenticate(Request(scope), "http")
    assert result.identity == Identity(subject="ada", method="user_key")
    assert result.presented == Credential("user_key", "path", "lrk_ada")
    assert scope["path"] == "/v1/messages"
    assert scope["raw_path"] == b"/v1/messages"


def test_header_key_is_extracted_and_scrubbed() -> None:
    scope = _scope("/v1/messages", [(b"X-LLM-Redact-User", b" lrk_grace "), (b"a", b"b")])
    result = _chain().authenticate(Request(scope), "http")
    assert result.identity == Identity(subject="grace", method="user_key")
    assert result.presented is not None and result.presented.channel == "header"
    assert scope["headers"] == [(b"a", b"b")]


def test_path_wins_and_both_channels_are_scrubbed() -> None:
    scope = _scope("/u/lrk_ada/v1/messages", [(USER_KEY_HEADER.encode(), b"lrk_grace")])
    result = _chain().authenticate(Request(scope), "http")
    assert result.identity is not None and result.identity.subject == "ada"
    assert scope["headers"] == []
    assert scope["path"] == "/v1/messages"


def test_percent_encoded_key_is_scrubbed_from_raw_path() -> None:
    # Forwarding builds the upstream URL from raw_path; an encoded key
    # segment must not survive there (the segment strip is encoding-agnostic).
    scope = _scope("/u/lrk_ada/v1/messages", raw_path=b"/u/lrk_%61da/v1/messages")
    assert _chain().authenticate(Request(scope), "http").identity is not None
    assert scope["raw_path"] == b"/v1/messages"


def test_encoded_prefix_rebuilds_raw_path_from_scrubbed_path() -> None:
    scope = _scope("/u/lrk_ada/v1/messages", raw_path=b"/%75/lrk_ada/v1/messages")
    _chain().authenticate(Request(scope), "http")
    assert scope["raw_path"] == b"/v1/messages"


def test_scrub_preserves_encoded_remainder() -> None:
    scope = _scope(
        "/u/lrk_ada/model/arn:aws:bedrock/invoke",
        raw_path=b"/u/lrk_ada/model/arn%3Aaws%3Abedrock/invoke",
    )
    _chain().authenticate(Request(scope), "http")
    assert scope["raw_path"] == b"/model/arn%3Aaws%3Abedrock/invoke"


def test_bare_key_path_scrubs_to_root() -> None:
    scope = _scope("/u/lrk_ada")
    _chain().authenticate(Request(scope), "http")
    assert scope["path"] == "/"
    assert scope["raw_path"] == b"/"


def test_scope_without_raw_path() -> None:
    scope = _scope("/u/lrk_ada/v1/messages")
    del scope["raw_path"]
    assert _chain().authenticate(Request(scope), "http").identity is not None
    assert "raw_path" not in scope


@pytest.mark.parametrize("path", ["/u/", "/u//v1/messages", "/v1/u/lrk_ada"])
def test_empty_or_non_prefix_path_is_not_a_credential(path: str) -> None:
    scope = _scope(path)
    result = _chain().authenticate(Request(scope), "http")
    assert result == AuthResult(identity=None, presented=None)
    assert scope["path"] == path


def test_empty_header_is_scrubbed_but_not_a_credential() -> None:
    scope = _scope("/v1/messages", [(USER_KEY_HEADER.encode(), b"  ")])
    assert _chain().authenticate(Request(scope), "http").presented is None
    assert scope["headers"] == []


def test_cached_headers_and_url_are_dropped() -> None:
    # Code that read request.headers / request.url BEFORE the scrub must not
    # keep seeing the credential through Starlette's per-object caches.
    scope = _scope("/u/lrk_ada/v1/messages", [(USER_KEY_HEADER.encode(), b"lrk_ada")])
    request = Request(scope)
    assert USER_KEY_HEADER in request.headers
    assert request.url.path.startswith("/u/")
    _chain().authenticate(request, "http")
    assert USER_KEY_HEADER not in request.headers
    assert request.url.path == "/v1/messages"


def test_websocket_surface_reads_the_header_only() -> None:
    # Unchanged behavior: WS upgrades never accepted the path prefix, so a
    # /u/ path is left for the WS router (which refuses unknown paths).
    scope = _scope("/u/lrk_ada/v1/realtime", [(USER_KEY_HEADER.encode(), b"lrk_grace")])
    scope["type"] = "websocket"

    async def never() -> Any:
        raise AssertionError

    websocket = WebSocket(scope, never, never)
    result = _chain().authenticate(websocket, "websocket")
    assert result.identity is not None and result.identity.subject == "grace"
    assert scope["path"] == "/u/lrk_ada/v1/realtime"
    assert scope["headers"] == []


def test_unknown_key_is_presented_but_rejected() -> None:
    result = _chain().authenticate(Request(_scope("/u/lrk_nobody/v1/messages")), "http")
    assert result.identity is None
    assert result.rejected


def test_no_registry_means_nothing_verifies() -> None:
    scope = _scope("/u/lrk_ada/v1/messages")
    result = _chain(keys=None).authenticate(Request(scope), "http")
    assert result.rejected
    assert scope["path"] == "/v1/messages"  # still scrubbed


def test_credential_repr_never_shows_the_secret() -> None:
    assert "lrk_ada" not in repr(Credential("user_key", "path", "lrk_ada"))


def test_legacy_extract_user_key_helper() -> None:
    # llm-redact-pro's tests import this name; it keeps its contract.
    scope = _scope("/u/lrk_ada/v1/messages")
    assert _extract_user_key(Request(scope)) == "lrk_ada"
    assert _extract_user_key(Request(_scope("/v1/messages"))) is None


# --- the chain's decision rule ----------------------------------------------


class _Recording:
    """A fake method: presents `secret` when set, verifies to `subject`."""

    def __init__(self, method: str, secret: str | None, subject: str | None) -> None:
        self.method = method
        self.secret = secret
        self.subject = subject
        self.extracted = False
        self.verified = False

    def extract(self, conn: Any, surface: Surface) -> Credential | None:
        self.extracted = True
        return Credential(self.method, "fake", self.secret) if self.secret else None

    def verify(self, credential: Credential) -> Identity | None:
        self.verified = True
        return Identity(self.subject, self.method) if self.subject else None


def test_first_presented_credential_decides_and_never_falls_through() -> None:
    bad = _Recording("first", "wrong", None)
    good = _Recording("second", "right", "someone")
    chain = AuthChain((bad, good))
    result = chain.authenticate(Request(_scope("/")), "http")
    assert result.identity is None and result.rejected
    assert result.presented is not None and result.presented.method == "first"
    # Every method still extracted (= scrubbed) its channel; only the
    # decider verified.
    assert good.extracted and not good.verified
    assert chain.methods == ("first", "second")


def test_methods_without_a_credential_are_skipped() -> None:
    absent = _Recording("first", None, None)
    present = _Recording("second", "right", "someone")
    result = AuthChain((absent, present)).authenticate(Request(_scope("/")), "http")
    assert result.identity == Identity("someone", "second")
    assert not absent.verified


# --- through the proxy -------------------------------------------------------


@pytest.fixture
def users_registry(monkeypatch: pytest.MonkeyPatch) -> Registry:
    reg = Registry()
    reg.resolve_license = lambda *args, **kwargs: resolved("team")
    reg.build_users_store = lambda cfg, tier: FakeUsersStore(KEYS)
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg


def _app(received: list[httpx.Request]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        return httpx.Response(
            200,
            json={"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            headers={"content-type": "application/json"},
        )

    config = Config(providers={"anthropic": ProviderConfig(upstream_base_url=UPSTREAM)})
    return create_app(config, upstream_transport=httpx.MockTransport(handler))


BODY = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hello"}],
}


async def _post(app: Any, path: str, headers: dict[str, str] | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        return await client.post(path, json=BODY, headers=headers)


async def test_proxy_attributes_and_never_forwards_the_key(users_registry: Registry) -> None:
    received: list[httpx.Request] = []
    app = _app(received)
    assert (await _post(app, "/u/lrk_ada/v1/messages")).status_code == 200
    assert (await _post(app, "/v1/messages", {USER_KEY_HEADER: "lrk_grace"})).status_code == 200
    for request in received:
        assert request.url.path == "/v1/messages"
        assert USER_KEY_HEADER not in request.headers
        assert b"lrk_" not in request.url.raw_path
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
        rows = (await client.get("/__llm-redact/recent")).json()["entries"]
    assert [row["user"] for row in rows] == ["grace", "ada"]
    assert "lrk_" not in str(rows)


@pytest.mark.parametrize(
    ("path", "headers"),
    [
        ("/v1/messages", None),
        ("/u/lrk_nobody/v1/messages", None),
        # A bad path key is not rescued by a good header key.
        ("/u/lrk_nobody/v1/messages", {USER_KEY_HEADER: "lrk_ada"}),
    ],
)
async def test_proxy_refuses_without_a_valid_key(
    users_registry: Registry, path: str, headers: dict[str, str] | None
) -> None:
    received: list[httpx.Request] = []
    response = await _post(_app(received), path, headers)
    assert response.status_code == 403
    assert "named-user key" in response.text
    assert "lrk_" not in response.text
    assert received == []
