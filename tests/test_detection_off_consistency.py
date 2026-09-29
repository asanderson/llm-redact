"""``detection = false`` turns off redaction, never the ownership check.

The session router's ownership check (llm-redact-pro named users) and
stored-object tracking read the PARSED request body. With ``[providers.NAME]
detection = false`` the proxy forwarded the ORIGINAL bytes, so the two
could disagree:

- a repeated JSON key parses to its LAST occurrence while a first-wins
  upstream acts on the FIRST: the router approved ``previous_response_id``
  = one of the user's own responses while the upstream continued another
  user's (or stored a completion the proxy saw as ``store: false``);
- under ``auth = "identity"`` a body the router cannot read at all (gzip,
  non-JSON bytes) reached it as None and was signed with the proxy's own
  identity — the upstream decodes what the ownership check never saw.

A repeated-key body is now always forwarded re-serialized (exactly what was
checked), and under identity the body rule applies whatever ``detection``
says: the proxy's identity only carries a body the router could inspect.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import pytest

from llm_redact.config import ProviderConfig
from test_object_access_seams import AZURE, REFUSAL, ScriptedRouter, Upstream, _app, _client
from test_upstream_auth import VERTEX, _install

DUPLICATE = (
    b'{"model": "gpt-4.1", "previous_response_id": "resp_other",'
    b' "previous_response_id": "resp_mine", "input": "hi"}'
)


class RefusesOther(ScriptedRouter):
    """Refuses any request citing another user's response."""

    def object_access_refusal(
        self, adapter_name: str | None, method: str, path: str, body: Any, *, identity: bool
    ) -> Any:
        super().object_access_refusal(adapter_name, method, path, body, identity=identity)
        cited = body.get("previous_response_id") if isinstance(body, dict) else None
        return REFUSAL if cited == "resp_other" else None


async def test_repeated_key_forwarded_as_checked_when_detection_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = RefusesOther()
    upstream = Upstream({"id": "resp_2", "object": "response"})
    app = _app(
        monkeypatch,
        router,
        upstream,
        providers={"openai": ProviderConfig("https://up.test", detection=False)},
    )
    async with _client(app) as client:
        plain = await client.post("/v1/responses", json={"previous_response_id": "resp_other"})
        response = await client.post(
            "/v1/responses", content=DUPLICATE, headers={"content-type": "application/json"}
        )
    assert plain.status_code == 403
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert b"resp_other" not in sent.content
    assert json.loads(sent.content) == router.checks[-1][3]  # exactly what the router saw


async def test_repeated_key_signed_as_checked_under_identity_with_detection_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = RefusesOther()
    upstream = Upstream({"id": "resp_2", "object": "response"})
    registry, built = _install(monkeypatch)
    app = _app(
        monkeypatch,
        router,
        upstream,
        registry=registry,
        providers={"azure": ProviderConfig(AZURE, auth="identity", detection=False)},
    )
    async with _client(app) as client:
        response = await client.post(
            "/openai/v1/responses", content=DUPLICATE, headers={"content-type": "application/json"}
        )
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert b"resp_other" not in sent.content
    assert built[0].calls[0][3] == sent.content


@pytest.mark.parametrize(
    ("body", "headers", "status"),
    [
        (
            gzip.compress(json.dumps({"cachedContent": "cachedContents/cache-a"}).encode()),
            {"content-type": "application/json", "content-encoding": "gzip"},
            415,
        ),
        (
            b"\x08\x96\x01 cachedContents/cache-a",
            {"content-type": "application/x-protobuf"},
            400,
        ),
    ],
    ids=["gzip", "non-json"],
)
async def test_identity_never_signs_a_body_the_ownership_check_could_not_read(
    monkeypatch: pytest.MonkeyPatch, body: bytes, headers: dict[str, str], status: int
) -> None:
    router = ScriptedRouter()
    upstream = Upstream({})
    registry, built = _install(monkeypatch)
    app = _app(
        monkeypatch,
        router,
        upstream,
        registry=registry,
        providers={"vertex": ProviderConfig(VERTEX, auth="identity", detection=False)},
    )
    async with _client(app) as client:
        response = await client.post(
            "/v1/projects/p/locations/us-east5/publishers/google/models/g:generateContent",
            content=body,
            headers=headers,
        )
    assert response.status_code == status
    assert "proxy's own identity" in response.json()["error"]["message"]
    assert upstream.requests == [] and built[0].calls == []


async def test_identity_with_detection_off_still_forwards_json_unredacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ScriptedRouter()
    upstream = Upstream({})
    registry, built = _install(monkeypatch)
    app = _app(
        monkeypatch,
        router,
        upstream,
        registry=registry,
        providers={"vertex": ProviderConfig(VERTEX, auth="identity", detection=False)},
    )
    body = json.dumps({"contents": [{"parts": [{"text": "mail jane.doe@corp.example"}]}]}).encode()
    async with _client(app) as client:
        response = await client.post(
            "/v1/projects/p/locations/us-east5/publishers/google/models/g:generateContent",
            content=body,
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 200
    (sent,) = upstream.requests
    assert sent.content == body and built[0].calls[0][3] == body  # unredacted, byte-identical
    assert router.checks[-1][3] == json.loads(body)
