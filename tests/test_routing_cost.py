"""Routing a request path costs time linear in its length, whatever its shape.

Everything that decides where a request goes — attribution, the adapters'
``matches`` and the prefix mixins' tail search (``_canonical``), and the
misaddressed-path check (``proxy._misaddressed``) — runs before the proxy
answers any refusal, for any client (a web page's no-CORS ``<img>`` GET
included). A tail search that tried every tail of every tail was cubic in
the number of segments: a 5 KB target blocked the event loop for seconds,
one at uvicorn's 16 KB head limit for minutes.

Each shape below is the worst case of one step at 16 KiB (uvicorn's h11
head limit; ASGITransport has none): it must be answered in well under a
second, and — the deterministic half, which also covers a server without
that limit (httptools) — the number of adapter matches it costs must not
grow with the path's length.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

import llm_redact.registry as registry_mod
from license_fixtures import resolved
from llm_redact.config import Config, ProviderConfig
from llm_redact.plugin_api import Admission
from llm_redact.providers.openai import OpenAIAdapter
from llm_redact.providers.openai_responses import OpenAIResponsesAdapter
from llm_redact.proxy import ProxyState, create_app
from llm_redact.registry import Registry

PROXY = "http://127.0.0.1:8787"
EMAIL = "jane.doe@corp.example"
OPENAI = {"authorization": "Bearer sk-proj-FAKEFAKE"}
# A web page's request: refused by the request-origin rule, which must not
# wait for any routing work beyond what attributes the request.
EVIL_ORIGIN = {"origin": "https://evil.example"}
# uvicorn's h11 head limit (the request line and headers together); the
# in-process transport takes a path this long on its own.
TARGET_LENGTH = 16 * 1024
# Generous: every shape is answered in a few milliseconds; the cubic tail
# search took 3 s for a 5 KB target and minutes at this length.
BOUND_SECONDS = 0.5


def _fill(head: str, unit: str, tail: str = "", *, length: int = TARGET_LENGTH) -> str:
    """``head`` + as many ``unit`` repeats as fit in ``length`` + ``tail``."""
    return head + unit * max(1, (length - len(head) - len(tail)) // len(unit)) + tail


# name -> (head, repeated unit, tail): the worst shape of each routing step.
SHAPES: dict[str, tuple[str, str, str]] = {
    # The reported shape: every tail under an extra prefix starts a Gemini
    # OpenAI-compatible path, each of which searched every one of ITS tails.
    "gemini-openai-repeats": ("/x" + "/v1beta/openai" * 150, "/a", ""),
    # One prefixed path with every segment a candidate tail.
    "gemini-openai-segments": ("/v1beta/openai", "/a", ""),
    "custom-segments": ("/custom/lm", "/a", ""),
    "custom-gemini-repeats": ("/custom/lm", "/v1beta/openai", ""),
    "custom-models": ("/custom/lm", "/models", ""),
    "custom-v1-models": ("/custom/lm", "/v1/models", ""),
    "custom-conversations-lookalike": ("/custom/lm", "/conversationsx", ""),
    # Extra-prefix candidates: every tail starts an API path.
    "v1-repeats": ("/x", "/v1", ""),
    "v1-chat-repeats": ("/x", "/v1/chat", ""),
    "custom-repeats": ("/x", "/custom/lm", ""),
    "api-repeats": ("/x", "/api", ""),
    "model-repeats": ("/x", "/model/m", ""),
    # Spellings: a trailing '/' and another case double every step, and each
    # front-end normalization (\ for /, ;params, trailing whitespace and
    # dots, a second decoding, NFKC, .NET case folding) adds its own.
    "upper-trailing-slash": ("/X", "/V1BETA/OPENAI", "/"),
    "backslashes": ("/x", "%5Cv1", ""),
    "params": ("/x", "/v1;a", ""),
    "trailing-whitespace": ("/x", "/v1%20%09.", ""),
    "double-encoded": ("/x", "%255Cv1", ""),
    "percent-u": ("/x", "/%u0076%u0031", ""),
    "non-ascii": ("/x", "/%C4%B1%EF%BD%83", "/"),
    "every-normalization": ("/X%5C", "V1BETA;a/OPENAI%20%09./%C4%B1%255C", "/"),
    # A greedy slash-bearing id (Bedrock) and a path family of its own.
    "bedrock-id": ("/model", "/a", "/x"),
    "azure-family": ("/openai", "/a", ""),
    "vertex-family": ("/v1/projects", "/a", ""),
    "long-segment": ("/v1beta/openai/", "a", ""),
}


def _config() -> Config:
    providers = dict(Config().providers)
    providers["azure"] = ProviderConfig("https://res.openai.azure.com")
    providers["bedrock"] = ProviderConfig("https://bedrock-runtime.us-east-1.amazonaws.com")
    providers["vertex"] = ProviderConfig("https://us-central1-aiplatform.googleapis.com")
    providers["custom:lm"] = ProviderConfig("http://lm.local")
    return Config(providers=providers)


def _app() -> Any:
    return create_app(
        _config(), upstream_transport=httpx.MockTransport(lambda _: httpx.Response(404))
    )


async def _send(app: Any, method: str, path: str, headers: dict[str, str]) -> httpx.Response:
    content = b'{"input": "mail %s"}' % EMAIL.encode() if method == "POST" else None
    request_headers = {**headers, "content-type": "application/json"} if content else headers
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        return await client.request(method, PROXY + path, headers=request_headers, content=content)


async def _timed(app: Any, method: str, path: str, headers: dict[str, str]) -> float:
    started = time.perf_counter()
    response = await _send(app, method, path, headers)
    elapsed = time.perf_counter() - started
    assert response.status_code != 500, response.text
    return elapsed


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
@pytest.mark.parametrize("headers", [OPENAI, EVIL_ORIGIN], ids=["client", "web-page"])
async def test_every_path_shape_is_routed_in_bounded_time(
    shape: str, method: str, headers: dict[str, str]
) -> None:
    path = _fill(*SHAPES[shape])
    assert TARGET_LENGTH - 64 < len(path) <= TARGET_LENGTH
    elapsed = await _timed(_app(), method, path, headers)
    assert elapsed < BOUND_SECONDS, f"{shape} {method}: {elapsed:.2f} s"


@pytest.fixture
def match_counts(monkeypatch: pytest.MonkeyPatch) -> Iterator[Counter[str]]:
    """Counts ProxyState.route calls and the wrapped OpenAI matchers the
    prefix mixins' tail search runs (``super().matches`` resolves to the
    patched attribute at call time)."""
    counts: Counter[str] = Counter()
    for owner, name in (
        (OpenAIAdapter, "matches"),
        (OpenAIResponsesAdapter, "matches"),
        (ProxyState, "route"),
    ):
        original = getattr(owner, name)

        def counted(*args: Any, _original: Any = original, _key: str = name, **kw: Any) -> Any:
            counts[_key] += 1
            return _original(*args, **kw)

        monkeypatch.setattr(owner, name, counted)
    yield counts


class _RefusingGate:
    """An access gate (llm-redact-pro's seam) that refuses every client."""

    def admit(self, conn: Any, surface: str) -> Admission:
        return Admission(refusal="a test key is required")

    def status(self) -> dict[str, Any]:
        return {}

    async def handle(self, request: Any, host: Any) -> Any:
        raise AssertionError("no gate path is requested")

    def close(self) -> None:
        pass


async def test_a_refused_request_is_answered_without_the_misaddressing_work(
    monkeypatch: pytest.MonkeyPatch, match_counts: Counter[str]
) -> None:
    """A web page's request and a client the access gate refuses cost the
    one match that attributes them: the misaddressing check (its spellings,
    the missing /v1, an extra prefix) runs only for an admitted request.
    Their refusal is what an unmisaddressed path would get."""
    misaddressed = "/v1/v1/chat/completions"
    response = await _send(_app(), "POST", misaddressed, EVIL_ORIGIN)
    assert response.status_code == 403
    assert match_counts["route"] == 1

    registry = Registry()
    registry.resolve_license = lambda *args, **kwargs: resolved("team")
    registry.build_access_gate = lambda config, license: _RefusingGate()
    monkeypatch.setattr(registry_mod, "_registry", registry)
    match_counts.clear()
    response = await _send(_app(), "POST", misaddressed, OPENAI)
    assert response.status_code == 403
    assert "a test key is required" in response.text
    assert match_counts["route"] == 1
    # Admitted, the same path is refused as misaddressed.
    monkeypatch.setattr(registry_mod, "_registry", Registry())
    match_counts.clear()
    response = await _send(_app(), "POST", misaddressed, OPENAI)
    assert response.status_code == 404 and "extra prefix" in response.text
    assert match_counts["route"] > 1


@pytest.mark.parametrize("shape", SHAPES)
async def test_routing_work_does_not_grow_with_the_path(
    shape: str, match_counts: Counter[str]
) -> None:
    """Deterministic: the matches a path costs are the same at 4 KB and at
    16 KiB — a bounded number of tails is ever tried."""
    app = _app()
    observed = []
    for length in (4_000, TARGET_LENGTH):
        match_counts.clear()
        for method in ("GET", "POST", "DELETE"):
            await _timed(app, method, _fill(*SHAPES[shape], length=length), OPENAI)
        observed.append(dict(match_counts))
    short, long = observed
    assert long == short, f"{shape}: {short} at 4 KB, {long} at 16 KiB"
