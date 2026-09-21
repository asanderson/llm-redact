"""The routed request path beyond the §8 walkthroughs (docs/routing.md):
no-route 502, model restore across SSE/NDJSON chunk boundaries, the
transport-fault chain, expose_models, the editor's preserved sections,
recent rows, log lines, reissue/debug headers, unrouted providers, reload
hot-swaps, spend-store selection, metrics, the Gemini path model, and the
hop-loop bounds. Shares the walkthrough harness and fake upstream."""

import dataclasses
import json
import logging
import tomllib
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.proxy as proxy_module
from llm_redact.config import Config, ConfigError, ProviderConfig, parse_config
from llm_redact.config_write import emit_config_toml
from llm_redact.proxy import (
    _FILE_PRESERVED_KEYS,
    CSRF_HEADER,
    ProxyState,
    RouteDelivery,
    _restore_model_payload,
    _restore_model_text,
    _rewrite_gemini_path,
    create_app,
    gemini_path_model,
)
from llm_redact.routing import HOPS_HEADER, REISSUE_HEADER, UPSTREAM_HEADER, UpstreamConfig
from llm_redact.spend import InMemorySpendStore, SqliteSpendStore
from test_routing_walkthroughs import (
    ANTHROPIC_KEY,
    EMAIL,
    GATEWAY_HEADERS,
    OAUTH_HEADERS,
    Harness,
    Scenario,
    fake_upstream,
    messages_body,
    route_of,
    set_env_keys,
    spec_config,
    spec_toml,
)


def _raw(vault_path: Path) -> dict[str, Any]:
    return tomllib.loads(spec_toml(vault_path))


def _config(raw: dict[str, Any]) -> Config:
    return parse_config(raw, "<routing-proxy>")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    set_env_keys(monkeypatch)
    return monkeypatch


# ---------------------------------------------------------------------------
# No route: neither rule nor default for the protocol → 502, never guessed.
# ---------------------------------------------------------------------------


async def test_no_route_502(env: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["default_upstream"] = {"anthropic": "ollama"}  # no openai default
    raw["routing"]["rule"] = [
        r for r in raw["routing"]["rule"] if r["match"]["protocol"] != "openai"
    ]
    harness = Harness(_config(raw))
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]},
        )
    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "message": (
                "llm-redact routing: no rule matched and no default_upstream for protocol openai"
            ),
            "type": "invalid_request_error",
            "code": None,
        }
    }
    assert not harness.scenarios  # nothing was contacted
    assert route_of(harness) == {
        "rule": None,
        "upstream": None,
        "hops": 0,
        "auth": "none",
        "class": "no_route",
        "reissue": "no",
    }
    assert "POST /v1/chat/completions -> 502 rule=- upstream=- hops=0 auth=none" in caplog.text
    assert "class=no_route reissue=no" in caplog.text
    await harness.aclose()


# ---------------------------------------------------------------------------
# Model restore across SSE chunk boundaries (a chunked transport).
# ---------------------------------------------------------------------------


class _ChunkedUpstream(httpx.AsyncBaseTransport):
    """Streams exact byte chunks — ASGITransport would coalesce them."""

    def __init__(self, chunks: list[bytes], content_type: str) -> None:
        self._chunks = chunks
        self._content_type = content_type
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)

        async def parts() -> Any:
            for chunk in self._chunks:
                yield chunk

        return httpx.Response(
            200, headers={"content-type": self._content_type}, content=parts(), request=request
        )


def _sse_bytes(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    return b"".join(
        f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()
        for name, payload in events
    )


@pytest.mark.parametrize("split_at", [5, 60, 95, 140, 260])
async def test_model_restored_across_sse_chunk_boundaries(
    env: Any, tmp_path: Path, split_at: int
) -> None:
    stream = _sse_bytes(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {"model": "muse-64k", "usage": {"input_tokens": 7}},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hi «EMA"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "IL_001» ok"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "usage": {"output_tokens": 3}}),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    upstream = _ChunkedUpstream([stream[:split_at], stream[split_at:]], "text/event-stream")
    harness = Harness(spec_config(tmp_path / "v.db"), upstream_transport=upstream)
    headers = {**OAUTH_HEADERS, "x-claude-code-agent-id": "Explore"}
    response = await harness.client.post(
        "/v1/messages", json=messages_body(stream=True), headers=headers
    )
    assert response.status_code == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0]["message"]["model"] == "claude-sonnet-5"
    assert "muse-64k" not in response.text
    # Rehydration across the split still holds alongside the restore.
    assert (
        "".join(e["delta"]["text"] for e in events if e.get("type") == "content_block_delta")
        == f"hi {EMAIL} ok"
    )
    # The request carried the rewritten model; usage was tracked from the
    # stream (message_start input + message_delta output) and recorded
    # against the zero-cost upstream at hop 1.
    sent = json.loads(upstream.requests[0].content)
    assert sent["model"] == "muse-64k"
    totals = harness.state.budget_ledger.totals("ollama")
    assert (totals.in_tokens, totals.out_tokens, totals.rows) == (7, 3, 1)
    await harness.aclose()


async def test_model_restored_on_ndjson_stream(env: Any, tmp_path: Path) -> None:
    # Ollama native protocol: the rule rewrites the model; every NDJSON line's
    # `model` comes back as the original, split across network chunks.
    raw = _raw(tmp_path / "v.db")
    raw["upstreams"]["ollama_native"] = {
        "protocol": "ollama",
        "base_url": "http://ollama-native",
        "credential": "none",
        "cost": "zero",
    }
    raw["routing"]["rule"].append(
        {
            "id": "native",
            "match": {"protocol": "ollama", "model": "big-*"},
            "upstream": "ollama_native",
            "model_rewrite": "small",
        }
    )
    lines = b"".join(
        json.dumps(
            {"model": "small", "message": {"role": "assistant", "content": c}, "done": False}
        ).encode()
        + b"\n"
        for c in ("saw «EMA", "IL_001» end")
    )
    lines += (
        b'{"model":"small","message":{"role":"assistant","content":""},"done":true,'
        b'"prompt_eval_count":4,"eval_count":2}\n'
    )
    upstream = _ChunkedUpstream([lines[:37], lines[37:]], "application/x-ndjson")
    harness = Harness(_config(raw), upstream_transport=upstream)
    response = await harness.client.post(
        "/api/chat",
        json={"model": "big-model", "messages": [{"role": "user", "content": f"mail {EMAIL}"}]},
    )
    assert response.status_code == 200
    rows = [json.loads(line) for line in response.content.splitlines() if line.strip()]
    assert {row["model"] for row in rows} == {"big-model"}
    assert "".join(row["message"]["content"] for row in rows) == f"saw {EMAIL} end"
    assert json.loads(upstream.requests[0].content)["model"] == "small"
    totals = harness.state.budget_ledger.totals("ollama_native")
    assert (totals.in_tokens, totals.out_tokens) == (4, 2)
    assert route_of(harness)["rule"] == "native"
    await harness.aclose()


async def test_model_restored_in_buffered_json_and_gemini_payloads() -> None:
    # The buffered path's helper: dict, Gemini's modelVersion, and the
    # buffered array form; unparsable or unchanged text stays identical.
    payload: dict[str, Any] = {"model": "x", "message": {"model": "x"}, "modelVersion": "x"}
    assert _restore_model_payload(payload, "orig", "gemini")
    assert payload == {"model": "orig", "message": {"model": "orig"}, "modelVersion": "orig"}
    array: list[Any] = [{"modelVersion": "x"}, {"modelVersion": "orig"}, "text"]
    assert _restore_model_payload(array, "orig", "gemini")
    assert array[0]["modelVersion"] == "orig"
    assert not _restore_model_payload({"modelVersion": "x"}, "orig", "anthropic")
    assert _restore_model_text("not json", "orig", "anthropic") == "not json"
    assert _restore_model_text('{"model": "orig"}', "orig", "openai") == '{"model": "orig"}'
    assert _restore_model_text('{"model": "x"}', "orig", "openai") == '{"model": "orig"}'


# ---------------------------------------------------------------------------
# Gemini: the model id lives in the path (decision 15b).
# ---------------------------------------------------------------------------


def test_gemini_path_helpers() -> None:
    assert (
        gemini_path_model("/v1beta/models/gemini-2.5-flash:generateContent") == "gemini-2.5-flash"
    )
    assert gemini_path_model("/v1/tunedModels/my-tuned:streamGenerateContent") == "my-tuned"
    assert gemini_path_model("/v1beta/cachedContents") is None
    assert (
        _rewrite_gemini_path("/v1beta/models/gemini-2.5-flash:generateContent", "gemma 3")
        == "/v1beta/models/gemma%203:generateContent"
    )
    assert _rewrite_gemini_path("/v1beta/cachedContents", "gemma") == "/v1beta/cachedContents"


async def test_gemini_rule_matches_path_model_and_rewrites_it(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["upstreams"]["gemini_key"] = {
        "protocol": "gemini",
        "base_url": "http://gemini",
        "credential": "env:GEMINI_API_KEY",
    }
    raw["routing"]["rule"].append(
        {
            "id": "gemini-native",
            "match": {"protocol": "gemini", "model": "gemini-*"},
            "upstream": "gemini_key",
            "model_rewrite": "gemini-2.5-flash-lite",
        }
    )
    harness = Harness(_config(raw), {"gemini": Scenario(echo_model=True)})
    response = await harness.client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent?key=client-key-EXAMPLE",
        json={"contents": [{"role": "user", "parts": [{"text": f"mail {EMAIL}"}]}]},
    )
    assert response.status_code == 200
    [seen] = harness.received("gemini")
    assert seen.path == "/v1beta/models/gemini-2.5-flash-lite:generateContent"
    assert "model" not in seen.body  # never added to a Gemini body
    assert seen.headers["x-goog-api-key"] == "AIza-test-not-a-real-gemini-key-EXAMPLE"
    assert "client-key-EXAMPLE" not in json.dumps(seen.headers)
    body = response.json()
    assert body["modelVersion"] == "gemini-2.5-pro"
    assert EMAIL in body["candidates"][0]["content"]["parts"][0]["text"]
    assert route_of(harness)["rule"] == "gemini-native"
    totals = harness.state.budget_ledger.totals("gemini_key")
    assert totals.rows == 1 and totals.in_tokens == 10
    await harness.aclose()


# ---------------------------------------------------------------------------
# Transport faults: classified "502", chained, counted by upstream name.
# ---------------------------------------------------------------------------


class _FaultingByHost(httpx.AsyncBaseTransport):
    """Connection refused for `hosts`; everything else reaches the fake."""

    def __init__(self, inner: httpx.AsyncBaseTransport, hosts: set[str]) -> None:
        self._inner = inner
        self._hosts = hosts

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host in self._hosts:
            raise httpx.ConnectError("refused")
        return await self._inner.handle_async_request(request)


async def test_transport_fault_falls_back_along_the_5xx_chain(
    env: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    scenarios: dict[str, Any] = {}
    fake = fake_upstream.build_app(scenarios)
    harness = Harness(
        spec_config(tmp_path / "v.db"),
        scenarios,
        upstream_transport=_FaultingByHost(httpx.ASGITransport(app=fake), {"key"}),
    )
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages", json=messages_body(), headers=GATEWAY_HEADERS
        )
    assert response.status_code == 200
    assert response.headers[UPSTREAM_HEADER] == "ollama"
    assert response.headers[HOPS_HEADER] == "2"
    assert harness.state.upstream_errors == {"anthropic_key": 1}
    assert not harness.state.routing_state.healthy("anthropic_key")
    assert "upstream anthropic_key fault (ConnectError)" in caplog.text
    assert route_of(harness)["class"] == "ok" and route_of(harness)["reissue"] == "yes"
    await harness.aclose()


async def test_transport_fault_without_chain_is_a_recorded_502(
    env: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["rule"][2]["on_status"] = {"429": ["ollama"]}  # no 5xx chain
    scenarios: dict[str, Any] = {}
    fake = fake_upstream.build_app(scenarios)
    harness = Harness(
        _config(raw),
        scenarios,
        upstream_transport=_FaultingByHost(httpx.ASGITransport(app=fake), {"key"}),
    )
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages", json=messages_body(), headers=GATEWAY_HEADERS
        )
    assert response.status_code == 502
    assert response.json()["error"] == {
        "type": "api_error",
        "message": "llm-redact: upstream request failed",
    }
    assert route_of(harness)["class"] == "transport"
    assert "-> 502 rule=anthropic-key-lane upstream=anthropic_key hops=1" in caplog.text
    assert harness.state.metrics.requests[("anthropic", "502")] == 1
    # An unlisted transport fault: the primary is not put into cooldown.
    assert harness.state.routing_state.healthy("anthropic_key")
    await harness.aclose()


async def test_buffered_read_fault_on_a_routed_response(env: Any, tmp_path: Path) -> None:
    class _DropsBody(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            async def body() -> Any:
                yield b'{"content": '
                raise httpx.ReadError("dropped")

            return httpx.Response(
                200, headers={"content-type": "application/json"}, content=body(), request=request
            )

    harness = Harness(spec_config(tmp_path / "v.db"), upstream_transport=_DropsBody())
    response = await harness.client.post(
        "/v1/messages", json=messages_body(), headers=OAUTH_HEADERS
    )
    assert response.status_code == 502
    assert route_of(harness)["class"] == "transport"
    assert harness.state.upstream_errors == {"anthropic_oauth": 1}
    await harness.aclose()


async def test_vanished_env_credential_fails_that_hop(
    env: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    harness = Harness(spec_config(tmp_path / "v.db"))
    env.delenv("ANTHROPIC_API_KEY")
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages", json=messages_body(), headers=GATEWAY_HEADERS
        )
    # The 5xx chain took over (the missing key is a "502" for chain lookup).
    assert response.status_code == 200 and response.headers[UPSTREAM_HEADER] == "ollama"
    assert "environment variable ANTHROPIC_API_KEY is unset or empty" in caplog.text
    assert ANTHROPIC_KEY not in caplog.text
    assert harness.state.upstream_errors == {"anthropic_key": 1}
    await harness.aclose()


# ---------------------------------------------------------------------------
# expose_models: both shapes, answered before adapter routing.
# ---------------------------------------------------------------------------


async def test_expose_models_both_shapes(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["expose_models"] = True
    raw["routing"]["model_catalog"] = ["claude-opus-5"]
    raw["routing"]["rule"].append(
        {
            "id": "literal",
            "match": {"protocol": "anthropic", "model": ["claude-haiku-4-5", "claude-opus-5"]},
            "upstream": "anthropic_key",
        }
    )
    harness = Harness(_config(raw))
    anthropic = await harness.client.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
    assert anthropic.status_code == 200
    assert anthropic.json() == {
        "data": [
            {
                "type": "model",
                "id": model,
                "display_name": model,
                "created_at": "2026-01-01T00:00:00Z",
            }
            for model in ("claude-opus-5", "claude-haiku-4-5")
        ],
        "has_more": False,
        "first_id": "claude-opus-5",
        "last_id": "claude-haiku-4-5",
    }
    openai = await harness.client.get("/v1/models")
    assert openai.json() == {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": 0, "owned_by": "llm-redact"}
            for model in ("claude-opus-5", "claude-haiku-4-5")
        ],
    }
    assert not harness.scenarios  # never forwarded
    assert harness.state.recent[-1]["provider"] == "routing"
    assert harness.state.recent[-1]["status"] == 200
    await harness.aclose()


async def test_expose_models_off_passes_through(env: Any, tmp_path: Path) -> None:
    harness = Harness(spec_config(tmp_path / "v.db"))
    response = await harness.client.get("/v1/models")
    # Pass-through openai traffic is routed to the openai default (the fake
    # has no /v1/models route → its 404), never answered locally.
    assert response.status_code == 404
    assert route_of(harness)["upstream"] == "ollama_openai"
    await harness.aclose()


# ---------------------------------------------------------------------------
# Editor: the routing sections are preserved from file truth, never edited.
# ---------------------------------------------------------------------------


async def test_editor_preserves_routing_sections(env: Any, tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(spec_toml(tmp_path / "v.db"))
    config = parse_config(tomllib.loads(config_file.read_text()), str(config_file))
    app = create_app(
        config,
        upstream_transport=httpx.ASGITransport(app=fake_upstream.build_app()),
        config_path=config_file,
    )
    state: ProxyState = app.state.proxy
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")
    token = (await client.get("/__llm-redact/config")).json()["csrf_token"]
    headers = {CSRF_HEADER: token, "content-type": "application/json"}

    for key in sorted(_FILE_PRESERVED_KEYS):
        refused = await client.post(
            "/__llm-redact/config", json={"config": {key: {}}}, headers=headers
        )
        assert refused.status_code == 400
        assert "edit the file and reload" in refused.json()["error"]

    # (max_body_bytes rather than inject_system_note: the latter is the
    # default every upstream's own inject_system_note resolves from, so
    # editing it legitimately changes the parsed [upstreams].)
    applied = await client.post(
        "/__llm-redact/config", json={"config": {"max_body_bytes": 123456}}, headers=headers
    )
    assert applied.status_code == 200, applied.text
    written = tomllib.loads(config_file.read_text())
    assert written["max_body_bytes"] == 123456
    # Every routing section survived byte-for-byte in meaning.
    reparsed = parse_config(written, str(config_file))
    assert reparsed.routing == config.routing and reparsed.prices == config.prices
    assert set(written) >= {"upstreams", "routing", "prices"}
    assert written["upstreams"]["anthropic_key"]["credential"] == "env:ANTHROPIC_API_KEY"
    assert state.config.routing == config.routing
    # The emitter round trip of the applied config still holds.
    assert (
        parse_config(tomllib.loads(emit_config_toml(state.config)), "<rt>").routing
        == config.routing
    )
    await client.aclose()
    state.spend_store.close()
    state.vault_manager.close()


# ---------------------------------------------------------------------------
# Reissue / debug headers and the unrouted providers.
# ---------------------------------------------------------------------------


async def test_debug_headers_on_hop_one(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["debug_headers"] = True
    harness = Harness(_config(raw))
    response = await harness.client.post(
        "/v1/messages", json=messages_body(), headers=OAUTH_HEADERS
    )
    assert response.headers[UPSTREAM_HEADER] == "anthropic_oauth"
    assert response.headers[HOPS_HEADER] == "1"
    assert REISSUE_HEADER not in response.headers
    await harness.aclose()


async def test_no_candidate_returns_last_response_with_header(
    env: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    harness = Harness(
        spec_config(tmp_path / "v.db"),
        {"oauth": Scenario(status=429, plan_limit=True), "key": Scenario(status=503)},
    )
    # Park both chain members: anthropic_key in cooldown, ollama out via a
    # runtime passthrough swap (the parse-time rule made a config one
    # impossible; the loop must still refuse it).
    harness.state.routing_state.mark_unhealthy("anthropic_key", 60.0, "503")
    harness.state.routing_upstreams["ollama"] = dataclasses.replace(
        harness.state.routing_upstreams["ollama"], credential="passthrough"
    )
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages", json=messages_body(), headers=OAUTH_HEADERS
        )
    assert response.status_code == 429
    assert response.headers[REISSUE_HEADER] == "skipped; reason=no-candidate"
    assert response.headers["anthropic-ratelimit-unified-status"] == "rejected"
    assert "chain member ollama is a passthrough upstream; skipped" in caplog.text
    assert not harness.received("key") and not harness.received("ollama")
    assert route_of(harness)["reissue"] == "skipped:no-candidate"
    assert route_of(harness)["class"] == "plan_limit_429"
    await harness.aclose()


async def test_chain_skips_count_tokens_false_members(env: Any, tmp_path: Path) -> None:
    # The 5xx chain of anthropic-key-lane is [ollama] and ollama cannot count
    # tokens: nothing eligible remains, so the 503 comes back unchanged.
    harness = Harness(spec_config(tmp_path / "v.db"), {"key": Scenario(status=503)})
    response = await harness.client.post(
        "/v1/messages/count_tokens", json=messages_body(), headers=GATEWAY_HEADERS
    )
    assert response.status_code == 503
    assert response.headers[REISSUE_HEADER] == "skipped; reason=no-candidate"
    assert not harness.received("ollama")
    await harness.aclose()


async def test_budget_chain_with_no_candidate_is_402(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["upstreams"]["anthropic_key"]["monthly_budget_tokens"] = 1
    harness = Harness(_config(raw))
    body = messages_body()
    assert (
        await harness.client.post("/v1/messages", json=body, headers=GATEWAY_HEADERS)
    ).status_code == 200
    harness.state.routing_state.mark_unhealthy("ollama", 60.0, "503")
    response = await harness.client.post("/v1/messages", json=body, headers=GATEWAY_HEADERS)
    assert response.status_code == 402
    assert response.headers[REISSUE_HEADER] == "skipped; reason=no-candidate"
    assert len(harness.received("key")) == 1
    await harness.aclose()


@pytest.mark.parametrize(
    ("path", "provider", "body"),
    [
        ("/openai/deployments/d/chat/completions?api-version=1", "azure", {"messages": []}),
        ("/v2/chat", "cohere", {"model": "command", "messages": []}),
        ("/custom/vllm/v1/chat/completions", "custom:vllm", {"model": "m", "messages": []}),
    ],
)
async def test_unrouted_providers_keep_the_legacy_path(
    env: Any, tmp_path: Path, path: str, provider: str, body: dict[str, Any]
) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["providers"] = {
        "azure": {"upstream_base_url": "http://azure-legacy"},
        "cohere": {"upstream_base_url": "http://cohere-legacy"},
        "custom": {"vllm": {"upstream_base_url": "http://vllm-legacy"}},
    }
    scenarios: dict[str, Any] = {}
    harness = Harness(_config(raw), scenarios)
    response = await harness.client.post(path, json=body, headers=GATEWAY_HEADERS)
    # The fake has no such routes (404) — what matters is WHERE it went.
    host = {"azure": "azure-legacy", "cohere": "cohere-legacy", "custom:vllm": "vllm-legacy"}[
        provider
    ]
    assert host in scenarios or response.status_code == 404
    assert UPSTREAM_HEADER not in response.headers
    assert harness.state.recent[-1]["route"] is None
    assert harness.state.recent[-1]["provider"] == provider
    await harness.aclose()


async def test_detection_off_provider_is_still_routed(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["providers"] = {"anthropic": {"detection": False}}
    harness = Harness(_config(raw), {"ollama": Scenario(echo_model=True)})
    headers = {**OAUTH_HEADERS, "x-claude-code-agent-id": "Explore"}
    response = await harness.client.post("/v1/messages", json=messages_body(), headers=headers)
    assert response.status_code == 200
    [seen] = harness.received("ollama")
    assert EMAIL in seen.body["messages"][0]["content"]  # unredacted, as documented
    assert seen.body["model"] == "muse-64k"  # routing still rewrote the model
    assert response.json()["model"] == "claude-sonnet-5"
    await harness.aclose()


# ---------------------------------------------------------------------------
# Hop-loop bounds: throttle cap, max_hops, deadline, reissue_policy never,
# body_defaults on the default upstream, stream_options injection.
# ---------------------------------------------------------------------------


async def test_throttle_wait_over_cap_returns_429_without_waiting(
    env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def never(_seconds: float) -> None:
        raise AssertionError("must not sleep")

    monkeypatch.setattr(proxy_module, "_RETRY_SLEEP", never)
    harness = Harness(
        spec_config(tmp_path / "v.db"), {"oauth": Scenario(status=429, retry_after=60)}
    )
    response = await harness.client.post(
        "/v1/messages", json=messages_body(), headers=OAUTH_HEADERS
    )
    assert response.status_code == 429 and len(harness.received("oauth")) == 1
    assert route_of(harness)["hops"] == 1
    await harness.aclose()


async def test_max_hops_bounds_the_chain(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["max_hops"] = 2
    harness = Harness(
        _config(raw), {"openai-key": Scenario(status=503), "gemini-compat": Scenario(status=503)}
    )
    response = await harness.client.post(
        "/v1/chat/completions", json={"model": "gpt-5", "messages": []}, headers=GATEWAY_HEADERS
    )
    assert response.status_code == 503
    assert response.headers[HOPS_HEADER] == "2"
    assert response.headers[UPSTREAM_HEADER] == "gemini_openai_compat"
    assert not harness.received("ollama-openai")
    assert route_of(harness) == {
        "rule": "openai-lane",
        "upstream": "gemini_openai_compat",
        "hops": 2,
        "auth": "gateway-key",
        "class": "5xx",
        "reissue": "yes",
    }
    # Both members failed with a listed status → both in cooldown.
    assert not harness.state.routing_state.healthy("openai_key")
    assert not harness.state.routing_state.healthy("gemini_openai_compat")
    await harness.aclose()


async def test_deadline_stops_reissues(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["request_deadline_seconds"] = 0.000001
    harness = Harness(_config(raw), {"openai-key": Scenario(status=503)})
    response = await harness.client.post(
        "/v1/chat/completions", json={"model": "gpt-5", "messages": []}, headers=GATEWAY_HEADERS
    )
    assert response.status_code == 503 and route_of(harness)["hops"] == 1
    assert not harness.received("gemini-compat")
    await harness.aclose()


async def test_reissue_policy_never_delivers_the_error(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["rule"][3]["reissue_policy"] = "never"
    harness = Harness(_config(raw), {"openai-key": Scenario(status=503)})
    response = await harness.client.post(
        "/v1/chat/completions", json={"model": "gpt-5", "messages": []}, headers=GATEWAY_HEADERS
    )
    assert response.status_code == 503 and route_of(harness)["reissue"] == "no"
    assert not harness.received("gemini-compat")
    await harness.aclose()


async def test_default_upstream_gets_body_defaults_and_stream_usage(
    env: Any, tmp_path: Path
) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["default_upstream"] = {"anthropic": "ollama", "openai": "openrouter"}
    harness = Harness(_config(raw))
    body = {
        "model": "mistral-large",  # matches no openai rule → the default
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = await harness.client.post(
        "/v1/chat/completions", json=body, headers={"authorization": "Bearer gateway-local"}
    )
    assert response.status_code == 200
    [seen] = harness.received("openrouter")
    assert seen.path == "/api/v1/chat/completions"
    assert seen.body["provider"] == {"only": ["anthropic", "openai"], "data_collection": "deny"}
    assert seen.body["stream_options"] == {"include_usage": True}
    assert seen.headers["http-referer"] == "http://localhost"
    assert seen.headers["x-title"] == "llm-redact"
    assert seen.headers["authorization"] == "Bearer sk-or-test-not-a-real-key-EXAMPLE"
    assert route_of(harness)["rule"] is None and route_of(harness)["upstream"] == "openrouter"
    # The final usage chunk was tracked and recorded.
    assert harness.state.budget_ledger.totals("openrouter").rows == 1
    assert harness.state.metrics.routed[("openrouter", "-")] == 1
    await harness.aclose()


async def test_passthrough_stream_is_forwarded_byte_identical(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["upstreams"]["openai_pt"] = {"protocol": "openai", "base_url": "http://openai-pt"}
    raw["routing"]["rule"].insert(
        0, {"id": "pt", "match": {"protocol": "openai", "model": "gpt-pt"}, "upstream": "openai_pt"}
    )
    harness = Harness(_config(raw))
    original = b'{"model":"gpt-pt", "stream":true,  "messages":[{"role":"user","content":"hi"}]}'
    response = await harness.client.post(
        "/v1/chat/completions",
        content=original,
        headers={"authorization": "Bearer gateway-local", "content-type": "application/json"},
    )
    assert response.status_code == 200
    [seen] = harness.received("openai-pt")
    assert "stream_options" not in seen.body  # never injected on passthrough
    assert seen.headers["authorization"] == "Bearer gateway-local"
    await harness.aclose()


# ---------------------------------------------------------------------------
# ProxyState: credentials, warnings, spend store selection, reload.
# ---------------------------------------------------------------------------


def test_state_init_resolves_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_env_keys(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY \\(upstream 'openrouter'\\)"):
        ProxyState(spec_config(tmp_path / "v.db"), None)


def test_state_init_logs_routing_warnings(
    env: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["default_upstream"] = {"anthropic": "anthropic_key"}  # metered default
    with caplog.at_level(logging.WARNING, logger="llm_redact"):
        state = ProxyState(_config(raw), None)
    assert state.config.routing.warnings
    assert all(f"routing: {w}" in caplog.text for w in state.config.routing.warnings)
    state.spend_store.close()
    state.vault_manager.close()


def test_spend_store_selection(env: Any, tmp_path: Path) -> None:
    sqlite_state = ProxyState(spec_config(tmp_path / "v.db"), None)
    assert isinstance(sqlite_state.spend_store, SqliteSpendStore)
    assert sqlite_state.spend_store.path == tmp_path / "v.db"
    sqlite_state.spend_store.close()
    sqlite_state.vault_manager.close()

    memory_raw = _raw(tmp_path / "unused.db")
    memory_raw["vault"] = {"backend": "memory"}
    memory_state = ProxyState(_config(memory_raw), None)
    assert isinstance(memory_state.spend_store, InMemorySpendStore)

    # Routing disabled: the sqlite vault file is never touched for spend.
    disabled_raw = _raw(tmp_path / "untouched.db")
    disabled_raw["routing"]["enabled"] = False
    disabled_state = ProxyState(_config(disabled_raw), None)
    assert isinstance(disabled_state.spend_store, InMemorySpendStore)
    disabled_state.vault_manager.close()
    assert ProxyState(Config(), None).config.routing.enabled is False


def test_reload_hot_swaps_routing_and_prices(
    env: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(spec_toml(tmp_path / "v.db"))
    state = ProxyState(
        parse_config(tomllib.loads(config_file.read_text()), "<f>"), None, config_path=config_file
    )
    state.routing_state.mark_unhealthy("openrouter", 60.0, "503")
    state.routing_state.mark_unhealthy("anthropic_key", 60.0, "503")
    old_price = state.price_table.lookup("claude-sonnet-5")
    assert old_price is not None and old_price.input == 2.0

    # Drop openrouter, raise anthropic_key's budget, change the override.
    raw = tomllib.loads(config_file.read_text())
    del raw["upstreams"]["openrouter"]
    raw["upstreams"]["anthropic_key"]["monthly_budget_usd"] = 999
    raw["prices"]["override"]["claude-sonnet-5"]["input"] = 4.0
    raw["routing"]["budget_reset_day"] = 5
    config_file.write_text(emit_config_toml(parse_config(raw, "<f>")))
    state.reload()
    assert "openrouter" not in state.routing_upstreams
    assert state.routing_state.cooldown_remaining("openrouter") == 0.0  # pruned
    assert not state.routing_state.healthy("anthropic_key")  # kept
    new_price = state.price_table.lookup("claude-sonnet-5")
    assert new_price is not None and new_price.input == 4.0
    assert state.budget_ledger.snapshot()["anthropic_key"]["budget_usd"] == 999
    assert state.budget_ledger.period[0].day == 5

    # A reload whose env credential is missing keeps the old config.
    env.delenv("GEMINI_API_KEY")
    before = state.config
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        state.reload()
    assert state.config is before
    assert "GEMINI_API_KEY (upstream 'gemini_openai_compat')" in caplog.text
    state.spend_store.close()
    state.vault_manager.close()


def test_reload_enabling_routing_opens_the_sqlite_spend_table(env: Any, tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["enabled"] = False
    config_file.write_text(emit_config_toml(_config(raw)))
    state = ProxyState(_config(raw), None, config_path=config_file)
    assert isinstance(state.spend_store, InMemorySpendStore)
    raw["routing"]["enabled"] = True
    config_file.write_text(emit_config_toml(_config(raw)))
    state.reload()
    assert state.config.routing.enabled
    assert isinstance(state.spend_store, SqliteSpendStore)
    state.spend_store.close()
    state.vault_manager.close()


# ---------------------------------------------------------------------------
# /status block, metrics, recent rows.
# ---------------------------------------------------------------------------


async def test_status_routing_block_shape(env: Any, tmp_path: Path) -> None:
    harness = Harness(spec_config(tmp_path / "v.db"))
    await harness.client.post("/v1/messages", json=messages_body(), headers=OAUTH_HEADERS)
    block = (await harness.client.get("/__llm-redact/status")).json()["routing"]
    assert set(block) == {
        "enabled",
        "default_upstreams",
        "rules",
        "reissues_last_hour",
        "plan_limit_detection",
        "expose_models",
        "upstreams",
        "unpriced_models",
        "warnings",
    }
    assert block["enabled"] is True and block["rules"] == 6
    assert block["default_upstreams"] == {"anthropic": "ollama", "openai": "ollama_openai"}
    assert block["plan_limit_detection"] == "headers" and block["expose_models"] is False
    assert set(block["upstreams"]) == set(harness.state.config.routing.upstream_names())
    oauth = block["upstreams"]["anthropic_oauth"]
    assert set(oauth) == {
        "protocol",
        "credential",
        "cost",
        "legacy",
        "state",
        "cooldown_remaining_seconds",
        "requests",
        "reissues_last_hour",
        "last_error_class",
        "last_error_at",
        "spend",
    }
    assert oauth["credential"] == "passthrough" and oauth["requests"] == 1
    assert oauth["state"] == "healthy" and oauth["legacy"] is False
    assert set(oauth["spend"]) == {
        "period",
        "in_tokens",
        "out_tokens",
        "cache_read",
        "cache_write",
        "usd",
        "budget_usd",
        "budget_tokens",
        "remaining_usd",
        "remaining_tokens",
        "unpriced_rows",
        "reissue_usd",
        "reissue_tokens",
    }
    assert oauth["spend"]["in_tokens"] == 10 and oauth["spend"]["budget_usd"] is None
    key = block["upstreams"]["anthropic_key"]
    assert key["credential"] == "env" and key["spend"]["budget_usd"] == 100
    assert key["spend"]["remaining_usd"] == 100
    # Credential VALUES and env var names never appear.
    flat = json.dumps(block)
    assert "ANTHROPIC_API_KEY" not in flat and ANTHROPIC_KEY not in flat
    await harness.aclose()


async def test_metrics_render_routing_counters(env: Any, tmp_path: Path) -> None:
    harness = Harness(
        spec_config(tmp_path / "v.db"), {"oauth": Scenario(status=429, plan_limit=True)}
    )
    await harness.client.post("/v1/messages", json=messages_body(), headers=OAUTH_HEADERS)
    text = (await harness.client.get("/__llm-redact/metrics")).text
    assert "# TYPE llm_redact_routed_requests_total counter" in text
    assert 'llm_redact_routed_requests_total{upstream="anthropic_key",rule="max-lane"} 1' in text
    assert "# TYPE llm_redact_reissues_total counter" in text
    assert (
        'llm_redact_reissues_total{from_upstream="anthropic_oauth",to_upstream="anthropic_key"} 1'
        in text
    )
    await harness.aclose()


async def test_recent_rows_carry_route_and_events_feed(env: Any, tmp_path: Path) -> None:
    harness = Harness(spec_config(tmp_path / "v.db"))
    await harness.client.post("/v1/messages", json=messages_body(), headers=OAUTH_HEADERS)
    recent = (
        await harness.client.get("/__llm-redact/recent", headers={"host": "127.0.0.1"})
    ).json()
    row = recent["entries"][0]
    assert row["route"] == {
        "rule": "max-lane",
        "upstream": "anthropic_oauth",
        "hops": 1,
        "auth": "oauth",
        "class": "ok",
        "reissue": "no",
    }
    assert "route" in row and row["provider"] == "anthropic"
    await harness.aclose()


def test_route_delivery_row_and_line_helpers() -> None:
    upstream = UpstreamConfig(name="u", protocol="anthropic", base_url="http://u")
    delivery = RouteDelivery(
        rule_id=None,
        upstream=upstream,
        hops=1,
        auth="none",
        status_class="ok",
        reissue="no",
        protocol="anthropic",
        original_model="orig",
        sent_model="new",
        headers={},
        method="POST",
        path="/v1/messages",
    )
    assert delivery.as_row()["rule"] is None
    # observe_line: non-UTF-8 bytes pass untouched; a restored line re-encodes.
    assert delivery.observe_line(b"\xff\xfe") == b"\xff\xfe"
    assert delivery.observe_line(b'{"model": "new"}') == b'{"model": "orig"}'
    assert delivery.observe_line(b'{"model": "orig"}') == b'{"model": "orig"}'
    assert (
        proxy_module._route_log_suffix(delivery.as_row())
        == " rule=- upstream=u hops=1 auth=none class=ok reissue=no"
    )


def test_legacy_provider_config_default_state_has_no_routing() -> None:
    # The unrouted default: memory spend store, no upstreams, routing off —
    # and a plain Config builds without any env credential.
    state = ProxyState(
        Config(providers={**Config().providers, "anthropic": ProviderConfig("http://up")}), None
    )
    assert state.routing_upstreams == {} and isinstance(state.spend_store, InMemorySpendStore)
    assert state.config.routing.enabled is False
