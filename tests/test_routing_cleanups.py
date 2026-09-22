"""The whole-feature review's code cleanups (docs/routing.md), pinned:

- ONE precedence ladder: `RouteRule.resolve` is the only special > exact >
  class lookup; `chain_for` and the proxy's hop loop both go through it
  (the proxy's private copy is gone).
- `GET /v1/models` (expose_models) is a LOCAL answer placed after the
  disabled-provider 502 and the named-user 403 — never forwarded, never
  answered to a client the gates would refuse.
- `request_deadline_seconds` bounds each hop's own send/read (a per-hop
  httpx timeout derived from the remaining budget), not only the checks
  between re-issues: a hop that accepts and hangs fails INSIDE the deadline.
- A transport fault that resolves through a chain but is stopped by the
  reissue policy is recorded class=transport (reissue=no /
  skipped:stateful), never as the chain key it resolved through.
- `routes test`'s count_tokens 404 note uses the proxy's own gate.
- The config-derived helpers have one home each (routing.py / spend.py);
  the CLIs and the proxy import the same implementation.
- The request-path spend store waits milliseconds, not seconds, for a
  locked vault file (the synchronous INSERT is documented; a stall is not).
- A routed pass-through JSON body with nothing to bill or restore is
  forwarded without being parsed.
"""

import asyncio
import inspect
import json
import sqlite3
import time
import tomllib
from pathlib import Path
from typing import Any

import httpx
import pytest

import llm_redact.proxy as proxy_module
import llm_redact.routing as routing_module
import llm_redact.spend as spend_module
from llm_redact import routes_cli
from llm_redact.config import parse_config
from llm_redact.pricing import Usage
from llm_redact.routing import RETRY_SAME, RouteRule, RuleMatch
from llm_redact.spend import REQUEST_PATH_BUSY_TIMEOUT_MS, SpendRow, SqliteSpendStore, format_ts
from test_routing_walkthroughs import (
    GATEWAY_HEADERS,
    Harness,
    Scenario,
    fake_upstream,
    messages_body,
    route_of,
    set_env_keys,
    spec_toml,
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    set_env_keys(monkeypatch)
    return monkeypatch


def _raw(vault_path: Path) -> dict[str, Any]:
    return tomllib.loads(spec_toml(vault_path))


def _config(raw: dict[str, Any]) -> Any:
    return parse_config(raw, "<routing-cleanups>")


# ---------------------------------------------------------------------------
# (1) RouteRule.resolve is THE precedence ladder.
# ---------------------------------------------------------------------------


_RULE = RouteRule(
    id="r",
    match=RuleMatch(protocol="anthropic"),
    upstream="a",
    on_status=(
        ("plan_limit_429", ("b",)),
        ("429", ("c",)),
        ("4xx", ("d",)),
        ("529", ("e",)),
        ("5xx", ("f",)),
        ("throttle_429", RETRY_SAME),
    ),
)


@pytest.mark.parametrize(
    ("status_key", "expected"),
    [
        ("plan_limit_429", ("plan_limit_429", ("b",))),
        ("throttle_429", ("throttle_429", RETRY_SAME)),
        ("429", ("429", ("c",))),
        ("403", ("4xx", ("d",))),
        ("529", ("529", ("e",))),
        ("502", ("5xx", ("f",))),
        ("302", None),
        ("weird", None),
    ],
)
def test_resolve_returns_the_matched_key_and_action(
    status_key: str, expected: tuple[str, Any] | None
) -> None:
    assert _RULE.resolve(status_key) == expected
    # chain_for is resolve without the key — never a second ladder.
    assert _RULE.chain_for(status_key) == (None if expected is None else expected[1])


def test_special_key_falls_back_to_exact_then_class() -> None:
    only_class = RouteRule(
        id="r", match=RuleMatch(protocol="anthropic"), upstream="a", on_status=(("4xx", ("d",)),)
    )
    assert only_class.resolve("plan_limit_429") == ("4xx", ("d",))
    only_exact = RouteRule(
        id="r", match=RuleMatch(protocol="anthropic"), upstream="a", on_status=(("429", ("c",)),)
    )
    assert only_exact.resolve("throttle_429") == ("429", ("c",))
    assert only_exact.resolve("plan_limit_429") == ("429", ("c",))


def test_proxy_has_no_private_ladder() -> None:
    assert not hasattr(proxy_module, "_resolved_status_key")
    source = inspect.getsource(proxy_module)
    assert "rule.resolve(key)" in source
    # The candidate tuples exist in exactly one module.
    assert '"429", "4xx"' not in source
    assert inspect.getsource(routing_module).count('"429", "4xx"') == 1


# ---------------------------------------------------------------------------
# (2) GET /v1/models sits behind the disabled-provider and named-user gates.
# ---------------------------------------------------------------------------


def _models_config(vault_path: Path, **providers: dict[str, Any]) -> Any:
    raw = _raw(vault_path)
    raw["routing"]["expose_models"] = True
    raw["routing"]["model_catalog"] = ["claude-opus-5"]
    if providers:
        raw["providers"] = providers
    return _config(raw)


async def test_expose_models_after_named_user_enforcement(env: Any, tmp_path: Path) -> None:
    harness = Harness(_models_config(tmp_path / "v.db"))
    # 2+ verified users (the pro registry) — the gate every real API path
    # sits behind; the local model answer must not leak past it.
    harness.state.user_enforcement_required = lambda: True  # type: ignore[method-assign]
    response = await harness.client.get("/v1/models")
    assert response.status_code == 403
    assert "named-user key" in response.text
    assert not harness.scenarios  # never forwarded
    assert harness.state.recent[-1]["status"] == 403
    assert harness.state.recent[-1]["provider"] != "routing"
    await harness.aclose()


async def test_expose_models_after_disabled_provider(env: Any, tmp_path: Path) -> None:
    harness = Harness(_models_config(tmp_path / "v.db", openai={"enabled": False}))
    response = await harness.client.get("/v1/models")  # inferred openai
    assert response.status_code == 502
    assert "disabled" in response.text
    # The path infers openai whatever the headers say (provider_for), so
    # the Anthropic-shaped call is refused by the same gate.
    anthropic = await harness.client.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
    assert anthropic.status_code == 502
    assert not harness.scenarios
    assert harness.state.recent[-1]["provider"] == "openai"
    await harness.aclose()
    # Another provider disabled: the openai-inferred discovery call is
    # still answered locally.
    harness = Harness(_models_config(tmp_path / "v2.db", anthropic={"enabled": False}))
    response = await harness.client.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
    assert response.status_code == 200
    assert [m["id"] for m in response.json()["data"]] == ["claude-opus-5"]
    assert not harness.scenarios  # a local answer, never forwarded
    assert harness.state.recent[-1]["provider"] == "routing"
    await harness.aclose()


async def test_expose_models_still_a_local_answer(env: Any, tmp_path: Path) -> None:
    harness = Harness(_models_config(tmp_path / "v.db"))
    response = await harness.client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "claude-opus-5"
    assert not harness.scenarios
    # POST /v1/models is not the discovery call: it passes through as before.
    posted = await harness.client.post("/v1/models", json={})
    assert posted.status_code == 404  # the fake's 404, i.e. forwarded
    await harness.aclose()


# ---------------------------------------------------------------------------
# (3) request_deadline_seconds bounds each hop's own send/read.
# ---------------------------------------------------------------------------


class _HangingTransport(httpx.AsyncBaseTransport):
    """An upstream that accepts the connection and never answers: it sleeps
    `hang` seconds unless the request carries a shorter httpx read timeout,
    in which case it fails with ReadTimeout when that elapses — exactly what
    httpcore does with `request.extensions["timeout"]`. Records the read
    timeout each hop was sent with."""

    def __init__(self, hang: float) -> None:
        self._hang = hang
        self.read_timeouts: list[float | None] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        timeouts = request.extensions.get("timeout", {})
        read = timeouts.get("read")
        self.read_timeouts.append(read)
        if read is not None and read < self._hang:
            await asyncio.sleep(read)
            raise httpx.ReadTimeout("read timed out", request=request)
        await asyncio.sleep(self._hang)
        return httpx.Response(200, json={"never": "reached"})


async def test_hung_hop_fails_inside_the_deadline(env: Any, tmp_path: Path) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["request_deadline_seconds"] = 0.3
    upstream = _HangingTransport(hang=30.0)
    harness = Harness(_config(raw), upstream_transport=upstream)
    started = time.monotonic()
    response = await harness.client.post(
        "/v1/messages", json=messages_body(), headers=GATEWAY_HEADERS
    )
    elapsed = time.monotonic() - started
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    # The hop was cut off by the deadline, not by the client's 600 s.
    assert elapsed < 5.0
    # The primary's read timeout was the remaining budget (<= deadline);
    # the 5xx chain's next member (ollama) would only be tried while budget
    # remains — after a full-budget hang there is none, so ONE hop was sent.
    assert upstream.read_timeouts == [pytest.approx(0.3, abs=0.05)]
    row = route_of(harness)
    assert row["class"] == "transport" and row["hops"] == 1
    assert harness.state.upstream_errors == {"anthropic_key": 1}
    await harness.aclose()


async def test_second_hop_gets_the_remaining_budget(
    env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _HangThenServe(_HangingTransport):
        """The primary hangs; the chain member (ollama) answers at once."""

        def __init__(self) -> None:
            super().__init__(hang=30.0)
            self._inner = httpx.ASGITransport(app=fake_upstream.build_app())

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "ollama":
                self.read_timeouts.append(request.extensions.get("timeout", {}).get("read"))
                return await self._inner.handle_async_request(request)
            return await super().handle_async_request(request)

    raw = _raw(tmp_path / "v.db")
    raw["routing"]["request_deadline_seconds"] = 2.0
    upstream = _HangThenServe()
    # The primary's hang eats 0.15 s of budget via its read timeout.
    harness = Harness(_config(raw), upstream_transport=upstream)
    original_hop_timeout = proxy_module._hop_timeout

    def shortened(remaining: float) -> httpx.Timeout:
        # First hop: pretend only 0.15 s remain so the test stays fast; the
        # second hop sees the REAL remaining budget (< 2 s, > 0).
        if not upstream.read_timeouts:
            return original_hop_timeout(min(remaining, 0.15))
        return original_hop_timeout(remaining)

    monkeypatch.setattr(proxy_module, "_hop_timeout", shortened)
    response = await harness.client.post(
        "/v1/messages", json=messages_body(), headers=GATEWAY_HEADERS
    )
    assert response.status_code == 200
    first, second = upstream.read_timeouts
    assert first == pytest.approx(0.15, abs=0.02)
    assert second is not None and 0.0 < second < 2.0 - 0.15 + 0.01
    assert route_of(harness) == {
        "rule": "anthropic-key-lane",
        "upstream": "ollama",
        "hops": 2,
        "auth": "gateway-key",
        "class": "ok",
        "reissue": "yes",
    }
    await harness.aclose()


def test_hop_timeout_never_exceeds_the_client_caps() -> None:
    full = proxy_module._hop_timeout(600.0)
    # Identical to the shared client's own timeout: the default deadline
    # changes nothing.
    assert full == httpx.Timeout(600.0, connect=10.0)
    assert proxy_module._hop_timeout(10_000.0) == full
    short = proxy_module._hop_timeout(0.5)
    assert (short.read, short.write, short.connect, short.pool) == (0.5, 0.5, 0.5, 0.5)
    # An expired budget is a floor, never httpx's "0 = no timeout".
    assert proxy_module._hop_timeout(-1.0).read == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# (4) Transport fault through a chain, stopped by the reissue policy.
# ---------------------------------------------------------------------------


class _Refusing(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, hosts: set[str]) -> None:
        self._inner, self._hosts = inner, hosts

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.host in self._hosts:
            raise httpx.ConnectError("refused")
        return await self._inner.handle_async_request(request)


@pytest.mark.parametrize(
    ("policy", "stateful", "reissue"),
    [("never", False, "no"), ("stateless-only", True, "skipped:stateful")],
)
async def test_policy_stopped_transport_fault_keeps_class_transport(
    env: Any,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    policy: str,
    stateful: bool,
    reissue: str,
) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["rule"][2]["reissue_policy"] = policy  # anthropic-key-lane, 5xx = ["ollama"]
    scenarios: dict[str, Any] = {}
    fake = fake_upstream.build_app(scenarios)
    harness = Harness(
        _config(raw),
        scenarios,
        upstream_transport=_Refusing(httpx.ASGITransport(app=fake), {"key"}),
    )
    with caplog.at_level("INFO", logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages", json=messages_body(stateful=stateful), headers=GATEWAY_HEADERS
        )
    assert response.status_code == 502
    assert not harness.received("ollama")  # the chain was resolved but never entered
    assert route_of(harness) == {
        "rule": "anthropic-key-lane",
        "upstream": "anthropic_key",
        "hops": 1,
        "auth": "gateway-key",
        "class": "transport",
        "reissue": reissue,
    }
    assert f"class=transport reissue={reissue}" in caplog.text
    # A LISTED fault still parks the upstream (decision 10) — the class is
    # about what happened, the cooldown about what the rule lists.
    assert not harness.state.routing_state.healthy("anthropic_key")
    await harness.aclose()


async def test_listed_status_stopped_by_policy_still_reports_its_key(
    env: Any, tmp_path: Path
) -> None:
    raw = _raw(tmp_path / "v.db")
    raw["routing"]["rule"][2]["reissue_policy"] = "never"
    harness = Harness(_config(raw), {"key": Scenario(status=503)})
    response = await harness.client.post(
        "/v1/messages", json=messages_body(), headers=GATEWAY_HEADERS
    )
    assert response.status_code == 503
    assert route_of(harness)["class"] == "5xx" and route_of(harness)["reissue"] == "no"
    await harness.aclose()


# ---------------------------------------------------------------------------
# (5) routes test's count_tokens note uses the proxy's gate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("protocol", "path", "expected"),
    [
        ("anthropic", "/v1/messages/count_tokens", True),
        ("anthropic", "/v1/messages/count_tokens/", False),
        ("anthropic", "/v1/x/count_tokens", False),
        ("openai", "/v1/x/count_tokens", False),
        ("gemini", "/v1beta/models/g:countTokens", False),
        (None, "/v1/messages/count_tokens", False),
    ],
)
def test_count_tokens_gate_is_shared(
    tmp_path: Path, protocol: str | None, path: str, expected: bool
) -> None:
    assert routing_module.is_count_tokens_path(protocol, path) is expected
    assert "is_count_tokens_path(plan.protocol, path)" in inspect.getsource(proxy_module)
    if protocol not in ("anthropic", "openai"):
        return  # the spec config has no gemini default; the gate itself is pinned above
    config = _config(_raw(tmp_path / "v.db"))
    # Force the decision onto ollama (count_tokens = false) via the default.
    decision = routes_cli.route_decision(
        config.routing,
        protocol=protocol,
        model="unrouted-model",
        headers={},
        path=path,
        auth="none",
    )
    assert decision["upstream"]["name"] in ("ollama", "ollama_openai")
    assert decision["count_tokens_404"] is (expected and decision["upstream"]["name"] == "ollama")


# ---------------------------------------------------------------------------
# (6) One home per shared helper.
# ---------------------------------------------------------------------------


def test_shared_helpers_have_one_implementation() -> None:
    assert routes_cli.build_price_table is spend_module.build_price_table
    assert routes_cli.budgets_for is spend_module.budgets_for
    assert routes_cli.model_from_gemini_path is routing_module.gemini_path_model
    assert proxy_module.gemini_path_model is routing_module.gemini_path_model
    assert proxy_module._build_price_table is spend_module.build_price_table
    assert proxy_module._budgets_for is spend_module.budgets_for
    assert proxy_module._rewrite_gemini_path is routing_module.rewrite_gemini_path
    # No lazy imports of the proxy's private names remain in the CLI.
    cli_source = inspect.getsource(routes_cli)
    assert "from llm_redact.proxy import _" not in cli_source
    assert "from llm_redact.proxy import gemini_path_model" not in cli_source


def test_gemini_path_helpers_round_trip() -> None:
    path = "/v1beta/models/gemini-2.5-flash:generateContent"
    assert routing_module.gemini_path_model(path) == "gemini-2.5-flash"
    rewritten = routing_module.rewrite_gemini_path(path, "muse/64k")
    assert rewritten == "/v1beta/models/muse%2F64k:generateContent"
    assert routing_module.gemini_path_model(rewritten) == "muse%2F64k"
    assert routing_module.rewrite_gemini_path("/v1beta/cachedContents", "x") == (
        "/v1beta/cachedContents"
    )


# ---------------------------------------------------------------------------
# (7) The request-path spend store waits milliseconds for a locked file.
# ---------------------------------------------------------------------------


def test_request_path_busy_timeout_is_short(tmp_path: Path) -> None:
    assert REQUEST_PATH_BUSY_TIMEOUT_MS <= 500
    store = SqliteSpendStore(tmp_path / "vault.db")
    try:
        (value,) = store._conn.execute("PRAGMA busy_timeout").fetchone()
        assert value == REQUEST_PATH_BUSY_TIMEOUT_MS
        # An offline caller may ask for the CLI's patience.
        longer = SqliteSpendStore(tmp_path / "vault.db", busy_timeout_ms=5000)
        assert longer._conn.execute("PRAGMA busy_timeout").fetchone() == (5000,)
        longer.close()
    finally:
        store.close()


def test_locked_vault_file_costs_milliseconds_not_seconds(tmp_path: Path) -> None:
    path = tmp_path / "vault.db"
    store = SqliteSpendStore(path)
    other = sqlite3.connect(path, isolation_level=None)
    try:
        other.execute("BEGIN EXCLUSIVE")  # another process holding the write lock
        row = SpendRow(
            ts=format_ts(spend_module.datetime.now(spend_module.UTC)),
            upstream="anthropic_key",
            model="claude-sonnet-5",
            hop=1,
            usage=Usage(input_tokens=1, output_tokens=1),
            usd=0.0,
        )
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError):
            store.record(row)
        elapsed = time.monotonic() - started
        assert REQUEST_PATH_BUSY_TIMEOUT_MS / 1000.0 * 0.5 <= elapsed < 2.0
    finally:
        other.execute("ROLLBACK")
        other.close()
        store.close()


# ---------------------------------------------------------------------------
# (8) A routed pass-through JSON body with nothing to do is not parsed.
# ---------------------------------------------------------------------------


class _CountingJson:
    """`json` as proxy.py sees it, recording what `loads` was given."""

    def __init__(self) -> None:
        self.loaded: list[Any] = []

    def loads(self, text: Any, *args: Any, **kwargs: Any) -> Any:
        self.loaded.append(text)
        return json.loads(text, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(json, name)


async def test_routed_passthrough_json_is_not_parsed(
    env: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(_config(_raw(tmp_path / "v.db")))
    counting = _CountingJson()
    monkeypatch.setattr(proxy_module, "json", counting)
    # Anthropic batch poll: pass-through (kind NONE) routed to the default
    # anthropic upstream; the fake answers JSON with no model to restore.
    response = await harness.client.get(
        "/v1/messages/batches/msgbatch_fake", headers={"anthropic-version": "2023-06-01"}
    )
    assert response.status_code == 200
    body = response.content
    assert response.json() == {"id": "msgbatch_fake", "processing_status": "ended"}
    assert route_of(harness)["upstream"] == "ollama"
    assert not any((item == body or item == body.decode("utf-8")) for item in counting.loaded), (
        "the pass-through body was parsed for nothing"
    )
    await harness.aclose()
