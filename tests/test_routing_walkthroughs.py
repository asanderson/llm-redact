"""The ten §8 behaviour walkthroughs of the routing spec, reproduced as
integration tests: the real proxy app, the full §2.2-style config (with the
spec's own I-5 correction), env credentials via monkeypatch, and ONE
`scripts/fake_upstream.py` build_app serving every upstream base URL
(`http://oauth`, `http://key`, `http://ollama`, …) through a single
ASGITransport, distinguished by Host (docs/routing.md)."""

import dataclasses
import importlib.util
import json
import logging
import re
import sys
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_redact.config import Config, ConfigError, parse_config
from llm_redact.proxy import ProxyState, create_app
from llm_redact.routing import HOPS_HEADER, REISSUE_HEADER, UPSTREAM_HEADER
from llm_redact.spend import Budget, BudgetLedger, report

_SCRIPT = Path(__file__).parents[1] / "scripts" / "fake_upstream.py"
_spec = importlib.util.spec_from_file_location("fake_upstream", _SCRIPT)
assert _spec is not None and _spec.loader is not None
fake_upstream = importlib.util.module_from_spec(_spec)
sys.modules["fake_upstream"] = fake_upstream
_spec.loader.exec_module(fake_upstream)
Scenario = fake_upstream.Scenario

EMAIL = "jane.doe@corp.example"

# Fake credentials: secret-SHAPED but never real. They live in env only (the
# built-in detectors match sk-ant-…, so they must never enter a BODY).
ANTHROPIC_KEY = "sk-ant-test-not-a-real-key-EXAMPLE"
OPENAI_KEY = "sk-test-not-a-real-openai-key-EXAMPLE"
GEMINI_KEY = "AIza-test-not-a-real-gemini-key-EXAMPLE"
OPENROUTER_KEY = "sk-or-test-not-a-real-key-EXAMPLE"
OAUTH_TOKEN = "sk-ant-oat01-test-not-a-real-oauth-token-EXAMPLE"
ENV_KEYS = {
    "ANTHROPIC_API_KEY": ANTHROPIC_KEY,
    "OPENAI_API_KEY": OPENAI_KEY,
    "GEMINI_API_KEY": GEMINI_KEY,
    "OPENROUTER_API_KEY": OPENROUTER_KEY,
}
ALL_SECRETS = (*ENV_KEYS.values(), OAUTH_TOKEN)

OAUTH_MARKER = "oauth-2025-04-20"
OAUTH_HEADERS = {
    "authorization": f"Bearer {OAUTH_TOKEN}",
    "anthropic-beta": f"{OAUTH_MARKER},interleaved-thinking-2025-05-14",
    "anthropic-version": "2023-06-01",
    "anthropic-dangerous-direct-browser-access": "true",
    "user-agent": "claude-cli/2.0.0 (external, cli)",
    "x-app": "cli",
    "x-claude-code-session-id": "sess_EXAMPLE",
    "x-stainless-lang": "js",
}
GATEWAY_HEADERS = {
    "authorization": "Bearer gateway-local",
    "anthropic-version": "2023-06-01",
}

# The spec's §2.2 example against the fake upstreams (base URLs are hosts
# the single fake app tells apart), with the spec's own I-5 slip corrected:
# `anthropic-key-lane` chains to an anthropic-protocol member (ollama), not
# to the openai-protocol `openrouter`. `count_tokens = false` on ollama per
# decision 7; the table form of default_upstream per decision 2.
SPEC_TOML = """
[vault]
backend = "sqlite"
path = "{vault_path}"

[upstreams.anthropic_oauth]
protocol   = "anthropic"
base_url   = "http://oauth"
credential = "passthrough"
inject_system_note = false

[upstreams.anthropic_key]
protocol   = "anthropic"
base_url   = "http://key"
credential = "env:ANTHROPIC_API_KEY"
monthly_budget_usd = 100
cooldown_seconds   = 60

[upstreams.openai_key]
protocol   = "openai"
base_url   = "http://openai-key/v1"
credential = "env:OPENAI_API_KEY"
monthly_budget_usd = 50
cooldown_seconds   = 60

[upstreams.gemini_openai_compat]
protocol   = "openai"
base_url   = "http://gemini-compat/v1beta/openai"
credential = "env:GEMINI_API_KEY"
monthly_budget_usd = 30

[upstreams.openrouter]
protocol   = "openai"
base_url   = "http://openrouter/api/v1"
credential = "env:OPENROUTER_API_KEY"
monthly_budget_usd = 25
extra_headers = {{ "HTTP-Referer" = "http://localhost", "X-Title" = "llm-redact" }}
body_defaults = {{ provider = {{ only = ["anthropic", "openai"], data_collection = "deny" }} }}

[upstreams.ollama]
protocol   = "anthropic"
base_url   = "http://ollama"
credential = "none"
cost       = "zero"
count_tokens = false

[upstreams.ollama_openai]
protocol   = "openai"
base_url   = "http://ollama-openai/v1"
credential = "none"
cost       = "zero"

[routing]
enabled          = true
default_upstream = {{ anthropic = "ollama", openai = "ollama_openai" }}
max_hops         = 3
request_deadline_seconds = 600
plan_limit_detection = "headers"

[[routing.rule]]
id       = "explore-local"
match    = {{ protocol = "anthropic", headers = {{ "x-claude-code-agent-id" = "Explore" }} }}
upstream = "ollama"
model_rewrite = "muse-64k"

[[routing.rule]]
id       = "max-lane"
match    = {{ protocol = "anthropic", model = "claude-*", auth = "oauth" }}
upstream = "anthropic_oauth"
reissue_policy = "stateless-only"
[routing.rule.on_status]
plan_limit_429 = ["anthropic_key", "ollama"]
529 = ["anthropic_key"]
throttle_429 = "retry-same"

[[routing.rule]]
id       = "anthropic-key-lane"
match    = {{ protocol = "anthropic", model = "claude-*", auth = "gateway-key" }}
upstream = "anthropic_key"
on_status = {{ 429 = ["ollama"], 5xx = ["ollama"] }}
{budget_chain}

[[routing.rule]]
id       = "openai-lane"
match    = {{ protocol = "openai", model = "gpt-*" }}
upstream = "openai_key"
[routing.rule.on_status]
429 = ["gemini_openai_compat", "ollama_openai"]
5xx = ["gemini_openai_compat", "ollama_openai"]

[[routing.rule]]
id       = "gemini-lane"
match    = {{ protocol = "openai", model = "gemini-*" }}
upstream = "gemini_openai_compat"
on_status = {{ 429 = ["ollama_openai"] }}

[[routing.rule]]
id       = "local-openai"
match    = {{ protocol = "openai", model = ["muse-*", "gemma4*", "gpt-oss*"] }}
upstream = "ollama_openai"

[prices]
table    = "builtin"
[prices.override."claude-sonnet-5"]
input = 2.0
output = 10.0
cache_read = 0.2
cache_write = 2.5
"""


def spec_toml(vault_path: Path, *, budget_chain: bool = True) -> str:
    chain = 'on_budget_exhausted = ["ollama"]' if budget_chain else ""
    return SPEC_TOML.format(vault_path=vault_path, budget_chain=chain)


Mutator = Callable[[dict[str, Any]], None]


def spec_config(
    vault_path: Path, *, budget_chain: bool = True, mutate: Mutator | None = None
) -> Config:
    raw = tomllib.loads(spec_toml(vault_path, budget_chain=budget_chain))
    if mutate is not None:
        mutate(raw)
    return parse_config(raw, "<walkthrough>")


def token_budget(raw: dict[str, Any]) -> None:
    """anthropic_key with a 20-token monthly budget (the fake's 15-token
    replies cross it on the second request) instead of the USD one."""
    del raw["upstreams"]["anthropic_key"]["monthly_budget_usd"]
    raw["upstreams"]["anthropic_key"]["monthly_budget_tokens"] = 20


def set_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in ENV_KEYS.items():
        monkeypatch.setenv(name, value)


class Harness:
    """A proxy over one fake-upstream app; `scenarios` is the per-Host table
    the walkthroughs program before sending."""

    def __init__(
        self,
        config: Config,
        scenarios: dict[str, Any] | None = None,
        *,
        upstream_transport: httpx.AsyncBaseTransport | None = None,
        raise_app_exceptions: bool = True,
    ) -> None:
        self.scenarios: dict[str, Any] = scenarios if scenarios is not None else {}
        self.fake = fake_upstream.build_app(self.scenarios)
        transport = upstream_transport or httpx.ASGITransport(app=self.fake)
        self.app = create_app(config, upstream_transport=transport)
        self.state: ProxyState = self.app.state.proxy
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=raise_app_exceptions),
            base_url="http://proxy",
        )

    def received(self, host: str) -> list[Any]:
        scenario = self.scenarios.get(host)
        return list(scenario.received) if scenario is not None else []

    async def aclose(self) -> None:
        await self.client.aclose()
        self.state.spend_store.close()
        self.state.vault_manager.close()


@pytest.fixture
def harness_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    set_env_keys(monkeypatch)
    created: list[Harness] = []

    def make(
        scenarios: dict[str, Any] | None = None,
        *,
        budget_chain: bool = True,
        mutate: Mutator | None = None,
        config: Config | None = None,
        upstream_transport: httpx.AsyncBaseTransport | None = None,
        raise_app_exceptions: bool = True,
    ) -> Harness:
        if config is None:
            config = spec_config(tmp_path / "vault.db", budget_chain=budget_chain, mutate=mutate)
        harness = Harness(
            config,
            scenarios,
            upstream_transport=upstream_transport,
            raise_app_exceptions=raise_app_exceptions,
        )
        created.append(harness)
        return harness

    return make


@pytest.fixture
async def harness(harness_factory: Any) -> Any:
    h = harness_factory()
    yield h
    await h.aclose()


def messages_body(
    *,
    model: str = "claude-sonnet-5",
    stream: bool = False,
    stateful: bool = False,
    text: str = f"please mail {EMAIL} today",
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": text}]
    if stateful:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "signed", "signature": "EqQBCkYIBEXAMPLE"},
                    {"type": "text", "text": "ok"},
                ],
            }
        )
        messages.append({"role": "user", "content": "continue"})
    return {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "system": [
            {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}}
        ],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
        "metadata": {"user_id": "u_EXAMPLE"},
        "messages": messages,
    }


def route_of(harness: Harness) -> dict[str, Any]:
    return dict(harness.state.recent[-1]["route"])


def assert_no_secret_leak(received: list[Any], forbidden: tuple[str, ...]) -> None:
    """No forbidden credential VALUE appears in any header the upstream saw."""
    for item in received:
        flat = json.dumps(item.headers)
        for secret in forbidden:
            assert secret not in flat


# ---------------------------------------------------------------------------
# 1. Max lane, healthy.
# ---------------------------------------------------------------------------


async def test_walkthrough_1_max_lane_healthy(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    body = messages_body(stream=True)
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await harness.client.post("/v1/messages", json=body, headers=OAUTH_HEADERS)
    assert response.status_code == 200
    text = response.text
    # SSE relayed with its ping comment, the placeholder restored on the way back.
    assert ": ping" in text
    assert EMAIL in text and "«EMAIL_001»" not in text
    # Rate-limit / request-id headers of the upstream relayed.
    assert response.headers["anthropic-ratelimit-unified-status"] == "allowed"
    assert response.headers["request-id"] == "req_fake"
    # Hop 1, no debug headers.
    assert UPSTREAM_HEADER not in response.headers and HOPS_HEADER not in response.headers

    [seen] = harness.received("oauth")
    # Headers forwarded byte-exact (R-13): auth, beta flags, stainless, app…
    for name, value in OAUTH_HEADERS.items():
        assert seen.headers[name] == value
    assert "x-api-key" not in seen.headers
    # system / tools / metadata untouched; only message content redacted (R-12).
    assert seen.body["system"] == body["system"]
    assert seen.body["tools"] == body["tools"]
    assert seen.body["metadata"] == body["metadata"]
    assert seen.body["max_tokens"] == 64 and seen.body["stream"] is True
    assert seen.body["messages"][0]["content"] == "please mail «EMAIL_001» today"
    assert not harness.received("key") and not harness.received("ollama")

    assert (
        "POST /v1/messages -> 200 rule=max-lane upstream=anthropic_oauth hops=1 auth=oauth"
        " class=ok reissue=no redacted: EMAIL×1"
    ) in caplog.text
    assert route_of(harness) == {
        "rule": "max-lane",
        "upstream": "anthropic_oauth",
        "hops": 1,
        "auth": "oauth",
        "class": "ok",
        "reissue": "no",
    }
    # Passthrough spend is recorded in tokens (subscription usage is visible).
    totals = harness.state.budget_ledger.totals("anthropic_oauth")
    assert totals.rows == 1 and totals.in_tokens == 10 and totals.out_tokens == 5


# ---------------------------------------------------------------------------
# 2. Max lane, plan-limit 429, stateless subagent request → anthropic_key.
# ---------------------------------------------------------------------------


async def test_walkthrough_2_plan_limit_reissues_to_api_key(
    harness_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    harness = harness_factory(
        {
            "oauth": Scenario(status=429, plan_limit=True, retry_after=120),
            "key": Scenario(echo_model=True),
        }
    )
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages", json=messages_body(), headers=OAUTH_HEADERS
        )
    assert response.status_code == 200
    assert response.headers[UPSTREAM_HEADER] == "anthropic_key"
    assert response.headers[HOPS_HEADER] == "2"
    assert REISSUE_HEADER not in response.headers
    assert response.json()["model"] == "claude-sonnet-5"
    assert EMAIL in response.json()["content"][0]["text"]

    [oauth_seen] = harness.received("oauth")
    [key_seen] = harness.received("key")
    # I-3 both ways: the OAuth token never reaches the env: upstream, and the
    # server-held key never reaches the passthrough upstream.
    assert key_seen.headers["x-api-key"] == ANTHROPIC_KEY
    assert "authorization" not in key_seen.headers
    assert key_seen.headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"
    assert key_seen.headers["anthropic-version"] == "2023-06-01"
    assert oauth_seen.headers["authorization"] == f"Bearer {OAUTH_TOKEN}"
    assert "x-api-key" not in oauth_seen.headers
    assert_no_secret_leak([oauth_seen], (ANTHROPIC_KEY,))
    assert_no_secret_leak([key_seen], (OAUTH_TOKEN,))
    # The redacted body is reused verbatim on the second hop (R-18).
    assert key_seen.body == oauth_seen.body

    # Spend attributed to anthropic_key with hop 2; oauth recorded nothing.
    ledger = harness.state.budget_ledger
    assert ledger.totals("anthropic_key").rows == 1
    assert ledger.totals("anthropic_key").reissue_rows == 1
    assert ledger.totals("anthropic_key").usd > 0  # priced via the override
    assert ledger.totals("anthropic_oauth").rows == 0
    # anthropic_oauth in cooldown for retry-after (120 > cooldown_seconds 60).
    remaining = harness.state.routing_state.cooldown_remaining("anthropic_oauth")
    assert 118.0 < remaining <= 120.0
    assert harness.state.routing_state.reissues_last_hour("anthropic_oauth") == 1
    assert harness.state.metrics.reissues[("anthropic_oauth", "anthropic_key")] == 1
    assert (
        "rule=max-lane upstream=anthropic_key hops=2 auth=oauth class=ok reissue=yes" in caplog.text
    )

    status = (await harness.client.get("/__llm-redact/status")).json()["routing"]
    oauth_status = status["upstreams"]["anthropic_oauth"]
    assert oauth_status["state"] == "cooldown"
    assert oauth_status["last_error_class"] == "plan_limit_429"
    assert oauth_status["last_error_at"] is not None
    assert oauth_status["reissues_last_hour"] == 1
    assert status["reissues_last_hour"] == 1
    assert status["upstreams"]["anthropic_key"]["spend"]["reissue_tokens"] == 15
    await harness.aclose()


# ---------------------------------------------------------------------------
# 3. Max lane, plan-limit 429, stateful main-thread request → returned as-is.
# ---------------------------------------------------------------------------


async def test_walkthrough_3_stateful_request_is_not_reissued(
    harness_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    harness = harness_factory({"oauth": Scenario(status=429, plan_limit=True, retry_after=30)})
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages", json=messages_body(stateful=True), headers=OAUTH_HEADERS
        )
    assert response.status_code == 429
    assert response.headers[REISSUE_HEADER] == "skipped; reason=stateful"
    # The upstream's own 429 body and rate-limit headers, unchanged.
    assert response.json()["error"]["type"] == "rate_limit_error"
    assert response.headers["anthropic-ratelimit-unified-status"] == "rejected"
    assert response.headers["retry-after"] == "30"
    assert UPSTREAM_HEADER not in response.headers
    # No credential swap happened: nothing reached anthropic_key or ollama.
    assert len(harness.received("oauth")) == 1
    assert not harness.received("key") and not harness.received("ollama")
    assert "class=plan_limit_429 reissue=skipped:stateful" in caplog.text
    assert route_of(harness)["reissue"] == "skipped:stateful"
    # The listed status still put the failed upstream into cooldown (R-19).
    assert not harness.state.routing_state.healthy("anthropic_oauth")
    await harness.aclose()


# ---------------------------------------------------------------------------
# 4. Max lane, throttle 429 → retry-same once, then the 429 to the client.
# ---------------------------------------------------------------------------


async def test_walkthrough_4_throttle_retries_same_upstream_once(
    harness_factory: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import llm_redact.proxy as proxy_module

    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(proxy_module, "_RETRY_SLEEP", fake_sleep)
    harness = harness_factory({"oauth": Scenario(status=429, retry_after=1)})
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages", json=messages_body(), headers=OAUTH_HEADERS
        )
    assert response.status_code == 429
    assert waits == [2.0]  # max(retry-after 1 s, 2 s)
    assert len(harness.received("oauth")) == 2
    assert not harness.received("key") and not harness.received("ollama")
    assert REISSUE_HEADER not in response.headers
    # A throttle never puts the upstream into cooldown.
    assert harness.state.routing_state.healthy("anthropic_oauth")
    assert "class=throttle_429 reissue=no" in caplog.text
    assert route_of(harness)["hops"] == 2
    await harness.aclose()


# ---------------------------------------------------------------------------
# 5. Explore subagent → local Ollama with the model rewritten and restored.
# ---------------------------------------------------------------------------


async def test_walkthrough_5_explore_subagent_goes_local(harness_factory: Any) -> None:
    harness = harness_factory({"ollama": Scenario(echo_model=True)})
    headers = {**OAUTH_HEADERS, "x-claude-code-agent-id": "Explore"}
    response = await harness.client.post("/v1/messages", json=messages_body(), headers=headers)
    assert response.status_code == 200
    [seen] = harness.received("ollama")
    assert seen.body["model"] == "muse-64k"
    # credential = "none": the OAuth token never reaches the local upstream.
    assert "authorization" not in seen.headers and "x-api-key" not in seen.headers
    assert OAUTH_MARKER not in seen.headers.get("anthropic-beta", "")
    # The response echoes muse-64k; the proxy restored the original id.
    assert response.json()["model"] == "claude-sonnet-5"
    assert not harness.received("oauth") and not harness.received("key")
    assert route_of(harness)["rule"] == "explore-local"
    # Zero-cost: recorded in tokens, no USD, no budget.
    totals = harness.state.budget_ledger.totals("ollama")
    assert totals.rows == 1 and totals.usd == 0.0

    # count_tokens for this rule → a local 404, no upstream contact.
    response = await harness.client.post(
        "/v1/messages/count_tokens", json=messages_body(), headers=headers
    )
    assert response.status_code == 404
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "not_found_error",
            "message": "llm-redact routing: upstream ollama does not implement count_tokens",
        },
    }
    assert len(harness.received("ollama")) == 1
    row = harness.state.recent[-1]
    assert row["status"] == 404 and row["route"]["class"] == "404"
    await harness.aclose()


# ---------------------------------------------------------------------------
# 6. OpenAI lane, 5xx chain: openai_key 503 → gemini compat → ollama.
# ---------------------------------------------------------------------------


async def test_walkthrough_6_openai_lane_5xx_chain(harness_factory: Any) -> None:
    harness = harness_factory(
        {
            "openai-key": Scenario(status=503),
            "gemini-compat": Scenario(status=400),  # rejects the gpt-* id
            "ollama-openai": Scenario(echo_model=True),
        }
    )
    body = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": f"mail {EMAIL}"}],
    }
    response = await harness.client.post(
        "/v1/chat/completions", json=body, headers={"authorization": "Bearer gateway-local"}
    )
    assert response.status_code == 200
    assert response.headers[UPSTREAM_HEADER] == "ollama_openai"
    assert response.headers[HOPS_HEADER] == "3"
    assert EMAIL in response.json()["choices"][0]["message"]["content"]
    [openai_seen] = harness.received("openai-key")
    [gemini_seen] = harness.received("gemini-compat")
    [ollama_seen] = harness.received("ollama-openai")
    # Each env: upstream got ITS key; the client's gateway token reached none.
    assert openai_seen.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert gemini_seen.headers["authorization"] == f"Bearer {GEMINI_KEY}"
    assert "authorization" not in ollama_seen.headers
    assert_no_secret_leak([openai_seen], (GEMINI_KEY, "gateway-local"))
    assert_no_secret_leak([gemini_seen], (OPENAI_KEY, "gateway-local"))
    # The model id is unchanged along the chain (no model_rewrite on the rule).
    assert gemini_seen.body["model"] == "gpt-5" and ollama_seen.body["model"] == "gpt-5"
    # Base paths folded per decision 6.
    assert gemini_seen.path == "/v1beta/openai/chat/completions"
    assert openai_seen.path == "/v1/chat/completions"
    # openai_key in cooldown (listed 5xx); gemini's unlisted 400 is not.
    assert not harness.state.routing_state.healthy("openai_key")
    assert harness.state.routing_state.healthy("gemini_openai_compat")
    status = (await harness.client.get("/__llm-redact/status")).json()["routing"]
    assert status["upstreams"]["openai_key"]["last_error_class"] == "503"
    assert route_of(harness) == {
        "rule": "openai-lane",
        "upstream": "ollama_openai",
        "hops": 3,
        "auth": "gateway-key",
        "class": "ok",
        "reissue": "yes",
    }
    await harness.aclose()


# ---------------------------------------------------------------------------
# 7. Budget exhausted: 402 direct, chains skip it, next month resets.
# ---------------------------------------------------------------------------


async def test_walkthrough_7_budget_exhausted(harness_factory: Any) -> None:
    harness = harness_factory(budget_chain=False, mutate=token_budget)
    # The ledger's clock is injectable: the same budgets, a steerable month.
    ledger_clock = [datetime(2026, 9, 21, 12, 0, tzinfo=UTC)]
    harness.state.budget_ledger = BudgetLedger(
        harness.state.spend_store,
        {"anthropic_key": Budget(usd=None, tokens=20)},
        reset_day=1,
        clock=lambda: ledger_clock[0],
    )
    body = messages_body()
    for _ in range(2):
        response = await harness.client.post("/v1/messages", json=body, headers=GATEWAY_HEADERS)
        assert response.status_code == 200
    assert len(harness.received("key")) == 2
    assert harness.state.budget_ledger.exhausted("anthropic_key")

    # Direct request to the exhausted primary (no on_budget_exhausted) → 402.
    response = await harness.client.post("/v1/messages", json=body, headers=GATEWAY_HEADERS)
    assert response.status_code == 402
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "billing_error",
            "message": (
                "llm-redact routing: upstream anthropic_key budget exhausted for this period"
            ),
        },
    }
    assert len(harness.received("key")) == 2  # no upstream contact
    assert route_of(harness)["class"] == "budget_exhausted"

    status = (await harness.client.get("/__llm-redact/status")).json()["routing"]
    key_status = status["upstreams"]["anthropic_key"]
    assert key_status["state"] == "budget_exhausted"
    assert key_status["spend"]["budget_tokens"] == 20
    assert key_status["spend"]["remaining_tokens"] == 0
    assert key_status["spend"]["in_tokens"] == 20 and key_status["spend"]["out_tokens"] == 10
    # The CLI report reads the same store.
    spent = report(
        harness.state.spend_store,
        month=None,
        budgets={"anthropic_key": Budget(usd=None, tokens=20)},
        reset_day=1,
        now=ledger_clock[0],
    )
    assert spent["upstreams"]["anthropic_key"]["exhausted"] is True
    assert spent["upstreams"]["anthropic_key"]["hops"]["primary"]["rows"] == 2

    # Chains skip the exhausted member: the max lane's plan-limit chain
    # [anthropic_key, ollama] lands on ollama.
    harness.scenarios["oauth"] = Scenario(status=429, plan_limit=True)
    response = await harness.client.post("/v1/messages", json=body, headers=OAUTH_HEADERS)
    assert response.status_code == 200
    assert response.headers[UPSTREAM_HEADER] == "ollama"
    assert len(harness.received("key")) == 2

    # Next month: the period rolls over and the budget resets.
    ledger_clock[0] = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
    assert not harness.state.budget_ledger.exhausted("anthropic_key")
    response = await harness.client.post("/v1/messages", json=body, headers=GATEWAY_HEADERS)
    assert response.status_code == 200 and len(harness.received("key")) == 3
    await harness.aclose()


async def test_walkthrough_7b_budget_chain_continues(harness_factory: Any) -> None:
    harness = harness_factory(budget_chain=True, mutate=token_budget)
    body = messages_body()
    for _ in range(2):
        assert (
            await harness.client.post("/v1/messages", json=body, headers=GATEWAY_HEADERS)
        ).status_code == 200
    response = await harness.client.post("/v1/messages", json=body, headers=GATEWAY_HEADERS)
    assert response.status_code == 200
    assert response.headers[UPSTREAM_HEADER] == "ollama"
    assert response.headers[HOPS_HEADER] == "2"
    assert len(harness.received("key")) == 2 and len(harness.received("ollama")) == 1
    assert route_of(harness)["reissue"] == "yes"
    assert harness.state.metrics.reissues[("anthropic_key", "ollama")] == 1
    await harness.aclose()


# ---------------------------------------------------------------------------
# 8. Invariant violations fail serve --check with a specific message each.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            "chain_to_second_passthrough",
            "member 'anthropic_oauth2' is a passthrough upstream",
        ),
        (
            "inject_note_on_passthrough",
            "inject_system_note = true is not allowed on a passthrough upstream",
        ),
        ("budget_on_passthrough", r"monthly_budget_\* is not allowed on a passthrough upstream"),
    ],
)
def test_walkthrough_8_invariant_violations_fail_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    expected: str,
) -> None:
    from llm_redact.cli import main

    set_env_keys(monkeypatch)
    raw = tomllib.loads(spec_toml(tmp_path / "vault.db"))
    if mutation == "chain_to_second_passthrough":
        raw["upstreams"]["anthropic_oauth2"] = {
            "protocol": "anthropic",
            "base_url": "http://oauth2",
            "credential": "passthrough",
        }
        raw["routing"]["rule"][1]["on_status"]["529"] = ["anthropic_oauth2"]
    elif mutation == "inject_note_on_passthrough":
        raw["upstreams"]["anthropic_oauth"]["inject_system_note"] = True
    else:
        raw["upstreams"]["anthropic_oauth"]["monthly_budget_usd"] = 10
    with pytest.raises(ConfigError, match=expected):
        parse_config(raw, "<walkthrough>")
    # serve --check runs the same parse and fails closed.
    text = spec_toml(tmp_path / "vault.db")
    if mutation == "inject_note_on_passthrough":
        text = text.replace("inject_system_note = false", "inject_system_note = true")
    elif mutation == "budget_on_passthrough":
        text = text.replace(
            "[upstreams.anthropic_oauth]\n",
            "[upstreams.anthropic_oauth]\nmonthly_budget_usd = 10\n",
        )
    else:
        text = text.replace(
            "[upstreams.anthropic_oauth]\n",
            '[upstreams.anthropic_oauth2]\nprotocol = "anthropic"\nbase_url = "http://oauth2"\n'
            'credential = "passthrough"\n\n[upstreams.anthropic_oauth]\n',
        ).replace('529 = ["anthropic_key"]', '529 = ["anthropic_oauth2"]')
    config_file = tmp_path / "config.toml"
    config_file.write_text(text)
    with pytest.raises(SystemExit) as excinfo:
        main(["serve", "--check", "--config", str(config_file)])
    assert excinfo.value.code == 1
    assert re.search(expected, capsys.readouterr().err)


# ---------------------------------------------------------------------------
# 9. Legacy config: only [providers.*] — byte-identical, routing disabled.
# ---------------------------------------------------------------------------


async def test_walkthrough_9_legacy_config_unchanged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from llm_redact.config import ProviderConfig

    config = Config(providers={**Config().providers, "anthropic": ProviderConfig("http://legacy")})
    harness = Harness(config, {"legacy": Scenario()})
    assert harness.state.config.routing.enabled is False
    original = b'{"messages" :[{"role":"user","content":"hello there"}],   "model":"m"}'
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages",
            content=original,
            headers={**OAUTH_HEADERS, "content-type": "application/json"},
        )
    assert response.status_code == 200
    [seen] = harness.received("legacy")
    # Forwarded exactly as before: same bytes, same headers, no routing headers.
    assert seen.body == json.loads(original)
    assert seen.headers["authorization"] == OAUTH_HEADERS["authorization"]
    assert UPSTREAM_HEADER not in response.headers and HOPS_HEADER not in response.headers
    assert "rule=" not in caplog.text and "POST /v1/messages -> 200" in caplog.text
    assert harness.state.recent[-1]["route"] is None
    status = (await harness.client.get("/__llm-redact/status")).json()
    assert status["routing"] == {"enabled": False}
    await harness.aclose()


# ---------------------------------------------------------------------------
# 10. Mid-stream failure: propagated as-is, no re-issue, class=stream_error.
# ---------------------------------------------------------------------------


class AbortingTransport(httpx.AsyncBaseTransport):
    """Wraps the fake app's ASGITransport (which buffers whole bodies) and,
    when the fake marks a response `x-fake-abort-after-first-event`, hands
    the proxy a body that yields what the fake produced and then drops the
    connection — a real mid-stream fault."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if response.headers.get(fake_upstream.ABORT_HEADER) != "1":
            return response
        produced = await response.aread()

        async def dropping() -> Any:
            yield produced
            raise httpx.ReadError("connection dropped mid-stream")

        return httpx.Response(
            response.status_code, headers=response.headers, content=dropping(), request=request
        )


async def test_walkthrough_10_mid_stream_failure_is_not_reissued(
    harness_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    scenarios = {"oauth": Scenario(abort_mid_stream=True)}
    fake = fake_upstream.build_app(scenarios)
    harness = harness_factory(
        scenarios,
        upstream_transport=AbortingTransport(httpx.ASGITransport(app=fake)),
        raise_app_exceptions=False,
    )
    with caplog.at_level(logging.INFO, logger="llm_redact"):
        response = await harness.client.post(
            "/v1/messages", json=messages_body(stream=True), headers=OAUTH_HEADERS
        )
    # The client saw the truncated stream: the first content event arrived,
    # the terminating events never did.
    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "event: content_block_delta" in response.text
    assert "message_stop" not in response.text
    # No re-issue after the first byte (R-17): only the oauth upstream was contacted.
    assert len(scenarios["oauth"].received) == 1
    assert "key" not in scenarios and "ollama" not in scenarios
    assert "class=stream_error" in caplog.text
    row = harness.state.recent[-1]
    assert row["streamed"] is True and row["route"]["class"] == "stream_error"
    assert row["route"]["upstream"] == "anthropic_oauth" and row["route"]["reissue"] == "no"
    await harness.aclose()


# ---------------------------------------------------------------------------
# The walkthrough config itself: no fake key value ever appears in what a
# passthrough upstream receives (I-3), pinned across every lane above.
# ---------------------------------------------------------------------------


def test_spec_config_shape(tmp_path: Path) -> None:
    config = spec_config(tmp_path / "vault.db")
    routing = config.routing
    assert routing.enabled and routing.present
    assert dict(routing.default_upstreams) == {"anthropic": "ollama", "openai": "ollama_openai"}
    assert [rule.id for rule in routing.rules] == [
        "explore-local",
        "max-lane",
        "anthropic-key-lane",
        "openai-lane",
        "gemini-lane",
        "local-openai",
    ]
    assert routing.upstream("ollama").count_tokens is False
    assert dataclasses.asdict(routing.upstream("anthropic_oauth"))["inject_system_note"] is False
