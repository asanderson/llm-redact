"""Pure routing logic: auth classification, rule selection, chain precedence,
status classification, statelessness, headers/URL/body per upstream, model
restore, and the in-process RoutingState (docs/routing.md)."""

import json
from typing import Any

import pytest

from llm_redact.routing import (
    DEFAULT_PLAN_LIMIT_HEADERS,
    OAUTH_BETA_MARKER,
    PROTOCOLS,
    RETRY_SAME,
    MissingCredential,
    RouteRule,
    RoutingConfig,
    RoutingState,
    RuleMatch,
    UpstreamConfig,
    apply_body_rewrites,
    beta_has_marker,
    classify_auth,
    is_plan_limit,
    is_stateful_request,
    literal_models,
    outbound_headers,
    parse_retry_after,
    request_protocol,
    restore_model,
    restore_model_in_json_text,
    select_rule,
    status_key,
    strip_beta_marker,
    upstream_url,
)

OAUTH = UpstreamConfig(name="oauth", protocol="anthropic", base_url="http://oauth.example")
KEY = UpstreamConfig(
    name="key",
    protocol="anthropic",
    base_url="http://key.example",
    credential="env:ANTHROPIC_API_KEY",
    monthly_budget_usd=10.0,
)
OLLAMA = UpstreamConfig(
    name="ollama",
    protocol="anthropic",
    base_url="http://ollama.example:11434",
    credential="none",
    cost="zero",
    count_tokens=False,
)
OPENAI_KEY = UpstreamConfig(
    name="openai_key",
    protocol="openai",
    base_url="https://api.openai.com/v1",
    credential="env:OPENAI_API_KEY",
    extra_headers=(("http-referer", "http://localhost"), ("x-title", "llm-redact")),
    body_defaults_json=json.dumps({"provider": {"only": ["openai"]}}, sort_keys=True),
)
GEMINI_KEY = UpstreamConfig(
    name="gemini_key",
    protocol="gemini",
    base_url="https://generativelanguage.googleapis.com",
    credential="env:GEMINI_API_KEY",
)
OLLAMA_NATIVE = UpstreamConfig(
    name="ollama_native",
    protocol="ollama",
    base_url="http://127.0.0.1:11434",
    credential="env:OLLAMA_KEY",
)


def _rule(rule_id: str, **kwargs: Any) -> RouteRule:
    match = kwargs.pop("match")
    return RouteRule(id=rule_id, match=match, **kwargs)


CONFIG = RoutingConfig(
    enabled=True,
    present=True,
    default_upstreams=(("anthropic", "ollama"),),
    upstreams=(KEY, OLLAMA, OAUTH, OPENAI_KEY),
    model_catalog=("claude-sonnet-5",),
    rules=(
        _rule(
            "explore-local",
            match=RuleMatch(protocol="anthropic", headers=(("x-claude-code-agent-id", "Explore"),)),
            upstream="ollama",
            model_rewrite="muse-64k",
        ),
        _rule(
            "max-lane",
            match=RuleMatch(protocol="anthropic", models=("claude-*",), auth="oauth"),
            upstream="oauth",
            on_status=(
                ("plan_limit_429", ("key", "ollama")),
                ("529", ("key",)),
                ("throttle_429", RETRY_SAME),
                ("5xx", ("ollama",)),
            ),
        ),
        _rule(
            "key-lane",
            match=RuleMatch(protocol="anthropic", models=("claude-*",), auth="gateway-key"),
            upstream="key",
            on_status=(("429", ("ollama",)), ("4xx", ("ollama",))),
            reissue_policy="always",
        ),
        _rule(
            "count-only",
            match=RuleMatch(protocol="anthropic", path="/v1/messages/count_tokens"),
            upstream="key",
        ),
        _rule(
            "openai-lane",
            match=RuleMatch(
                protocol="openai", models=("gpt-*", "o3", "gpt-oss*", "claude-sonnet-5")
            ),
            upstream="openai_key",
        ),
    ),
)


# --- request_protocol ---------------------------------------------------------


@pytest.mark.parametrize("name", PROTOCOLS)
def test_request_protocol_adapter_family(name: str) -> None:
    assert request_protocol(name, "anthropic") == name


@pytest.mark.parametrize("name", ["azure", "vertex", "bedrock", "cohere", "custom:vllm"])
def test_request_protocol_other_adapters_are_legacy(name: str) -> None:
    # The adapter decides; an inferred provider never overrides a matched one.
    assert request_protocol(name, "anthropic") is None


def test_request_protocol_passthrough_uses_inference() -> None:
    assert request_protocol(None, "openai") == "openai"
    assert request_protocol(None, "vertex") is None
    assert request_protocol(None, "custom:x") is None


# --- auth classification (R-6) -------------------------------------------------


def test_classify_auth_oauth_needs_bearer_and_marker() -> None:
    headers = {"Authorization": "Bearer tok", "Anthropic-Beta": "oauth-2025-04-20"}
    assert classify_auth(headers, OAUTH_BETA_MARKER) == "oauth"


def test_classify_auth_marker_case_and_whitespace() -> None:
    headers = {
        "authorization": "bearer tok",
        "anthropic-beta": "prompt-caching-2024-07-31 ,  OAUTH-2025-04-20 , fine-grained",
    }
    assert classify_auth(headers, OAUTH_BETA_MARKER) == "oauth"
    assert classify_auth(headers, "other-marker") == "gateway-key"


def test_classify_auth_gateway_key_forms() -> None:
    assert classify_auth({"x-api-key": "sk-ant-not-real"}, OAUTH_BETA_MARKER) == "gateway-key"
    assert classify_auth({"authorization": "Bearer gateway-local"}, OAUTH_BETA_MARKER) == (
        "gateway-key"
    )
    # Bearer without the marker is a gateway key; marker without Bearer too.
    assert classify_auth(
        {"authorization": "Basic abc", "anthropic-beta": OAUTH_BETA_MARKER}, OAUTH_BETA_MARKER
    ) == ("gateway-key")
    assert classify_auth({"x-api-key": "k", "anthropic-beta": OAUTH_BETA_MARKER}, "x") == (
        "gateway-key"
    )


def test_classify_auth_none() -> None:
    assert classify_auth({}, OAUTH_BETA_MARKER) == "none"
    assert classify_auth({"authorization": "   ", "x-api-key": ""}, OAUTH_BETA_MARKER) == "none"
    # "Bearer" with nothing after it is present (gateway-key), never oauth.
    assert classify_auth(
        {"authorization": "Bearer ", "anthropic-beta": OAUTH_BETA_MARKER}, OAUTH_BETA_MARKER
    ) == ("gateway-key")


def test_beta_has_marker() -> None:
    assert beta_has_marker("a, oauth-2025-04-20", OAUTH_BETA_MARKER)
    assert not beta_has_marker(None, OAUTH_BETA_MARKER)
    assert not beta_has_marker("", OAUTH_BETA_MARKER)
    assert not beta_has_marker("oauth-2025-04-20x", OAUTH_BETA_MARKER)


def test_strip_beta_marker() -> None:
    assert strip_beta_marker("oauth-2025-04-20", OAUTH_BETA_MARKER) is None
    assert strip_beta_marker(" OAUTH-2025-04-20 ,", OAUTH_BETA_MARKER) is None
    assert strip_beta_marker("a,oauth-2025-04-20, b", OAUTH_BETA_MARKER) == "a, b"
    assert strip_beta_marker("a, b", OAUTH_BETA_MARKER) == "a, b"


# --- select_rule (R-5, R-10) ---------------------------------------------------


def _select(**kwargs: Any) -> str | None:
    defaults: dict[str, Any] = {
        "protocol": "anthropic",
        "model": "claude-sonnet-5",
        "headers": {},
        "path": "/v1/messages",
        "auth": "oauth",
    }
    defaults.update(kwargs)
    rule = select_rule(CONFIG, **defaults)
    return rule.id if rule is not None else None


def test_select_rule_first_match_wins() -> None:
    # explore-local precedes max-lane: the header wins even for OAuth Claude.
    assert _select(headers={"X-Claude-Code-Agent-Id": "Explore"}) == "explore-local"
    assert _select() == "max-lane"


def test_select_rule_and_semantics_and_auth() -> None:
    assert _select(auth="gateway-key") == "key-lane"
    assert _select(auth="none") is None  # only auth=any rules would match
    assert _select(model="gpt-5") is None  # protocol anthropic, model glob fails
    assert _select(model=None) is None  # a rule constraining model needs one


def test_select_rule_auth_any_query_is_wildcard() -> None:
    # `routes test --auth any`: the first rule whatever its auth constraint.
    assert _select(auth="any") == "max-lane"


def test_select_rule_glob_and_path() -> None:
    assert _select(model="claude-opus-5-20260101", auth="gateway-key") == "key-lane"
    assert _select(model="gpt-5", protocol="openai") == "openai-lane"
    assert _select(model="gpt-oss:20b", protocol="openai") == "openai-lane"
    assert _select(model="gemini-2.5", protocol="openai") is None
    assert _select(model="llama", path="/v1/messages/count_tokens", auth="none") == "count-only"


def test_select_rule_header_names_case_insensitive_values_exact() -> None:
    assert _select(headers={"x-claude-code-agent-id": "Explore"}) == "explore-local"
    assert _select(headers={"X-CLAUDE-CODE-AGENT-ID": "Explore"}) == "explore-local"
    assert _select(headers={"x-claude-code-agent-id": "explore"}) == "max-lane"
    assert _select(headers={"x-other": "Explore"}) == "max-lane"


def test_select_rule_header_glob() -> None:
    config = RoutingConfig(
        upstreams=(OAUTH,),
        rules=(
            _rule(
                "ua",
                match=RuleMatch(protocol="anthropic", headers=(("user-agent", "claude-cli/*"),)),
                upstream="oauth",
            ),
        ),
    )
    rule = select_rule(
        config,
        protocol="anthropic",
        model=None,
        headers={"User-Agent": "claude-cli/2.0"},
        path="/v1/messages",
        auth="none",
    )
    assert rule is not None and rule.id == "ua"
    assert (
        select_rule(
            config,
            protocol="anthropic",
            model=None,
            headers={},
            path="/v1/messages",
            auth="none",
        )
        is None
    )


def test_select_rule_empty_config() -> None:
    assert (
        select_rule(
            RoutingConfig(), protocol="anthropic", model="x", headers={}, path="/", auth="none"
        )
        is None
    )


# --- chain precedence (R-7) ----------------------------------------------------


def test_chain_for_precedence_special_over_exact_over_class() -> None:
    max_lane = CONFIG.rules[1]
    assert max_lane.chain_for("plan_limit_429") == ("key", "ollama")
    assert max_lane.chain_for("throttle_429") == RETRY_SAME
    assert max_lane.chain_for("529") == ("key",)
    assert max_lane.chain_for("503") == ("ollama",)  # 5xx class
    assert max_lane.chain_for("502") == ("ollama",)  # transport faults classify as 502
    assert max_lane.chain_for("404") is None
    key_lane = CONFIG.rules[2]
    # A special key with no special entry falls back to the exact status,
    # then the class.
    assert key_lane.chain_for("plan_limit_429") == ("ollama",)
    assert key_lane.chain_for("throttle_429") == ("ollama",)
    assert key_lane.chain_for("403") == ("ollama",)
    assert key_lane.chain_for("500") is None
    assert key_lane.chain_for("transport") is None


# --- status_key (R-20) ---------------------------------------------------------


def test_status_key_plan_limit_vs_throttle() -> None:
    plan = {"Anthropic-Ratelimit-Unified-Status": "rejected", "retry-after": "60"}
    warn = {"anthropic-ratelimit-unified-status": "allowed_warning"}
    assert status_key(429, protocol="anthropic", response_headers=plan, config=CONFIG) == (
        "plan_limit_429"
    )
    assert status_key(429, protocol="anthropic", response_headers=warn, config=CONFIG) == (
        "throttle_429"
    )
    assert status_key(429, protocol="anthropic", response_headers={}, config=CONFIG) == (
        "throttle_429"
    )
    # 5h/7d window headers count on their own; values compare case-insensitively.
    assert is_plan_limit(
        {"anthropic-ratelimit-unified-7d-status": " REJECTED "}, DEFAULT_PLAN_LIMIT_HEADERS
    )
    assert not is_plan_limit({"anthropic-ratelimit-unified-7d-status": "allowed"}, ())


def test_status_key_detection_off_and_other_protocols() -> None:
    off = RoutingConfig(plan_limit_detection="off")
    plan = {"anthropic-ratelimit-unified-status": "rejected"}
    assert status_key(429, protocol="anthropic", response_headers=plan, config=off) == "429"
    assert status_key(429, protocol="openai", response_headers=plan, config=CONFIG) == "429"
    assert status_key(503, protocol="anthropic", response_headers=plan, config=CONFIG) == "503"
    assert status_key(200, protocol="anthropic", response_headers={}, config=CONFIG) == "200"


def test_status_key_custom_table() -> None:
    config = RoutingConfig(plan_limit_headers=(("x-quota", ("exhausted", "over")),))
    assert status_key(
        429, protocol="anthropic", response_headers={"X-Quota": "over"}, config=config
    ) == ("plan_limit_429")
    assert status_key(
        429,
        protocol="anthropic",
        response_headers={"anthropic-ratelimit-unified-status": "rejected"},
        config=config,
    ) == ("throttle_429")


# --- statelessness guard (R-22) -----------------------------------------------


def test_is_stateful_request() -> None:
    thinking = {"type": "thinking", "thinking": "…", "signature": "sig"}
    redacted = {"type": "redacted_thinking", "data": "opaque"}
    text = {"type": "text", "text": "hi"}
    assert is_stateful_request(
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [thinking]},
            ]
        }
    )
    assert is_stateful_request({"messages": [{"role": "assistant", "content": [text, redacted]}]})
    # String content, user-turn thinking blocks, and cache_control are not.
    assert not is_stateful_request({"messages": [{"role": "assistant", "content": "plain"}]})
    assert not is_stateful_request({"messages": [{"role": "user", "content": [thinking]}]})
    assert not is_stateful_request(
        {"messages": [{"role": "assistant", "content": [{**text, "cache_control": {}}]}]}
    )
    assert not is_stateful_request({"messages": [{"role": "assistant"}, "junk", 3]})
    assert not is_stateful_request({"messages": "nope"})
    assert not is_stateful_request(["not", "a", "dict"])
    assert not is_stateful_request({"input": "responses api"})


# --- parse_retry_after ---------------------------------------------------------


def test_parse_retry_after_int_and_date_and_garbage() -> None:
    assert parse_retry_after("120") == 120.0
    assert parse_retry_after(" 7 ") == 7.0
    assert parse_retry_after("0") == 0.0
    # HTTP-date relative to an injected `now`; the past clamps to 0.
    now = 1_699_999_980.0  # 2023-11-14T22:13:00Z
    assert parse_retry_after("Tue, 14 Nov 2023 22:14:00 GMT", now=now) == 60.0
    assert parse_retry_after("Tue, 14 Nov 2023 22:12:00 GMT", now=now) == 0.0
    assert parse_retry_after("Tue, 14 Nov 2023 22:14:00 -0000", now=now) == 60.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("-5") is None
    assert parse_retry_after("1.5") is None
    # Without `now` the wall clock applies; a far-future date stays positive.
    assert (parse_retry_after("Fri, 01 Jan 2100 00:00:00 GMT") or 0.0) > 0.0


# --- outbound_headers (R-13, R-14, I-3) ---------------------------------------

INBOUND: list[tuple[str, str]] = [
    ("host", "127.0.0.1:8787"),
    ("authorization", "Bearer client-oauth-token"),
    ("x-api-key", "client-key"),
    ("x-goog-api-key", "client-goog-key"),
    ("anthropic-beta", "prompt-caching-2024-07-31, oauth-2025-04-20"),
    ("anthropic-version", "2023-06-01"),
    ("x-stainless-lang", "js"),
    ("user-agent", "claude-cli/2.0"),
]
ENV = {
    "ANTHROPIC_API_KEY": "sk-ant-test-not-a-real-key",
    "OPENAI_API_KEY": "sk-test-not-real",
    "GEMINI_API_KEY": "AIzaTestNotReal",
    "OLLAMA_KEY": "ollama-local",
}


def test_outbound_headers_passthrough_is_identical() -> None:
    out = outbound_headers(INBOUND, OAUTH, environ=ENV, oauth_marker=OAUTH_BETA_MARKER)
    assert out == INBOUND
    assert out is not INBOUND  # a copy the caller may extend


def test_outbound_headers_env_anthropic() -> None:
    out = outbound_headers(INBOUND, KEY, environ=ENV, oauth_marker=OAUTH_BETA_MARKER)
    names = [name for name, _ in out]
    assert "authorization" not in names
    assert "x-goog-api-key" not in names
    assert names.count("x-api-key") == 1
    assert ("x-api-key", "sk-ant-test-not-a-real-key") in out
    assert ("anthropic-beta", "prompt-caching-2024-07-31") in out
    # Non-credential headers survive in order.
    assert ("anthropic-version", "2023-06-01") in out
    assert ("x-stainless-lang", "js") in out
    assert ("user-agent", "claude-cli/2.0") in out


def test_outbound_headers_env_openai_extra_headers_and_bearer() -> None:
    out = outbound_headers(INBOUND, OPENAI_KEY, environ=ENV, oauth_marker=OAUTH_BETA_MARKER)
    assert ("authorization", "Bearer sk-test-not-real") in out
    assert ("http-referer", "http://localhost") in out
    assert ("x-title", "llm-redact") in out
    assert ("x-api-key", "client-key") not in out
    # extra_headers precede the injected credential (last wins in httpx).
    assert out.index(("x-title", "llm-redact")) < out.index(
        ("authorization", "Bearer sk-test-not-real")
    )


def test_outbound_headers_env_gemini_and_ollama() -> None:
    out = outbound_headers(INBOUND, GEMINI_KEY, environ=ENV, oauth_marker=OAUTH_BETA_MARKER)
    assert ("x-goog-api-key", "AIzaTestNotReal") in out
    assert [n for n, _ in out].count("x-goog-api-key") == 1
    out = outbound_headers(INBOUND, OLLAMA_NATIVE, environ=ENV, oauth_marker=OAUTH_BETA_MARKER)
    assert ("authorization", "Bearer ollama-local") in out


def test_outbound_headers_none_drops_credentials_and_empty_beta() -> None:
    inbound = [*INBOUND[:4], ("anthropic-beta", "oauth-2025-04-20"), ("x-app", "cli")]
    out = outbound_headers(inbound, OLLAMA, environ={}, oauth_marker=OAUTH_BETA_MARKER)
    assert out == [("host", "127.0.0.1:8787"), ("x-app", "cli")]


def test_outbound_headers_missing_credential_names_var_only() -> None:
    with pytest.raises(MissingCredential) as excinfo:
        outbound_headers(INBOUND, KEY, environ={}, oauth_marker=OAUTH_BETA_MARKER)
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)
    assert "client" not in str(excinfo.value)
    with pytest.raises(MissingCredential):
        outbound_headers(
            INBOUND, KEY, environ={"ANTHROPIC_API_KEY": ""}, oauth_marker=OAUTH_BETA_MARKER
        )


# --- upstream_url (decision 6) -------------------------------------------------


def test_upstream_url_folds_v1_for_openai_only() -> None:
    assert (
        upstream_url(OPENAI_KEY, "/v1/chat/completions", "")
        == "https://api.openai.com/v1/chat/completions"
    )
    assert upstream_url(OPENAI_KEY, "/v1", "") == "https://api.openai.com/v1"
    assert upstream_url(OPENAI_KEY, "/v1/models", "a=1&b=2") == (
        "https://api.openai.com/v1/models?a=1&b=2"
    )
    no_v1 = UpstreamConfig(name="o", protocol="openai", base_url="http://127.0.0.1:11434")
    assert upstream_url(no_v1, "/v1/chat/completions", "") == (
        "http://127.0.0.1:11434/v1/chat/completions"
    )
    # An anthropic-protocol base ending in /v1 is left alone (no folding).
    anth = UpstreamConfig(name="a", protocol="anthropic", base_url="http://h.example/v1")
    assert upstream_url(anth, "/v1/messages", "") == "http://h.example/v1/v1/messages"
    assert upstream_url(OLLAMA, "/v1/messages", "beta=true") == (
        "http://ollama.example:11434/v1/messages?beta=true"
    )


# --- apply_body_rewrites (R-9, R-24, decision 13) ------------------------------

RULE_REWRITE = CONFIG.rules[0]  # model_rewrite = muse-64k
RULE_PLAIN = CONFIG.rules[2]


def test_apply_body_rewrites_model_rewrite_never_mutates() -> None:
    body = {"model": "claude-sonnet-5", "messages": [], "stream": True}
    out = apply_body_rewrites(body, upstream=OLLAMA, rule=RULE_REWRITE)
    assert out == {"model": "muse-64k", "messages": [], "stream": True}
    assert body["model"] == "claude-sonnet-5"
    # Already the target: nothing changes.
    assert apply_body_rewrites({"model": "muse-64k"}, upstream=OLLAMA, rule=RULE_REWRITE) is None
    # Model rewrite applies on passthrough too (it is not a system mutation).
    assert apply_body_rewrites({"model": "x"}, upstream=OAUTH, rule=RULE_REWRITE) == {
        "model": "muse-64k"
    }


def test_apply_body_rewrites_none_when_unchanged() -> None:
    assert apply_body_rewrites({"model": "claude"}, upstream=KEY, rule=RULE_PLAIN) is None
    assert apply_body_rewrites(
        {"model": "gpt", "stream": False}, upstream=OPENAI_KEY, rule=RULE_PLAIN
    ) == {
        "model": "gpt",
        "stream": False,
        "provider": {"only": ["openai"]},
    }


def test_apply_body_rewrites_body_defaults_absent_only() -> None:
    body: dict[str, Any] = {"model": "gpt", "provider": {"only": ["azure"]}}
    assert apply_body_rewrites(body, upstream=OPENAI_KEY, rule=RULE_PLAIN) is None
    body = {"model": "gpt", "provider": None}  # present (null) is still present
    assert apply_body_rewrites(body, upstream=OPENAI_KEY, rule=RULE_PLAIN) is None
    out = apply_body_rewrites({"model": "gpt"}, upstream=OPENAI_KEY, rule=RULE_PLAIN)
    assert out is not None
    # The merged value is a fresh object, never the upstream's cached one.
    out["provider"]["only"].append("mutated")
    assert OPENAI_KEY.body_defaults() == {"provider": {"only": ["openai"]}}


def test_apply_body_rewrites_include_usage_openai_stream_only() -> None:
    body = {"model": "gpt", "stream": True, "provider": {}}
    out = apply_body_rewrites(body, upstream=OPENAI_KEY, rule=RULE_PLAIN)
    assert out == {**body, "stream_options": {"include_usage": True}}
    assert "stream_options" not in body
    # Present options gain the key without losing others; explicit false stays.
    body = {"model": "gpt", "stream": True, "provider": {}, "stream_options": {"x": 1}}
    out = apply_body_rewrites(body, upstream=OPENAI_KEY, rule=RULE_PLAIN)
    assert out is not None and out["stream_options"] == {"x": 1, "include_usage": True}
    assert body["stream_options"] == {"x": 1}
    body = {
        "model": "gpt",
        "stream": True,
        "provider": {},
        "stream_options": {"include_usage": False},
    }
    assert apply_body_rewrites(body, upstream=OPENAI_KEY, rule=RULE_PLAIN) is None
    # A non-table stream_options is replaced.
    body = {"model": "gpt", "stream": True, "provider": {}, "stream_options": None}
    out = apply_body_rewrites(body, upstream=OPENAI_KEY, rule=RULE_PLAIN)
    assert out is not None and out["stream_options"] == {"include_usage": True}
    # Not for stream: "true" (string), not for anthropic, never on passthrough.
    assert (
        apply_body_rewrites(
            {"stream": "true", "provider": {}}, upstream=OPENAI_KEY, rule=RULE_PLAIN
        )
        is None
    )
    assert apply_body_rewrites({"stream": True}, upstream=KEY, rule=RULE_PLAIN) is None
    openai_pass = UpstreamConfig(name="p", protocol="openai", base_url="http://p.example")
    assert apply_body_rewrites({"stream": True}, upstream=openai_pass, rule=RULE_PLAIN) is None


# --- restore_model (R-9) ------------------------------------------------------


def test_restore_model_shapes() -> None:
    start = {"type": "message_start", "message": {"id": "m", "model": "muse-64k", "usage": {}}}
    assert restore_model(start, "claude-sonnet-5")
    assert start["message"]["model"] == "claude-sonnet-5"
    top = {"id": "m", "model": "muse-64k", "role": "assistant"}
    assert restore_model(top, "claude-sonnet-5") and top["model"] == "claude-sonnet-5"
    chunk = {"object": "chat.completion.chunk", "model": "gemma", "choices": []}
    assert restore_model(chunk, "gpt-5") and chunk["model"] == "gpt-5"
    line = {"model": "muse", "message": {"role": "assistant", "content": "x"}, "done": True}
    assert restore_model(line, "claude-haiku-4-5") and line["model"] == "claude-haiku-4-5"
    assert line["message"] == {"role": "assistant", "content": "x"}
    # Unchanged / absent / non-string / non-dict.
    assert not restore_model({"model": "gpt-5"}, "gpt-5")
    assert not restore_model({"type": "ping"}, "gpt-5")
    assert not restore_model({"model": None, "message": {"model": 3}}, "gpt-5")
    assert not restore_model(["model"], "gpt-5")


def test_restore_model_in_json_text() -> None:
    text = '{"type":"message_start","message":{"model":"muse-64k","content":[]}}'
    out = restore_model_in_json_text(text, "claude-sonnet-5")
    assert json.loads(out) == {
        "type": "message_start",
        "message": {"model": "claude-sonnet-5", "content": []},
    }
    # Unparsable, non-object, and unchanged inputs come back byte-identical.
    assert restore_model_in_json_text("[DONE]", "m") == "[DONE]"
    assert restore_model_in_json_text("", "m") == ""
    assert restore_model_in_json_text("[1, 2]", "m") == "[1, 2]"
    assert restore_model_in_json_text('{"model":"m"}', "m") == '{"model":"m"}'
    # Non-ASCII survives (ensure_ascii=False, like the adapters).
    out = restore_model_in_json_text('{"model":"x","text":"żółć"}', "m")
    assert out == '{"model": "m", "text": "żółć"}'


# --- literal_models (R-15) ---------------------------------------------------


def test_literal_models_catalog_then_glob_free_rule_models_deduped() -> None:
    # catalog first; "claude-*", "gpt-*", "gpt-oss*" are globs; "o3" literal;
    # "claude-sonnet-5" appears in both and is kept once.
    assert literal_models(CONFIG) == ["claude-sonnet-5", "o3"]
    assert literal_models(RoutingConfig()) == []
    config = RoutingConfig(
        model_catalog=("b", "a"),
        rules=(
            _rule("r", match=RuleMatch(protocol="openai", models=("a", "c", "d?")), upstream="x"),
        ),
    )
    assert literal_models(config) == ["b", "a", "c"]


# --- RoutingConfig helpers -----------------------------------------------------


def test_routing_config_lookups() -> None:
    assert CONFIG.upstream("key") is KEY
    with pytest.raises(KeyError):
        CONFIG.upstream("nope")
    assert CONFIG.upstream_names() == ("key", "ollama", "oauth", "openai_key")
    assert CONFIG.default_for("anthropic") == "ollama"
    assert CONFIG.default_for("openai") is None


def test_upstream_config_properties() -> None:
    assert (OAUTH.credential_mode, OAUTH.env_var, OAUTH.is_passthrough) == (
        "passthrough",
        None,
        True,
    )
    assert (KEY.credential_mode, KEY.env_var, KEY.is_passthrough) == (
        "env",
        "ANTHROPIC_API_KEY",
        False,
    )
    assert (OLLAMA.credential_mode, OLLAMA.zero_cost, OLLAMA.has_budget) == ("none", True, False)
    assert KEY.has_budget and not OAUTH.has_budget and not OPENAI_KEY.has_budget
    tokens_only = UpstreamConfig(
        name="t", protocol="openai", base_url="http://t.example", monthly_budget_tokens=5
    )
    assert tokens_only.has_budget
    zero_with_budget = UpstreamConfig(
        name="z", protocol="openai", base_url="http://z", cost="zero", monthly_budget_usd=1.0
    )
    assert not zero_with_budget.has_budget
    assert OAUTH.body_defaults() == {}
    assert (
        UpstreamConfig(
            name="j", protocol="openai", base_url="http://j", body_defaults_json="[]"
        ).body_defaults()
        == {}
    )


# --- RoutingState (R-19, R-30) -----------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_routing_state_cooldown_with_injected_clock() -> None:
    clock = _Clock()
    state = RoutingState(clock, wall_clock=lambda: 1_700_000_000.0)
    assert state.healthy("oauth") and state.cooldown_remaining("oauth") == 0.0
    state.mark_unhealthy("oauth", 60.0, "plan_limit_429")
    assert not state.healthy("oauth")
    assert state.cooldown_remaining("oauth") == 60.0
    clock.now += 59.5
    assert state.cooldown_remaining("oauth") == 0.5
    clock.now += 0.5
    assert state.healthy("oauth")
    snap = state.snapshot("oauth")
    assert snap["state"] == "healthy"
    assert snap["last_error_class"] == "plan_limit_429"
    assert snap["last_error_at"] == "2023-11-14T22:13:20+00:00"
    # The cap and the floor both apply.
    state.mark_unhealthy("oauth", 99999.0, "503")
    assert state.cooldown_remaining("oauth") == 3600.0
    state.mark_unhealthy("oauth", -5.0, "503")
    assert state.healthy("oauth")


def test_routing_state_counters_reissue_window_and_snapshot() -> None:
    clock = _Clock()
    state = RoutingState(clock)
    state.record_request("oauth")
    state.record_request("oauth")
    state.record_request("key")
    state.record_reissue("oauth", "key")
    clock.now += 1800.0
    state.record_reissue("oauth", "ollama")
    assert state.reissues_last_hour() == 2
    assert state.reissues_last_hour("oauth") == 2
    assert state.reissues_last_hour("key") == 0  # counts the FROM side only
    clock.now += 1800.0  # first reissue is now exactly one hour old: expired
    assert state.reissues_last_hour() == 1
    snap = state.snapshot("oauth")
    assert snap == {
        "state": "healthy",
        "cooldown_remaining_seconds": 0.0,
        "requests": 2,
        "reissues_last_hour": 1,
        "last_error_class": None,
        "last_error_at": None,
    }
    state.mark_unhealthy("key", 10.0, "429")
    assert state.snapshot("key")["state"] == "cooldown"
    assert state.snapshot("key")["cooldown_remaining_seconds"] == 10.0
    assert state.snapshot("never-seen")["requests"] == 0


def test_routing_state_prune_to_drops_removed_names() -> None:
    clock = _Clock()
    state = RoutingState(clock)
    state.record_request("gone")
    state.mark_unhealthy("gone", 30.0, "503")
    state.record_request("kept")
    state.record_reissue("gone", "kept")
    state.record_reissue("kept", "kept")
    state.prune_to(["kept", "new"])
    assert state.snapshot("gone") == state.snapshot("never-seen")
    assert state.snapshot("kept")["requests"] == 1
    assert state.reissues_last_hour() == 1


def test_routing_state_default_clocks() -> None:
    state = RoutingState()
    state.mark_unhealthy("x", 5.0, "503")
    assert 0.0 < state.cooldown_remaining("x") <= 5.0
    assert state.snapshot("x")["last_error_at"] is not None
