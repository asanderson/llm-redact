"""Free-tier session routing: the static router and the fail-closed factory.

The Free core serves one shared placeholder namespace. Per-conversation
routing is a paid feature (llm_redact_pro.sessions, covered by
tests/pro/test_sessions_pro.py); here we pin the Free surface — the static router
always resolves to the configured session, its inert stubs, and the
fail-closed ``build_session_router`` that refuses per-conversation without the
pro package rather than silently downgrading to the shared namespace.
"""

import pytest

from llm_redact.config import ConfigError, VaultConfig
from llm_redact.sessions import StaticSessionRouter, build_session_router


def test_static_router_mode_and_always_fallback() -> None:
    router = StaticSessionRouter("default")
    assert router.mode == "static"
    body = {"messages": [{"role": "user", "content": "anything"}]}
    # Every shape resolves to the one configured session — no anchor hashing.
    assert router.resolve("anthropic", "POST", "/v1/messages", body) == "default"
    assert router.resolve("openai", "POST", "/v1/responses", {"input": "x"}) == "default"
    assert router.resolve("openai", "GET", "/v1/responses/resp_1", {}) == "default"
    assert router.resolve(None, "POST", "/v1/unknown", {}) == "default"


def test_static_router_record_response_id_is_inert() -> None:
    # The proxy never calls record_response_id in static mode; the stub exists
    # only to satisfy the plugin_api.SessionRouter contract and must be a no-op.
    router = StaticSessionRouter("default")
    assert router.record_response_id("resp_1", "conv-x") is None
    # Recording changed nothing: resolution is still the static fallback.
    assert router.resolve("openai", "POST", "/v1/responses", {"input": "x"}) == "default"


def test_build_session_router_static_returns_static_router() -> None:
    router = build_session_router(VaultConfig(session_mode="static", session="acct"))
    assert isinstance(router, StaticSessionRouter)
    assert router.mode == "static"
    assert router.resolve("anthropic", "POST", "/v1/messages", {}) == "acct"


def test_build_session_router_per_conversation_fails_closed_without_pro() -> None:
    # The Free default fails closed — never a silent downgrade to the shared
    # static namespace (which would put every conversation's secrets in one
    # namespace, the opposite of the isolation per-conversation promises). The
    # installed pro package overrides this via the registry; here we call the
    # Free factory directly to pin its fail-closed contract.
    cfg = VaultConfig(session_mode="per-conversation", session="default")
    with pytest.raises(ConfigError, match="requires the llm-redact-pro package"):
        build_session_router(cfg)
