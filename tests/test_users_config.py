"""``[users] unrecorded_objects``: the shape of llm-redact-pro's opt-out from
its unknown-owner refusal (the core parses, validates and emits it; the
implementation — and every decision it drives — is llm-redact-pro's).

"refuse" (the default): under a credential the proxy holds, a named user's
reference to a stored object NO user is recorded creating is refused.
"allow": it is forwarded, and whatever it reads is served sealed. Like the
rest of ``[users]`` it is restart-only, and without llm-redact-pro a
non-default ``[users]`` fails closed.
"""

from __future__ import annotations

import tomllib

import pytest

from license_fixtures import FREE
from llm_redact.config import (
    RESTART_ONLY_KEYS,
    UNRECORDED_OBJECT_POLICIES,
    Config,
    ConfigError,
    UsersConfig,
    parse_config,
)
from llm_redact.config_write import emit_config_toml
from llm_redact.registry import Registry


def test_the_default_is_refuse() -> None:
    assert UsersConfig().unrecorded_objects == "refuse"
    assert UNRECORDED_OBJECT_POLICIES == ("refuse", "allow")
    assert parse_config({}, "c.toml").users == UsersConfig()
    parsed = parse_config({"users": {"unrecorded_objects": "refuse"}}, "c.toml")
    assert parsed.users == UsersConfig()


def test_allow_is_parsed() -> None:
    raw = {"users": {"path": "/srv/users.db", "unrecorded_objects": "allow"}}
    users = parse_config(raw, "c.toml").users
    assert users == UsersConfig(path="/srv/users.db", unrecorded_objects="allow")


@pytest.mark.parametrize("value", ["Allow", "deny", "", 1, True, ["allow"]])
def test_anything_else_is_refused(value: object) -> None:
    with pytest.raises(ConfigError, match=r"\[users\] unrecorded_objects must be"):
        parse_config({"users": {"unrecorded_objects": value}}, "c.toml")


def test_an_unknown_users_key_is_still_refused() -> None:
    with pytest.raises(ConfigError, match=r"\[users\]"):
        parse_config({"users": {"unrecorded": "allow"}}, "c.toml")


@pytest.mark.parametrize(
    "users",
    [
        UsersConfig(unrecorded_objects="allow"),
        UsersConfig(path="/srv/users.db", unrecorded_objects="allow"),
        UsersConfig(path="/srv/users.db"),
    ],
)
def test_the_setting_round_trips(users: UsersConfig) -> None:
    text = emit_config_toml(Config(users=users))
    assert parse_config(tomllib.loads(text), "round-trip").users == users
    assert ("unrecorded_objects" in text) is (users.unrecorded_objects != "refuse")


def test_it_is_restart_only_like_the_rest_of_users() -> None:
    assert "users" in RESTART_ONLY_KEYS


def test_without_llm_redact_pro_a_non_default_users_section_fails_closed() -> None:
    reg = Registry()
    with pytest.raises(ConfigError, match="llm-redact-pro"):
        reg.build_access_gate(Config(users=UsersConfig(unrecorded_objects="allow")), FREE)
    assert reg.build_access_gate(Config(users=UsersConfig()), FREE) is None
