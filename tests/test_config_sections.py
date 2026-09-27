"""The config seam (plugin_api.ConfigSection): a plugin claims its own
top-level tables; the core parses, emits and pins them restart-only, and
keeps rejecting every section nobody claimed.

Keyless: a scripted fake section on a bare Registry stands in for
llm-redact-pro's [auth].
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

import llm_redact.registry as registry_mod
from llm_redact.config import Config, ConfigError, parse_config
from llm_redact.config_write import emit_config_toml
from llm_redact.proxy import ProxyState
from llm_redact.registry import Registry


@dataclass(frozen=True)
class Widget:
    size: int
    tags: tuple[str, ...]
    owner: str | None
    limits: tuple[tuple[str, int], ...]
    rules: tuple[tuple[str, str], ...]


class WidgetSection:
    name = "widget"

    def parse(self, raw: Any, where: str) -> Widget:
        if not isinstance(raw, dict):
            raise ConfigError(f"{where} must be a table")
        unknown = set(raw) - {"size", "tags", "owner", "limits", "rule"}
        if unknown:
            raise ConfigError(f"unknown key(s) {sorted(unknown)} in {where}")
        size = raw.get("size", 1)
        if not isinstance(size, int) or size < 1:
            raise ConfigError(f"{where} size must be a positive integer")
        return Widget(
            size=size,
            tags=tuple(raw.get("tags", ())),
            owner=raw.get("owner"),
            limits=tuple(sorted(raw.get("limits", {}).items())),
            rules=tuple((r["match"], r["action"]) for r in raw.get("rule", ())),
        )

    def emit(self, value: Widget) -> Mapping[str, Any]:
        return {
            "size": value.size,
            "tags": list(value.tags),
            "owner": value.owner,  # None is omitted by the emitter
            "limits": dict(value.limits),
            "rule": [{"match": m, "action": a} for m, a in value.rules],
        }


class ShadowSection(WidgetSection):
    name = "vault"  # collides with a core section: ignored


@pytest.fixture
def sections(monkeypatch: pytest.MonkeyPatch) -> Registry:
    reg = Registry()
    reg.config_sections = [WidgetSection(), ShadowSection()]
    monkeypatch.setattr(registry_mod, "_registry", reg)
    return reg


RAW = {
    "widget": {
        "size": 3,
        "tags": ["a", 'q"uote'],
        "limits": {"x": 1, "y z": 2},
        "rule": [{"match": "m1", "action": "allow"}, {"match": "m2", "action": "deny"}],
    }
}


def test_unclaimed_section_fails_closed_naming_the_package() -> None:
    with pytest.raises(ConfigError) as err:
        parse_config({"auth": {"methods": ["user_key"]}}, "config.toml")
    message = str(err.value)
    assert "['auth']" in message
    assert "llm-redact-pro" in message


def test_claimed_section_is_parsed_into_extensions(sections: Registry) -> None:
    config = parse_config(RAW, "config.toml")
    widget = config.extensions["widget"]
    assert isinstance(widget, Widget)
    assert widget.size == 3
    assert widget.rules == (("m1", "allow"), ("m2", "deny"))
    assert parse_config({}, "config.toml").extensions == {}  # absent => not stored


def test_section_errors_propagate(sections: Registry) -> None:
    with pytest.raises(ConfigError, match=r"\[widget\] in config.toml size"):
        parse_config({"widget": {"size": 0}}, "config.toml")


def test_plugin_cannot_shadow_a_core_section(sections: Registry) -> None:
    config = parse_config({"vault": {"backend": "memory"}}, "config.toml")
    assert "vault" not in config.extensions
    assert config.vault.backend == "memory"


def test_emitter_round_trips_the_section(sections: Registry) -> None:
    config = parse_config({**RAW, "routing": {"enabled": False}}, "config.toml")
    text = emit_config_toml(config)
    assert "[widget]" in text
    assert "[[widget.rule]]" in text
    reparsed = parse_config(tomllib.loads(text), "emitted")
    assert reparsed.extensions == config.extensions


def test_reload_pins_plugin_sections(sections: Registry) -> None:
    state = ProxyState(parse_config(RAW, "config.toml"), upstream_transport=None)
    changed = parse_config({"widget": {"size": 9}}, "config.toml")
    restart = state.apply_config(changed)
    assert restart == ["widget"]
    assert state.config.extensions["widget"].size == 3  # the running value is kept
    assert state.apply_config(parse_config(RAW, "config.toml")) == []


def test_default_config_has_no_extensions() -> None:
    assert Config().extensions == {}
