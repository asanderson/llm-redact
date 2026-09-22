from pathlib import Path

import pytest

from llm_redact.config import load_config
from llm_redact.proxy import ProxyState


def _write(path: Path, body: str) -> None:
    path.write_text(body)


def _state(config_path: Path) -> ProxyState:
    return ProxyState(load_config(config_path), None, config_path=config_path)


def test_reload_swaps_detection_rules(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write(config_path, '[detection]\nenabled = ["email"]\n')
    state = _state(config_path)
    assert len(state.detectors) == 1

    _write(config_path, '[detection]\nenabled = ["email", "ipv4", "jwt"]\n')
    state.reload()
    assert len(state.detectors) == 3
    assert tuple(state.config.detection.enabled) == ("email", "ipv4", "jwt")
    # The static context redactor was rebuilt on the new detector set.
    assert state._static_context.redactor is state.redactor


def test_reload_bad_toml_keeps_old_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write(config_path, '[detection]\nenabled = ["email"]\n')
    state = _state(config_path)
    before = state.config

    _write(config_path, "this is not toml [ [")
    state.reload()
    assert state.config is before  # untouched


def test_reload_unknown_key_keeps_old_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write(config_path, "port = 8787\n")
    state = _state(config_path)
    _write(config_path, "not_a_real_key = true\n")
    state.reload()
    assert state.config.port == 8787


def test_reload_vault_change_requires_restart(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config_path = tmp_path / "config.toml"
    _write(config_path, '[vault]\nsession = "one"\n')
    state = _state(config_path)

    _write(config_path, '[vault]\nsession = "two"\n')
    with caplog.at_level("WARNING", logger="llm_redact"):
        state.reload()
    assert any("require restart" in record.getMessage() for record in caplog.records)
    assert state.config.vault.session == "one"  # kept


def test_reload_fuzzy_flag_applies(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write(config_path, "[rehydration]\nfuzzy = true\n")
    state = _state(config_path)
    _write(config_path, "[rehydration]\nfuzzy = false\n")
    state.reload()
    assert state.config.rehydration.fuzzy is False


def test_reload_unchanged_detection_reuses_detectors(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write(config_path, '[detection]\nenabled = ["email"]\n')
    state = _state(config_path)
    detectors_before = state.detectors
    _write(config_path, 'inject_system_note = false\n\n[detection]\nenabled = ["email"]\n')
    state.reload()
    # Detection config unchanged: the (possibly expensive NER) detector list
    # is reused, while the other hot flag applied.
    assert state.detectors is detectors_before
    assert state.config.inject_system_note is False


def test_apply_config_reports_and_pins_restart_only_fields(tmp_path: Path) -> None:
    import dataclasses

    config_path = tmp_path / "config.toml"
    _write(config_path, "port = 8787\n")
    state = _state(config_path)
    fresh = dataclasses.replace(state.config, port=9999, inject_system_note=False)
    restart_required = state.apply_config(fresh)
    assert restart_required == ["port"]
    assert state.config.port == 8787  # pinned to the running value
    assert state.config.inject_system_note is False  # hot-applied


def test_reload_swaps_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write(config_path, '[detection]\nenabled = ["email"]\n')
    state = _state(config_path)
    assert state.modes == {}

    _write(
        config_path,
        '[detection]\nenabled = ["email"]\n\n[detection.modes]\nemail = "warn"\n',
    )
    state.reload()
    assert state.modes == {"EMAIL": "warn"}
    # The rebuilt static-context redactor dispatches on the new map.
    out = state.redactor.redact_text("mail jane@corp.example")
    assert out == "mail jane@corp.example"
    assert state.warn_counts["EMAIL"] == 1


def test_reload_unchanged_detection_reuses_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write(config_path, '[detection.modes]\nemail = "warn"\n')
    state = _state(config_path)
    modes_before = state.modes
    detectors_before = state.detectors
    _write(config_path, 'inject_system_note = false\n\n[detection.modes]\nemail = "warn"\n')
    state.reload()
    assert state.modes is modes_before
    assert state.detectors is detectors_before


def test_reload_bad_modes_keeps_old_config(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Unknown rule names pass parse_config (validation lives in build_modes,
    # mirroring `enabled`/build_detectors) — reload must survive them.
    config_path = tmp_path / "config.toml"
    _write(config_path, '[detection.modes]\nemail = "warn"\n')
    state = _state(config_path)
    before = state.config

    _write(config_path, '[detection.modes]\nno_such_rule = "block"\n')
    with caplog.at_level("ERROR", logger="llm_redact"):
        state.reload()
    assert state.config is before
    assert state.modes == {"EMAIL": "warn"}
    assert any("reload failed" in record.getMessage() for record in caplog.records)


def test_reload_bad_enabled_rule_keeps_old_config(tmp_path: Path) -> None:
    # Same deferred-validation path as modes, via build_detectors.
    config_path = tmp_path / "config.toml"
    _write(config_path, '[detection]\nenabled = ["email"]\n')
    state = _state(config_path)
    before = state.config

    _write(config_path, '[detection]\nenabled = ["no_such_rule"]\n')
    state.reload()
    assert state.config is before


def test_reload_picks_up_deny_strings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write(config_path, '[detection]\nenabled = ["email"]\n')
    state = _state(config_path)
    assert state.redactor.redact_text("ship aurora now") == "ship aurora now"

    _write(config_path, '[detection]\nenabled = ["email"]\ndeny = ["aurora"]\n')
    state.reload()
    assert state.redactor.redact_text("ship AURORA now") == "ship «DENY_001» now"

    _write(config_path, '[detection]\nenabled = ["email"]\n')
    state.reload()
    assert state.redactor.redact_text("ship aurora again") == "ship aurora again"


def test_reload_reconfigures_running_router(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from fake_router import ROUTED_TOML, FakeRouter, install

    router = FakeRouter()
    install(monkeypatch, router)
    config_path = tmp_path / "config.toml"
    _write(config_path, ROUTED_TOML)
    state = _state(config_path)
    assert state.router is router

    _write(config_path, ROUTED_TOML + "max_hops = 4\n")
    with caplog.at_level("WARNING", logger="llm_redact"):
        state.reload()
    # Reconfigured in place — never rebuilt, never closed (D10: a running
    # router is not re-gated on reload).
    assert state.router is router and router.closed == 0
    assert [cfg.routing.max_hops for cfg in router.reconfigured] == [4]
    assert state.config.routing.max_hops == 4
    # The parser's own routing warnings are logged by the core on every
    # reload (D20) — here the metered default_upstream.
    assert any(
        record.getMessage().startswith("routing: ")
        for record in caplog.records
        if record.levelname == "WARNING"
    )


def test_reload_builds_router_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fake_router import ROUTED_TOML, FakeRouter, install

    router = FakeRouter()
    _reg, calls = install(monkeypatch, router)
    config_path = tmp_path / "config.toml"
    _write(config_path, ROUTED_TOML.replace("enabled = true", "enabled = false"))
    state = _state(config_path)
    assert state.router is None
    assert len(calls) == 1  # the startup factory call returned None (disabled)

    _write(config_path, ROUTED_TOML)
    state.reload()
    # A reload that turns routing ON builds through the registry with the
    # re-resolved tier (keyless: free).
    assert state.router is router
    assert len(calls) == 2
    built_config, tier = calls[-1]
    assert built_config.routing.enabled is True and tier == "free"
    assert router.reconfigured == []


def test_reload_drops_and_closes_router_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fake_router import ROUTED_TOML, FakeRouter, install

    router = FakeRouter()
    install(monkeypatch, router)
    config_path = tmp_path / "config.toml"
    _write(config_path, ROUTED_TOML)
    state = _state(config_path)
    assert state.router is router

    _write(config_path, ROUTED_TOML.replace("enabled = true", "enabled = false"))
    state.reload()
    assert state.router is None
    assert router.closed == 1  # closed AT SWAP TIME, after the other builds
    assert state.config.routing.enabled is False


def test_raising_reconfigure_keeps_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from fake_router import ROUTED_TOML, FakeRouter, install
    from llm_redact.config import ConfigError

    router = FakeRouter(reconfigure_error=ConfigError("price table: nope"))
    install(monkeypatch, router)
    config_path = tmp_path / "config.toml"
    _write(config_path, ROUTED_TOML)
    state = _state(config_path)
    before = state.config

    _write(config_path, ROUTED_TOML + "max_hops = 4\n")
    with caplog.at_level("ERROR", logger="llm_redact"):
        state.reload()
    # reconfigure raised (ConfigError is a ValueError) before the swap: the
    # running config and router are untouched, nothing closed.
    assert state.config is before
    assert state.router is router and router.closed == 0
    assert any(
        "reload failed" in record.getMessage() and "price table: nope" in record.getMessage()
        for record in caplog.records
    )


def test_failed_router_build_does_not_half_apply_the_adapter_list(tmp_path: Path) -> None:
    # A reload that changes the provider set AND enables routing without the
    # pro package: the router build raises after the new adapter list was
    # built. The adapter list must be swapped only with the rest of the
    # config — a half-applied reload (old config still naming custom:x, new
    # adapters without it) would forward /custom/x/ traffic unredacted.
    config_path = tmp_path / "config.toml"
    _write(
        config_path,
        '[providers.custom.x]\nupstream_base_url = "http://x.example/v1"\n',
    )
    state = _state(config_path)
    before_config = state.config
    before_adapters = state.adapters
    assert any(a.name == "custom:x" for a in before_adapters)

    _write(
        config_path,
        '[upstreams.u]\nprotocol = "anthropic"\nbase_url = "http://u.example"\n'
        '[routing]\nenabled = true\ndefault_upstream = "u"\n',
    )
    state.reload()  # the Free default raises: routing needs llm-redact-pro

    assert state.config is before_config
    assert state.adapters is before_adapters
    assert any(a.name == "custom:x" for a in state.adapters)
