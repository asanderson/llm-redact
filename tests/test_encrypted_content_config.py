"""Explicit, value-free consent for the encrypted-content scan opt-out."""

import argparse
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from llm_redact import config as config_module
from llm_redact.config import Config, ConfigError, load_config, parse_config
from llm_redact.config_write import emit_config_toml
from llm_redact.detection.engine import DetectionConfig
from llm_redact.init_cli import run_init


def test_encrypted_content_defaults_are_disabled() -> None:
    detection = parse_config({}, "test").detection
    assert detection.encrypted_content_passthrough is False
    assert detection.acknowledge_encrypted_content_risk is False


@pytest.mark.parametrize("enabled,acknowledged", [(False, False), (False, True), (True, True)])
def test_encrypted_content_config_round_trip(enabled: bool, acknowledged: bool) -> None:
    config = parse_config(
        {
            "detection": {
                "encrypted_content_passthrough": enabled,
                "acknowledge_encrypted_content_risk": acknowledged,
            }
        },
        "test",
    )
    text = emit_config_toml(config)
    assert parse_config(tomllib.loads(text), "round-trip") == config
    assert f"encrypted_content_passthrough = {str(enabled).lower()}" in text
    assert f"acknowledge_encrypted_content_risk = {str(acknowledged).lower()}" in text
    assert "possible plaintext" in text
    assert "field name" in text
    assert "provider can decrypt" in text


@pytest.mark.parametrize(
    "key", ["encrypted_content_passthrough", "acknowledge_encrypted_content_risk"]
)
@pytest.mark.parametrize("bad", ["private-marker-fake", 1, 0, [], None])
def test_encrypted_content_config_rejects_nonboolean_without_values(key: str, bad: object) -> None:
    with pytest.raises(ConfigError) as caught:
        parse_config({"detection": {key: bad}}, "test")
    assert "must be a boolean" in str(caught.value)
    assert key in str(caught.value)
    assert "private-marker-fake" not in str(caught.value)


def test_encrypted_content_enabling_requires_acknowledgement() -> None:
    with pytest.raises(ConfigError) as caught:
        parse_config({"detection": {"encrypted_content_passthrough": True}}, "test")
    assert config_module.ENCRYPTED_CONTENT_RISK_WARNING in str(caught.value)
    assert "acknowledge_encrypted_content_risk = true" in str(caught.value)


def test_programmatic_config_has_reusable_consent_validation() -> None:
    config_module.validate_encrypted_content_passthrough(DetectionConfig())
    unsafe = replace(DetectionConfig(), encrypted_content_passthrough=True)
    with pytest.raises(ConfigError):
        config_module.validate_encrypted_content_passthrough(unsafe)


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "xdg" / "llm-redact" / "config.toml"
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return target


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "tools": "codex",
        "vault": "memory",
        "encryption": "none",
        "port": 8787,
        "yes": True,
        "force": False,
        "encrypted_content_passthrough": True,
        "accept_encrypted_content_risk": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("yes", [False, True])
def test_headless_init_requires_specific_acceptance(
    yes: bool,
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert run_init(_args(yes=yes)) != 0
    assert not config_path.exists()
    assert config_module.ENCRYPTED_CONTENT_RISK_WARNING in capsys.readouterr().out


def test_headless_init_acceptance_saves_acknowledged_option(
    config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_init(_args(accept_encrypted_content_risk=True)) == 0
    saved = load_config(config_path).detection
    assert saved.encrypted_content_passthrough is True
    assert saved.acknowledge_encrypted_content_risk is True
    out = capsys.readouterr().out
    assert out.index(config_module.ENCRYPTED_CONTENT_RISK_WARNING) < out.index("wrote ")


@pytest.mark.parametrize("answer", ["n", "", "eof"])
@pytest.mark.parametrize("existing", [False, True])
def test_interactive_init_decline_or_eof_does_not_write(
    answer: str,
    existing: bool,
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    if existing:
        config_path.parent.mkdir(parents=True)
        config_path.write_text("# existing config must survive\n")

    def respond(prompt: str) -> str:
        assert config_module.ENCRYPTED_CONTENT_RISK_WARNING in capsys.readouterr().out
        if answer == "eof":
            raise EOFError
        return answer

    monkeypatch.setattr("builtins.input", respond)
    assert run_init(_args(yes=False, force=True)) != 0
    if existing:
        assert config_path.read_text() == "# existing config must survive\n"
        assert not config_path.with_suffix(".toml.bak").exists()
    else:
        assert not config_path.exists()


def test_interactive_init_explicit_acceptance(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert run_init(_args(yes=False)) == 0
    assert load_config(config_path).detection.acknowledge_encrypted_content_risk is True


def test_default_init_needs_no_consent_prompt(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def unexpected_prompt(prompt: str) -> str:
        pytest.fail("default init must not prompt for this opt-out")

    monkeypatch.setattr("builtins.input", unexpected_prompt)
    assert run_init(_args(yes=False, encrypted_content_passthrough=False)) == 0
    assert load_config(config_path).detection == Config().detection
    assert "possible plaintext" not in capsys.readouterr().out
