"""init / service / completions subcommands (no real launchd/systemd)."""

import argparse
import sys
from pathlib import Path

import pytest

from llm_redact import service_cli
from llm_redact.cli import build_parser
from llm_redact.completions import COMMANDS, script_for
from llm_redact.config import load_config
from llm_redact.init_cli import run_init

# ---- init ----


def _init_args(**overrides) -> argparse.Namespace:
    defaults = dict(tools=None, vault=None, encryption=None, port=None, yes=True, force=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def xdg_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "llm-redact" / "config.toml"


def test_init_noninteractive_defaults(xdg_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_init(_init_args()) == 0
    config = load_config(xdg_home)
    assert config.port == 8787
    # 3.3: sqlite is the non-interactive default — it is Free (since 3.2) and
    # the memory backend made restarts silently unrestorable for old tokens.
    assert config.vault.backend == "sqlite"
    assert config.vault.encryption == "none"
    out = capsys.readouterr().out
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8787" in out
    assert "llm-redact serve" in out


def test_init_flags(xdg_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _init_args(tools="codex,gemini", vault="sqlite", encryption="fernet", port=9000)
    assert run_init(args) == 0
    config = load_config(xdg_home)
    assert config.port == 9000
    assert config.vault.backend == "sqlite"
    assert config.vault.encryption == "fernet"
    out = capsys.readouterr().out
    assert "OPENAI_BASE_URL=http://127.0.0.1:9000" in out
    assert "GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:9000" in out
    assert "ANTHROPIC_BASE_URL" not in out
    assert "gen-key" in out


def test_init_refuses_overwrite_without_force(
    xdg_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_init(_init_args()) == 0
    assert run_init(_init_args(port=9999)) == 1
    assert load_config(xdg_home).port == 8787  # untouched
    assert run_init(_init_args(port=9999, force=True)) == 0
    assert load_config(xdg_home).port == 9999
    assert xdg_home.with_name("config.toml.bak").exists()


def test_init_rejects_bad_combos(xdg_home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_init(_init_args(tools="notepad")) == 2
    assert not xdg_home.exists()


def test_init_accepts_encrypted_memory_vault(xdg_home: Path) -> None:
    # fernet applies to BOTH backends since the encrypted in-memory vault
    # landed (ciphertext in RAM); init must not reject the combination.
    assert run_init(_init_args(vault="memory", encryption="fernet")) == 0
    config = load_config(xdg_home)
    assert config.vault.backend == "memory"
    assert config.vault.encryption == "fernet"


# ---- service ----


def _service_args(command: str, print_only: bool = False) -> argparse.Namespace:
    return argparse.Namespace(service_command=command, print_only=print_only)


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="service units are launchd/systemd; win32 prints schtasks guidance (its own test below)",
)
def test_service_print_only_changes_nothing(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert service_cli.run_service(_service_args("install", print_only=True)) == 0
    out = capsys.readouterr().out
    assert "llm_redact" in out or "llm-redact" in out
    assert "serve" in out
    assert not list(fake_home.rglob("*.service")) and not list(fake_home.rglob("*.plist"))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="service units are launchd/systemd; win32 prints schtasks guidance (its own test below)",
)
def test_service_install_uninstall_roundtrip(
    fake_home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(service_cli, "_run", lambda cmd: calls.append(cmd) or 0)

    assert service_cli.run_service(_service_args("install")) == 0
    unit = service_cli._unit_path()
    assert unit.exists()
    unit_text = unit.read_text()
    assert "serve" in unit_text
    if sys.platform != "darwin":
        # Linux systemd unit carries the sandboxing directives.
        for directive in ("NoNewPrivileges=yes", "ProtectSystem=strict", "CapabilityBoundingSet="):
            assert directive in unit_text, f"systemd unit missing {directive}"
    assert calls  # the loader was invoked

    calls.clear()
    assert service_cli.run_service(_service_args("status")) == 0
    assert calls

    calls.clear()
    assert service_cli.run_service(_service_args("uninstall")) == 0
    assert not unit.exists()
    assert calls


def test_service_config_env_propagated(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_REDACT_CONFIG", "/etc/custom/config.toml")
    command = service_cli._command_line()
    assert command[-2:] == ["--config", "/etc/custom/config.toml"]


def test_launchd_plist_escapes_xml_metacharacters() -> None:
    # Security review 3.1.1: a config path with &/</> must be XML-escaped so
    # it can't break the plist or inject extra elements.
    plist = service_cli._launchd_plist(["llm-redact", "serve", "--config", "/a & b/<x>.toml"])
    assert "/a & b/<x>.toml" not in plist
    assert "&amp;" in plist and "&lt;x&gt;" in plist


def test_service_unit_rejects_control_chars_in_command() -> None:
    # A newline in the command would terminate the unit-file line and could
    # inject directives; both builders must refuse it.
    with pytest.raises(ValueError, match="control character"):
        service_cli._launchd_plist(["llm-redact", "serve", "--config", "/x\n.toml"])
    with pytest.raises(ValueError, match="control character"):
        service_cli._systemd_unit(["llm-redact", "serve", "--config", "/x\n.toml"])


# ---- completions ----


def test_completions_cover_the_real_parser() -> None:
    parser = build_parser()
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    real = set(subparsers.choices)
    assert set(COMMANDS) == real, "completions.COMMANDS drifted from the argparse parser"
    for name, sub in subparsers.choices.items():
        real_opts = {
            opt
            for action in sub._actions
            for opt in action.option_strings
            if opt.startswith("--") and opt != "--help"
        }
        declared_subs, declared_opts = COMMANDS[name]
        nested = [a for a in sub._actions if isinstance(a, argparse._SubParsersAction)]
        real_subs = set(nested[0].choices) if nested else set()
        # Nested subcommand options (sessions list/prune etc.) are folded
        # into the parent's option list for completion purposes.
        for nested_sub in nested[0].choices.values() if nested else ():
            real_opts |= {
                opt
                for action in nested_sub._actions
                for opt in action.option_strings
                if opt.startswith("--") and opt != "--help"
            }
        positional_choices = {
            str(c)
            for action in sub._actions
            for c in (action.choices or ())
            if not action.option_strings and not isinstance(action, argparse._SubParsersAction)
        }
        assert set(declared_subs) == real_subs | positional_choices, name
        assert set(declared_opts) == real_opts, name


def test_completion_scripts_render() -> None:
    bash = script_for("bash")
    zsh = script_for("zsh")
    fish = script_for("fish")
    assert "complete -F _llm_redact llm-redact" in bash
    assert "#compdef llm-redact" in zsh
    assert "complete -c llm-redact" in fish
    for name in COMMANDS:
        assert name in bash and name in zsh and name in fish
    # Every flag reaches fish too (bash/zsh embed them via the case block).
    assert "-l check" in fish and "-l set-env" in fish
    with pytest.raises(ValueError):
        script_for("powershell")


def test_service_windows_prints_schtasks_guidance(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys as _sys

    from llm_redact.service_cli import run_service

    monkeypatch.setattr(_sys, "platform", "win32")
    args = argparse.Namespace(service_command="install", print_only=False)
    assert run_service(args) == 2  # nothing is written — guidance only
    out = capsys.readouterr().out
    assert "schtasks /Create" in out and "schtasks /Delete" in out
    assert "launchd" in out and "systemd" in out
