"""`llm-redact serve --check`: serve's full startup build, minus the socket.

The deploy/reload gate: if --check exits 0, serve would start — so it must
run the SAME sequence (config load + CLI overrides, bind policy, complete
app build including the vault open) and fail on exactly what serve fails on.
"""

from pathlib import Path

import pytest

from llm_redact.cli import main


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    return int(excinfo.value.code or 0)


def _write(tmp_path: Path, body: str) -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text(body)
    return config_file


def test_check_passes_on_a_working_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = _write(tmp_path, 'host = "127.0.0.1"\n')
    assert _run(["serve", "--check", "--config", str(config_file)]) == 0
    assert "OK" in capsys.readouterr().out


def test_free_tier_sqlite_vault_passes_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # 3.2.0: an UNENCRYPTED SQLite vault needs no license — serve --check
    # (the full startup gate incl. check_license) must pass on the Free tier.
    monkeypatch.delenv("LLM_REDACT_LICENSE_KEY", raising=False)
    db = tmp_path / "vault.db"
    config_file = _write(
        tmp_path, f'[vault]\nbackend = "sqlite"\npath = "{db.as_posix()}"\n[log]\nformat = "json"\n'
    )
    assert _run(["serve", "--check", "--config", str(config_file)]) == 0
    assert "OK" in capsys.readouterr().out


def test_free_tier_encrypted_sqlite_still_fails_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The encryption gate is independent: an ENCRYPTED sqlite vault stays Pro,
    # so serve --check fails closed for a Free deployment.
    pytest.importorskip("cryptography")
    monkeypatch.delenv("LLM_REDACT_LICENSE_KEY", raising=False)
    from llm_redact.vault_crypto import ENV_KEY, generate_key

    monkeypatch.setenv(ENV_KEY, generate_key())
    db = tmp_path / "vault.db"
    config_file = _write(
        tmp_path,
        f'[vault]\nbackend = "sqlite"\nencryption = "fernet"\npath = "{db.as_posix()}"\n',
    )
    assert _run(["serve", "--check", "--config", str(config_file)]) == 1
    # Package-presence failure, never a tier one: the FOSS core has no tier
    # gates — what is missing is the implementation.
    assert "llm-redact-pro" in capsys.readouterr().err


def test_check_fails_on_parse_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_file = _write(tmp_path, "[detection]\nunknown_key = 1\n")
    assert _run(["serve", "--check", "--config", str(config_file)]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_check_fails_on_build_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Parses clean, fails the detector build — the exact hole the gate
    # exists for (doctor now catches it too; --check is the serve-fidelity
    # variant that also opens the vault).
    config_file = _write(tmp_path, '[detection.modes]\nno_such_rule = "warn"\n')
    assert _run(["serve", "--check", "--config", str(config_file)]) == 1
    assert "no_such_rule" in capsys.readouterr().err


def test_check_enforces_bind_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_REDACT_INSECURE_BIND", raising=False)
    monkeypatch.delenv("LLM_REDACT_HOST", raising=False)
    config_file = _write(tmp_path, 'host = "0.0.0.0"\n')
    assert _run(["serve", "--check", "--config", str(config_file)]) == 1
    assert "mutual TLS" in capsys.readouterr().err


def test_invalid_custom_rule_regex_fails_named_not_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 3.2.1: an invalid user regex used to escape as a raw re.error traceback
    # from the deploy gate itself; now it is a named FAIL like any other
    # config problem.
    config_file = _write(
        tmp_path,
        '[[detection.custom_rules]]\nname = "broken"\ntype = "TICKET"\npattern = "PROJ-(\\\\d+"\n',
    )
    assert _run(["serve", "--check", "--config", str(config_file)]) == 1
    err = capsys.readouterr().err
    assert "custom rule 'broken': invalid pattern" in err
    assert "Traceback" not in err


def test_invalid_allowlist_pattern_fails_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = _write(tmp_path, '[detection]\nallowlist_patterns = ["PROJ-(\\\\d+"]\n')
    assert _run(["serve", "--check", "--config", str(config_file)]) == 1
    err = capsys.readouterr().err
    assert "allowlist_patterns entry" in err
    assert "invalid regex" in err
