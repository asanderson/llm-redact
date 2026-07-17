"""`llm-redact doctor`: read-only diagnostics, non-zero exit on FAILs."""

import argparse
import socket
import sys
from pathlib import Path

import pytest

from llm_redact.doctor_cli import run_doctor


def _args(config: Path | None) -> argparse.Namespace:
    return argparse.Namespace(config=config)


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _write(tmp_path: Path, body: str) -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text(body)
    return config_file


def test_healthy_defaults_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    config_file = _write(tmp_path, f"port = {_free_port()}\n")
    assert run_doctor(_args(config_file)) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "config" in out and "parses" in out
    assert "not running" in out  # proxy absence is a WARN, not a FAIL
    assert "port" in out and "free" in out


def test_broken_config_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_file = _write(tmp_path, "[detection]\nunknown_key = 1\n")
    assert run_doctor(_args(config_file)) == 1
    assert "FAIL" in capsys.readouterr().out


def test_parse_clean_build_broken_config_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The preflight hole this check exists for: an unknown rule name in
    # [detection.modes] parses fine but fails the detector BUILD — serve
    # would refuse it and a SIGHUP reload would silently keep the old
    # config. doctor must FAIL, not report a green "parses".
    config_file = _write(
        tmp_path,
        f'port = {_free_port()}\n[detection.modes]\nno_such_rule = "warn"\n',
    )
    assert run_doctor(_args(config_file)) == 1
    out = capsys.readouterr().out
    assert "does not BUILD" in out
    assert "no_such_rule" in out


def test_build_broken_custom_validator_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = _write(
        tmp_path,
        f"""port = {_free_port()}
[[detection.custom_rules]]
name = "internal_id"
type = "INTERNAL_ID"
pattern = "ID-[0-9]+"
validator = "no_such_validator"
""",
    )
    assert run_doctor(_args(config_file)) == 1
    assert "does not BUILD" in capsys.readouterr().out


def test_build_dry_run_never_builds_ner(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # With NER enabled, the build dry-run must swap NER out (doctor never
    # loads or downloads models) and say so honestly; importability is the
    # separate ner check. gliner is absent in the test env, so the ner
    # check FAILs while the build line still reports the swap.
    try:
        import gliner  # noqa: F401

        pytest.skip("gliner installed; missing-extra path not testable")
    except ImportError:
        pass
    config_file = _write(
        tmp_path,
        f'port = {_free_port()}\n[detection.ner]\nenabled = true\nbackend = "gliner"\n',
    )
    assert run_doctor(_args(config_file)) == 1  # the ner FAIL, not a crash
    out = capsys.readouterr().out
    assert "NER backends not built" in out
    assert "detectors build" in out


def test_multi_backend_ner_check_covers_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # backends = [stanza, hf] used to crash doctor with a KeyError (the
    # module map only knew the original three); every active backend gets
    # its own importability line now.
    config_file = _write(
        tmp_path,
        f'port = {_free_port()}\n[detection.ner]\nenabled = true\nbackends = ["stanza", "hf"]\n',
    )
    exit_code = run_doctor(_args(config_file))
    out = capsys.readouterr().out
    assert "stanza" in out and "hf" in out
    # In the dev env neither extra is installed, so both FAIL; if someone
    # installs them locally the lines flip to PASS and the exit changes.
    if "uv sync --extra stanza" in out:
        assert exit_code == 1


def test_doctor_json_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    config_file = _write(tmp_path, f"port = {_free_port()}\n")
    exit_code = run_doctor(argparse.Namespace(config=config_file, json=True))
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"version", "failed", "checks"}
    assert payload["failed"] is (exit_code == 1)
    assert {"level", "area", "message"} == set(payload["checks"][0])
    assert any(row["area"] == "build" for row in payload["checks"])


def test_doctor_json_mode_on_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    config_file = _write(tmp_path, f'port = {_free_port()}\n[detection.modes]\nnope = "warn"\n')
    assert run_doctor(argparse.Namespace(config=config_file, json=True)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["failed"] is True
    assert any(row["level"] == "FAIL" for row in payload["checks"])


def test_config_show_roundtrips_and_names_the_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.config import parse_config
    from llm_redact.doctor_cli import run_config_show

    monkeypatch.setenv("LLM_REDACT_PORT", "9123")
    config_file = _write(tmp_path, 'port = 8000\n[detection]\ndeny = ["hunter2"]\n')
    assert run_config_show(argparse.Namespace(config=config_file, path=False)) == 0
    out = capsys.readouterr().out
    assert f"# source: {config_file}" in out
    assert "LLM_REDACT_PORT" in out  # active override named, already applied
    # The emitted body is valid TOML that reparses to the effective config.
    import tomllib

    body = "\n".join(line for line in out.splitlines() if not line.startswith("# "))
    effective = parse_config(tomllib.loads(body), "show")
    assert effective.port == 9123  # env override baked into the shown config
    assert ("hunter2", False) in effective.detection.deny_strings or any(
        "hunter2" in str(d) for d in effective.detection.deny_strings
    )


def test_config_show_path_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.doctor_cli import run_config_show

    config_file = _write(tmp_path, "port = 8000\n")
    assert run_config_show(argparse.Namespace(config=config_file, path=True)) == 0
    assert capsys.readouterr().out.strip() == str(config_file)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX mode bits are synthetic on Windows; doctor WARNs about NTFS"
    " ACLs instead (test_windows_platform_note_and_no_false_perm_fails)",
)
def test_loose_vault_permissions_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    vault_dir = tmp_path / "data"
    vault_dir.mkdir(mode=0o700)
    vault_db = vault_dir / "vault.db"
    vault_db.touch(mode=0o644)  # group/world readable: the secrets file!
    config_file = _write(
        tmp_path,
        f'port = {_free_port()}\n[vault]\nbackend = "sqlite"\npath = "{vault_db.as_posix()}"\n',
    )
    assert run_doctor(_args(config_file)) == 1
    out = capsys.readouterr().out
    assert "world accessible" in out


def test_missing_ner_extra_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    try:
        import gliner  # noqa: F401

        pytest.skip("gliner installed; missing-extra path not testable")
    except ImportError:
        pass
    config_file = _write(
        tmp_path,
        f'port = {_free_port()}\n[detection.ner]\nenabled = true\nbackend = "gliner"\n',
    )
    assert run_doctor(_args(config_file)) == 1
    assert "uv sync --extra gliner" in capsys.readouterr().out


def test_licensed_features_line_reports_not_installed() -> None:
    # Free suite: the pro package is genuinely absent, so the line is a PASS
    # that says so (never a FAIL — the Free core is complete on its own).
    from llm_redact.doctor_cli import _check_licensed_features, _Report

    report = _Report(json_mode=True)
    _check_licensed_features(report)
    assert report.failed is False
    assert report.rows == [
        {
            "level": "PASS",
            "area": "license",
            "message": "licensed-features package not installed"
            " (FOSS core is complete; pro-only config fails closed)",
        }
    ]


def test_licensed_features_line_reports_installed_and_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_redact.registry as registry_mod
    from llm_redact.doctor_cli import _check_licensed_features, _Report

    monkeypatch.setattr(registry_mod, "pro_package_installed", lambda: True)
    monkeypatch.setattr(registry_mod, "get_registry", lambda: None)
    monkeypatch.setattr(registry_mod, "loaded_plugins", lambda: ["llm_redact_pro"])
    report = _Report(json_mode=True)
    _check_licensed_features(report)
    assert report.failed is False
    assert report.rows[0]["level"] == "PASS"
    assert "installed (llm_redact_pro active)" in report.rows[0]["message"]


def test_licensed_features_line_warns_when_installed_but_not_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The one loud case: the package is present but its plugin did not register,
    # so paid features are silently OFF — a WARN, never a silent pass.
    import llm_redact.registry as registry_mod
    from llm_redact.doctor_cli import _check_licensed_features, _Report

    monkeypatch.setattr(registry_mod, "pro_package_installed", lambda: True)
    monkeypatch.setattr(registry_mod, "get_registry", lambda: None)
    monkeypatch.setattr(registry_mod, "loaded_plugins", lambda: [])
    report = _Report(json_mode=True)
    _check_licensed_features(report)
    assert report.failed is False  # WARN, not FAIL
    assert report.rows[0]["level"] == "WARN"
    assert "did not register" in report.rows[0]["message"]


def test_fernet_without_key_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_REDACT_VAULT_KEY", raising=False)
    config_file = _write(
        tmp_path,
        f'port = {_free_port()}\n[vault]\nbackend = "sqlite"\nencryption = "fernet"\n',
    )
    assert run_doctor(_args(config_file)) == 1
    assert "vault gen-key" in capsys.readouterr().out


def test_missing_tls_files_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_file = _write(
        tmp_path,
        f"port = {_free_port()}\n[tls]\n"
        f'certfile = "{tmp_path.as_posix()}/absent.crt"\n'
        f'keyfile = "{tmp_path.as_posix()}/absent.key"\n',
    )
    assert run_doctor(_args(config_file)) == 1
    assert "missing or unreadable" in capsys.readouterr().out


def test_nonloopback_without_mtls_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_REDACT_INSECURE_BIND", raising=False)
    monkeypatch.delenv("LLM_REDACT_HOST", raising=False)
    config_file = _write(tmp_path, 'host = "0.0.0.0"\n')
    assert run_doctor(_args(config_file)) == 1
    assert "mutual TLS" in capsys.readouterr().out


def test_audit_s3_credentials_check(monkeypatch: pytest.MonkeyPatch) -> None:
    # The S3 audit sink is a Pro feature (a Free-tier config enabling it fails
    # the license gate, so run_doctor would exit 1 on the gate regardless of
    # credentials). The credential CHECK itself is Free code, so drive it
    # directly through _check_audit_s3 — decoupled from the license gate.
    from llm_redact.config import AuditConfig, Config, S3AuditConfig
    from llm_redact.doctor_cli import _check_audit_s3, _Report

    config = Config(
        audit=AuditConfig(
            enabled=True,
            s3=S3AuditConfig(
                enabled=True,
                provider="minio",
                bucket="audit",
                endpoint_url="http://127.0.0.1:9000",
            ),
        )
    )

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    missing = _Report(json_mode=True)
    _check_audit_s3(missing, config)
    assert missing.failed
    fail = next(r for r in missing.rows if r["level"] == "FAIL")
    assert "AWS_ACCESS_KEY_ID" in fail["message"] and "never the config file" in fail["message"]

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLEONLY00000")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-a-real-secret")
    present = _Report(json_mode=True)
    _check_audit_s3(present, config)
    assert not present.failed
    joined = " ".join(r["message"] for r in present.rows)
    assert "metadata rows leave this machine" in joined
    assert "not-a-real-secret" not in joined  # presence only, never values


def test_posture_clean_when_no_opt_outs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    config_file = _write(tmp_path, f"port = {_free_port()}\n")
    assert run_doctor(_args(config_file)) == 0
    out = capsys.readouterr().out
    assert "posture" in out and "no coverage opt-outs" in out


def test_posture_warns_on_every_opt_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    config_file = _write(
        tmp_path,
        f"port = {_free_port()}\n"
        "[detection]\n"
        'languages = ["en"]\n'
        "[detection.modes]\n"
        'phone_number = "warn"\n'
        "[detection.mcp]\n"
        'exempt_servers = ["github"]\n'
        "[providers.gemini]\n"
        'upstream_base_url = "https://example.invalid"\n'
        "detection = false\n",
    )
    # Every posture line is a WARN (deliberate opt-out), so the exit stays 0.
    assert run_doctor(_args(config_file)) == 0
    out = capsys.readouterr().out
    assert "warn mode on phone_number" in out and "FORWARDED" in out
    assert "detection disabled for provider(s) gemini" in out
    assert "MCP server(s) exempt" in out
    assert "french_nir" in out and "unbuilt" in out  # language scope drops fr rules


def test_windows_platform_note_and_no_false_perm_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On win32 POSIX mode bits are synthetic (files commonly stat as 666):
    doctor must say NTFS ACLs govern instead of raising false FAILs, and
    the platform note must state the supported scope + reload path."""
    import sys as _sys

    monkeypatch.setattr(_sys, "platform", "win32")
    monkeypatch.delenv("LLM_REDACT_CONFIG", raising=False)
    vault_db = tmp_path / "vault.db"
    vault_db.write_text("")
    vault_db.chmod(0o666)  # would FAIL the POSIX check if it ran
    config_file = _write(
        tmp_path,
        f'port = {_free_port()}\n[vault]\nbackend = "sqlite"\npath = "{vault_db.as_posix()}"\n',
    )
    assert run_doctor(_args(config_file)) == 0
    out = capsys.readouterr().out
    assert "NTFS" in out
    assert "config editor" in out and "SIGHUP" in out
    # The POSIX mode FAIL never fired — assert on its signature, not the bare
    # "666" (a random free port can also contain those digits and flake this).
    assert "group/world accessible" not in out
