"""``llm-redact override``: the approval needs a person at the terminal.

The confirmation is read from the controlling terminal (a fake one here),
never from stdin; without a terminal the command refuses. Listing and
revocation are value-free."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from llm_redact import override_cli
from llm_redact.cli import main
from llm_redact.overrides import OverrideStore

EMAIL = "jane.doe@corp.example"


class FakeTty(io.StringIO):
    def __init__(self, answer: str) -> None:
        super().__init__()
        self.answer = answer
        self.shown = ""

    def readline(self, *args: Any) -> str:  # type: ignore[override]
        self.shown = self.getvalue()
        return self.answer


def _run(*argv: str) -> int:
    with pytest.raises(SystemExit) as exited:
        main(["override", *argv])
    return int(exited.value.code or 0)


def _pending(db: Path, subject: str = "", kind: str = "block") -> str:
    values = [("EMAIL", EMAIL)] if kind == "block" else []
    return OverrideStore(db).record_pending(kind, subject, "openai", "POST", "/v1/x", values)


def test_approve_once_after_typing_allow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "o.db"
    code = _pending(db)
    tty = FakeTty("allow\n")
    monkeypatch.setattr(override_cli, "_open_tty", lambda: tty)
    # Anything on stdin is ignored: only the terminal answers.
    monkeypatch.setattr("sys.stdin", io.StringIO("allow\n"))
    assert _run(code.lower(), "--once", "--db", str(db)) == 0
    assert "kind block, types EMAIL" in tty.shown and "next such request" in tty.shown
    out = capsys.readouterr().out
    assert "approved once" in out and EMAIL not in out and code not in out + tty.shown
    assert [e.state for e in OverrideStore(db).entries()] == ["once"]


def test_approve_always_warns_that_values_are_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "o.db"
    code = _pending(db)
    tty = FakeTty("ALLOW\n")
    monkeypatch.setattr(override_cli, "_open_tty", lambda: tty)
    assert _run(code, "--always", "--db", str(db)) == 0
    assert "forward these exact values UNREDACTED" in tty.shown
    route_code = _pending(db, kind="unscanned_body")
    tty = FakeTty("allow")
    monkeypatch.setattr(override_cli, "_open_tty", lambda: tty)
    assert _run(route_code, "--always", "--db", str(db)) == 0
    assert "forwarded UNSCANNED" in tty.shown


def test_anything_but_allow_declines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "o.db"
    code = _pending(db)
    monkeypatch.setattr(override_cli, "_open_tty", lambda: FakeTty("yes\n"))
    assert _run(code, "--once", "--db", str(db)) == 1
    assert "not approved" in capsys.readouterr().err
    assert [e.state for e in OverrideStore(db).entries()] == ["pending"]


def test_no_terminal_no_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "o.db"
    code = _pending(db)

    def no_tty() -> Any:
        raise OSError("no controlling terminal")

    monkeypatch.setattr(override_cli, "_open_tty", no_tty)
    monkeypatch.setattr("sys.stdin", io.StringIO("allow\n"))
    assert _run(code, "--once", "--db", str(db)) == 1
    assert "needs a terminal" in capsys.readouterr().err
    assert [e.state for e in OverrideStore(db).entries()] == ["pending"]


def test_a_named_users_refusal_is_not_the_operators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "o.db"
    code = _pending(db, subject="alice")
    monkeypatch.setattr(override_cli, "_open_tty", lambda: FakeTty("allow\n"))
    assert _run(code, "--once", "--db", str(db)) == 1
    assert "named user" in capsys.readouterr().err


def test_usage_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "o.db"
    assert _run("not-a-code", "--once", "--db", str(db)) == 2
    code = _pending(db)
    assert _run(code, "--db", str(db)) == 2
    assert "--once or --always" in capsys.readouterr().err
    assert _run("revoke", "--db", str(db)) == 2
    assert _run("0" * 12, "--once", "--db", str(db)) == 1  # unknown code
    with pytest.raises(SystemExit):
        main(["override", code, "--once", "--always"])


def test_list_and_revoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "o.db"
    assert _run("list", "--db", str(db)) == 0
    assert "no pending refusals" in capsys.readouterr().out
    code = _pending(db, subject="alice")
    OverrideStore(db).approve("always", approver="alice", code=code)
    _pending(db)
    assert _run("list", "--db", str(db)) == 0
    table = capsys.readouterr().out
    assert "always" in table and "alice" in table and "operator" in table
    assert EMAIL not in table
    assert _run("list", "--json", "--db", str(db)) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [e["state"] for e in listed] == ["pending", "always"]
    assert EMAIL not in json.dumps(listed)
    rule = listed[1]["id"]
    assert _run("revoke", rule, "--db", str(db)) == 0
    assert _run("revoke", rule, "--db", str(db)) == 1
    assert [e.state for e in OverrideStore(db).entries()] == ["pending"]


def test_the_store_path_comes_from_the_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = tmp_path / "from-config.db"
    config = tmp_path / "config.toml"
    config.write_text(f'[overrides]\npath = "{db}"\n')
    _pending(db)
    assert _run("list", "--json", "--config", str(config)) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1


def test_the_real_terminal_opener_fails_without_one(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no tty")

    monkeypatch.setattr("builtins.open", refuse)
    with pytest.raises(OSError):
        override_cli._open_tty()


# --- doctor and the status posture ----------------------------------------------------


def _doctor(config: Any) -> list[dict[str, str]]:
    from llm_redact.doctor_cli import _check_overrides, _Report

    report = _Report(json_mode=True)
    _check_overrides(report, config)
    return report.rows


def test_doctor_reports_approved_overrides_as_counts(tmp_path: Path) -> None:
    from llm_redact.config import Config, OverridesConfig

    db = tmp_path / "o.db"
    config = Config(overrides=OverridesConfig(path=str(db)))
    (none,) = _doctor(config)
    assert none["level"] == "PASS" and "none approved" in none["message"]
    assert not db.exists()  # doctor never creates the store
    OverrideStore(db).approve("always", approver=None, code=_pending(db))
    (warn,) = _doctor(config)
    assert warn["level"] == "WARN" and "1 every-time rule(s)" in warn["message"]
    assert EMAIL not in warn["message"]
    (off,) = _doctor(Config(overrides=OverridesConfig(enabled=False)))
    assert off["level"] == "PASS" and "disabled" in off["message"]
    (broken,) = _doctor(Config(overrides=OverridesConfig(path=str(tmp_path))))
    assert broken["level"] == "WARN" and "could not be read" in broken["message"]


def test_doctor_posture_counts_overrides_as_an_opt_out(tmp_path: Path) -> None:
    from llm_redact.config import Config, OverridesConfig
    from llm_redact.doctor_cli import _check_posture, _Report

    db = tmp_path / "o.db"
    OverrideStore(db).approve("always", approver=None, code=_pending(db))
    report = _Report(json_mode=True)
    _check_posture(report, Config(overrides=OverridesConfig(path=str(db))))
    assert not any("no coverage opt-outs" in row["message"] for row in report.rows)


def test_status_posture_line(capsys: pytest.CaptureFixture[str]) -> None:
    from llm_redact.cli import _print_posture

    base = {"warnings_total": {}, "detection": {}, "audit": {}}
    _print_posture({**base, "overrides": {"enabled": True, "always": 0, "used_total": {}}})
    assert "overrides" not in capsys.readouterr().out
    _print_posture({**base, "overrides": {"always": 2, "used_total": {"once": 1}}})
    out = capsys.readouterr().out
    assert "2 every-time rule(s), uses once×1" in out and "FORWARDED" in out
    _print_posture({**base, "overrides": {"always": 1, "used_total": {}}})
    assert "uses none yet" in capsys.readouterr().out
