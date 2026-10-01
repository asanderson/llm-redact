"""``llm-redact override``: the approval needs a person at the terminal.

The confirmation is read from the controlling terminal (a fake one here),
never from stdin; without a terminal the command refuses. Listing and
revocation are value-free."""

from __future__ import annotations

import io
import json
import os
import select
import sys
import time
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

    monkeypatch.setattr(override_cli.io, "FileIO", refuse)
    with pytest.raises(OSError):
        override_cli._open_tty()


def test_the_terminal_opener_closes_the_terminal_when_it_cannot_wrap_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    class Raw:
        def close(self) -> None:
            closed.append(True)

    def unwrappable(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("no text layer")

    monkeypatch.setattr(override_cli.io, "FileIO", lambda *args: Raw())
    monkeypatch.setattr(override_cli.io, "TextIOWrapper", unwrappable)
    with pytest.raises(ValueError):
        override_cli._open_tty()
    assert closed == [True]


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


def _pty_run(argv: list[str], answer: bytes | None) -> tuple[int, str]:
    """Run the real CLI with a pseudo-terminal as its controlling terminal
    (``pty.fork``), typing ``answer`` once it asks; (exit code, output)."""
    import pty

    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover — the child execs at once
        os.execv(
            sys.executable,
            [sys.executable, "-c", "from llm_redact.cli import main; main()", *argv],
        )
    out = b""
    deadline = time.monotonic() + 60
    typed = answer is None
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 1024)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
        if not typed and b"to confirm" in out:
            assert answer is not None
            os.write(fd, answer)
            typed = True
    os.close(fd)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), out.decode("utf-8", "replace")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs a pseudo-terminal")
def test_the_real_terminal_opener_reads_a_typed_confirmation(tmp_path: Path) -> None:
    """``_open_tty`` itself (no fake) on a real controlling terminal: it once
    opened /dev/tty as a seekable buffered file, which every terminal
    refuses, so the command always answered 'needs a terminal'."""
    db = tmp_path / "o.db"
    code = _pending(db)
    status, out = _pty_run(["override", code, "--once", "--db", str(db)], b"allow\n")
    assert status == 0, out
    assert "kind block, types EMAIL" in out and "approved once" in out
    assert [e.state for e in OverrideStore(db).entries()] == ["once"]
    refused = _pending(db)
    status, out = _pty_run(["override", refused, "--always", "--db", str(db)], b"no\n")
    assert status == 1 and "not approved" in out


def test_listing_and_doctor_never_write_the_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`override list` and doctor read the store read-only: no schema, no
    key, no journal-mode change, no chmod — the file's bytes and its
    directory's mode stay as they were, and a file that is not a store
    (here: empty) is never turned into one."""
    import hashlib

    from llm_redact.config import Config, OverridesConfig
    from llm_redact.doctor_cli import _check_overrides, _Report

    store_dir = tmp_path / "data"
    db = store_dir / "o.db"
    writer = OverrideStore(db)
    code = writer.record_pending("block", "", "openai", "POST", "/v1/x", [("EMAIL", EMAIL)])
    writer.approve("always", approver=None, code=code)
    writer.close()
    os.chmod(store_dir, 0o755)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert _run("list", "--db", str(db)) == 0
    assert "always" in capsys.readouterr().out
    report = _Report()
    assert _check_overrides(report, Config(overrides=OverridesConfig(path=str(db)))) is True
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    assert (store_dir.stat().st_mode & 0o777) == 0o755
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    assert _run("list", "--db", str(empty)) == 0
    assert "no pending refusals" in capsys.readouterr().out
    assert _check_overrides(_Report(), Config(overrides=OverridesConfig(path=str(empty)))) is False
    assert empty.read_bytes() == b""
