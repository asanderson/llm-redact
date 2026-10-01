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


@pytest.fixture(autouse=True)
def _overrides_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The commands below run under a config that turns overrides ON (the
    config search reads LLM_REDACT_CONFIG; the pseudo-terminal child inherits
    it). Off — the default — every form exits 1 naming the setting:
    test_every_form_names_the_setting_while_overrides_are_off."""
    config = tmp_path / "overrides-on.toml"
    config.write_text("[overrides]\nenabled = true\n")
    monkeypatch.setenv("LLM_REDACT_CONFIG", str(config))


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
    # A TOML literal string: a Windows path's backslashes are not escapes.
    config.write_text(f"[overrides]\nenabled = true\npath = '{db}'\n")
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
    config = Config(overrides=OverridesConfig(enabled=True, path=str(db)))
    (none,) = _doctor(config)
    assert none["level"] == "PASS" and "none approved" in none["message"]
    assert not db.exists()  # doctor never creates the store
    OverrideStore(db).approve("always", approver=None, code=_pending(db))
    (warn,) = _doctor(config)
    assert warn["level"] == "WARN" and "1 every-time rule(s)" in warn["message"]
    assert EMAIL not in warn["message"]
    (off,) = _doctor(Config(overrides=OverridesConfig(enabled=False)))
    assert off["level"] == "PASS" and "off (the default)" in off["message"]
    assert "[overrides] enabled = true" in off["message"]
    (broken,) = _doctor(Config(overrides=OverridesConfig(enabled=True, path=str(tmp_path))))
    assert broken["level"] == "WARN" and "could not be read" in broken["message"]


def test_doctor_posture_counts_overrides_as_an_opt_out(tmp_path: Path) -> None:
    from llm_redact.config import Config, OverridesConfig
    from llm_redact.doctor_cli import _check_posture, _Report

    db = tmp_path / "o.db"
    OverrideStore(db).approve("always", approver=None, code=_pending(db))
    report = _Report(json_mode=True)
    _check_posture(report, Config(overrides=OverridesConfig(enabled=True, path=str(db))))
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
    assert (
        _check_overrides(report, Config(overrides=OverridesConfig(enabled=True, path=str(db))))
        is True
    )
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    if sys.platform != "win32":  # POSIX mode bits are synthetic on Windows
        assert (store_dir.stat().st_mode & 0o777) == 0o755
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    assert _run("list", "--db", str(empty)) == 0
    assert "no pending refusals" in capsys.readouterr().out
    assert (
        _check_overrides(
            _Report(), Config(overrides=OverridesConfig(enabled=True, path=str(empty)))
        )
        is False
    )
    assert empty.read_bytes() == b""


def test_a_route_is_never_printed_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The route (and requester) come from the request: an escape sequence
    in a path (cursor-up, erase-line, a bidi override) must not rewrite or
    hide the kind and types the person confirms — in the prompt, the
    listing and its JSON form alike (OVR-4)."""
    db = tmp_path / "o.db"
    route = "/v1/conversations/conv\x1b[1A\x1b[2K\rkind block, types EMAIL\u202e\x85/items"
    escaped = r"/v1/conversations/conv\x1b[1A\x1b[2K\x0dkind block, types EMAIL\u202e\x85/items"
    store = OverrideStore(db)
    code = store.record_pending("block", "", "openai", "POST", route, [("AWS_KEY", "k")])
    store.record_pending("block", "bob\x1b]0;x\x07", "openai", "POST", route, [("EMAIL", EMAIL)])
    tty = FakeTty("no\n")
    monkeypatch.setattr(override_cli, "_open_tty", lambda: tty)
    assert _run(code, "--once", "--db", str(db)) == 1
    assert "kind block, types AWS_KEY" in tty.shown and escaped in tty.shown
    capsys.readouterr()
    assert _run("list", "--db", str(db)) == 0
    assert _run("list", "--json", "--db", str(db)) == 0
    out = capsys.readouterr().out
    for raw in ("\x1b", "\r", "\u202e", "\x85", "\x07"):
        assert raw not in out and raw not in tty.shown
    assert r"bob\x1b]0;x\x07" in out
    assert {entry.route for entry in OverrideStore(db).entries()} == {f"POST openai {escaped}"}


def test_a_long_route_never_pushes_the_kind_and_types_off_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A printable route of thousands of characters ending in a fake "kind
    block, types EMAIL" once left the prompt line ending in the requester's
    own text, the real kind and types scrolled away: a listed route (and
    requester) is cut to ``SHOWN_CHARS`` with its length named, and the
    line the person answers repeats the real kind and types."""
    db = tmp_path / "o.db"
    route = "/v1/conversations/" + "x" * 6000 + " kind block, types EMAIL, route POST /v1/items"
    code = OverrideStore(db).record_pending(
        "block", "", "openai", "POST", route, [("AWS_KEY", "k")]
    )
    tty = FakeTty("no\n")
    monkeypatch.setattr(override_cli, "_open_tty", lambda: tty)
    assert _run(code, "--once", "--db", str(db)) == 1
    assert "types EMAIL" not in tty.shown and "x" * 200 not in tty.shown
    answered = tty.shown.rstrip().splitlines()[-2]
    assert answered.startswith("Approve once for kind block, types AWS_KEY:")
    (entry,) = OverrideStore(db).entries()
    assert len(entry.route) < 200 and entry.route.endswith(f"({len(route) + 12} characters)")
    capsys.readouterr()
    assert _run("list", "--json", "--db", str(db)) == 0
    assert "x" * 200 not in capsys.readouterr().out


# --- overrides off (the default) --------------------------------------------------------


def _store_digest(db: Path) -> str:
    """The store file's bytes (a reader's WAL index files aside: a
    read-only connection to a WAL database may create them)."""
    import hashlib

    return hashlib.sha256(db.read_bytes()).hexdigest()


def test_every_form_names_the_setting_while_overrides_are_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Off (a config that does not opt in): approve and revoke exit 1 naming
    `[overrides] enabled`, never ask the terminal and touch nothing; list
    prints what the store holds (value-free) with a line saying it is inert,
    and exits 1 too; a config that cannot be parsed exits 2."""
    off = tmp_path / "off.toml"
    off.write_text("")  # the default: overrides off
    db = tmp_path / "o.db"
    writer = OverrideStore(db)
    code = writer.record_pending("block", "", "openai", "POST", "/v1/x", [("EMAIL", EMAIL)])
    rule_code = writer.record_pending("block", "", "openai", "POST", "/v1/x", [("EMAIL", "b@c.d")])
    writer.approve("always", approver=None, code=rule_code)
    writer.close()
    before = _store_digest(db)
    asked: list[bool] = []

    def tty() -> FakeTty:
        asked.append(True)
        return FakeTty("allow\n")

    monkeypatch.setattr(override_cli, "_open_tty", tty)
    capsys.readouterr()
    for scope in ("--once", "--always"):
        assert _run(code, scope, "--config", str(off), "--db", str(db)) == 1
        err = capsys.readouterr().err
        assert "[overrides] enabled = true" in err and "off" in err
        assert code not in err and EMAIL not in err
    assert asked == []
    (rule,) = [e for e in OverrideStore(db, read_only=True).entries() if e.state == "always"]
    assert _run("revoke", rule.id, "--config", str(off), "--db", str(db)) == 1
    assert "[overrides] enabled = true" in capsys.readouterr().err
    assert _run("revoke", "--config", str(off), "--db", str(db)) == 1
    capsys.readouterr()

    assert _run("list", "--config", str(off), "--db", str(db)) == 1
    listed = capsys.readouterr()
    assert "pending" in listed.out and "always" in listed.out and EMAIL not in listed.out
    assert "inert" in listed.err and "[overrides] enabled = true" in listed.err
    assert _run("list", "--json", "--config", str(off), "--db", str(db)) == 1
    listed = capsys.readouterr()
    assert [e["state"] for e in json.loads(listed.out)] == ["pending", "always"]
    assert "inert" in listed.err
    empty = tmp_path / "none.db"
    assert _run("list", "--config", str(off), "--db", str(empty)) == 1
    listed = capsys.readouterr()
    assert "no pending refusals" in listed.out and "inert" in listed.err
    assert not empty.exists()
    # Nothing was written: the same records, the same bytes.
    assert _store_digest(db) == before
    assert [e.state for e in OverrideStore(db, read_only=True).entries()] == ["pending", "always"]

    bad = tmp_path / "bad.toml"
    bad.write_text("[overrides]\nenabled = 'yes'\n")
    assert _run("list", "--config", str(bad), "--db", str(db)) == 2
    assert "[overrides]" in capsys.readouterr().err
    # Turned on, the same store is approvable again.
    on = tmp_path / "on.toml"
    on.write_text("[overrides]\nenabled = true\n")
    assert _run(code, "--once", "--config", str(on), "--db", str(db)) == 0
    assert asked == [True]


def test_doctor_warns_about_approvals_kept_while_overrides_are_off(tmp_path: Path) -> None:
    """Off, the store's approvals are inert — but they apply again the
    moment overrides are turned on: doctor WARNs with the counts and the
    file (never a value), reads the store read-only and never creates one,
    and does not count it as a coverage opt-out."""
    from llm_redact.config import Config, OverridesConfig
    from llm_redact.doctor_cli import _check_overrides, _check_posture, _Report

    db = tmp_path / "o.db"
    off = Config(overrides=OverridesConfig(enabled=False, path=str(db)))
    (quiet,) = _doctor(off)
    assert quiet["level"] == "PASS" and "off (the default)" in quiet["message"]
    assert not db.exists()
    writer = OverrideStore(db)
    for scope in ("always", "once"):
        code = writer.record_pending("block", "", "openai", "POST", "/v1/x", [("EMAIL", EMAIL)])
        writer.approve(scope, approver=None, code=code)
    writer.close()
    before = _store_digest(db)
    (warn,) = _doctor(off)
    assert warn["level"] == "WARN" and "inert" in warn["message"]
    assert "1 every-time rule(s) and 1 one-time grant(s)" in warn["message"]
    assert str(db) in warn["message"] and "[overrides] enabled = true" in warn["message"]
    assert EMAIL not in warn["message"]
    assert _check_overrides(_Report(json_mode=True), off) is False
    report = _Report(json_mode=True)
    _check_posture(report, off)
    assert any("no coverage opt-outs" in row["message"] for row in report.rows)
    assert _store_digest(db) == before
    # An unreadable store while off: a WARN naming the exception type only.
    (broken,) = _doctor(Config(overrides=OverridesConfig(enabled=False, path=str(tmp_path))))
    assert broken["level"] == "WARN" and "could not be read" in broken["message"]
