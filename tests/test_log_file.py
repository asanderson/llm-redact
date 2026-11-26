"""`serve` logs to a rotating file (`[log] file`, `serve --log-file`) and
starts without a console (pythonw, a hidden logon task), logging to
``paths.log_dir()/proxy.log`` from its first line.

Lines carry exactly the stderr content (text: level, logger and message;
json: the fixed value-free key set), a text line in a file prefixed with its
time; UTF-8 with LF line ends on every platform."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import socket
import stat
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

import pytest

from llm_redact import log, paths
from llm_redact.config import Config, ConfigError, LogConfig, parse_config
from llm_redact.config_write import emit_config_toml


def _record(message: str, *args: object) -> logging.LogRecord:
    return logging.LogRecord("llm_redact", logging.INFO, __file__, 1, message, args, None)


def _write(handler: logging.Handler, *records: logging.LogRecord) -> None:
    try:
        for record in records:
            handler.handle(record)
    finally:
        handler.close()


def test_a_text_line_in_the_file_is_the_stderr_line_with_its_time(tmp_path: Path) -> None:
    path = tmp_path / "proxy.log"
    _write(log.log_handler("text", path), _record("POST %s -> %d redacted: EMAIL×%d", "/p", 200, 2))
    data = path.read_bytes()
    assert b"\r" not in data
    line = data.decode("utf-8")
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00"
        r" INFO llm_redact: POST /p -> 200 redacted: EMAIL×2\n",
        line,
    ), line


def test_a_json_line_in_the_file_has_the_fixed_keys(tmp_path: Path) -> None:
    path = tmp_path / "proxy.log"
    _write(log.log_handler("json", path), _record("POST %s -> %d", "/p", 200))
    (line,) = path.read_text(encoding="utf-8").splitlines()
    parsed = json.loads(line)
    assert set(parsed) == {"ts", "level", "logger", "service", "version", "message"}
    assert parsed["message"] == "POST /p -> 200"


def test_the_file_rotates_at_its_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(log, "LOG_FILE_BYTES", 400)
    monkeypatch.setattr(log, "LOG_FILE_BACKUPS", 2)
    path = tmp_path / "proxy.log"
    _write(log.log_handler("text", path), *(_record("request %d", n) for n in range(60)))
    names = sorted(entry.name for entry in tmp_path.iterdir())
    assert names == ["proxy.log", "proxy.log.1", "proxy.log.2"]
    for name in names:
        assert (tmp_path / name).stat().st_size <= 400
    # The newest lines are in the current file.
    assert "request 59" in path.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_the_file_is_private_to_its_owner_after_a_rotation_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(log, "LOG_FILE_BYTES", 200)
    path = tmp_path / "logs" / "proxy.log"
    _write(log.log_handler("text", path), *(_record("request %d", n) for n in range(20)))
    assert stat.S_IMODE((tmp_path / "logs").stat().st_mode) == 0o700
    for entry in (tmp_path / "logs").iterdir():
        assert stat.S_IMODE(entry.stat().st_mode) == 0o600, entry.name


def test_a_line_the_encoding_cannot_hold_is_escaped_not_lost(tmp_path: Path) -> None:
    path = tmp_path / "proxy.log"
    _write(log.log_handler("text", path), _record("path %s", "/a\udcff"))
    assert path.read_text(encoding="utf-8").endswith("path /a\\udcff\n")


def test_where_serve_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert log.log_destination(None, console=True) is None
    assert log.log_destination(None, console=False) == paths.log_dir() / "proxy.log"
    assert log.log_destination(str(tmp_path / "a.log"), console=True) == tmp_path / "a.log"
    assert log.log_destination("~/a.log", console=False) == Path.home() / "a.log"


def test_the_config_names_an_absolute_log_file(tmp_path: Path) -> None:
    file = str(tmp_path / "proxy.log")
    assert parse_config({"log": {"file": file}}, "t").log.file == file
    assert parse_config({"log": {"file": "~/proxy.log"}}, "t").log.file == "~/proxy.log"
    assert parse_config({}, "t").log.file is None
    for bad, message in (
        ("proxy.log", "absolute"),
        ("logs/proxy.log", "absolute"),
        ("", "non-empty string"),
        (7, "non-empty string"),
        ("/var/log/a\nb.log", "control characters"),
    ):
        with pytest.raises(ConfigError, match=message):
            parse_config({"log": {"file": bad}}, "t")


@pytest.mark.parametrize(
    "file", ["C:\\Users\\Ada\\AppData\\Local\\llm-redact\\logs\\proxy.log", "/var/log/llm.log"]
)
def test_the_emitter_round_trips_the_log_file(file: str) -> None:
    text = emit_config_toml(Config(log=LogConfig(format="json", file=file)))
    assert tomllib.loads(text)["log"] == {"format": "json", "file": file}
    assert "file =" not in emit_config_toml(Config())


def _serve_args(config: Path, log_file: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=config, log_format=None, log_file=log_file, session=None, port=None, check=False
    )


def test_serve_log_file_beats_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_redact.cli import _serve_config

    config = tmp_path / "config.toml"
    config.write_text(f"[log]\nfile = {json.dumps(str(tmp_path / 'a.log'))}\n", encoding="utf-8")
    assert _serve_config(_serve_args(config)).log.file == str(tmp_path / "a.log")
    monkeypatch.chdir(tmp_path)
    # A relative flag is read against the directory serve starts in.
    chosen = _serve_config(_serve_args(config, "b.log")).log.file
    assert chosen == str(tmp_path / "b.log")


def test_serve_check_fails_on_a_log_file_it_cannot_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llm_redact.cli import main

    blocker = tmp_path / "not-a-folder"
    blocker.write_text("", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(f"[log]\nfile = {json.dumps(str(blocker / 'proxy.log'))}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_:
        main(["serve", "--check", "--config", str(config)])
    assert exit_.value.code == 1
    err = capsys.readouterr().err
    assert "serve --check: FAIL" in err and "[log] file" in err


@pytest.mark.parametrize(
    ("raised", "said"),
    [
        (None, None),
        (ConfigError("[detection] unknown key: no_such_key"), "serve did not start: [detection]"),
        (RuntimeError("boom"), "serve did not start"),
    ],
)
def test_a_start_without_a_console_says_why_it_failed(
    raised: Exception | None,
    said: str | None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # In-process (the subprocess tests below run it for real): the log
    # line, and the exit code serve returns.
    from llm_redact import cli

    def run_serve(args: argparse.Namespace) -> None:
        if raised is not None:
            raise raised

    monkeypatch.setattr(cli, "_run_serve", run_serve)
    with caplog.at_level(logging.ERROR, logger="llm_redact"):
        code = cli._serve_without_console(_serve_args(Path("unused.toml")))
    assert code == (0 if raised is None else 1)
    if said is None:
        assert "serve did not start" not in caplog.text
    else:
        assert said in caplog.text


def test_a_delayed_log_file_never_written_leaves_nothing(tmp_path: Path) -> None:
    log.log_handler("text", tmp_path / "logs" / "proxy.log", delay=True).close()
    assert list(tmp_path.iterdir()) == []


def _free_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _serves(port: int, process: subprocess.Popen[bytes], seconds: float = 60) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            url = f"http://127.0.0.1:{port}/__llm-redact/healthz"
            with urllib.request.urlopen(url, timeout=2) as answer:  # noqa: S310 — loopback
                if answer.status == 200:
                    return True
        except OSError:
            time.sleep(0.2)
    return False


def _serve_without_console(
    command: list[str], port: int, env: dict[str, str], log_file: Path, *, handles: bool = True
) -> str:
    # pythonw with standard handles (DEVNULL) gets a sys.stderr writing to
    # NUL; started as Task Scheduler starts it, with none, it has no
    # sys.stdout or sys.stderr at all.
    streams = (
        {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if handles
        else {}
    )
    process = subprocess.Popen(command, env=env, **streams)
    try:
        assert _serves(port, process), (
            log_file.read_text(encoding="utf-8") if log_file.exists() else "no log"
        )
    finally:
        process.terminate()
        process.wait(timeout=30)
    return log_file.read_text(encoding="utf-8")


def test_serve_starts_without_a_console_and_logs_to_the_default_file(tmp_path: Path) -> None:
    # What pythonw (or a hidden logon task) gives a program: no stdout and
    # no stderr. uvicorn's default log config read sys.stdout.isatty() and
    # failed to start ("Unable to configure formatter 'default'").
    port = _free_port()
    config = tmp_path / "config.toml"
    config.write_text(f"port = {port}\n", encoding="utf-8")
    env = dict(os.environ, XDG_STATE_HOME=str(tmp_path / "state"))
    script = (
        "import sys; sys.stdout = sys.stderr = None;"
        " from llm_redact.cli import main; main(sys.argv[1:])"
    )
    log_file = tmp_path / "state" / "llm-redact" / "proxy.log"
    text = _serve_without_console(
        [sys.executable, "-c", script, "serve", "--config", str(config)], port, env, log_file
    )
    assert f"serving on http://127.0.0.1:{port}" in text


def test_serve_without_a_console_logs_why_it_did_not_start(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[detection]\nno_such_key = 1\n", encoding="utf-8")
    env = dict(os.environ, XDG_STATE_HOME=str(tmp_path / "state"))
    script = (
        "import sys; sys.stdout = sys.stderr = None;"
        " from llm_redact.cli import main; main(sys.argv[1:])"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, "serve", "--config", str(config)],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert done.returncode != 0
    text = (tmp_path / "state" / "llm-redact" / "proxy.log").read_text(encoding="utf-8")
    assert "serve did not start" in text and "no_such_key" in text


@pytest.mark.skipif(sys.platform != "win32", reason="pythonw.exe is the Windows GUI interpreter")
def test_pythonw_serves_and_logs_under_localappdata(tmp_path: Path) -> None:
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pytest.skip(f"no pythonw.exe beside {sys.executable}")
    port = _free_port()
    config = tmp_path / "config.toml"
    config.write_text(f"port = {port}\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("XDG_")}
    env["LOCALAPPDATA"] = str(tmp_path / "AppData-Local")
    log_file = tmp_path / "AppData-Local" / "llm-redact" / "logs" / "proxy.log"
    text = _serve_without_console(
        [str(pythonw), "-m", "llm_redact", "serve", "--config", str(config)],
        port,
        env,
        log_file,
        handles=False,
    )
    assert f"serving on http://127.0.0.1:{port}" in text
