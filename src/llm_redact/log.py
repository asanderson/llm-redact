"""Logging for ``serve``: text or JSON lines, to stderr or a rotating file.

The JSON formatter serializes only a FIXED, value-free field set — timestamp,
level, logger name, the fully interpolated message, plus static `service`
and `version` identifiers for filtering in aggregated multi-service log
streams (k8s, journald), and exception text when present. It deliberately
ignores everything else on the record: log content is value-free by design
(paths, statuses, detection counts), and JSON mode must change the framing,
never widen the content — in particular it never serializes arbitrary record
attributes someone attaches later.

A log file (``[log] file``, ``serve --log-file``; without a console —
pythonw, a hidden logon task — ``paths.log_dir()/proxy.log``) holds the same
lines, a text line prefixed with its time (stderr has none: journald and
container runtimes add their own). It rotates at ``LOG_FILE_BYTES`` keeping
``LOG_FILE_BACKUPS`` older files, is UTF-8 with LF line ends on every
platform, and is created private to its owner (0600 in a 0700 folder on
POSIX).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from llm_redact import __version__, paths, winacl
from llm_redact.jsonwalk import json_text

_SERVICE = "llm-redact"
TEXT_FORMAT = "%(levelname)s %(name)s: %(message)s"
DEFAULT_LOG_FILE = "proxy.log"
# One file holds this much before it rotates; this many older ones are kept.
LOG_FILE_BYTES = 10 * 1024 * 1024
LOG_FILE_BACKUPS = 5


def _timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": _timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "service": _SERVICE,
            "version": __version__,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json_text(entry)


class _FileTextFormatter(logging.Formatter):
    """The stderr text line, prefixed with its time (UTC, as JSON's ``ts``)."""

    def __init__(self) -> None:
        super().__init__(f"%(asctime)s {TEXT_FORMAT}")

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # noqa: N802
        return _timestamp(record)


class LogFile(logging.handlers.RotatingFileHandler):
    """A rotating log file, created private to its owner (0600 in a 0700
    folder it makes; on Windows a protected access control list,
    ``winacl.make_private``, refusing a file another account owns), UTF-8
    with LF line ends on every platform, a character the file cannot hold
    escaped rather than the line lost."""

    def __init__(self, path: Path, *, delay: bool = False) -> None:
        super().__init__(
            path,
            maxBytes=LOG_FILE_BYTES,
            backupCount=LOG_FILE_BACKUPS,
            encoding="utf-8",
            delay=delay,
            errors="backslashreplace",
        )

    def _open(self) -> TextIO:  # type: ignore[override]
        # Its folder made here, so a delayed file that is never written
        # leaves nothing behind.
        folder = Path(self.baseFilename).parent
        if not folder.exists():
            folder.mkdir(parents=True, mode=0o700)
            winacl.make_private(folder, folder=True)  # Windows' 0700
        # O_BINARY: the C runtime would turn "\n" into "\r\n" on Windows.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        fd = os.open(self.baseFilename, flags, 0o600)
        try:
            winacl.make_private(Path(self.baseFilename))  # Windows' 0600
        except BaseException:
            os.close(fd)
            raise
        return os.fdopen(fd, "a", encoding="utf-8", errors="backslashreplace", newline="\n")


def default_log_file() -> Path:
    """Where ``serve`` logs when it has no console and no file is named."""
    return paths.log_dir() / DEFAULT_LOG_FILE


def log_destination(configured: str | None, *, console: bool) -> Path | None:
    """The file ``serve`` logs to: the configured one (``~`` expanded), else,
    without a console, ``default_log_file()``; None = stderr."""
    if configured is not None:
        return Path(configured).expanduser()
    return None if console else default_log_file()


def log_handler(
    log_format: str, log_file: Path | None = None, *, delay: bool = False
) -> logging.Handler:
    """The handler ``setup_logging`` installs: stderr, or ``log_file``."""
    handler: logging.Handler
    if log_file is None:
        handler = logging.StreamHandler()
        formatter: logging.Formatter = logging.Formatter(TEXT_FORMAT)
    else:
        handler = LogFile(log_file, delay=delay)
        formatter = _FileTextFormatter()
    handler.setFormatter(JsonFormatter() if log_format == "json" else formatter)
    return handler


# The handler setup_logging installed last: a later call replaces it (serve
# without a console logs to the default file until its config names one).
_installed: logging.Handler | None = None


def setup_logging(log_format: str, log_file: Path | None = None, *, delay: bool = False) -> None:
    """Root logging for `serve`: text (human) or json (one object/line), to
    stderr or ``log_file``. Like ``logging.basicConfig``, it installs nothing
    when the root logger already has handlers someone else set up."""
    global _installed
    root = logging.getLogger()
    if _installed is not None:
        root.removeHandler(_installed)
        _installed.close()
        _installed = None
    handler = log_handler(log_format, log_file, delay=delay)
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    if handler in root.handlers:
        _installed = handler
    else:
        handler.close()
    # httpx logs full request URLs at INFO — query strings can carry
    # provider API keys (Gemini ?key=), so it stays at WARNING. websockets
    # logs the whole upgrade request at DEBUG (the Authorization header, the
    # ?key= query) — pinned at WARNING too, so a root DEBUG level set later
    # can never switch it on.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
