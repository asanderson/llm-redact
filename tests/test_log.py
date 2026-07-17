"""JSON log framing: every line one valid object, content unchanged."""

import json
import logging

import pytest

from llm_redact.config import ConfigError, parse_config
from llm_redact.log import JsonFormatter


def _record(
    message: str, *args: object, level: int = logging.INFO, exc_info: object = None
) -> logging.LogRecord:
    return logging.LogRecord(
        name="llm_redact",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=args,
        exc_info=exc_info,  # type: ignore[arg-type]
    )


def test_lines_are_valid_json_with_interpolated_message() -> None:
    formatted = JsonFormatter().format(_record("POST %s -> %d redacted: EMAIL×%d", "/p", 200, 2))
    parsed = json.loads(formatted)
    assert parsed["message"] == "POST /p -> 200 redacted: EMAIL×2"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "llm_redact"
    assert parsed["ts"].endswith("+00:00")
    assert parsed["service"] == "llm-redact"
    assert parsed["version"]  # static build id for log correlation
    # A FIXED, value-free key set — the formatter must never widen beyond this.
    assert set(parsed) == {"ts", "level", "logger", "service", "version", "message"}


def test_exception_text_included() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record("failed", level=logging.ERROR, exc_info=sys.exc_info())
    parsed = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in parsed["exception"]


def test_extra_record_attributes_never_widen_output() -> None:
    # Log content is value-free by design; the formatter must not start
    # serializing arbitrary record attributes someone attaches later.
    record = _record("hello")
    record.secret_value = "hunter2"  # type: ignore[attr-defined]
    parsed = json.loads(JsonFormatter().format(record))
    assert "hunter2" not in json.dumps(parsed)


def test_log_config_parses_and_validates() -> None:
    assert parse_config({"log": {"format": "json"}}, "test").log.format == "json"
    assert parse_config({}, "test").log.format == "text"
    with pytest.raises(ConfigError, match="text.*json"):
        parse_config({"log": {"format": "yaml"}}, "test")
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config({"log": {"fromat": "json"}}, "test")
