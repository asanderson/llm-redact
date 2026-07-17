"""Structured logging: one JSON object per line, for log shippers.

The formatter serializes only a FIXED, value-free field set — timestamp,
level, logger name, the fully interpolated message, plus static `service`
and `version` identifiers for filtering in aggregated multi-service log
streams (k8s, journald), and exception text when present. It deliberately
ignores everything else on the record: log content is value-free by design
(paths, statuses, detection counts), and JSON mode must change the framing,
never widen the content — in particular it never serializes arbitrary record
attributes someone attaches later.
"""

import json
import logging
from datetime import UTC, datetime

from llm_redact import __version__

_SERVICE = "llm-redact"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "service": _SERVICE,
            "version": __version__,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(log_format: str) -> None:
    """Root logging for `serve`: text (human) or json (one object/line)."""
    if log_format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logging.basicConfig(level=logging.INFO, handlers=[handler])
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # httpx logs full request URLs at INFO — query strings can carry
    # provider API keys (Gemini ?key=), so it stays at WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
