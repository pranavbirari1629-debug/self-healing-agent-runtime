"""Structured JSON logging with correlation ID support for the shar runtime.

Provides a standard library based JSON logging formatter and setup utility
to ensure all runtime events are machine-readable and traceable.
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
from typing import Any, Optional


class JsonFormatter(logging.Formatter):
    """Custom logging formatter that serializes log records into JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp_iso = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.timezone.utc
        ).isoformat()

        log_data: dict[str, Any] = {
            "timestamp": timestamp_iso,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include correlation_id if provided on the record
        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Include optional contextual keys if present
        for attr in ("task_name", "outcome", "failure_type", "execution_time_ms"):
            val = getattr(record, attr, None)
            if val is not None:
                log_data[attr] = val

        # Handle exception information if attached
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logger(
    name: str = "shar",
    level: int = logging.INFO,
    stream: Optional[Any] = None,
) -> logging.Logger:
    """Configures and returns a logger with structured JSON formatting.

    Args:
        name: The logger name hierarchy (defaults to "shar").
        level: Logging threshold level (defaults to INFO).
        stream: Optional output stream (defaults to sys.stdout).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    # Prevent propagating to root logger to avoid double logging with standard formatters
    logger.propagate = False
    return logger
