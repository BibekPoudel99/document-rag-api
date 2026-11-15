import json
import logging
import sys

# Libraries that log every HTTP request at INFO; keep them quiet unless they warn.
_NOISY_LOGGERS = (
    "httpx",
    "httpx2",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "urllib3",
    "filelock",
)


class JsonFormatter(logging.Formatter):
    """Emit each log record as one JSON line (easy to grep and ship)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, str] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)