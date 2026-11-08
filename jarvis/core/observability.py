"""Structured logging, secret redaction and trace context.

The redactor is process-global: any secret value read from the secret store (and
any token minted by the auth service) is registered here, and every log record,
event payload and audit entry is passed through :func:`redact` before it is
written anywhere.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("jarvis_trace_id", default=None)

# Patterns for credentials that may appear in arbitrary text (tool output, errors).
_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),  # JWT
    re.compile(r"jv[dsp]_[A-Za-z0-9_\-]{20,}"),  # JARVIS device/session/pairing tokens
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=]{12,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password|passwd)\s*[=:]\s*)[\"']?[^\s\"',;]{6,}"),
]

SENSITIVE_KEYS = {"api_key", "apikey", "token", "secret", "password", "passwd", "authorization", "access_token", "refresh_token", "device_token", "session_token", "pairing_code", "code"}

REDACTED = "[REDACTED]"


class Redactor:
    def __init__(self) -> None:
        self._values: set[str] = set()
        self._lock = threading.Lock()

    def register(self, value: str | None) -> None:
        if value and len(value) >= 6:
            with self._lock:
                self._values.add(value)

    def redact_text(self, text: str) -> str:
        if not text:
            return text
        with self._lock:
            values = sorted(self._values, key=len, reverse=True)
        for v in values:
            if v in text:
                text = text.replace(v, REDACTED)
        for pat in _PATTERNS:
            if pat.groups:
                text = pat.sub(lambda m: m.group(1) + REDACTED, text)
            else:
                text = pat.sub(REDACTED, text)
        return text

    def redact(self, obj: Any, _depth: int = 0) -> Any:
        if _depth > 12:
            return obj
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.lower() in SENSITIVE_KEYS and isinstance(v, str) and v:
                    out[k] = REDACTED
                else:
                    out[k] = self.redact(v, _depth + 1)
            return out
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.redact(v, _depth + 1) for v in obj)
        return obj


redactor = Redactor()


def redact(obj: Any) -> Any:
    return redactor.redact(obj)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def get_trace_id() -> str | None:
    return _trace_id.get()


def set_trace_id(trace_id: str | None) -> contextvars.Token:
    return _trace_id.set(trace_id)


def reset_trace_id(token: contextvars.Token) -> None:
    _trace_id.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        trace = get_trace_id()
        if trace:
            payload["trace_id"] = trace
        extra = getattr(record, "data", None)
        if extra is not None:
            payload["data"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redactor.redact_text(json.dumps(redact(payload), default=str))


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = getattr(record, "data", None)
        if extra is not None:
            base += " " + json.dumps(extra, default=str)
        return redactor.redact_text(base)


def setup_logging(level: str = "INFO", log_dir: Path | None = None, json_logs: bool = True) -> None:
    root = logging.getLogger("jarvis")
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(TextFormatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(console)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "jarvis.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(JsonFormatter() if json_logs else TextFormatter())
        root.addHandler(fh)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("jarvis") else f"jarvis.{name}")


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def ms(self) -> float:
        return round((time.perf_counter() - self.start) * 1000, 1)
