"""Stdio transport discipline for the MCP server."""

from __future__ import annotations

import builtins
import io
import logging
import re
import sys
from typing import TextIO

_MAX_ERROR_LEN = 500
_PATH_PATTERN = re.compile(r"(?:/|\\)[^\s\"']+")
_SECRET_PATTERN = re.compile(
    r"(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+",
    re.IGNORECASE,
)
_BUILTIN_PRINT = builtins.print
_INSTALLED = False


def redact_error_message(message: str, *, max_len: int = _MAX_ERROR_LEN) -> str:
    text = str(message or "")
    text = _SECRET_PATTERN.sub("[REDACTED]", text)
    text = _PATH_PATTERN.sub("[PATH]", text)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _safe_print(*args: object, **kwargs: object) -> None:
    file = kwargs.get("file", sys.stdout)
    if file is sys.stdout:
        kwargs = dict(kwargs)
        kwargs["file"] = sys.stderr
    _BUILTIN_PRINT(*args, **kwargs)


def _configure_logging_to_stderr() -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(handler)
        root.setLevel(logging.WARNING)
        return
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            handler.stream = sys.stderr


def install_mcp_stdio_discipline() -> None:
    """Route operational output away from the MCP protocol stdout stream."""
    global _INSTALLED
    if _INSTALLED:
        return
    _configure_logging_to_stderr()
    builtins.print = _safe_print
    _INSTALLED = True


def is_mcp_stdio_discipline_installed() -> bool:
    return _INSTALLED


class StdioPollutionCapture:
    """Capture writes that would pollute the MCP protocol stdout stream."""

    def __init__(self) -> None:
        self._buffer = io.StringIO()
        self._real_stdout = sys.stdout

    def write(self, text: str) -> int:
        self._buffer.write(text)
        return len(text)

    def flush(self) -> None:
        return None

    @property
    def pollution(self) -> str:
        return self._buffer.getvalue()

    def assert_clean(self) -> None:
        captured = self.pollution.strip()
        if captured:
            raise AssertionError(f"protocol-hostile stdout output detected: {captured[:200]}")


def assert_stdout_reserved_for_protocol(action: object) -> None:
    """Fail when *action* writes non-empty content to stdout."""
    capture = StdioPollutionCapture()
    original_stdout = sys.stdout
    sys.stdout = capture  # type: ignore[assignment]
    try:
        action()  # type: ignore[operator]
    finally:
        sys.stdout = original_stdout
    capture.assert_clean()


def operational_stream() -> TextIO:
    return sys.stderr


__all__ = [
    "StdioPollutionCapture",
    "assert_stdout_reserved_for_protocol",
    "install_mcp_stdio_discipline",
    "is_mcp_stdio_discipline_installed",
    "operational_stream",
    "redact_error_message",
]
