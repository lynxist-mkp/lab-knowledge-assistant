"""Structured MCP tool errors for ask-path saturation."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError

from wenmai.http.ask_governor import AskSaturationCode, AskSaturationError
from wenmai.mcp.stdio_safety import redact_error_message


class AskSaturationToolError(ToolError):
    """MCP tool error that preserves ask-governor saturation metadata."""

    def __init__(self, exc: AskSaturationError) -> None:
        detail = exc.as_dict()
        detail["message"] = redact_error_message(str(detail["message"]))
        self.code: AskSaturationCode = exc.code
        self.detail = detail
        super().__init__(json.dumps(detail, ensure_ascii=False))

    def as_dict(self) -> dict[str, Any]:
        return dict(self.detail)


def saturation_detail_from_tool_error(exc: ToolError) -> dict[str, Any] | None:
    """Parse structured saturation detail from an MCP tool error message."""
    message = str(exc)
    prefix = "Error executing tool "
    if prefix in message:
        message = message.split(": ", 1)[-1]
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "code" not in payload:
        return None
    return payload


__all__ = [
    "AskSaturationToolError",
    "saturation_detail_from_tool_error",
]
