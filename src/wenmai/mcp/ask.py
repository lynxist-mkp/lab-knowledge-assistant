from __future__ import annotations

from typing import Any

from wenmai.config import Settings
from wenmai.pipelines.query import QueryGenerationError, ask_question


class AskWenmaiError(Exception):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


def ask_wenmai(
    question: str,
    settings: Settings | None = None,
    culture_domain: str | None = None,
) -> dict[str, Any]:
    """MCP tool handler: ask via query pipeline and return structured result."""
    resolved = settings or Settings.load()
    try:
        result = ask_question(question, resolved, culture_domain=culture_domain)
    except QueryGenerationError as exc:
        raise AskWenmaiError(str(exc), exc.trace_id) from exc
    return result.as_dict()
