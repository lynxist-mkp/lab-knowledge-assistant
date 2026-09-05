from __future__ import annotations

from typing import Any

from wenmai.config import Settings
from wenmai.http.ask_service import run_ask
from wenmai.pipelines.query import QueryGenerationError


class AskWenmaiError(Exception):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


def ask_wenmai(
    question: str,
    settings: Settings | None = None,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
) -> dict[str, Any]:
    """Legacy MCP ask helper: return answer, citations, and trace_id."""
    resolved = settings or Settings.load()
    try:
        result = run_ask(
            question,
            resolved,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
        )
    except QueryGenerationError as exc:
        raise AskWenmaiError(str(exc), exc.trace_id) from exc
    return result.as_dict()
