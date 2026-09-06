from __future__ import annotations

from typing import Any

from wenmai.ask_surface import AskSurfaceError, ask_surface
from wenmai.config import Settings
from wenmai.knowledge.collections import UnknownCollectionError


class AskWenmaiError(Exception):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


def ask_wenmai(
    question: str,
    settings: Settings | None = None,
    collection_id: str | None = None,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
) -> dict[str, Any]:
    """Legacy MCP ask helper: return answer, citations, and trace_id."""
    resolved = settings or Settings.load()
    try:
        result = ask_surface(
            question,
            resolved,
            collection_id=collection_id,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            entrypoint="mcp-legacy",
        )
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc
    except AskSurfaceError as exc:
        raise AskWenmaiError(str(exc), exc.trace_id) from exc
    return result.result.as_dict()
