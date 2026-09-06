from __future__ import annotations

from typing import Any

from lab_knowledge.config import Settings
from lab_knowledge.generation import QueryGenerationError
from lab_knowledge.http.ask_service import run_ask
from lab_knowledge.knowledge.collections import UnknownCollectionError


class AskLabKnowledgeError(Exception):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


def ask_lab_knowledge(
    question: str,
    settings: Settings | None = None,
    collection_id: str | None = None,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
) -> dict[str, Any]:
    """Primary MCP ask helper: return answer, citations, and trace_id."""
    resolved = settings or Settings.load()
    try:
        result = run_ask(
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
    except QueryGenerationError as exc:
        raise AskLabKnowledgeError(str(exc), exc.trace_id) from exc
    return result.as_dict()


AskWenmaiError = AskLabKnowledgeError


def ask_wenmai(
    question: str,
    settings: Settings | None = None,
    collection_id: str | None = None,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
) -> dict[str, Any]:
    """Legacy compatibility alias for older MCP clients."""
    return ask_lab_knowledge(
        question,
        settings=settings,
        collection_id=collection_id,
        culture_domain=culture_domain,
        retrieval_mode=retrieval_mode,
        rerank_enabled=rerank_enabled,
    )
