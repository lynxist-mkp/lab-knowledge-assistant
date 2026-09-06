from __future__ import annotations

from time import perf_counter
from typing import Any

from lab_knowledge.config import Settings
from lab_knowledge.generation import QueryGenerationError
from lab_knowledge.http.ask_service import run_ask
from lab_knowledge.knowledge.collections import (
    UnknownCollectionError,
    resolve_routable_collection_scope,
)
from lab_knowledge.knowledge.store import Knowledge
from lab_knowledge.mcp.envelope import McpMeta, envelope, refs_from_ask_result, scope_for


class AskAnswerError(Exception):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


def ask_answer(
    question: str,
    settings: Settings,
    *,
    collection_id: str | None = None,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
    knowledge: Knowledge | None = None,
) -> dict[str, Any]:
    try:
        scope = resolve_routable_collection_scope(settings, collection_id)
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc

    try:
        started = perf_counter()
        result = run_ask(
            question,
            scope.settings,
            collection_id=scope.collection_id,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            entrypoint="mcp",
        )
        elapsed_ms = (perf_counter() - started) * 1000
    except QueryGenerationError as exc:
        raise AskAnswerError(str(exc), exc.trace_id) from exc

    return envelope(
        data=result.as_dict(),
        scope=scope_for(
            scope.settings,
            collection_id=scope.collection_id,
            culture_domain=culture_domain,
        ),
        refs=refs_from_ask_result(result),
        meta=McpMeta(elapsed_ms=elapsed_ms, mode=retrieval_mode),
    ).as_dict()
