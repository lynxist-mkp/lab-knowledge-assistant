from __future__ import annotations

import time
from typing import Any

from wenmai.config import Settings
from wenmai.http.ask_service import run_ask
from wenmai.knowledge.collections import UnknownCollectionError, resolve_collection_id
from wenmai.knowledge.store import Knowledge
from wenmai.mcp.envelope import McpMeta, envelope, refs_from_ask_result, scope_for
from wenmai.pipelines.query import QueryGenerationError


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
        resolve_collection_id(settings, collection_id)
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc

    started = time.perf_counter()
    try:
        result = run_ask(
            question,
            settings,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
        )
    except QueryGenerationError as exc:
        raise AskAnswerError(str(exc), exc.trace_id) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    return envelope(
        data=result.as_dict(),
        scope=scope_for(
            settings,
            collection_id=collection_id,
            culture_domain=culture_domain,
        ),
        refs=refs_from_ask_result(result),
        meta=McpMeta(elapsed_ms=elapsed_ms, mode=retrieval_mode),
    ).as_dict()
