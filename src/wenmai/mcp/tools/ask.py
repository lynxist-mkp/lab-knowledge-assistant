from __future__ import annotations

from typing import Any

from wenmai.ask_surface import AskSurfaceError, ask_surface
from wenmai.config import Settings
from wenmai.knowledge.collections import UnknownCollectionError, resolve_routable_collection_scope
from wenmai.knowledge.store import Knowledge
from wenmai.mcp.envelope import McpMeta, envelope, refs_from_ask_result, scope_for


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
        ask = ask_surface(
            question,
            scope.settings,
            collection_id=scope.collection_id,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            entrypoint="mcp",
        )
    except AskSurfaceError as exc:
        raise AskAnswerError(str(exc), exc.trace_id) from exc

    return envelope(
        data=ask.result.as_dict(),
        scope=scope_for(
            scope.settings,
            collection_id=scope.collection_id,
            culture_domain=culture_domain,
        ),
        refs=refs_from_ask_result(ask.result),
        meta=McpMeta(elapsed_ms=ask.elapsed_ms, mode=retrieval_mode),
    ).as_dict()
