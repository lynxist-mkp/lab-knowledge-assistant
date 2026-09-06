from __future__ import annotations

from wenmai.config import Settings
from wenmai.http.ask_governor import get_ask_governor
from wenmai.knowledge.collections import resolve_collection_scope
from wenmai.knowledge.store import Knowledge
from wenmai.models import AskResult
from wenmai.pipelines.query_orchestration import AskPipelineInput, ask_pipeline_single


def run_ask(
    question: str,
    settings: Settings,
    *,
    collection_id: str | None = None,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
    knowledge: Knowledge | None = None,
    entrypoint: str = "service",
) -> AskResult:
    """Stable service-layer seam for external ask entrypoints.

    HTTP handlers and MCP tools should call here so they share one outward
    contract before delegating into the deeper 提问编排 implementation.
    """
    scope = resolve_collection_scope(settings, collection_id)
    governor = get_ask_governor(scope.settings)
    with governor.acquire(scope.settings, entrypoint=entrypoint):
        return ask_pipeline_single(
            AskPipelineInput(
                question=question,
                settings=scope.settings,
                culture_domain=culture_domain,
                retrieval_mode=retrieval_mode,
                rerank_enabled=rerank_enabled,
                knowledge=knowledge,
            ),
            phase_batch=settings.resources.query_phase_batch,
        )
