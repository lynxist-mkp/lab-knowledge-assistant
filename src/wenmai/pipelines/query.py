"""Compatibility shim for legacy single-question query imports.

The formal outward seam for one 提问 is `wenmai.http.ask_service.run_ask()`.
This module remains only for legacy/test imports that still need direct access
to a minimal helper or query-normalization utilities.
"""

from __future__ import annotations

from wenmai.config import Settings
from wenmai.generation import QueryGenerationError
from wenmai.knowledge import Knowledge
from wenmai.models import AskResult
from wenmai.pipelines.query_batch import get_query_coordinator, window_batch_enabled
from wenmai.pipelines.query_orchestration import (
    AskPipelineInput,
    ask_pipeline_single,
    normalize_question,
)

__all__ = ["QueryGenerationError", "ask_question", "normalize_question"]


def ask_question(
    question: str,
    settings: Settings,
    culture_domain: str | None = None,
    *,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
    knowledge: Knowledge | None = None,
    record_trace: bool = True,
) -> AskResult:
    if record_trace and not window_batch_enabled(settings):
        # Keep the common path aligned with the formal service seam.
        from wenmai.http.ask_service import run_ask

        return run_ask(
            question,
            settings,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            entrypoint="query-compat",
        )

    if window_batch_enabled(settings):
        return get_query_coordinator(settings).submit(
            question,
            settings,
            culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            record_trace=record_trace,
        )

    phase_batch = settings.resources.query_phase_batch
    return ask_pipeline_single(
        AskPipelineInput(
            question=question,
            settings=settings,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            record_trace=record_trace,
        ),
        phase_batch=phase_batch,
    )
