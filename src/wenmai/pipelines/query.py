from __future__ import annotations

from wenmai.config import Settings
from wenmai.generation import QueryGenerationError
from wenmai.knowledge import Knowledge
from wenmai.models import AskResult
from wenmai.pipelines.query_core import (
    AskPipelineInput,
    ask_pipeline_single,
    normalize_question,
)
from wenmai.pipelines.query_batch import get_query_coordinator, window_batch_enabled

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
