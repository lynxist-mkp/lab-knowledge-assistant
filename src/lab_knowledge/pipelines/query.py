"""Compatibility shim for legacy single-question query imports.

The formal outward seam for one 提问 is `lab_knowledge.http.ask_service.run_ask()`.
This module remains only for legacy/test imports that still need direct access
to a minimal helper or query-normalization utilities.
"""

from __future__ import annotations

from lab_knowledge.config import Settings
from lab_knowledge.generation import QueryGenerationError
from lab_knowledge.knowledge import Knowledge
from lab_knowledge.models import AskResult
from lab_knowledge.pipelines.query_orchestration import normalize_question

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
    from lab_knowledge.http.ask_service import run_ask

    return run_ask(
        question,
        settings,
        culture_domain=culture_domain,
        retrieval_mode=retrieval_mode,
        rerank_enabled=rerank_enabled,
        knowledge=knowledge,
        record_trace=record_trace,
        entrypoint="query-compat",
    )
