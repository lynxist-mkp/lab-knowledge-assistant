"""检索 seam helpers — mode resolve + Trace attach (fusion is the deep 检索 interface)."""

from __future__ import annotations

from dataclasses import replace

from wenmai.config import Settings
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.retrieval.fusion import RetrievalResult
from wenmai.tracing.stages.retrieval import stages_from_fusion

_RETRIEVAL_MODES = frozenset({"rrf", "dense_only", "sparse_only"})


def attach_retrieval_trace_stages(
    result: RetrievalResult,
    settings: Settings,
    *,
    knowledge: Knowledge | None = None,
    culture_domain: str | None = None,
) -> RetrievalResult:
    """Attach fusion trace stages at the 提问编排 seam."""
    knowledge = knowledge or create_knowledge(settings)
    stages = stages_from_fusion(knowledge, settings, result, culture_domain)
    return replace(result, stages=stages)


def _resolve_mode(settings: Settings, retrieval_mode: str | None) -> str:
    mode = settings.retrieval.mode if retrieval_mode is None else retrieval_mode
    if mode not in _RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval_mode: {mode!r}")
    return mode


def resolve_retrieval_mode(
    settings: Settings,
    retrieval_mode: str | None,
    rerank_enabled: bool | None,
) -> tuple[str, bool]:
    mode = _resolve_mode(settings, retrieval_mode)
    rerank = settings.retrieval.rerank_enabled if rerank_enabled is None else rerank_enabled
    if mode != "rrf":
        rerank = False
    return mode, bool(rerank)


__all__ = [
    "attach_retrieval_trace_stages",
    "resolve_retrieval_mode",
]
