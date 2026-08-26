"""Ablation config mapping and run artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from wenmai.config import Settings


@dataclass(frozen=True)
class AblationSpec:
    mode: str
    rerank_enabled: bool


_ABLATION_MAP: dict[str, AblationSpec] = {
    "dense_only": AblationSpec(mode="dense_only", rerank_enabled=False),
    "sparse_only": AblationSpec(mode="sparse_only", rerank_enabled=False),
    "rrf": AblationSpec(mode="rrf", rerank_enabled=False),
    "rrf_rerank": AblationSpec(mode="rrf", rerank_enabled=True),
}


def resolve_ablation(group: str) -> AblationSpec:
    spec = _ABLATION_MAP.get(group)
    if spec is None:
        raise ValueError(f"unknown ablation group: {group!r}")
    return spec


def apply_ablation(settings: Settings, group: str) -> Settings:
    spec = resolve_ablation(group)
    retrieval = replace(
        settings.retrieval,
        mode=spec.mode,
        rerank_enabled=spec.rerank_enabled,
    )
    return replace(settings, retrieval=retrieval)


def config_snapshot(settings: Settings, group: str) -> dict[str, Any]:
    spec = resolve_ablation(group)
    return {
        "ablation_group": group,
        "retrieval": {
            "mode": spec.mode,
            "rerank_enabled": spec.rerank_enabled,
            "dense_k": settings.retrieval.dense_k,
            "sparse_k": settings.retrieval.sparse_k,
            "rrf_k": settings.retrieval.rrf_k,
            "fused_k": settings.retrieval.fused_k,
            "rerank_top": settings.retrieval.rerank_top,
        },
        "providers": {
            "embedding": settings.providers.embedding,
            "reranker": settings.providers.reranker,
            "llm": settings.providers.llm,
        },
    }


def group_metrics_payload(
    hit_at_5: float,
    mrr: float,
    refusal_accuracy: float,
    citation_coverage: float,
    answerable_count: int,
    unanswerable_count: int,
) -> dict[str, Any]:
    return {
        "hit_at_5": round(hit_at_5, 6),
        "mrr": round(mrr, 6),
        "refusal_accuracy": round(refusal_accuracy, 6),
        "citation_coverage": round(citation_coverage, 6),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
    }
