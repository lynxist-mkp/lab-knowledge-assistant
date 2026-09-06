"""Build query Trace stages from fusion metrics at the orchestration seam."""

from __future__ import annotations

from lab_knowledge.config import Settings
from lab_knowledge.knowledge import Knowledge
from lab_knowledge.retrieval.fusion import RetrievalResult
from lab_knowledge.tracing.context import StageRecord
from lab_knowledge.tracing.stages.query import QueryStage


def stages_from_fusion(
    knowledge: Knowledge,
    settings: Settings,
    fusion_result: RetrievalResult,
    culture_domain: str | None,
) -> list[StageRecord]:
    mode = fusion_result.mode
    if mode == "dense_only":
        return [
            QueryStage.dense(
                knowledge,
                settings,
                fusion_result.dense_chunks,
                fusion_result.dense_elapsed_ms,
                culture_domain,
            )
        ]
    if mode == "sparse_only":
        return [
            QueryStage.sparse(
                settings,
                fusion_result.sparse_chunks,
                fusion_result.sparse_elapsed_ms,
                culture_domain,
            )
        ]

    dense_chunks = fusion_result.dense_chunks
    sparse_chunks = fusion_result.sparse_chunks
    return [
        QueryStage.dense(
            knowledge,
            settings,
            dense_chunks,
            fusion_result.dense_elapsed_ms,
            culture_domain,
        ),
        QueryStage.sparse(
            settings,
            sparse_chunks,
            fusion_result.sparse_elapsed_ms,
            culture_domain,
        ),
        QueryStage.fusion(
            settings,
            dense_chunks,
            sparse_chunks,
            fusion_result.chunks,
            fusion_result.fusion_elapsed_ms,
            query_path_counts=fusion_result.query_path_counts,
        ),
    ]


__all__ = ["stages_from_fusion"]
