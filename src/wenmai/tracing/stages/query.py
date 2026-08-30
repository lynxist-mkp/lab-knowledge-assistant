from __future__ import annotations

from wenmai.config import Settings
from wenmai.knowledge import Knowledge
from wenmai.models import ScoredChunk
from wenmai.tracing.context import StageRecord


def candidate_records(scored_chunks: list[ScoredChunk]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": item.chunk.chunk_id,
            "score": round(item.score, 6),
        }
        for item in scored_chunks
    ]


def _dense_input_summary(settings: Settings, culture_domain: str | None) -> str:
    summary = f"k={settings.retrieval.dense_k}"
    if culture_domain is not None:
        summary += f" culture_domain={culture_domain}"
    return summary


def _sparse_input_summary(settings: Settings, culture_domain: str | None) -> str:
    summary = f"k={settings.retrieval.sparse_k}"
    if culture_domain is not None:
        summary += f" culture_domain={culture_domain}"
    return summary


class QueryStage:
    @staticmethod
    def query_processing(
        *,
        question: str,
        normalized: str,
        elapsed_ms: float,
        culture_domain: str | None = None,
    ) -> StageRecord:
        return StageRecord(
            name="query_processing",
            method="normalize",
            provider="local",
            elapsed_ms=elapsed_ms,
            input_summary=question,
            output_summary=normalized,
            candidate_count=1,
            culture_domain=culture_domain,
        )

    @staticmethod
    def dense(
        knowledge: Knowledge,
        settings: Settings,
        scored_chunks: list[ScoredChunk],
        elapsed_ms: float,
        culture_domain: str | None = None,
    ) -> StageRecord:
        return StageRecord(
            name="dense",
            method="vector_query",
            provider=knowledge.dense_provider,
            elapsed_ms=elapsed_ms,
            input_summary=_dense_input_summary(settings, culture_domain),
            output_summary=f"retrieved {len(scored_chunks)} chunks",
            candidate_count=len(scored_chunks),
            candidates=candidate_records(scored_chunks),
        )

    @staticmethod
    def sparse(
        settings: Settings,
        scored_chunks: list[ScoredChunk],
        elapsed_ms: float,
        culture_domain: str | None = None,
    ) -> StageRecord:
        return StageRecord(
            name="sparse",
            method="bm25",
            provider="local",
            elapsed_ms=elapsed_ms,
            input_summary=_sparse_input_summary(settings, culture_domain),
            output_summary=f"retrieved {len(scored_chunks)} chunks",
            candidate_count=len(scored_chunks),
            candidates=candidate_records(scored_chunks),
        )

    @staticmethod
    def fusion(
        settings: Settings,
        dense_chunks: list[ScoredChunk],
        sparse_chunks: list[ScoredChunk],
        fused_chunks: list[ScoredChunk],
        elapsed_ms: float,
    ) -> StageRecord:
        return StageRecord(
            name="fusion",
            method="rrf",
            provider="local",
            elapsed_ms=elapsed_ms,
            input_summary=(
                f"dense={len(dense_chunks)} sparse={len(sparse_chunks)} "
                f"k={settings.retrieval.rrf_k}"
            ),
            output_summary=f"fused {len(fused_chunks)} chunks",
            candidate_count=len(fused_chunks),
            candidates=candidate_records(fused_chunks),
            dense_candidates=candidate_records(dense_chunks),
            sparse_candidates=candidate_records(sparse_chunks),
        )

    @staticmethod
    def rerank_success(
        *,
        provider: str,
        elapsed_ms: float,
        pre_rerank: list[ScoredChunk],
        reranked: list[ScoredChunk],
        rerank_top: int,
        rank_changes: list[dict[str, object]],
    ) -> StageRecord:
        return StageRecord(
            name="rerank",
            method="cross_encoder",
            provider=provider,
            elapsed_ms=elapsed_ms,
            input_summary=f"{len(pre_rerank)} candidates, top={rerank_top}",
            output_summary=f"reranked to {len(reranked)} chunks",
            candidate_count=len(reranked),
            candidates=candidate_records(reranked),
            pre_rerank_candidates=candidate_records(pre_rerank),
            rank_changes=rank_changes,
        )

    @staticmethod
    def rerank_fallback(
        *,
        provider: str,
        elapsed_ms: float,
        pre_rerank: list[ScoredChunk],
        fallback: list[ScoredChunk],
        rerank_top: int,
        reason: str,
    ) -> StageRecord:
        return StageRecord(
            name="rerank",
            method="rrf_fallback",
            provider=provider,
            elapsed_ms=elapsed_ms,
            input_summary=f"{len(pre_rerank)} candidates, top={rerank_top}",
            output_summary=f"fallback to RRF top-{len(fallback)}",
            candidate_count=len(fallback),
            candidates=candidate_records(fallback),
            pre_rerank_candidates=candidate_records(pre_rerank),
            error=reason,
            fallback_reason=reason,
        )

    @staticmethod
    def generation(
        *,
        provider: str,
        elapsed_ms: float,
        input_summary: str,
        output_summary: str,
        candidate_count: int,
        error: str | None = None,
        expanded_from: list[str] | None = None,
        expanded_chunk_ids: list[str] | None = None,
    ) -> StageRecord:
        return StageRecord(
            name="generation",
            method="llm",
            provider=provider,
            elapsed_ms=elapsed_ms,
            input_summary=input_summary,
            output_summary=output_summary,
            candidate_count=candidate_count,
            error=error,
            expanded_from=expanded_from,
            expanded_chunk_ids=expanded_chunk_ids,
        )
