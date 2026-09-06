from __future__ import annotations

from lab_knowledge.tracing.context import StageRecord


class IngestionStage:
    @staticmethod
    def gray_review(
        *,
        provider: str,
        method: str,
        elapsed_ms: float,
        output_summary: str,
        error: str | None = None,
    ) -> StageRecord:
        return StageRecord(
            name="gray_review",
            method=method,
            provider=provider,
            elapsed_ms=elapsed_ms,
            input_summary="灰区复判",
            output_summary=output_summary,
            candidate_count=1,
            error=error,
        )

    @staticmethod
    def captioner(
        *,
        provider: str,
        elapsed_ms: float,
        chunk_count: int,
        output_summary: str,
        error: str | None = None,
    ) -> StageRecord:
        return StageRecord(
            name="captioner",
            method="vision",
            provider=provider,
            elapsed_ms=elapsed_ms,
            input_summary=f"{chunk_count} chunks",
            output_summary=output_summary,
            candidate_count=chunk_count,
            error=error,
        )

    @staticmethod
    def embed(
        *,
        provider: str,
        elapsed_ms: float,
        chunk_count: int,
        embed_dimension: int,
    ) -> StageRecord:
        return StageRecord(
            name="embed",
            method=provider,
            provider=provider,
            elapsed_ms=elapsed_ms,
            input_summary=f"{chunk_count} chunks",
            output_summary=f"dim={embed_dimension}",
            candidate_count=chunk_count,
        )

    @staticmethod
    def upsert(
        *,
        provider: str,
        elapsed_ms: float,
        chunk_count: int,
    ) -> StageRecord:
        return StageRecord(
            name="upsert",
            method=provider,
            provider=provider,
            elapsed_ms=elapsed_ms,
            input_summary=f"{chunk_count} chunks",
            output_summary=f"upserted {chunk_count}",
            candidate_count=chunk_count,
        )
