from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from wenmai.tracing.context import StageRecord

if TYPE_CHECKING:
    from wenmai.ingestion.quality import QualityGateResult, SourcePeek


class IngestionStage:
    @staticmethod
    def quality_gate(
        *,
        elapsed_ms: float,
        path: Path,
        gate_result: QualityGateResult,
        peek: SourcePeek,
        decision: str,
        gray_ran: bool,
        reject_below: float,
    ) -> StageRecord:
        output_summary = f"ratio={gate_result.ratio:.2f} band={gate_result.band}"
        if peek.defer_reject and gate_result.band == "gray":
            output_summary += " defer=scanned_pdf"
        error: str | None = None
        if decision == "rejected" and not gray_ran:
            error = (
                f"effective_char_ratio {gate_result.ratio:.2f} "
                f"below {reject_below:.2f}"
            )
        return StageRecord(
            name="quality_gate",
            method="effective_char_ratio",
            provider="config",
            elapsed_ms=elapsed_ms,
            input_summary=str(path),
            output_summary=output_summary,
            candidate_count=1,
            error=error,
        )

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
