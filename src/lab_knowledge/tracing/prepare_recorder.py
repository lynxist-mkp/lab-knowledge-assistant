"""PrepareTraceRecorder — trace adapter at the 入库 pipeline seam."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lab_knowledge.config import Settings
from lab_knowledge.ingestion.source_metadata import ResolvedSourceMetadata
from lab_knowledge.tracing.context import StageRecord, TraceContext
from lab_knowledge.tracing.stages.ingestion import IngestionStage
from lab_knowledge.tracing.store import save_trace


class PrepareTraceRecorder:
    def __init__(
        self,
        *,
        on_stage: Callable[[StageRecord], None] | None = None,
    ) -> None:
        self._context = TraceContext(trace_type="ingestion")
        self._context._on_stage = on_stage

    @property
    def trace_id(self) -> str:
        return self._context.trace_id

    @property
    def trace_context(self) -> TraceContext:
        return self._context

    @property
    def metadata(self) -> dict[str, Any]:
        return self._context.metadata

    def stage(self, *args: Any, **kwargs: Any):
        return self._context.stage(*args, **kwargs)

    def record_stage(self, *args: Any, **kwargs: Any) -> None:
        self._context.record_stage(*args, **kwargs)

    def append_stage(self, stage: StageRecord) -> None:
        self._context.append_stage(stage)

    def set_summary(
        self,
        *,
        source_path: str,
        document_id: str,
        title: str,
        status: str,
        chunk_count: int,
        chunks_with_images: int,
        source_metadata: ResolvedSourceMetadata | None = None,
    ) -> None:
        summary = {
            "source_path": source_path,
            "document_id": document_id,
            "title": title,
            "status": status,
            "chunk_count": chunk_count,
            "chunks_with_images": chunks_with_images,
        }
        if source_metadata is not None:
            summary.update(source_metadata.trace_fields())
        self._context.metadata.update(summary)

    def record_embed(
        self,
        *,
        provider: str,
        elapsed_ms: float,
        chunk_count: int,
        embed_dimension: int,
    ) -> None:
        self.append_stage(
            IngestionStage.embed(
                provider=provider,
                elapsed_ms=elapsed_ms,
                chunk_count=chunk_count,
                embed_dimension=embed_dimension,
            )
        )

    def record_upsert(self, *, provider: str, elapsed_ms: float, chunk_count: int) -> None:
        self.append_stage(
            IngestionStage.upsert(
                provider=provider,
                elapsed_ms=elapsed_ms,
                chunk_count=chunk_count,
            )
        )

    def close_and_save(self, settings: Settings) -> None:
        self._context.close()
        save_trace(settings, self._context)

    def save_on_error(self, settings: Settings) -> None:
        save_trace(settings, self._context)


__all__ = ["PrepareTraceRecorder"]
