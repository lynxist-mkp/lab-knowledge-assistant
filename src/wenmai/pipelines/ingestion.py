from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from wenmai.components.model_guard import ModelResource, phase_batch as model_phase_batch
from wenmai.config import Settings
from wenmai.ingestion.orchestrator import PreparedIngest, prepare_ingest
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import IngestResult
from wenmai.storage.document_images import IMAGE_PLACEHOLDER_RE
from wenmai.tracing import StageRecord, save_trace
from wenmai.tracing.stages.ingestion import IngestionStage


def _chunks_with_images(chunks: list) -> int:
    return sum(1 for chunk in chunks if IMAGE_PLACEHOLDER_RE.search(chunk.text))


def prepare_ingest_source(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
) -> PreparedIngest | IngestResult:
    """Phase-1 ingest: quality gate through transform; no embed/upsert."""
    return prepare_ingest(
        source_path,
        settings,
        pdf_load_mode=pdf_load_mode,
        on_stage=on_stage,
        knowledge=knowledge,
    )


def commit_prepared_ingest(
    prepared: PreparedIngest,
    settings: Settings,
    *,
    knowledge: Knowledge | None = None,
) -> IngestResult:
    """Phase-2 ingest: embed + upsert for a prepared document."""
    knowledge = knowledge or create_knowledge(settings)
    trace = prepared.trace

    try:
        trace.metadata.update(
            {
                "source_path": prepared.document_source_path,
                "document_id": prepared.document_id,
                "title": prepared.document_title,
                "status": prepared.status,
                "chunk_count": len(prepared.chunks),
                "chunks_with_images": _chunks_with_images(prepared.chunks),
            }
        )

        upserted = knowledge.commit_document(
            source_path=prepared.document_source_path,
            sha256=prepared.document_id,
            document_id=prepared.document_id,
            status=prepared.status,
            chunks=prepared.chunks,
            previous_document_id=prepared.previous_document_id,
        )
        trace.append_stage(
            IngestionStage.embed(
                provider=upserted.embed_provider,
                elapsed_ms=upserted.embed_elapsed_ms,
                chunk_count=upserted.chunk_count,
                embed_dimension=upserted.embed_dimension,
            )
        )
        trace.append_stage(
            IngestionStage.upsert(
                provider=upserted.upsert_provider,
                elapsed_ms=upserted.upsert_elapsed_ms,
                chunk_count=upserted.chunk_count,
            )
        )
        trace.close()
    finally:
        save_trace(settings, trace)

    return IngestResult(
        document_id=prepared.document_id,
        chunk_count=len(prepared.chunks),
        elapsed_ms=trace.total_elapsed_ms,
        trace_id=trace.trace_id,
        status=prepared.status,
    )


def ingest_source(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
) -> IngestResult:
    """Single-document ingest: Phase-1 (VLM/transform) then Phase-2 (embed)."""
    from wenmai.pipelines.ingest_batch import (
        get_ingest_coordinator,
        ingest_window_batch_enabled,
    )

    if ingest_window_batch_enabled(settings):
        return get_ingest_coordinator(settings).submit(
            source_path,
            settings,
            pdf_load_mode=pdf_load_mode,
            on_stage=on_stage,
            knowledge=knowledge,
        )

    knowledge = knowledge or create_knowledge(settings)

    with model_phase_batch(ModelResource.MLX_VLM, True):
        prepared = prepare_ingest_source(
            source_path,
            settings,
            pdf_load_mode=pdf_load_mode,
            on_stage=on_stage,
            knowledge=knowledge,
        )

    if isinstance(prepared, IngestResult):
        return prepared

    with model_phase_batch(ModelResource.BGE_M3, True):
        return commit_prepared_ingest(prepared, settings, knowledge=knowledge)


__all__ = ["PreparedIngest", "commit_prepared_ingest", "ingest_source", "prepare_ingest_source"]
