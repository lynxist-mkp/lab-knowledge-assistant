from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from wenmai.components.model_guard import ModelResource, phase_batch as model_phase_batch
from wenmai.config import Settings
from wenmai.ingestion.orchestrator import PreparedIngest, count_chunks_with_images, prepare_ingest
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import IngestResult
from wenmai.tracing import StageRecord


def prepare_ingest_source(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
) -> PreparedIngest | IngestResult:
    """Phase-1 入库: quality gate through transform; no embed/upsert."""
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
    """Phase-2 入库: embed + upsert for a prepared document."""
    knowledge = knowledge or create_knowledge(settings)
    body = prepared.body
    recorder = prepared.recorder

    try:
        recorder.set_summary(
            source_path=body.document_source_path,
            document_id=body.document_id,
            title=body.document_title,
            status=body.status,
            chunk_count=len(body.chunks),
            chunks_with_images=count_chunks_with_images(body.chunks),
        )

        upserted = knowledge.commit_document(
            source_path=body.document_source_path,
            sha256=body.document_id,
            document_id=body.document_id,
            status=body.status,
            chunks=body.chunks,
            previous_document_id=body.previous_document_id,
        )
        recorder.record_embed(
            provider=upserted.embed_provider,
            elapsed_ms=upserted.embed_elapsed_ms,
            chunk_count=upserted.chunk_count,
            embed_dimension=upserted.embed_dimension,
        )
        recorder.record_upsert(
            provider=upserted.upsert_provider,
            elapsed_ms=upserted.upsert_elapsed_ms,
            chunk_count=upserted.chunk_count,
        )
        recorder.close_and_save(settings)
    except Exception:
        recorder.save_on_error(settings)
        raise

    return IngestResult(
        document_id=body.document_id,
        chunk_count=len(body.chunks),
        elapsed_ms=recorder.trace_context.total_elapsed_ms,
        trace_id=recorder.trace_id,
        status=body.status,
    )


def run_prepare_commit(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
    phase_batch: bool = True,
) -> IngestResult:
    """Shared sequential 入库: VLM prepare then BGE commit."""
    knowledge = knowledge or create_knowledge(settings)

    with model_phase_batch(ModelResource.MLX_VLM, phase_batch):
        prepared = prepare_ingest_source(
            source_path,
            settings,
            pdf_load_mode=pdf_load_mode,
            on_stage=on_stage,
            knowledge=knowledge,
        )

    if isinstance(prepared, IngestResult):
        return prepared

    with model_phase_batch(ModelResource.BGE_M3, phase_batch):
        return commit_prepared_ingest(prepared, settings, knowledge=knowledge)


class PrepareCommitJob(Protocol):
    source_path: Path
    settings: Settings
    pdf_load_mode: str | None
    on_stage: Callable[[StageRecord], None] | None
    knowledge: Knowledge | None
    result: IngestResult | None
    error: BaseException | None


def run_prepare_commit_batch(
    jobs: list[PrepareCommitJob],
    *,
    batch_id: str,
    phase_batch: bool = True,
) -> None:
    """Shared batch 入库: VLM prepare all, then BGE commit all."""
    prepared_jobs: list[tuple[PrepareCommitJob, PreparedIngest | IngestResult]] = []
    with model_phase_batch(ModelResource.MLX_VLM, phase_batch):
        for job in jobs:
            try:
                knowledge = job.knowledge or create_knowledge(job.settings)
                prepared = prepare_ingest_source(
                    job.source_path,
                    job.settings,
                    pdf_load_mode=job.pdf_load_mode,
                    on_stage=job.on_stage,
                    knowledge=knowledge,
                )
                prepared_jobs.append((job, prepared))
            except BaseException as exc:
                job.error = exc

    commit_jobs: list[tuple[PrepareCommitJob, PreparedIngest]] = []
    for job, prepared in prepared_jobs:
        if job.error is not None:
            continue
        if isinstance(prepared, IngestResult):
            job.result = prepared
        else:
            commit_jobs.append((job, prepared))

    if commit_jobs:
        with model_phase_batch(ModelResource.BGE_M3, phase_batch):
            for job, prepared in commit_jobs:
                try:
                    knowledge = job.knowledge or create_knowledge(job.settings)
                    job.result = commit_prepared_ingest(
                        prepared,
                        job.settings,
                        knowledge=knowledge,
                    )
                except BaseException as exc:
                    job.error = exc

    for job in jobs:
        if job.result is None and job.error is None:
            job.error = RuntimeError(f"prepare-commit batch {batch_id} produced no result")


def ingest_source(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
) -> IngestResult:
    """Single-document 入库: Phase-1 (VLM/transform) then Phase-2 (embed)."""
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

    return run_prepare_commit(
        source_path,
        settings,
        pdf_load_mode=pdf_load_mode,
        on_stage=on_stage,
        knowledge=knowledge,
        phase_batch=True,
    )


__all__ = [
    "PreparedIngest",
    "PrepareCommitJob",
    "commit_prepared_ingest",
    "ingest_source",
    "prepare_ingest_source",
    "run_prepare_commit",
    "run_prepare_commit_batch",
]
