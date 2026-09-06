from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lab_knowledge.components.model_guard import ModelResource
from lab_knowledge.components.model_guard import phase_batch as model_phase_batch
from lab_knowledge.config import Settings
from lab_knowledge.ingestion.orchestrator import (
    PreparedIngest,
    prepare_ingest,
)
from lab_knowledge.ingestion.source_metadata import (
    SOURCE_KIND_GROUP,
    LiteratureMetadataOverrides,
    SourceKind,
)
from lab_knowledge.knowledge import Knowledge, create_knowledge
from lab_knowledge.models import BulkIngestItemResult, BulkIngestResult, IngestResult
from lab_knowledge.tracing import StageRecord

_INGESTABLE_SUFFIXES = {".md", ".pdf"}


@dataclass
class DirectoryPrepareCommitJob:
    source_path: Path
    settings: Settings
    pdf_load_mode: str | None
    source_kind: SourceKind
    literature: LiteratureMetadataOverrides | None
    on_stage: Callable[[StageRecord], None] | None
    knowledge: Knowledge | None
    result: IngestResult | None = None
    error: BaseException | None = None


def prepare_ingest_source(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    source_kind: SourceKind = SOURCE_KIND_GROUP,
    literature: LiteratureMetadataOverrides | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
) -> PreparedIngest | IngestResult:
    """Phase-1 入库: quality gate through transform; no embed/upsert."""
    return prepare_ingest(
        source_path,
        settings,
        pdf_load_mode=pdf_load_mode,
        source_kind=source_kind,
        literature=literature,
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
    return prepared.commit(settings, knowledge)


def run_prepare_commit(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    source_kind: SourceKind = SOURCE_KIND_GROUP,
    literature: LiteratureMetadataOverrides | None = None,
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
            source_kind=source_kind,
            literature=literature,
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
    source_kind: SourceKind
    literature: LiteratureMetadataOverrides | None
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
                    source_kind=getattr(job, "source_kind", SOURCE_KIND_GROUP),
                    literature=getattr(job, "literature", None),
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
    source_kind: SourceKind = SOURCE_KIND_GROUP,
    literature: LiteratureMetadataOverrides | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
) -> IngestResult:
    """Single-document 入库: Phase-1 (VLM/transform) then Phase-2 (embed)."""
    from lab_knowledge.pipelines.ingest_batch import (
        get_ingest_coordinator,
        ingest_window_batch_enabled,
    )

    if ingest_window_batch_enabled(settings):
        return get_ingest_coordinator(settings).submit(
            source_path,
            settings,
            pdf_load_mode=pdf_load_mode,
            source_kind=source_kind,
            literature=literature,
            on_stage=on_stage,
            knowledge=knowledge,
        )

    return run_prepare_commit(
        source_path,
        settings,
        pdf_load_mode=pdf_load_mode,
        source_kind=source_kind,
        literature=literature,
        on_stage=on_stage,
        knowledge=knowledge,
        phase_batch=True,
    )


def ingest_directory(
    source_root: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    source_kind: SourceKind = SOURCE_KIND_GROUP,
    knowledge: Knowledge | None = None,
) -> BulkIngestResult:
    """Directory ingest for bulk document sync using the shared prepare/commit seam."""
    files = _discover_ingestable_files(source_root)
    if not files:
        raise ValueError(f"no ingestable files found in directory: {source_root}")

    shared_knowledge = knowledge or create_knowledge(settings)
    jobs = [
        DirectoryPrepareCommitJob(
            source_path=path,
            settings=settings,
            pdf_load_mode=pdf_load_mode,
            source_kind=source_kind,
            literature=None,
            on_stage=None,
            knowledge=shared_knowledge,
        )
        for path in files
    ]
    run_prepare_commit_batch(
        jobs,
        batch_id=f"dir:{source_root.name}:{len(files)}",
        phase_batch=True,
    )

    results: list[BulkIngestItemResult] = []
    ingested_count = 0
    rebuilt_count = 0
    skipped_count = 0
    failed_count = 0
    for job in jobs:
        if job.error is not None:
            failed_count += 1
            results.append(
                BulkIngestItemResult(
                    source_path=str(job.source_path),
                    status="failed",
                    error=f"{type(job.error).__name__}: {job.error}",
                )
            )
            continue
        assert job.result is not None
        status = job.result.status
        if status == "ingested":
            ingested_count += 1
        elif status == "rebuilt":
            rebuilt_count += 1
        elif status == "skipped":
            skipped_count += 1
        results.append(
            BulkIngestItemResult(
                source_path=str(job.source_path),
                status=status,
                document_id=job.result.document_id,
                chunk_count=job.result.chunk_count,
                trace_id=job.result.trace_id,
            )
        )
    return BulkIngestResult(
        source_root=str(source_root),
        source_kind=source_kind,
        total_files=len(files),
        ingested_count=ingested_count,
        rebuilt_count=rebuilt_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        results=results,
    )


def _discover_ingestable_files(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _INGESTABLE_SUFFIXES
    )


__all__ = [
    "BulkIngestResult",
    "DirectoryPrepareCommitJob",
    "PreparedIngest",
    "PrepareCommitJob",
    "commit_prepared_ingest",
    "ingest_directory",
    "ingest_source",
    "prepare_ingest_source",
    "run_prepare_commit",
    "run_prepare_commit_batch",
]
