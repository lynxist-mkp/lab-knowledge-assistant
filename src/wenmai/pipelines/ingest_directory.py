"""Bulk ingest a local directory as personal literature."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from wenmai.config import Settings
from wenmai.ingestion.source_metadata import (
    SOURCE_KIND_PERSONAL,
    LiteratureMetadataOverrides,
    SourceKind,
)
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import IngestResult
from wenmai.pipelines.ingestion import run_prepare_commit_batch
from wenmai.tracing.context import StageRecord

_SUPPORTED_SUFFIXES = {".md", ".pdf"}


@dataclass(frozen=True)
class IngestFileOutcome:
    source_path: str
    status: str
    document_id: str | None = None
    chunk_count: int = 0
    trace_id: str | None = None
    elapsed_ms: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_path": self.source_path,
            "status": self.status,
        }
        if self.document_id is not None:
            payload["document_id"] = self.document_id
        if self.chunk_count:
            payload["chunk_count"] = self.chunk_count
        if self.trace_id is not None:
            payload["trace_id"] = self.trace_id
        if self.elapsed_ms:
            payload["elapsed_ms"] = self.elapsed_ms
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass
class IngestDirectoryResult:
    source_path: str
    source_kind: SourceKind
    total: int
    attempted: int
    ingested: int
    rebuilt: int
    skipped: int
    failed: int
    files: list[IngestFileOutcome] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "total": self.total,
            "attempted": self.attempted,
            "ingested": self.ingested,
            "rebuilt": self.rebuilt,
            "skipped": self.skipped,
            "failed": self.failed,
            "files": [item.as_dict() for item in self.files],
        }


@dataclass
class _DirectoryIngestJob:
    index: int
    source_path: Path
    settings: Settings
    source_kind: SourceKind
    literature: LiteratureMetadataOverrides | None
    pdf_load_mode: str | None
    knowledge: Knowledge | None = None
    on_stage: Callable[[StageRecord], None] | None = None
    result: IngestResult | None = None
    error: BaseException | None = None


def list_ingestable_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"source path is not a directory: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    )


def ingest_source_directory(
    directory: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    source_kind: SourceKind = SOURCE_KIND_PERSONAL,
    literature: LiteratureMetadataOverrides | None = None,
    knowledge: Knowledge | None = None,
    on_file_start: Callable[[int, int, Path], None] | None = None,
    on_file_done: Callable[[int, int, IngestFileOutcome], None] | None = None,
) -> IngestDirectoryResult:
    """Ingest every supported file under directory, preserving per-file dedupe semantics."""
    files = list_ingestable_files(directory)
    knowledge = knowledge or create_knowledge(settings)
    total = len(files)
    attempted = 0
    ingested = 0
    rebuilt = 0
    skipped = 0
    failed = 0
    outcomes: list[IngestFileOutcome] = []
    jobs: list[_DirectoryIngestJob] = []

    for index, source_path in enumerate(files, start=1):
        if on_file_start is not None:
            on_file_start(index, total, source_path)
        attempted += 1
        jobs.append(
            _DirectoryIngestJob(
                index=index,
                source_path=source_path,
                settings=settings,
                source_kind=source_kind,
                literature=literature,
                pdf_load_mode=pdf_load_mode,
                knowledge=knowledge,
            )
        )

    if jobs:
        run_prepare_commit_batch(jobs, batch_id="directory-import")

    for job in jobs:
        outcome = _job_outcome(job)
        outcomes.append(outcome)
        if outcome.status == "ingested":
            ingested += 1
        elif outcome.status == "rebuilt":
            rebuilt += 1
        elif outcome.status == "skipped":
            skipped += 1
        else:
            failed += 1
        if on_file_done is not None:
            on_file_done(job.index, total, outcome)

    return IngestDirectoryResult(
        source_path=str(directory),
        source_kind=source_kind,
        total=total,
        attempted=attempted,
        ingested=ingested,
        rebuilt=rebuilt,
        skipped=skipped,
        failed=failed,
        files=outcomes,
    )


def _job_outcome(job: _DirectoryIngestJob) -> IngestFileOutcome:
    source_path = str(job.source_path)
    if job.error is not None:
        return IngestFileOutcome(
            source_path=source_path,
            status="failed",
            error=f"{type(job.error).__name__}: {job.error}",
        )
    if job.result is None:
        return IngestFileOutcome(
            source_path=source_path,
            status="failed",
            error="no ingest result",
        )
    if job.result.status == "rejected":
        return IngestFileOutcome(
            source_path=source_path,
            status="failed",
            document_id=job.result.document_id,
            trace_id=job.result.trace_id,
            elapsed_ms=job.result.elapsed_ms,
            error="quality gate rejected",
        )
    return IngestFileOutcome(
        source_path=source_path,
        status=job.result.status,
        document_id=job.result.document_id,
        chunk_count=job.result.chunk_count,
        trace_id=job.result.trace_id,
        elapsed_ms=job.result.elapsed_ms,
    )


__all__ = [
    "IngestDirectoryResult",
    "IngestFileOutcome",
    "ingest_source_directory",
    "list_ingestable_files",
]
