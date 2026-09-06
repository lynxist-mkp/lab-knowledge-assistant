"""入库编排 — phased prepare flow with PrepareTraceRecorder at pipeline seam."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lab_knowledge.config import Settings
from lab_knowledge.factories import splitter as splitter_factory
from lab_knowledge.ingestion.admission import AdmissionGate
from lab_knowledge.ingestion.loaders import LoadedDocument, SourceLoadError, load_source
from lab_knowledge.ingestion.prepare import prepare_chunks
from lab_knowledge.ingestion.quality import peek_source
from lab_knowledge.ingestion.source_metadata import (
    SOURCE_KIND_GROUP,
    LiteratureMetadataOverrides,
    ResolvedSourceMetadata,
    SourceKind,
    resolve_source_metadata,
)
from lab_knowledge.knowledge import Knowledge, create_knowledge
from lab_knowledge.knowledge.domain import REVIEW_PENDING, REVIEW_STATUS_FIELD, stamp_review_status
from lab_knowledge.models import Chunk, IngestResult
from lab_knowledge.storage.document_images import IMAGE_PLACEHOLDER_RE
from lab_knowledge.task_progress import (
    IngestionTaskProgressConfig,
    TaskProgressRun,
    safe_persist_task_progress_outcome,
    safe_persist_task_progress_running,
)
from lab_knowledge.task_progress_builders import build_ingestion_progress
from lab_knowledge.tracing import StageRecord
from lab_knowledge.tracing.prepare_recorder import PrepareTraceRecorder
from lab_knowledge.tracing.stages.ingestion import IngestionStage


@dataclass
class PrepareBody:
    """Domain result from prepare phase — no trace."""

    source_path: Path
    document_id: str
    document_title: str
    document_source_path: str
    status: str
    gray_review: bool
    pdf_load_mode: str | None
    source_metadata: ResolvedSourceMetadata
    chunks: list[Chunk]
    previous_document_id: str | None


class PreparedIngest:
    """Opaque prepare result for one ingest run."""

    __slots__ = ("__body", "__lifecycle", "__recorder")

    def __init__(
        self,
        *,
        body: PrepareBody,
        lifecycle: IngestionLifecycle,
        recorder: PrepareTraceRecorder,
    ) -> None:
        self.__body = body
        self.__lifecycle = lifecycle
        self.__recorder = recorder

    def commit(self, settings: Settings, knowledge: Knowledge) -> IngestResult:
        body = self.__body
        try:
            upserted = knowledge.commit_document(
                source_path=body.document_source_path,
                sha256=body.document_id,
                document_id=body.document_id,
                status=body.status,
                chunks=body.chunks,
                previous_document_id=body.previous_document_id,
            )
            recorder = self.__recorder
            recorder.set_summary(
                source_path=body.document_source_path,
                document_id=body.document_id,
                title=body.document_title,
                status=body.status,
                chunk_count=len(body.chunks),
                chunks_with_images=count_chunks_with_images(body.chunks),
                source_metadata=body.source_metadata,
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
            persist_ingestion_outcome(
                settings,
                self.__lifecycle,
                recorder,
                document_id=body.document_id,
            )
        except Exception as exc:
            recorder = self.__recorder
            recorder.trace_context.error = f"{type(exc).__name__}: {exc}"
            recorder.trace_context.close()
            recorder.save_on_error(settings)
            persist_ingestion_outcome(
                settings,
                self.__lifecycle,
                recorder,
                document_id=body.document_id,
            )
            raise

        return IngestResult(
            document_id=body.document_id,
            chunk_count=len(body.chunks),
            elapsed_ms=self.__recorder.trace_context.total_elapsed_ms,
            trace_id=self.__recorder.trace_id,
            status=body.status,
        )


@dataclass(frozen=True)
class IngestionLifecycle:
    """Shared task-progress context for one ingestion run."""

    run: TaskProgressRun
    config: IngestionTaskProgressConfig


def count_chunks_with_images(chunks: list[Chunk]) -> int:
    return sum(1 for chunk in chunks if IMAGE_PLACEHOLDER_RE.search(chunk.text))


def build_ingestion_lifecycle(
    recorder: PrepareTraceRecorder,
    *,
    pdf_load_mode: str | None,
) -> IngestionLifecycle:
    return IngestionLifecycle(
        run=TaskProgressRun(
            task_id=f"ingestion:{recorder.trace_id}",
            task_type="ingestion",
            started_at=recorder.trace_context.started_at,
            trigger_source="ingest_api",
            owner_surface="ops",
            links={"trace_id": recorder.trace_id},
        ),
        config=IngestionTaskProgressConfig(pdf_load_mode=pdf_load_mode),
    )


def persist_ingestion_running(
    settings: Settings,
    lifecycle: IngestionLifecycle,
    *,
    total: int = 1,
) -> None:
    safe_persist_task_progress_running(
        settings,
        lifecycle.run,
        total=total,
        config=lifecycle.config,
    )


def persist_ingestion_outcome(
    settings: Settings,
    lifecycle: IngestionLifecycle,
    recorder: PrepareTraceRecorder,
    *,
    document_id: str | None = None,
) -> None:
    links = dict(lifecycle.run.links)
    if document_id is not None:
        links["document_id"] = document_id
    safe_persist_task_progress_outcome(
        settings,
        TaskProgressRun(
            task_id=lifecycle.run.task_id,
            task_type=lifecycle.run.task_type,
            started_at=lifecycle.run.started_at,
            trigger_source=lifecycle.run.trigger_source,
            owner_surface=lifecycle.run.owner_surface,
            links=links,
        ),
        config=lifecycle.config,
        outcome=build_ingestion_progress(recorder.trace_context.to_dict()),
    )


def _finish_rejected_ingest(
    lifecycle: IngestionLifecycle,
    recorder: PrepareTraceRecorder,
    source_path: Path,
    settings: Settings,
    *,
    document_source_path: str,
    pdf_load_mode: str | None,
    source_kind: SourceKind = SOURCE_KIND_GROUP,
    literature: LiteratureMetadataOverrides | None = None,
) -> IngestResult:
    document_id = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source_metadata = resolve_source_metadata(
        path=source_path,
        loaded_title=source_path.stem,
        source_kind=source_kind,
        overrides=literature,
    )
    recorder.set_summary(
        source_path=document_source_path,
        document_id=document_id,
        title=source_metadata.title,
        status="rejected",
        chunk_count=0,
        chunks_with_images=0,
        source_metadata=source_metadata,
    )
    recorder.close_and_save(settings)
    persist_ingestion_outcome(settings, lifecycle, recorder, document_id=document_id)
    return IngestResult(
        document_id=document_id,
        chunk_count=0,
        elapsed_ms=recorder.trace_context.total_elapsed_ms,
        trace_id=recorder.trace_id,
        status="rejected",
    )


def _chunk_metadata(
    *,
    document: LoadedDocument,
    index: int,
    gray_review: bool,
    source_metadata: ResolvedSourceMetadata,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "chunk_index": index,
        "document_id": document.document_id,
        "title": source_metadata.title,
        "url": document.url,
        "page": document.page,
        "source_path": document.source_path,
        **source_metadata.chunk_fields(),
        **{
            key: value
            for key, value in document.extra.items()
            if isinstance(value, (str, int, float, bool))
        },
    }
    if gray_review:
        metadata[REVIEW_STATUS_FIELD] = REVIEW_PENDING
    else:
        stamp_review_status(metadata)
    return metadata


def prepare_ingest(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    source_kind: SourceKind = SOURCE_KIND_GROUP,
    literature: LiteratureMetadataOverrides | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
    recorder: PrepareTraceRecorder | None = None,
) -> PreparedIngest | IngestResult:
    """Phase-1 ingest: 入库准入 through transform; no embed/upsert."""
    trace_recorder = recorder or PrepareTraceRecorder(on_stage=on_stage)
    lifecycle = build_ingestion_lifecycle(trace_recorder, pdf_load_mode=pdf_load_mode)
    persist_ingestion_running(settings, lifecycle)
    knowledge = knowledge or create_knowledge(settings)
    document_source_path = str(source_path)

    try:
        source_peek = peek_source(source_path, settings)
        admission_gate = AdmissionGate(settings)
        admission = admission_gate.decide(source_path, source_peek)

        with trace_recorder.stage(
            "quality_gate",
            method="effective_char_ratio",
            provider="config",
            input_summary=str(source_path),
        ) as gate_info:
            gate_result = admission.gate_result
            gate_info["output_summary"] = (
                f"ratio={gate_result.ratio:.2f} band={gate_result.band}"
            )
            if source_peek.defer_reject and gate_result.band == "gray":
                gate_info["output_summary"] += " defer=scanned_pdf"
            gate_info["candidate_count"] = 1
            if admission.decision == "rejected" and admission.gray_outcome is None:
                gate_info["error"] = (
                    f"effective_char_ratio {gate_result.ratio:.2f} "
                    f"below {settings.quality_gate.reject_below:.2f}"
                )

        if admission.decision == "rejected":
            if admission.gray_outcome is not None:
                trace_recorder.append_stage(
                    IngestionStage.gray_review(
                        provider=admission.gray_outcome.provider,
                        method=admission.gray_outcome.method,
                        elapsed_ms=admission.gray_elapsed_ms or 0.0,
                        output_summary=admission.gray_outcome.output_summary,
                        error=admission.gray_outcome.error,
                    )
                )
            return _finish_rejected_ingest(
                lifecycle,
                trace_recorder,
                source_path,
                settings,
                document_source_path=document_source_path,
                pdf_load_mode=pdf_load_mode,
                source_kind=source_kind,
                literature=literature,
            )

        if admission.gray_outcome is not None:
            trace_recorder.append_stage(
                IngestionStage.gray_review(
                    provider=admission.gray_outcome.provider,
                    method=admission.gray_outcome.method,
                    elapsed_ms=admission.gray_elapsed_ms or 0.0,
                    output_summary=admission.gray_outcome.output_summary,
                    error=admission.gray_outcome.error,
                )
            )

        stamp_pending_chunks = admission.stamp_pending_chunks

        with trace_recorder.stage(
            "load",
            method="pending",
            provider="pending",
            input_summary=str(source_path),
        ) as load_info:
            try:
                document = load_source(
                    source_path,
                    settings,
                    pdf_load_mode=pdf_load_mode,
                    images=knowledge.images,
                )
            except SourceLoadError as exc:
                load_info["method"] = exc.load_method
                load_info["provider"] = exc.load_provider
                if exc.__cause__ is not None:
                    raise exc.__cause__ from exc
                raise
            document_id = document.document_id
            document_title = document.title
            document_source_path = document.source_path
            source_metadata = resolve_source_metadata(
                path=source_path,
                loaded_title=document.title,
                front_matter=document.front_matter,
                embedded=document.embedded_literature,
                source_kind=source_kind,
                overrides=literature,
            )
            document_title = source_metadata.title
            load_info["output_summary"] = document_title
            load_info["candidate_count"] = 1
            load_info["method"] = document.load_method or (
                source_path.suffix.lower().lstrip(".") or "unknown"
            )
            load_info["provider"] = document.load_provider or "file"

        with trace_recorder.stage(
            "integrity",
            method="sha256",
            provider="knowledge",
            input_summary=document.document_id[:12],
        ) as integrity_info:
            prepared = knowledge.plan_document(
                source_path=document.source_path,
                sha256=document.document_id,
                document_id=document.document_id,
            )
            status = prepared.status
            previous_document_id = prepared.previous_document_id
            skipped = status == "skipped"
            if skipped:
                integrity_info["output_summary"] = "skipped: unchanged sha256"
                integrity_info["candidate_count"] = 0
            elif status == "rebuilt":
                prev = prepared.previous_document_id or ""
                integrity_info["output_summary"] = (
                    f"rebuilt: will replace {prev[:12]}..." if prev else "rebuilt"
                )
                integrity_info["candidate_count"] = 1
            else:
                integrity_info["output_summary"] = "new file"
                integrity_info["candidate_count"] = 1

        if skipped:
            trace_recorder.set_summary(
                source_path=document_source_path,
                document_id=document_id,
                title=document_title,
                status=status,
                chunk_count=0,
                chunks_with_images=0,
                source_metadata=source_metadata,
            )
            trace_recorder.close_and_save(settings)
            persist_ingestion_outcome(
                settings,
                lifecycle,
                trace_recorder,
                document_id=document_id,
            )
            return IngestResult(
                document_id=document_id,
                chunk_count=0,
                elapsed_ms=trace_recorder.trace_context.total_elapsed_ms,
                trace_id=trace_recorder.trace_id,
                status=status,
            )

        splitter = splitter_factory.create(settings)
        with trace_recorder.stage(
            "split",
            method=splitter.provider_name,
            provider=splitter.provider_name,
            input_summary=f"{len(document.text)} chars",
        ) as split_info:
            texts = splitter.split(document.text)
            split_info["candidate_count"] = len(texts)
            split_info["output_summary"] = f"{len(texts)} chunks"

        chunks = [
            Chunk(
                chunk_id=f"{document.document_id}:{index:04d}",
                document_id=document.document_id,
                text=text,
                metadata=_chunk_metadata(
                    document=document,
                    index=index,
                    gray_review=stamp_pending_chunks,
                    source_metadata=source_metadata,
                ),
            )
            for index, text in enumerate(texts)
        ]

        chunks = prepare_chunks(chunks, settings, trace_recorder)

        body = PrepareBody(
            source_path=source_path,
            document_id=document_id,
            document_title=document_title,
            document_source_path=document_source_path,
            status=status,
            gray_review=stamp_pending_chunks,
            pdf_load_mode=pdf_load_mode,
            source_metadata=source_metadata,
            chunks=chunks,
            previous_document_id=previous_document_id,
        )
        return PreparedIngest(body=body, lifecycle=lifecycle, recorder=trace_recorder)
    except Exception as exc:
        trace_recorder.trace_context.error = f"{type(exc).__name__}: {exc}"
        trace_recorder.trace_context.close()
        trace_recorder.save_on_error(settings)
        persist_ingestion_outcome(settings, lifecycle, trace_recorder)
        raise


__all__ = [
    "IngestionLifecycle",
    "PrepareBody",
    "PreparedIngest",
    "build_ingestion_lifecycle",
    "count_chunks_with_images",
    "persist_ingestion_outcome",
    "persist_ingestion_running",
    "prepare_ingest",
]
