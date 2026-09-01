from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from wenmai.components.model_guard import ModelResource, begin_batch, end_batch
from wenmai.config import Settings
from wenmai.factories import splitter as splitter_factory
from wenmai.ingestion.gray_review import run_gray_review
from wenmai.ingestion.loaders import LoadedDocument, SourceLoadError, load_source
from wenmai.ingestion.prepare import prepare_chunks
from wenmai.ingestion.quality import evaluate_quality_gate, peek_source
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.knowledge.domain import REVIEW_PENDING, REVIEW_STATUS_FIELD, stamp_review_status
from wenmai.models import Chunk, IngestResult
from wenmai.storage.document_images import IMAGE_PLACEHOLDER_RE
from wenmai.tracing import StageRecord, TraceContext, save_trace
from wenmai.tracing.stages.ingestion import IngestionStage


@dataclass
class PreparedIngest:
    trace: TraceContext
    source_path: Path
    document_id: str
    document_title: str
    document_source_path: str
    status: str
    gray_review: bool
    chunks: list[Chunk]
    previous_document_id: str | None


def _chunks_with_images(chunks: list[Chunk]) -> int:
    return sum(1 for chunk in chunks if IMAGE_PLACEHOLDER_RE.search(chunk.text))


def _set_trace_summary(
    trace: TraceContext,
    *,
    source_path: str,
    document_id: str,
    title: str,
    status: str,
    chunk_count: int,
    chunks_with_images: int,
) -> None:
    trace.metadata.update(
        {
            "source_path": source_path,
            "document_id": document_id,
            "title": title,
            "status": status,
            "chunk_count": chunk_count,
            "chunks_with_images": chunks_with_images,
        }
    )


def _finish_rejected_ingest(
    trace: TraceContext,
    source_path: Path,
    settings: Settings,
    *,
    document_source_path: str,
) -> IngestResult:
    document_id = hashlib.sha256(source_path.read_bytes()).hexdigest()
    _set_trace_summary(
        trace,
        source_path=document_source_path,
        document_id=document_id,
        title=source_path.stem,
        status="rejected",
        chunk_count=0,
        chunks_with_images=0,
    )
    trace.close()
    save_trace(settings, trace)
    return IngestResult(
        document_id=document_id,
        chunk_count=0,
        elapsed_ms=trace.total_elapsed_ms,
        trace_id=trace.trace_id,
        status="rejected",
    )


def _chunk_metadata(
    *,
    document: LoadedDocument,
    index: int,
    gray_review: bool,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "chunk_index": index,
        "document_id": document.document_id,
        "title": document.title,
        "url": document.url,
        "page": document.page,
        "source_path": document.source_path,
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


def prepare_ingest_source(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
) -> PreparedIngest | IngestResult:
    """Phase-1 ingest: quality gate through transform; no embed/upsert."""
    trace = TraceContext(trace_type="ingestion")
    trace._on_stage = on_stage
    knowledge = knowledge or create_knowledge(settings)
    gray_review = False
    document_id = ""
    document_title = ""
    document_source_path = str(source_path)
    status = "ingested"
    previous_document_id: str | None = None
    chunks: list[Chunk] = []

    try:
        with trace.stage(
            "quality_gate",
            method="effective_char_ratio",
            provider="config",
            input_summary=str(source_path),
        ) as gate_info:
            source_peek = peek_source(source_path, settings)
            gate_result = evaluate_quality_gate(
                source_peek.text,
                settings.quality_gate,
                defer_reject=source_peek.defer_reject,
            )
            gate_info["output_summary"] = (
                f"ratio={gate_result.ratio:.2f} band={gate_result.band}"
            )
            if source_peek.defer_reject and gate_result.band == "gray":
                gate_info["output_summary"] += " defer=scanned_pdf"
            gate_info["candidate_count"] = 1
            rejected = gate_result.band == "reject"
            if rejected:
                gate_info["error"] = (
                    f"effective_char_ratio {gate_result.ratio:.2f} "
                    f"below {settings.quality_gate.reject_below:.2f}"
                )
            else:
                gray_review = gate_result.band == "gray"

        if rejected:
            return _finish_rejected_ingest(
                trace,
                source_path,
                settings,
                document_source_path=document_source_path,
            )

        if gray_review and settings.quality_gate.gray_review:
            review_started = time.perf_counter()
            outcome = run_gray_review(source_path, source_peek, settings)
            elapsed_ms = (time.perf_counter() - review_started) * 1000
            trace.append_stage(
                IngestionStage.gray_review(
                    provider=outcome.provider,
                    method=outcome.method,
                    elapsed_ms=elapsed_ms,
                    output_summary=outcome.output_summary,
                    error=outcome.error,
                )
            )
            if outcome.hard_reject:
                return _finish_rejected_ingest(
                    trace,
                    source_path,
                    settings,
                    document_source_path=document_source_path,
                )
            gray_review = outcome.pending

        with trace.stage(
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
            load_info["output_summary"] = document.title
            load_info["candidate_count"] = 1
            load_info["method"] = document.load_method or (
                source_path.suffix.lower().lstrip(".") or "unknown"
            )
            load_info["provider"] = document.load_provider or "file"

        with trace.stage(
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
            _set_trace_summary(
                trace,
                source_path=document_source_path,
                document_id=document_id,
                title=document_title,
                status=status,
                chunk_count=0,
                chunks_with_images=0,
            )
            trace.close()
            save_trace(settings, trace)
            return IngestResult(
                document_id=document_id,
                chunk_count=0,
                elapsed_ms=trace.total_elapsed_ms,
                trace_id=trace.trace_id,
                status=status,
            )

        splitter = splitter_factory.create(settings)
        with trace.stage(
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
                    gray_review=gray_review,
                ),
            )
            for index, text in enumerate(texts)
        ]

        chunks = prepare_chunks(chunks, settings, trace)

        return PreparedIngest(
            trace=trace,
            source_path=source_path,
            document_id=document_id,
            document_title=document_title,
            document_source_path=document_source_path,
            status=status,
            gray_review=gray_review,
            chunks=chunks,
            previous_document_id=previous_document_id,
        )
    except Exception:
        save_trace(settings, trace)
        raise


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
        _set_trace_summary(
            trace,
            source_path=prepared.document_source_path,
            document_id=prepared.document_id,
            title=prepared.document_title,
            status=prepared.status,
            chunk_count=len(prepared.chunks),
            chunks_with_images=_chunks_with_images(prepared.chunks),
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

    begin_batch(ModelResource.MLX_VLM)
    try:
        prepared = prepare_ingest_source(
            source_path,
            settings,
            pdf_load_mode=pdf_load_mode,
            on_stage=on_stage,
            knowledge=knowledge,
        )
    finally:
        end_batch()

    if isinstance(prepared, IngestResult):
        return prepared

    begin_batch(ModelResource.BGE_M3)
    try:
        return commit_prepared_ingest(prepared, settings, knowledge=knowledge)
    finally:
        end_batch()
