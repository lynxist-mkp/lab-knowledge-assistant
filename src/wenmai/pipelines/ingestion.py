from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from wenmai.config import Settings
from wenmai.factories import splitter as splitter_factory
from wenmai.ingestion.loaders import LoadedDocument, SourceLoadError, load_source
from wenmai.ingestion.prepare import prepare_chunks
from wenmai.ingestion.quality import evaluate_quality_gate, peek_source
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.knowledge.domain import stamp_review_status
from wenmai.models import Chunk, IngestResult
from wenmai.storage.document_images import IMAGE_PLACEHOLDER_RE
from wenmai.tracing import StageRecord, TraceContext, save_trace
from wenmai.tracing.stages.ingestion import IngestionStage


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


def ingest_source(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
) -> IngestResult:
    trace = TraceContext(trace_type="ingestion")
    trace._on_stage = on_stage
    knowledge = knowledge or create_knowledge(settings)
    chunks: list[Chunk] = []
    status = "ingested"
    document_id = ""
    document_title = ""
    document_source_path = str(source_path)
    gray_review = False
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
            if source_peek.defer_reject and gate_result.ratio < settings.quality_gate.reject_below:
                gate_info["output_summary"] += " defer=scanned_pdf"
            gate_info["candidate_count"] = 1
            if gate_result.band == "reject":
                gate_info["error"] = (
                    f"effective_char_ratio {gate_result.ratio:.2f} "
                    f"below {settings.quality_gate.reject_below:.2f}"
                )
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
                return IngestResult(
                    document_id=document_id,
                    chunk_count=0,
                    elapsed_ms=trace.total_elapsed_ms,
                    trace_id=trace.trace_id,
                    status="rejected",
                )
            gray_review = gate_result.band == "gray"

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
            if status == "skipped":
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

        if status == "skipped":
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

        _set_trace_summary(
            trace,
            source_path=document_source_path,
            document_id=document_id,
            title=document_title,
            status=status,
            chunk_count=len(chunks),
            chunks_with_images=_chunks_with_images(chunks),
        )

        upserted = knowledge.commit_document(
            source_path=document.source_path,
            sha256=document.document_id,
            document_id=document.document_id,
            status=status,
            chunks=chunks,
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
    finally:
        save_trace(settings, trace)

    return IngestResult(
        document_id=document.document_id,
        chunk_count=len(chunks),
        elapsed_ms=trace.total_elapsed_ms,
        trace_id=trace.trace_id,
        status=status,
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
        metadata["审阅状态"] = "待审"
    else:
        stamp_review_status(metadata)
    return metadata
