"""入库编排 — phased prepare flow with PrepareTraceRecorder at pipeline seam."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from wenmai.config import Settings
from wenmai.factories import splitter as splitter_factory
from wenmai.ingestion.admission import AdmissionGate
from wenmai.ingestion.loaders import LoadedDocument, SourceLoadError, load_source
from wenmai.ingestion.prepare import prepare_chunks
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.knowledge.domain import REVIEW_PENDING, REVIEW_STATUS_FIELD, stamp_review_status
from wenmai.models import Chunk, IngestResult
from wenmai.storage.document_images import IMAGE_PLACEHOLDER_RE
from wenmai.tracing import StageRecord
from wenmai.tracing.prepare_recorder import PrepareTraceRecorder


@dataclass
class PrepareBody:
    """Domain result from prepare phase — no trace."""

    source_path: Path
    document_id: str
    document_title: str
    document_source_path: str
    status: str
    gray_review: bool
    chunks: list[Chunk]
    previous_document_id: str | None


@dataclass
class PreparedIngest:
    """Prepared document plus trace recorder owned by pipeline."""

    body: PrepareBody
    recorder: PrepareTraceRecorder


def count_chunks_with_images(chunks: list[Chunk]) -> int:
    return sum(1 for chunk in chunks if IMAGE_PLACEHOLDER_RE.search(chunk.text))


def _finish_rejected_ingest(
    recorder: PrepareTraceRecorder,
    source_path: Path,
    settings: Settings,
    *,
    document_source_path: str,
) -> IngestResult:
    document_id = hashlib.sha256(source_path.read_bytes()).hexdigest()
    recorder.set_summary(
        source_path=document_source_path,
        document_id=document_id,
        title=source_path.stem,
        status="rejected",
        chunk_count=0,
        chunks_with_images=0,
    )
    recorder.close_and_save(settings)
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


def prepare_ingest(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
    on_stage: Callable[[StageRecord], None] | None = None,
    knowledge: Knowledge | None = None,
    recorder: PrepareTraceRecorder | None = None,
) -> PreparedIngest | IngestResult:
    """Phase-1 ingest: 入库准入 through transform; no embed/upsert."""
    trace_recorder = recorder or PrepareTraceRecorder(on_stage=on_stage)
    knowledge = knowledge or create_knowledge(settings)
    document_source_path = str(source_path)

    try:
        admission = AdmissionGate(settings).admit(source_path)
        for stage in admission.stages:
            trace_recorder.append_stage(stage)

        if admission.decision == "rejected":
            return _finish_rejected_ingest(
                trace_recorder,
                source_path,
                settings,
                document_source_path=document_source_path,
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
            load_info["output_summary"] = document.title
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
            )
            trace_recorder.close_and_save(settings)
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
            chunks=chunks,
            previous_document_id=previous_document_id,
        )
        return PreparedIngest(body=body, recorder=trace_recorder)
    except Exception:
        trace_recorder.save_on_error(settings)
        raise


__all__ = ["PrepareBody", "PreparedIngest", "count_chunks_with_images", "prepare_ingest"]
