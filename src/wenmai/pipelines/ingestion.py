from __future__ import annotations

from pathlib import Path

from wenmai.config import Settings
from wenmai.factories import embedding as embedding_factory
from wenmai.factories import splitter as splitter_factory
from wenmai.factories import transform as transform_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.components.paddleocr.adapter import choose_pdf_route
from wenmai.ingestion.loaders import load_source
from wenmai.models import Chunk, IngestResult
from wenmai.storage.cleanup import delete_document_from_stores
from wenmai.storage.fingerprints import FingerprintStore
from wenmai.storage.paths import store_path
from wenmai.tracing.context import TraceContext
from wenmai.tracing.writer import JsonlTraceWriter


def ingest_markdown(
    source_path: Path,
    settings: Settings,
    *,
    pdf_load_mode: str | None = None,
) -> IngestResult:
    trace = TraceContext(trace_type="ingestion")
    writer = JsonlTraceWriter(store_path(settings, "traces"))
    fingerprint_store = FingerprintStore.from_settings(settings)
    chunks: list[Chunk] = []
    status = "ingested"
    try:
        with trace.stage(
            "load",
            method="pending",
            provider="pending",
            input_summary=str(source_path),
        ) as load_info:
            if source_path.suffix.lower() == ".pdf":
                route = choose_pdf_route(
                    source_path,
                    settings.pdf_load,
                    override_mode=pdf_load_mode,
                )
                if route == "markitdown":
                    load_info["method"] = "markitdown"
                    load_info["provider"] = "markitdown"
                else:
                    load_info["method"] = "paddleocr-vl"
                    load_info["provider"] = "mlx-vlm-server"
            document = load_source(source_path, settings, pdf_load_mode=pdf_load_mode)
            load_info["output_summary"] = document.title
            load_info["candidate_count"] = 1
            load_method = document.load_method or (
                source_path.suffix.lower().lstrip(".") or "unknown"
            )
            load_provider = document.load_provider or "file"
            load_info["method"] = load_method
            load_info["provider"] = load_provider

        previous = fingerprint_store.get_by_source_path(document.source_path)
        with trace.stage(
            "integrity",
            method="sha256",
            provider="sqlite",
            input_summary=document.document_id[:12],
        ) as integrity_info:
            if previous and previous.sha256 == document.document_id:
                status = "skipped"
                integrity_info["output_summary"] = "skipped: unchanged sha256"
                integrity_info["candidate_count"] = 0
            elif previous:
                status = "rebuilt"
                delete_document_from_stores(settings, previous.document_id)
                integrity_info["output_summary"] = (
                    f"rebuilt: replaced {previous.document_id[:12]}..."
                )
                integrity_info["candidate_count"] = 1
            else:
                integrity_info["output_summary"] = "new file"
                integrity_info["candidate_count"] = 1

        if status == "skipped":
            trace.close()
            return IngestResult(
                document_id=document.document_id,
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
                metadata={
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
                },
            )
            for index, text in enumerate(texts)
        ]

        chunks = transform_factory.run_registered(chunks, settings, trace)

        embedder = embedding_factory.create(settings)
        with trace.stage(
            "embed",
            method=embedder.provider_name,
            provider=embedder.provider_name,
            input_summary=f"{len(chunks)} chunks",
        ) as embed_info:
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                chunk.embedding = vector
            embed_info["candidate_count"] = len(chunks)
            embed_info["output_summary"] = f"dim={embedder.dimension}"

        store = vector_store_factory.create(settings)
        with trace.stage(
            "upsert",
            method=store.provider_name,
            provider=store.provider_name,
            input_summary=f"{len(chunks)} chunks",
        ) as upsert_info:
            store.upsert(chunks)
            upsert_info["candidate_count"] = len(chunks)
            upsert_info["output_summary"] = f"upserted {len(chunks)}"

        fingerprint_store.upsert(
            source_path=document.source_path,
            sha256=document.document_id,
            document_id=document.document_id,
            status=status,
        )
    finally:
        trace.close()
        writer.write(trace)
        fingerprint_store.close()

    return IngestResult(
        document_id=document.document_id,
        chunk_count=len(chunks),
        elapsed_ms=trace.total_elapsed_ms,
        trace_id=trace.trace_id,
        status=status,
    )
