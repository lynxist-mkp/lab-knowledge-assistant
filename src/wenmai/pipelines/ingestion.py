from __future__ import annotations

from pathlib import Path

from wenmai.config import Settings
from wenmai.factories import embedding as embedding_factory
from wenmai.factories import splitter as splitter_factory
from wenmai.factories import transform as transform_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.ingestion.loaders import load_source
from wenmai.models import Chunk, IngestResult
from wenmai.storage.paths import store_path
from wenmai.tracing.context import TraceContext
from wenmai.tracing.writer import JsonlTraceWriter


def ingest_markdown(source_path: Path, settings: Settings) -> IngestResult:
    trace = TraceContext(trace_type="ingestion")
    writer = JsonlTraceWriter(store_path(settings, "traces"))
    try:
        load_method = source_path.suffix.lower().lstrip(".") or "unknown"
        with trace.stage(
            "load", method=load_method, provider="file", input_summary=str(source_path)
        ) as load_info:
            document = load_source(source_path, settings)
            load_info["output_summary"] = document.title
            load_info["candidate_count"] = 1

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
    finally:
        trace.close()
        writer.write(trace)

    return IngestResult(
        document_id=document.document_id,
        chunk_count=len(chunks),
        elapsed_ms=trace.total_elapsed_ms,
        trace_id=trace.trace_id,
    )
