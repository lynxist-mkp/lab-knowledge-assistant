from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wenmai.config import Settings
from wenmai.factories import embedding as embedding_factory
from wenmai.factories import splitter as splitter_factory
from wenmai.factories import transform as transform_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.models import Chunk, IngestResult
from wenmai.storage.cleanup import delete_document_from_stores
from wenmai.storage.fingerprints import FingerprintStore
from wenmai.storage.paths import store_path
from wenmai.tracing.context import TraceContext
from wenmai.tracing.writer import JsonlTraceWriter


@dataclass
class LoadedDocument:
    document_id: str
    text: str
    title: str
    url: str
    page: int
    source_path: str
    extra: dict[str, Any]


def _first_heading(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def load_markdown(path: Path) -> LoadedDocument:
    raw_bytes = path.read_bytes()
    raw = raw_bytes.decode("utf-8")
    front_matter: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            parsed = yaml.safe_load(parts[1]) or {}
            if isinstance(parsed, dict):
                front_matter = parsed
            body = parts[2].lstrip("\n")
    title = str(front_matter.get("title") or _first_heading(body) or path.stem)
    url = str(front_matter.get("source_url") or front_matter.get("url") or "")
    page = int(front_matter.get("page") or 1)
    document_id = hashlib.sha256(raw_bytes).hexdigest()
    extra = {
        key: value
        for key, value in front_matter.items()
        if key not in {"title", "source_url", "url", "page"}
    }
    return LoadedDocument(
        document_id=document_id,
        text=body,
        title=title,
        url=url,
        page=page,
        source_path=str(path),
        extra=extra,
    )


def ingest_markdown(source_path: Path, settings: Settings) -> IngestResult:
    trace = TraceContext(trace_type="ingestion")
    writer = JsonlTraceWriter(store_path(settings, "traces"))
    fingerprint_store = FingerprintStore.from_settings(settings)
    document: LoadedDocument | None = None
    chunks: list[Chunk] = []
    status = "ingested"
    try:
        with trace.stage(
            "load", method="markdown", provider="file", input_summary=str(source_path)
        ) as load_info:
            document = load_markdown(source_path)
            load_info["output_summary"] = document.title
            load_info["candidate_count"] = 1

        assert document is not None
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
