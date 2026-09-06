from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    index: int
    chunk_id: str
    document_id: str
    title: str
    excerpt: str
    url: str = ""
    source_kind: str = "group_doc"
    source_label: str = "组内资料"
    authors: str = ""
    publication_year: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "excerpt": self.excerpt,
            "source_kind": self.source_kind,
            "source_label": self.source_label,
        }
        if self.url:
            payload["url"] = self.url
        if self.authors:
            payload["authors"] = self.authors
        if self.publication_year is not None:
            payload["publication_year"] = self.publication_year
        return payload


@dataclass
class AskResult:
    answer: str
    citations: list[Citation]
    trace_id: str
    refused: bool = False
    refusal_reason: str | None = None
    error: str | None = None
    ranked_chunks: list[ScoredChunk] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "answer": self.answer,
            "citations": [citation.as_dict() for citation in self.citations],
            "trace_id": self.trace_id,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class IngestResult:
    document_id: str
    chunk_count: int
    elapsed_ms: float
    trace_id: str
    status: str = "ingested"

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "elapsed_ms": self.elapsed_ms,
            "trace_id": self.trace_id,
            "status": self.status,
        }


@dataclass(frozen=True)
class BulkIngestItemResult:
    source_path: str
    status: str
    document_id: str | None = None
    chunk_count: int | None = None
    trace_id: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_path": self.source_path,
            "status": self.status,
        }
        if self.document_id is not None:
            payload["document_id"] = self.document_id
        if self.chunk_count is not None:
            payload["chunk_count"] = self.chunk_count
        if self.trace_id is not None:
            payload["trace_id"] = self.trace_id
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class BulkIngestResult:
    source_root: str
    source_kind: str
    total_files: int
    ingested_count: int
    rebuilt_count: int
    skipped_count: int
    failed_count: int
    results: list[BulkIngestItemResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_root": self.source_root,
            "source_kind": self.source_kind,
            "total_files": self.total_files,
            "ingested_count": self.ingested_count,
            "rebuilt_count": self.rebuilt_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "results": [item.as_dict() for item in self.results],
        }
