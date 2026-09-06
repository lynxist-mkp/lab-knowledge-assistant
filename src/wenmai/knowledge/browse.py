from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from wenmai.knowledge.domain import chunk_title, culture_domain, review_status, summary, tags, title
from wenmai.models import Chunk


@dataclass(frozen=True)
class ChunkDetail:
    chunk_id: str
    document_id: str
    title: str
    chunk_title: str
    culture_domain: str
    review_status: str
    text: str
    metadata: dict[str, Any]
    summary: str = ""
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "chunk_title": self.chunk_title,
            "culture_domain": self.culture_domain,
            "审阅状态": self.review_status,
            "text": self.text,
            "summary": self.summary,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ChunkSummary:
    chunk_id: str
    document_id: str
    preview: str
    review_status: str = "已通过"
    chunk_title: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "preview": self.preview,
            "审阅状态": self.review_status,
            "chunk_title": self.chunk_title,
            "summary": self.summary,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    title: str
    chunk_count: int
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    chunks: list[ChunkSummary] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "chunk_count": self.chunk_count,
            "summary": self.summary,
            "tags": list(self.tags),
            "chunks": [chunk.as_dict() for chunk in self.chunks],
        }


@dataclass(frozen=True)
class CultureDomainGroup:
    culture_domain: str
    document_count: int
    chunk_count: int
    documents: list[DocumentSummary] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "culture_domain": self.culture_domain,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "documents": [document.as_dict() for document in self.documents],
        }


@dataclass(frozen=True)
class OverviewStats:
    document_count: int
    chunk_count: int
    avg_query_latency_ms: float | None
    query_latency_p50_ms: float | None = None
    query_latency_p95_ms: float | None = None
    stage_latency: dict[str, dict[str, float | None]] | None = None

    def as_dict(self) -> dict[str, int | float | dict[str, dict[str, float | None]] | None]:
        return asdict(self)


def chunk_detail_from_chunk(chunk: Chunk) -> ChunkDetail:
    return ChunkDetail(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=title(chunk),
        chunk_title=chunk_title(chunk),
        culture_domain=culture_domain(chunk),
        review_status=review_status(chunk),
        text=chunk.text,
        summary=summary(chunk),
        tags=tags(chunk),
        metadata=chunk.metadata,
    )
