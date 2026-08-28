from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from wenmai.config import Settings
from wenmai.models import Chunk
from wenmai.tracing.store import average_query_latency_ms

_PREVIEW_CHARS = 120
_UNKNOWN_DOMAIN = "其他"


@dataclass(frozen=True)
class ChunkSummary:
    chunk_id: str
    document_id: str
    preview: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentSummary:
    document_id: str
    title: str
    chunk_count: int
    chunks: list[ChunkSummary] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "chunk_count": self.chunk_count,
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

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _culture_domain(chunk: Chunk) -> str:
    value = chunk.metadata.get("culture_domain")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _UNKNOWN_DOMAIN


def _title(chunk: Chunk) -> str:
    value = chunk.metadata.get("title")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return chunk.document_id


def _preview(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _PREVIEW_CHARS:
        return collapsed
    return collapsed[: _PREVIEW_CHARS - 1] + "…"


def browse_groups_from_chunks(chunks: list[Chunk]) -> list[CultureDomainGroup]:
    by_domain: dict[str, dict[str, list[Chunk]]] = {}
    for chunk in chunks:
        domain = _culture_domain(chunk)
        by_domain.setdefault(domain, {}).setdefault(chunk.document_id, []).append(chunk)

    groups: list[CultureDomainGroup] = []
    for culture_domain in sorted(by_domain):
        documents_for_domain = by_domain[culture_domain]
        documents: list[DocumentSummary] = []
        domain_chunk_count = 0
        for document_id in sorted(documents_for_domain):
            doc_chunks = sorted(documents_for_domain[document_id], key=lambda item: item.chunk_id)
            domain_chunk_count += len(doc_chunks)
            documents.append(
                DocumentSummary(
                    document_id=document_id,
                    title=_title(doc_chunks[0]),
                    chunk_count=len(doc_chunks),
                    chunks=[
                        ChunkSummary(
                            chunk_id=chunk.chunk_id,
                            document_id=chunk.document_id,
                            preview=_preview(chunk.text),
                        )
                        for chunk in doc_chunks
                    ],
                )
            )
        groups.append(
            CultureDomainGroup(
                culture_domain=culture_domain,
                document_count=len(documents),
                chunk_count=domain_chunk_count,
                documents=documents,
            )
        )
    return groups


def chunk_detail_from_chunk(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": _title(chunk),
        "culture_domain": _culture_domain(chunk),
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


def overview_from_chunks(chunks: list[Chunk], settings: Settings) -> OverviewStats:
    document_ids = {chunk.document_id for chunk in chunks if chunk.document_id}
    return OverviewStats(
        document_count=len(document_ids),
        chunk_count=len(chunks),
        avg_query_latency_ms=average_query_latency_ms(settings),
    )
