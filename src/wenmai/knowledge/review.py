from __future__ import annotations

from dataclasses import dataclass, field

from wenmai.knowledge.browse import ChunkSummary
from wenmai.knowledge.domain import (
    REVIEW_PENDING,
    culture_domain,
    preview,
    review_status,
    title,
)
from wenmai.models import Chunk


@dataclass(frozen=True)
class PendingReviewDocument:
    document_id: str
    title: str
    culture_domain: str
    chunk_count: int
    chunks: list[ChunkSummary] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "culture_domain": self.culture_domain,
            "chunk_count": self.chunk_count,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
        }


def collect_pending_review_documents(chunks: list[Chunk]) -> list[PendingReviewDocument]:
    by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if review_status(chunk) != REVIEW_PENDING:
            continue
        by_document.setdefault(chunk.document_id, []).append(chunk)

    pending: list[PendingReviewDocument] = []
    for document_id in sorted(by_document):
        doc_chunks = sorted(by_document[document_id], key=lambda item: item.chunk_id)
        first = doc_chunks[0]
        pending.append(
            PendingReviewDocument(
                document_id=document_id,
                title=title(first),
                culture_domain=culture_domain(first),
                chunk_count=len(doc_chunks),
                chunks=[
                    ChunkSummary(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        preview=preview(chunk.text),
                        review_status=review_status(chunk),
                    )
                    for chunk in doc_chunks
                ],
            )
        )
    return pending


def list_pending_from_catalog(catalog: object) -> list[PendingReviewDocument]:
    """Catalog-backed pending list — avoids scanning list_all()."""
    pending: list[PendingReviewDocument] = []
    for entry in catalog.list_pending_documents():
        doc_chunks = [
            chunk for chunk in entry.chunks if chunk.review_status == REVIEW_PENDING
        ]
        pending.append(
            PendingReviewDocument(
                document_id=entry.document_id,
                title=entry.title,
                culture_domain=entry.culture_domain,
                chunk_count=len(doc_chunks),
                chunks=doc_chunks,
            )
        )
    return pending
