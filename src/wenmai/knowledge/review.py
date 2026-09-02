from __future__ import annotations

from dataclasses import dataclass, field

from wenmai.knowledge.browse import ChunkSummary
from wenmai.knowledge.domain import REVIEW_PENDING


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
