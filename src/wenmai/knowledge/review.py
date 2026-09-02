from __future__ import annotations

from dataclasses import dataclass

from wenmai.knowledge.domain import (
    REVIEW_PENDING,
    culture_domain,
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


def collect_pending_review_documents(chunks: list[Chunk]) -> list[PendingReviewDocument]:
    by_document: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if review_status(chunk) != REVIEW_PENDING:
            continue
        by_document.setdefault(chunk.document_id, []).append(chunk)

    pending: list[PendingReviewDocument] = []
    for document_id in sorted(by_document):
        doc_chunks = by_document[document_id]
        first = doc_chunks[0]
        pending.append(
            PendingReviewDocument(
                document_id=document_id,
                title=title(first),
                culture_domain=culture_domain(first),
                chunk_count=len(doc_chunks),
            )
        )
    return pending
