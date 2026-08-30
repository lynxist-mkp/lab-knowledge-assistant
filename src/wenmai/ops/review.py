from __future__ import annotations

from dataclasses import dataclass

from wenmai.knowledge.document_card import DocumentNotFoundError
from wenmai.knowledge.domain import culture_domain, review_status, title
from wenmai.knowledge.store import Knowledge

_REVIEW_PENDING = "待审"
_REVIEW_APPROVED = "已通过"


@dataclass(frozen=True)
class PendingDocument:
    document_id: str
    title: str
    culture_domain: str
    chunk_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "culture_domain": self.culture_domain,
            "chunk_count": self.chunk_count,
        }


def list_pending_documents(knowledge: Knowledge) -> list[PendingDocument]:
    by_document: dict[str, list] = {}
    for chunk in knowledge.list_all():
        if review_status(chunk) != _REVIEW_PENDING:
            continue
        by_document.setdefault(chunk.document_id, []).append(chunk)

    pending: list[PendingDocument] = []
    for document_id in sorted(by_document):
        chunks = by_document[document_id]
        first = chunks[0]
        pending.append(
            PendingDocument(
                document_id=document_id,
                title=title(first),
                culture_domain=culture_domain(first),
                chunk_count=len(chunks),
            )
        )
    return pending


def approve_document(knowledge: Knowledge, document_id: str) -> None:
    if not knowledge.get_by_document_id(document_id):
        raise DocumentNotFoundError(document_id)
    knowledge.set_review_status(document_id, _REVIEW_APPROVED)


def reject_document(knowledge: Knowledge, document_id: str) -> None:
    if not knowledge.get_by_document_id(document_id):
        raise DocumentNotFoundError(document_id)
    knowledge.delete_document(document_id)
