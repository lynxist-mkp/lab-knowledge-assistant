from __future__ import annotations

from dataclasses import dataclass

from wenmai.knowledge.store import Knowledge


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
    return [
        PendingDocument(
            document_id=item.document_id,
            title=item.title,
            culture_domain=item.culture_domain,
            chunk_count=item.chunk_count,
        )
        for item in knowledge.list_pending_review_documents()
    ]


def approve_document(knowledge: Knowledge, document_id: str) -> None:
    knowledge.approve_review(document_id)


def reject_document(knowledge: Knowledge, document_id: str) -> None:
    knowledge.reject_review(document_id)
