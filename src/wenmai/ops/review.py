from __future__ import annotations

from wenmai.knowledge.review import PendingReviewDocument
from wenmai.knowledge.store import Knowledge


def list_pending_documents(knowledge: Knowledge) -> list[PendingReviewDocument]:
    return knowledge.list_pending_review_documents()


def approve_document(knowledge: Knowledge, document_id: str) -> None:
    knowledge.approve_review(document_id)


def reject_document(knowledge: Knowledge, document_id: str) -> None:
    knowledge.reject_review(document_id)


__all__ = ["approve_document", "list_pending_documents", "reject_document"]
