from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wenmai.config import Settings
from wenmai.knowledge.browse import CultureDomainGroup, DocumentSummary
from wenmai.knowledge.collections import (
    Collection,
    CollectionReadModel,
    CollectionStats,
    resolve_collection_id,
)
from wenmai.knowledge.document_card import DocumentCard, DocumentNotFoundError
from wenmai.knowledge.image_refs import ImageContent, ImageNotFoundError, ImageReferenceService
from wenmai.knowledge.review import PendingReviewDocument
from wenmai.knowledge.store import create_knowledge

if TYPE_CHECKING:
    from wenmai.knowledge.store import Knowledge


class DocumentManagement:
    """Document lifecycle facade above Knowledge read/write seams."""

    def __init__(self, settings: Settings, knowledge: Knowledge | None = None) -> None:
        self._settings = settings
        self._knowledge = knowledge or create_knowledge(settings)
        self._collections = CollectionReadModel(settings, self._knowledge)
        self._images = ImageReferenceService(settings)

    @property
    def knowledge(self) -> Knowledge:
        return self._knowledge

    @property
    def settings(self) -> Settings:
        return self._settings

    def list_collections(self) -> list[Collection]:
        return self._collections.list_collections()

    def get_collection_stats(self, collection_id: str | None = None) -> CollectionStats:
        return self._collections.get_stats(collection_id)

    def _resolve_collection_id(self, collection_id: str | None = None) -> str:
        return resolve_collection_id(self._settings, collection_id)

    def list_documents(
        self,
        *,
        collection_id: str | None = None,
        culture_domain: str | None = None,
    ) -> list[DocumentSummary]:
        self._resolve_collection_id(collection_id)
        groups = self._knowledge.browse_by_culture_domain()
        documents: list[DocumentSummary] = []
        for group in groups:
            if culture_domain is not None and group.culture_domain != culture_domain:
                continue
            documents.extend(group.documents)
        return documents

    def browse_groups(
        self, *, collection_id: str | None = None
    ) -> list[CultureDomainGroup]:
        self._resolve_collection_id(collection_id)
        return self._knowledge.browse_by_culture_domain()

    def get_document(self, document_id: str, *, collection_id: str | None = None) -> DocumentCard:
        self._resolve_collection_id(collection_id)
        return self._knowledge.document_card(document_id)

    def delete_document(self, document_id: str, *, collection_id: str | None = None) -> None:
        self._resolve_collection_id(collection_id)
        if not self._knowledge.get_by_document_id(document_id):
            raise DocumentNotFoundError(document_id)
        self._knowledge.delete_document(document_id)

    def list_pending_reviews(
        self, *, collection_id: str | None = None
    ) -> list[PendingReviewDocument]:
        self._resolve_collection_id(collection_id)
        return self._knowledge.list_pending_review_documents()

    def approve_review(self, document_id: str, *, collection_id: str | None = None) -> None:
        self._resolve_collection_id(collection_id)
        self._knowledge.approve_review(document_id)

    def reject_review(self, document_id: str, *, collection_id: str | None = None) -> None:
        self._resolve_collection_id(collection_id)
        self._knowledge.reject_review(document_id)

    def image_refs_for_document(
        self, document_id: str, *, collection_id: str | None = None
    ) -> list[dict[str, Any]]:
        self._resolve_collection_id(collection_id)
        return [ref.as_dict() for ref in self._images.list_refs_for_document(document_id)]

    def get_image_ref(self, image_id: str, *, collection_id: str | None = None) -> dict[str, Any]:
        self._resolve_collection_id(collection_id)
        ref = self._images.get_ref(image_id)
        if ref is None:
            raise ImageNotFoundError(image_id)
        return ref.as_dict()

    def get_image_content(
        self, image_id: str, *, collection_id: str | None = None
    ) -> ImageContent:
        self._resolve_collection_id(collection_id)
        return self._images.get_content(image_id)


def create_document_management(
    settings: Settings,
    knowledge: Knowledge | None = None,
) -> DocumentManagement:
    return DocumentManagement(settings, knowledge=knowledge)


__all__ = [
    "DocumentManagement",
    "create_document_management",
]
