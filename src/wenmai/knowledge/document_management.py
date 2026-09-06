from __future__ import annotations

from typing import TYPE_CHECKING

from wenmai.config import Settings
from wenmai.knowledge.browse import ChunkDetail, CultureDomainGroup, DocumentSummary
from wenmai.knowledge.collections import (
    Collection,
    CollectionReadModel,
    CollectionScope,
    CollectionStats,
    resolve_routable_collection_scope,
)
from wenmai.knowledge.document_card import DocumentCard, DocumentNotFoundError
from wenmai.knowledge.image_refs import (
    ImageContent,
    ImageNotFoundError,
    ImageRef,
    ImageReferenceService,
)
from wenmai.knowledge.review import PendingReviewDocument
from wenmai.knowledge.store import create_knowledge

if TYPE_CHECKING:
    from wenmai.knowledge.store import Knowledge


class DocumentManagement:
    """Document lifecycle facade above Knowledge read/write seams."""

    def __init__(
        self,
        settings: Settings,
        knowledge: Knowledge | None = None,
        *,
        scope: CollectionScope | None = None,
    ) -> None:
        self._settings = settings
        self._knowledge = knowledge or create_knowledge(settings)
        self._collections = CollectionReadModel(settings, self._knowledge)
        self._images = ImageReferenceService(settings)
        self._scope = scope

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def scope(self) -> CollectionScope:
        if self._scope is not None:
            return self._scope
        return self._collections.resolve_scope()

    def list_collections(self) -> list[Collection]:
        return self._collections.list_collections()

    def get_collection_stats(self, collection_id: str | None = None) -> CollectionStats:
        return self.for_collection(collection_id)._collections.get_stats()

    def for_collection(self, collection_id: str | None = None) -> DocumentManagement:
        resolved = resolve_routable_collection_scope(self._settings, collection_id)
        if self._scope is not None and self._scope.collection_id == resolved.collection_id:
            return self
        return DocumentManagement(resolved.settings, scope=resolved)

    def list_documents(
        self,
        *,
        collection_id: str | None = None,
        culture_domain: str | None = None,
    ) -> list[DocumentSummary]:
        scoped = self.for_collection(collection_id)
        groups = scoped._knowledge.browse_by_culture_domain()
        documents: list[DocumentSummary] = []
        for group in groups:
            if culture_domain is not None and group.culture_domain != culture_domain:
                continue
            documents.extend(group.documents)
        return documents

    def browse_groups(
        self, *, collection_id: str | None = None
    ) -> list[CultureDomainGroup]:
        return self.for_collection(collection_id)._knowledge.browse_by_culture_domain()

    def get_document(self, document_id: str, *, collection_id: str | None = None) -> DocumentCard:
        return self.for_collection(collection_id)._knowledge.document_card(document_id)

    def get_chunk_detail(
        self,
        chunk_id: str,
        *,
        collection_id: str | None = None,
    ) -> ChunkDetail | None:
        return self.for_collection(collection_id)._knowledge.chunk_detail(chunk_id)

    def delete_document(self, document_id: str, *, collection_id: str | None = None) -> None:
        scoped = self.for_collection(collection_id)
        if not scoped._knowledge.get_by_document_id(document_id):
            raise DocumentNotFoundError(document_id)
        scoped._knowledge.delete_document(document_id)

    def list_pending_reviews(
        self, *, collection_id: str | None = None
    ) -> list[PendingReviewDocument]:
        return self.for_collection(collection_id)._knowledge.list_pending_review_documents()

    def approve_review(self, document_id: str, *, collection_id: str | None = None) -> None:
        self.for_collection(collection_id)._knowledge.approve_review(document_id)

    def reject_review(self, document_id: str, *, collection_id: str | None = None) -> None:
        self.for_collection(collection_id)._knowledge.reject_review(document_id)

    def image_refs_for_document(
        self, document_id: str, *, collection_id: str | None = None
    ) -> list[ImageRef]:
        scoped = self.for_collection(collection_id)
        return scoped._images.list_refs_for_document(document_id)

    def get_image_ref(self, image_id: str, *, collection_id: str | None = None) -> ImageRef:
        ref = self.for_collection(collection_id)._images.get_ref(image_id)
        if ref is None:
            raise ImageNotFoundError(image_id)
        return ref

    def get_image_content(
        self, image_id: str, *, collection_id: str | None = None
    ) -> ImageContent:
        return self.for_collection(collection_id)._images.get_content(image_id)


def create_document_management(
    settings: Settings,
    knowledge: Knowledge | None = None,
    *,
    collection_id: str | None = None,
) -> DocumentManagement:
    if collection_id is None:
        return DocumentManagement(settings, knowledge=knowledge)
    scope = resolve_routable_collection_scope(settings, collection_id)
    return DocumentManagement(scope.settings, scope=scope)


__all__ = [
    "DocumentManagement",
    "create_document_management",
]
