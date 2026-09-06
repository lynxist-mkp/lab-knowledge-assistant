from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from wenmai.config import Settings
from wenmai.knowledge.domain import REVIEW_PENDING

if TYPE_CHECKING:
    from wenmai.knowledge.store import Knowledge


class UnknownCollectionError(ValueError):
    def __init__(self, collection_id: str) -> None:
        super().__init__(f"unknown collection: {collection_id}")
        self.collection_id = collection_id


@dataclass(frozen=True)
class CollectionScope:
    collection_id: str
    display_name: str
    settings: Settings


def resolve_collection_scope(
    settings: Settings, collection_id: str | None = None
) -> CollectionScope:
    resolved = resolve_collection_id(settings, collection_id)
    return CollectionScope(
        collection_id=resolved,
        display_name=settings.product.name,
        settings=_settings_for_collection(settings, resolved),
    )


def resolve_collection_id(settings: Settings, collection_id: str | None) -> str:
    default = settings.product.collection
    if collection_id is None or collection_id == default:
        return default
    raise UnknownCollectionError(collection_id)


def _settings_for_collection(settings: Settings, collection_id: str) -> Settings:
    # Return a collection-scoped settings view so callers can bind adapters once.
    return replace(
        settings,
        product=replace(settings.product, collection=collection_id),
    )


@dataclass(frozen=True)
class ReviewStatusCounts:
    approved_documents: int
    pending_documents: int
    approved_chunks: int
    pending_chunks: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CultureDomainStats:
    culture_domain: str
    document_count: int
    chunk_count: int
    pending_chunks: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class CollectionStats:
    document_count: int
    chunk_count: int
    review: ReviewStatusCounts
    by_culture_domain: list[CultureDomainStats]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "review": self.review.as_dict(),
            "by_culture_domain": [item.as_dict() for item in self.by_culture_domain],
        }


@dataclass(frozen=True)
class Collection:
    collection_id: str
    display_name: str
    stats: CollectionStats

    def as_dict(self) -> dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "display_name": self.display_name,
            "stats": self.stats.as_dict(),
        }


class CollectionReadModel:
    """First-class collection scope read model with status-layered statistics."""

    def __init__(self, settings: Settings, knowledge: Knowledge) -> None:
        self._settings = settings
        self._knowledge = knowledge

    @property
    def default_collection_id(self) -> str:
        return self._settings.product.collection

    def resolve_scope(self, collection_id: str | None = None) -> CollectionScope:
        return resolve_collection_scope(self._settings, collection_id)

    def list_collections(self) -> list[Collection]:
        return [self.get_collection(self.default_collection_id)]

    def get_collection(self, collection_id: str | None = None) -> Collection:
        scope = self.resolve_scope(collection_id)
        return Collection(
            collection_id=scope.collection_id,
            display_name=scope.display_name,
            stats=self.get_stats(scope.collection_id),
        )

    def get_stats(self, collection_id: str | None = None) -> CollectionStats:
        resolve_collection_id(self._settings, collection_id)
        groups = self._knowledge.browse_by_culture_domain()

        pending_documents = len(self._knowledge.list_pending_review_documents())
        approved_chunks = 0
        pending_chunks = 0
        by_domain: list[CultureDomainStats] = []

        for group in groups:
            domain_pending = 0
            for document in group.documents:
                for chunk in document.chunks:
                    if chunk.review_status == REVIEW_PENDING:
                        pending_chunks += 1
                        domain_pending += 1
                    else:
                        approved_chunks += 1
            by_domain.append(
                CultureDomainStats(
                    culture_domain=group.culture_domain,
                    document_count=group.document_count,
                    chunk_count=group.chunk_count,
                    pending_chunks=domain_pending,
                )
            )

        document_count = self._knowledge.document_count
        approved_documents = document_count - pending_documents

        return CollectionStats(
            document_count=document_count,
            chunk_count=self._knowledge.chunk_count,
            review=ReviewStatusCounts(
                approved_documents=approved_documents,
                pending_documents=pending_documents,
                approved_chunks=approved_chunks,
                pending_chunks=pending_chunks,
            ),
            by_culture_domain=by_domain,
        )


__all__ = [
    "Collection",
    "CollectionReadModel",
    "CollectionScope",
    "CollectionStats",
    "CultureDomainStats",
    "ReviewStatusCounts",
    "UnknownCollectionError",
    "resolve_collection_id",
    "resolve_collection_scope",
]
