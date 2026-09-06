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


def is_registered_collection(settings: Settings, collection_id: str) -> bool:
    return any(reg.collection_id == collection_id for reg in settings.collections)


def display_name_for_collection(settings: Settings, collection_id: str) -> str:
    for registration in settings.collections:
        if registration.collection_id == collection_id:
            return registration.display_name
    return settings.product.name


def iter_configured_collections(settings: Settings) -> list[tuple[str, str]]:
    """Return configured collections as (collection_id, display_name), default first."""
    default_id = settings.default_collection_id
    seen = {default_id}
    configured = [(default_id, display_name_for_collection(settings, default_id))]
    for registration in settings.collections:
        if registration.collection_id in seen:
            continue
        seen.add(registration.collection_id)
        configured.append((registration.collection_id, registration.display_name))
    return configured


def resolve_collection_scope(
    settings: Settings, collection_id: str | None = None
) -> CollectionScope:
    resolved = resolve_collection_id(settings, collection_id)
    return CollectionScope(
        collection_id=resolved,
        display_name=display_name_for_collection(settings, resolved),
        settings=_settings_for_collection(settings, resolved),
    )


def resolve_collection_id(settings: Settings, collection_id: str | None) -> str:
    default = settings.product.collection
    if collection_id is None or collection_id == default:
        return default
    if is_registered_collection(settings, collection_id):
        return collection_id
    raise UnknownCollectionError(collection_id)


def _settings_for_collection(settings: Settings, collection_id: str) -> Settings:
    # Return a collection-scoped settings view so callers can bind adapters once.
    return replace(
        settings,
        product=replace(settings.product, collection=collection_id),
    )


def resolve_routable_collection_scope(
    settings: Settings, collection_id: str | None = None
) -> CollectionScope:
    """Resolve a collection scope for routing to configured collection storage."""
    if collection_id is None:
        return resolve_collection_scope(settings, None)
    if collection_id == settings.default_collection_id:
        return resolve_collection_scope(settings, collection_id)
    if is_registered_collection(settings, collection_id):
        return CollectionScope(
            collection_id=collection_id,
            display_name=display_name_for_collection(settings, collection_id),
            settings=_settings_for_collection(settings, collection_id),
        )
    raise UnknownCollectionError(collection_id)


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
        return resolve_routable_collection_scope(self._settings, collection_id)

    def list_collections(self) -> list[Collection]:
        return [
            Collection(
                collection_id=collection_id,
                display_name=display_name,
                stats=self.get_stats(collection_id),
            )
            for collection_id, display_name in iter_configured_collections(self._settings)
        ]

    def get_collection(self, collection_id: str | None = None) -> Collection:
        scope = self.resolve_scope(collection_id)
        return Collection(
            collection_id=scope.collection_id,
            display_name=scope.display_name,
            stats=self.get_stats(scope.collection_id),
        )

    def get_stats(self, collection_id: str | None = None) -> CollectionStats:
        scope = resolve_routable_collection_scope(self._settings, collection_id)
        knowledge = self._knowledge_for_scope(scope)
        return self._stats_from_knowledge(knowledge)

    def _knowledge_for_scope(self, scope: CollectionScope) -> Knowledge:
        if scope.settings.product.collection == self._settings.product.collection:
            return self._knowledge
        from wenmai.knowledge.store import create_knowledge

        return create_knowledge(scope.settings)

    def _stats_from_knowledge(self, knowledge: Knowledge) -> CollectionStats:
        groups = knowledge.browse_by_culture_domain()

        pending_documents = len(knowledge.list_pending_review_documents())
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

        document_count = knowledge.document_count
        approved_documents = document_count - pending_documents

        return CollectionStats(
            document_count=document_count,
            chunk_count=knowledge.chunk_count,
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
    "display_name_for_collection",
    "is_registered_collection",
    "iter_configured_collections",
    "resolve_routable_collection_scope",
    "resolve_collection_id",
    "resolve_collection_scope",
]
