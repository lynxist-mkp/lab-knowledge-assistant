from __future__ import annotations

from typing import Any

from wenmai.knowledge.collections import UnknownCollectionError
from wenmai.knowledge.document_management import DocumentManagement
from wenmai.mcp.envelope import McpMeta, envelope, scope_for


def collections_list(document_management: DocumentManagement) -> dict[str, Any]:
    collections = document_management.list_collections()
    return envelope(
        data=[item.as_dict() for item in collections],
        scope=scope_for(document_management.settings),
        meta=McpMeta(count=len(collections)),
    ).as_dict()


def collections_get_stats(
    document_management: DocumentManagement,
    collection_id: str | None = None,
) -> dict[str, Any]:
    try:
        stats = document_management.get_collection_stats(collection_id)
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc
    resolved = collection_id or document_management.settings.product.collection
    return envelope(
        data=stats.as_dict(),
        scope=scope_for(document_management.settings, collection_id=resolved),
        meta=McpMeta(count=stats.document_count),
    ).as_dict()
