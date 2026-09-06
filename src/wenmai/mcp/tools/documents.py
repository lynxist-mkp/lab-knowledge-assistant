from __future__ import annotations

from typing import Any

from wenmai.knowledge.collections import UnknownCollectionError
from wenmai.knowledge.document_card import DocumentNotFoundError
from wenmai.knowledge.document_management import DocumentManagement
from wenmai.mcp.envelope import McpMeta, McpRefs, envelope, scope_for


def documents_list(
    document_management: DocumentManagement,
    *,
    collection_id: str | None = None,
    culture_domain: str | None = None,
) -> dict[str, Any]:
    try:
        documents = document_management.list_documents(
            collection_id=collection_id,
            culture_domain=culture_domain,
        )
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc
    data = [document.as_dict() for document in documents]
    refs = McpRefs(document_ids=[document.document_id for document in documents])
    return envelope(
        data=data,
        scope=scope_for(
            document_management.settings,
            collection_id=collection_id,
            culture_domain=culture_domain,
        ),
        refs=refs,
        meta=McpMeta(count=len(data)),
    ).as_dict()


def documents_get(
    document_management: DocumentManagement,
    document_id: str,
    *,
    collection_id: str | None = None,
) -> dict[str, Any]:
    try:
        card = document_management.get_document(document_id, collection_id=collection_id)
        image_refs = document_management.image_refs_for_document(
            document_id,
            collection_id=collection_id,
        )
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return envelope(
        data={**card.as_dict(), "image_refs": [ref.as_dict() for ref in image_refs]},
        scope=scope_for(
            document_management.settings,
            collection_id=collection_id,
            culture_domain=card.culture_domain,
        ),
        refs=McpRefs(document_ids=[document_id]),
        meta=McpMeta(count=1),
    ).as_dict()


def documents_delete(
    document_management: DocumentManagement,
    document_id: str,
    *,
    collection_id: str | None = None,
) -> dict[str, Any]:
    try:
        document_management.delete_document(
            document_id,
            collection_id=collection_id,
        )
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return envelope(
        data={"document_id": document_id, "deleted": True},
        scope=scope_for(document_management.settings, collection_id=collection_id),
        refs=McpRefs(document_ids=[document_id]),
        meta=McpMeta(count=1),
    ).as_dict()
