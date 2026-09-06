from __future__ import annotations

from typing import Any

from wenmai.knowledge.collections import UnknownCollectionError
from wenmai.knowledge.document_card import DocumentNotFoundError
from wenmai.knowledge.document_management import DocumentManagement
from wenmai.mcp.envelope import McpMeta, McpRefs, envelope, scope_for


def reviews_list_pending(
    document_management: DocumentManagement,
    *,
    collection_id: str | None = None,
) -> dict[str, Any]:
    try:
        pending = document_management.list_pending_reviews(collection_id=collection_id)
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc
    data = [item.as_dict() for item in pending]
    refs = McpRefs(document_ids=[item.document_id for item in pending])
    return envelope(
        data=data,
        scope=scope_for(document_management.settings, collection_id=collection_id),
        refs=refs,
        meta=McpMeta(count=len(data)),
    ).as_dict()


def reviews_approve(
    document_management: DocumentManagement,
    document_id: str,
    *,
    collection_id: str | None = None,
) -> dict[str, Any]:
    try:
        document_management.approve_review(document_id, collection_id=collection_id)
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return envelope(
        data={"document_id": document_id, "审阅状态": "已通过"},
        scope=scope_for(document_management.settings, collection_id=collection_id),
        refs=McpRefs(document_ids=[document_id]),
        meta=McpMeta(count=1),
    ).as_dict()


def reviews_reject(
    document_management: DocumentManagement,
    document_id: str,
    *,
    collection_id: str | None = None,
) -> dict[str, Any]:
    try:
        document_management.reject_review(document_id, collection_id=collection_id)
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return envelope(
        data={"document_id": document_id, "审阅状态": "已驳回"},
        scope=scope_for(document_management.settings, collection_id=collection_id),
        refs=McpRefs(document_ids=[document_id]),
        meta=McpMeta(count=1),
    ).as_dict()
