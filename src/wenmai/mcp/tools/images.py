from __future__ import annotations

from typing import Any

from wenmai.knowledge.document_management import DocumentManagement
from wenmai.knowledge.image_refs import ImageNotFoundError
from wenmai.mcp.envelope import McpMeta, McpRefs, envelope, scope_for


def images_get_ref(
    document_management: DocumentManagement,
    image_id: str,
    *,
    collection_id: str | None = None,
) -> dict[str, Any]:
    try:
        ref = document_management.get_image_ref(image_id, collection_id=collection_id)
    except ImageNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return envelope(
        data=ref,
        scope=scope_for(document_management.settings, collection_id=collection_id),
        refs=McpRefs(image_ids=[image_id]),
        meta=McpMeta(count=1),
    ).as_dict()


def images_get_content(
    document_management: DocumentManagement,
    image_id: str,
    *,
    collection_id: str | None = None,
) -> dict[str, Any]:
    try:
        content = document_management.get_image_content(image_id, collection_id=collection_id)
    except ImageNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return envelope(
        data=content.as_dict(),
        scope=scope_for(document_management.settings, collection_id=collection_id),
        refs=McpRefs(image_ids=[image_id]),
        meta=McpMeta(count=1),
    ).as_dict()
