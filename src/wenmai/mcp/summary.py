from __future__ import annotations

from typing import Any

from wenmai.config import Settings
from wenmai.knowledge.collections import UnknownCollectionError
from wenmai.knowledge.document_card import DocumentNotFoundError
from wenmai.knowledge.document_management import (
    DocumentManagement,
    create_document_management,
)


class GetDocumentSummaryError(Exception):
    def __init__(self, message: str, document_id: str) -> None:
        super().__init__(message)
        self.document_id = document_id


def get_document_summary(
    document_id: str,
    settings: Settings | None = None,
    document_management: DocumentManagement | None = None,
    collection_id: str | None = None,
) -> dict[str, Any]:
    """MCP adapter: fetch document card summary via DocumentManagement."""
    resolved = settings or Settings.load()
    mgmt = document_management or create_document_management(resolved)
    try:
        return mgmt.get_document_summary(document_id, collection_id=collection_id)
    except UnknownCollectionError as exc:
        raise ValueError(str(exc)) from exc
    except DocumentNotFoundError as exc:
        raise GetDocumentSummaryError(str(exc), document_id) from exc
