from __future__ import annotations

from typing import Any

from wenmai.config import Settings
from wenmai.knowledge.document_card import DocumentNotFoundError, build_document_card
from wenmai.knowledge.store import Knowledge, create_knowledge


class GetDocumentSummaryError(Exception):
    def __init__(self, message: str, document_id: str) -> None:
        super().__init__(message)
        self.document_id = document_id


def get_document_summary(
    document_id: str,
    settings: Settings | None = None,
    knowledge: Knowledge | None = None,
) -> dict[str, Any]:
    """MCP tool handler: fetch document card from catalog and chunk metadata."""
    resolved = settings or Settings.load()
    kb = knowledge or create_knowledge(resolved)
    try:
        card = build_document_card(kb, document_id)
    except DocumentNotFoundError as exc:
        raise GetDocumentSummaryError(str(exc), document_id) from exc
    return card.as_dict()
