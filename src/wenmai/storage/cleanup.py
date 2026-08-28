from __future__ import annotations

from wenmai.config import Settings
from wenmai.knowledge.store import create_knowledge


def delete_document_from_stores(settings: Settings, document_id: str) -> None:
    """Remove one document via Knowledge — sole coordinator for all ingestion stores."""
    create_knowledge(settings).delete_document(document_id)
