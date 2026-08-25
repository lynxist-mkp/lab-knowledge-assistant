from __future__ import annotations

from wenmai.config import Settings
from wenmai.factories import vector_store as vector_store_factory
from wenmai.storage.images import ImageStore
from wenmai.storage.paths import store_path


def delete_document_from_stores(settings: Settings, document_id: str) -> None:
    """Remove one document's chunks from every ingestion store that exists today."""
    vector_store_factory.create(settings).delete_by_document_id(document_id)
    _delete_bm25_document(settings, document_id)
    _delete_image_document(settings, document_id)


def _delete_bm25_document(settings: Settings, document_id: str) -> None:
    """Hook for ticket 14: delete BM25 postings keyed by document_id."""
    _ = store_path(settings, "bm25")
    _ = document_id


def _delete_image_document(settings: Settings, document_id: str) -> None:
    ImageStore(settings).delete_by_document_id(document_id)
