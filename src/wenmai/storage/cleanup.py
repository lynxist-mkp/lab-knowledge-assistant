from __future__ import annotations

from wenmai.config import Settings
from wenmai.factories import bm25 as bm25_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.storage.images import ImageStore


def delete_document_from_stores(settings: Settings, document_id: str) -> None:
    """Remove one document's chunks from every ingestion store that exists today."""
    vector_store_factory.create(settings).delete_by_document_id(document_id)
    _delete_bm25_document(settings, document_id)
    _delete_image_document(settings, document_id)


def _delete_bm25_document(settings: Settings, document_id: str) -> None:
    index = bm25_factory.create(settings)
    index.delete_by_document_id(document_id)
    index.save()


def _delete_image_document(settings: Settings, document_id: str) -> None:
    ImageStore(settings).delete_by_document_id(document_id)
