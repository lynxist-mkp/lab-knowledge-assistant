"""配图 lifecycle: attach → placeholder → delete with document."""

from __future__ import annotations

from pathlib import Path

from lab_knowledge.config import Settings
from lab_knowledge.storage.images import (
    IMAGE_PLACEHOLDER_RE,
    ImageStore,
    find_image_ids,
    format_image_placeholder,
)

__all__ = [
    "IMAGE_PLACEHOLDER_RE",
    "DocumentImages",
    "find_image_ids",
    "format_image_placeholder",
]


class DocumentImages:
    """Owns 配图 invariant: extract/save placeholders, delete with doc."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = ImageStore(settings)

    def attach(
        self,
        *,
        document_id: str,
        source_path: str,
        page: int,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> str:
        """Save image bytes and return the `[IMAGE: id]` placeholder for text."""
        image_id = self._store.save(
            document_id=document_id,
            source_path=source_path,
            page=page,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )
        return format_image_placeholder(image_id)

    def get(self, image_id: str) -> Path | None:
        record = self._store.get(image_id)
        if record is None:
            return None
        return Path(record.file_path)

    def exists(self, image_id: str) -> bool:
        return self._store.get(image_id) is not None

    def delete_for_document(self, document_id: str) -> None:
        self._store.delete_by_document_id(document_id)
