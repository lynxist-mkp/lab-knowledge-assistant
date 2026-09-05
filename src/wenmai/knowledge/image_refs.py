from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.storage.images import ImageRecord, ImageStore


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    document_id: str
    mime_type: str
    page: int | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "image_id": self.image_id,
            "document_id": self.document_id,
            "mime_type": self.mime_type,
        }
        if self.page is not None:
            payload["page"] = self.page
        return payload

    @classmethod
    def from_record(cls, record: ImageRecord) -> ImageRef:
        return cls(
            image_id=record.image_id,
            document_id=record.document_id,
            mime_type=record.mime_type,
            page=record.page,
        )


@dataclass(frozen=True)
class ImageContent:
    ref: ImageRef
    content_base64: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.as_dict(),
            "content_base64": self.content_base64,
            "mime_type": self.ref.mime_type,
        }


class ImageNotFoundError(LookupError):
    def __init__(self, image_id: str) -> None:
        super().__init__(f"image not found: {image_id}")
        self.image_id = image_id


class ImageReferenceService:
    """Reference-first image access; raw bytes are opt-in."""

    def __init__(self, settings: Settings) -> None:
        self._store = ImageStore(settings)

    def get_ref(self, image_id: str) -> ImageRef | None:
        record = self._store.get(image_id)
        if record is None:
            return None
        return ImageRef.from_record(record)

    def list_refs_for_document(self, document_id: str) -> list[ImageRef]:
        return [
            ImageRef.from_record(record)
            for record in self._store.list_by_document_id(document_id)
        ]

    def get_content(self, image_id: str) -> ImageContent:
        record = self._store.get(image_id)
        if record is None:
            raise ImageNotFoundError(image_id)
        path = Path(record.file_path)
        if not path.is_file():
            raise ImageNotFoundError(image_id)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return ImageContent(ref=ImageRef.from_record(record), content_base64=encoded)


__all__ = [
    "ImageContent",
    "ImageNotFoundError",
    "ImageRef",
    "ImageReferenceService",
]
