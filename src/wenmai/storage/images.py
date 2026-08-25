from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass

from wenmai.config import Settings
from wenmai.storage.paths import store_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    page INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    mime_type TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    document_id: str
    source_path: str
    page: int
    file_path: str
    mime_type: str


class ImageStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._collection = settings.product.collection
        self._images_root = store_path(settings, "images") / self._collection
        self._images_root.mkdir(parents=True, exist_ok=True)
        self._index_path = store_path(settings, "image_index")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._index_path) as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def save(
        self,
        *,
        document_id: str,
        source_path: str,
        page: int,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> str:
        image_id = hashlib.sha256(image_bytes).hexdigest()
        extension = _extension_for_mime(mime_type)
        file_path = self._images_root / f"{image_id}{extension}"
        if not file_path.exists():
            file_path.write_bytes(image_bytes)

        with sqlite3.connect(self._index_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO images
                    (image_id, document_id, source_path, page, file_path, mime_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (image_id, document_id, source_path, page, str(file_path), mime_type),
            )
            conn.commit()
        return image_id

    def get(self, image_id: str) -> ImageRecord | None:
        with sqlite3.connect(self._index_path) as conn:
            row = conn.execute(
                """
                SELECT image_id, document_id, source_path, page, file_path, mime_type
                FROM images
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchone()
        if row is None:
            return None
        return ImageRecord(
            image_id=row[0],
            document_id=row[1],
            source_path=row[2],
            page=row[3],
            file_path=row[4],
            mime_type=row[5],
        )


def _extension_for_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"
    return ".bin"
