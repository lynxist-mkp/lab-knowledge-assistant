from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from wenmai.config import Settings
from wenmai.storage.paths import store_path

_IMAGE_PLACEHOLDER_TEMPLATE = "[IMAGE: {image_id}]"
IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([a-f0-9]+)\s*\]")


def format_image_placeholder(image_id: str) -> str:
    return _IMAGE_PLACEHOLDER_TEMPLATE.format(image_id=image_id)


def find_image_ids(text: str) -> list[str]:
    return IMAGE_PLACEHOLDER_RE.findall(text)


def replace_image_placeholder(text: str, image_id: str, replacement: str) -> str:
    pattern = re.compile(rf"\[IMAGE:\s*{re.escape(image_id)}\s*\]")
    return pattern.sub(replacement, text)


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
        image_id = _image_id(document_id, page, image_bytes)
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
        return _row_to_record(row)

    def list_by_document_id(self, document_id: str) -> list[ImageRecord]:
        with sqlite3.connect(self._index_path) as conn:
            rows = conn.execute(
                """
                SELECT image_id, document_id, source_path, page, file_path, mime_type
                FROM images
                WHERE document_id = ?
                ORDER BY page, image_id
                """,
                (document_id,),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def delete_by_document_id(self, document_id: str) -> None:
        with sqlite3.connect(self._index_path) as conn:
            rows = conn.execute(
                "SELECT file_path FROM images WHERE document_id = ?",
                (document_id,),
            ).fetchall()
            conn.execute("DELETE FROM images WHERE document_id = ?", (document_id,))
            conn.commit()
        for (file_path,) in rows:
            path = Path(file_path)
            if path.is_file():
                path.unlink()


def _row_to_record(row: tuple[object, ...]) -> ImageRecord:
    return ImageRecord(
        image_id=str(row[0]),
        document_id=str(row[1]),
        source_path=str(row[2]),
        page=int(row[3]),
        file_path=str(row[4]),
        mime_type=str(row[5]),
    )


def _extension_for_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/png":
        return ".png"
    return ".bin"


def _image_id(document_id: str, page: int, image_bytes: bytes) -> str:
    # Image refs are document-scoped; identical bytes in different docs must not alias.
    digest = hashlib.sha256()
    digest.update(document_id.encode("utf-8"))
    digest.update(b":")
    digest.update(str(page).encode("ascii"))
    digest.update(b":")
    digest.update(image_bytes)
    return digest.hexdigest()
