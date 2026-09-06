from __future__ import annotations

import json
from collections.abc import Iterable

from wenmai.ingestion.source_metadata import SOURCE_KIND_GROUP, SOURCE_KIND_LABELS
from wenmai.models import Chunk

REVIEW_STATUS_FIELD = "审阅状态"
REVIEW_PENDING = "待审"
REVIEW_APPROVED = "已通过"
_PREVIEW_CHARS = 120
_UNKNOWN_DOMAIN = "其他"


def culture_domain(chunk: Chunk) -> str:
    value = chunk.metadata.get("culture_domain")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _UNKNOWN_DOMAIN


def title(chunk: Chunk) -> str:
    value = chunk.metadata.get("title")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return chunk.document_id


def chunk_title(chunk: Chunk) -> str:
    value = chunk.metadata.get("chunk_title")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return title(chunk)


def summary(chunk: Chunk) -> str:
    value = chunk.metadata.get("summary")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def tags(chunk: Chunk) -> list[str]:
    value = chunk.metadata.get("tags")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = [stripped]
    if not isinstance(value, Iterable) or isinstance(value, (bytes, dict)):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def source_kind(chunk: Chunk) -> str:
    value = chunk.metadata.get("source_kind")
    if value in SOURCE_KIND_LABELS:
        return str(value)
    return SOURCE_KIND_GROUP


def source_label(chunk: Chunk) -> str:
    value = chunk.metadata.get("source_label")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return SOURCE_KIND_LABELS[SOURCE_KIND_GROUP]


def authors(chunk: Chunk) -> str:
    value = chunk.metadata.get("authors")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def publication_year(chunk: Chunk) -> int | None:
    value = chunk.metadata.get("publication_year")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def preview(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _PREVIEW_CHARS:
        return collapsed
    return collapsed[: _PREVIEW_CHARS - 1] + "…"


def review_status(chunk: Chunk) -> str:
    value = chunk.metadata.get(REVIEW_STATUS_FIELD)
    if value == REVIEW_PENDING:
        return REVIEW_PENDING
    return REVIEW_APPROVED


def is_searchable(chunk: Chunk) -> bool:
    return review_status(chunk) == REVIEW_APPROVED


def stamp_review_status(metadata: dict[str, object]) -> dict[str, object]:
    if REVIEW_STATUS_FIELD not in metadata:
        metadata[REVIEW_STATUS_FIELD] = REVIEW_APPROVED
    return metadata
