from __future__ import annotations

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
