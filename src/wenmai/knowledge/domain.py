from __future__ import annotations

from wenmai.models import Chunk

_PREVIEW_CHARS = 120
_UNKNOWN_DOMAIN = "其他"
_REVIEW_STATUS_KEY = "审阅状态"
_REVIEW_APPROVED = "已通过"
_REVIEW_PENDING = "待审"


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
    value = chunk.metadata.get(_REVIEW_STATUS_KEY)
    if value == _REVIEW_PENDING:
        return _REVIEW_PENDING
    return _REVIEW_APPROVED


def is_searchable(chunk: Chunk) -> bool:
    return review_status(chunk) == _REVIEW_APPROVED


def stamp_review_status(metadata: dict[str, object]) -> dict[str, object]:
    if _REVIEW_STATUS_KEY not in metadata:
        metadata[_REVIEW_STATUS_KEY] = _REVIEW_APPROVED
    return metadata
