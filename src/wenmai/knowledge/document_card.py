from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class DocumentNotFoundError(Exception):
    def __init__(self, document_id: str) -> None:
        super().__init__(f"document not found: {document_id}")
        self.document_id = document_id


@dataclass(frozen=True)
class DocumentCard:
    document_id: str
    title: str
    culture_domain: str
    summary: str
    chunk_count: int
    tags: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
