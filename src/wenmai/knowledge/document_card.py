from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from wenmai.knowledge.store import Knowledge
from wenmai.models import Chunk


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _document_summary_from_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return ""
    summaries: list[str] = []
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        value = chunk.metadata.get("summary")
        if isinstance(value, str) and value.strip():
            summaries.append(value.strip())
    if not summaries:
        return ""
    return max(summaries, key=len)


def build_document_card(knowledge: Knowledge, document_id: str) -> DocumentCard:
    entry = knowledge.catalog.get_document(document_id)
    if entry is None:
        raise DocumentNotFoundError(document_id)
    chunks = knowledge.get_by_document_id(document_id)
    return DocumentCard(
        document_id=document_id,
        title=entry.title,
        culture_domain=entry.culture_domain,
        summary=_document_summary_from_chunks(chunks),
        chunk_count=entry.chunk_count,
    )
