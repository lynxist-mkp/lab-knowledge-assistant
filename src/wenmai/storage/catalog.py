from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.knowledge.browse import ChunkSummary, CultureDomainGroup, DocumentSummary
from wenmai.knowledge.domain import culture_domain, preview, review_status, title
from wenmai.models import Chunk
from wenmai.storage.paths import store_path


@dataclass(frozen=True)
class CatalogDocument:
    document_id: str
    culture_domain: str
    title: str
    chunk_count: int
    chunks: list[ChunkSummary]


class DocumentCatalog:
    """文档目录 sidecar: per-culture-domain document and chunk counts for browse/overview."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._documents: dict[str, dict[str, Any]] = {}
        if path.exists():
            self._load()

    @classmethod
    def from_settings(cls, settings: Settings) -> DocumentCatalog:
        return cls(store_path(settings, "catalog"))

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        documents = raw.get("documents")
        self._documents = dict(documents) if isinstance(documents, dict) else {}

    def _save(self) -> None:
        payload = {"documents": self._documents}
        temp_path = self._path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self._path)

    def upsert_document(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot catalog empty chunk list")
        document_id = chunks[0].document_id
        sorted_chunks = sorted(chunks, key=lambda item: item.chunk_id)
        self._documents[document_id] = {
            "culture_domain": culture_domain(sorted_chunks[0]),
            "title": title(sorted_chunks[0]),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "preview": preview(chunk.text),
                    "审阅状态": review_status(chunk),
                }
                for chunk in sorted_chunks
            ],
        }
        self._save()

    def remove_document(self, document_id: str) -> None:
        if document_id not in self._documents:
            return
        del self._documents[document_id]
        self._save()

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def chunk_count(self) -> int:
        return sum(len(doc.get("chunks") or []) for doc in self._documents.values())

    def browse_groups(self) -> list[CultureDomainGroup]:
        by_domain: dict[str, list[CatalogDocument]] = {}
        for document_id in sorted(self._documents):
            raw = self._documents[document_id]
            culture_domain = str(raw.get("culture_domain") or "其他")
            chunks_raw = raw.get("chunks") or []
            chunks = [
                ChunkSummary(
                    chunk_id=str(item["chunk_id"]),
                    document_id=str(item.get("document_id") or document_id),
                    preview=str(item.get("preview") or ""),
                    review_status=str(item.get("审阅状态") or "已通过"),
                )
                for item in chunks_raw
                if isinstance(item, dict) and item.get("chunk_id")
            ]
            entry = CatalogDocument(
                document_id=document_id,
                culture_domain=culture_domain,
                title=str(raw.get("title") or document_id),
                chunk_count=len(chunks),
                chunks=chunks,
            )
            by_domain.setdefault(culture_domain, []).append(entry)

        groups: list[CultureDomainGroup] = []
        for culture_domain in sorted(by_domain):
            documents = by_domain[culture_domain]
            groups.append(
                CultureDomainGroup(
                    culture_domain=culture_domain,
                    document_count=len(documents),
                    chunk_count=sum(doc.chunk_count for doc in documents),
                    documents=[
                        DocumentSummary(
                            document_id=doc.document_id,
                            title=doc.title,
                            chunk_count=doc.chunk_count,
                            chunks=doc.chunks,
                        )
                        for doc in documents
                    ],
                )
            )
        return groups
