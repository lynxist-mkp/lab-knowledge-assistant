from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.knowledge.browse import ChunkSummary, CultureDomainGroup, DocumentSummary
from wenmai.knowledge.domain import (
    REVIEW_PENDING,
    authors,
    chunk_title,
    culture_domain,
    preview,
    publication_year,
    review_status,
    source_kind,
    source_label,
    summary,
    tags,
    title,
)
from wenmai.models import Chunk
from wenmai.storage.paths import collection_storage_bindings


@dataclass(frozen=True)
class CatalogDocument:
    document_id: str
    culture_domain: str
    title: str
    chunk_count: int
    summary: str
    tags: list[str]
    source_kind: str
    source_label: str
    authors: str
    publication_year: int | None
    chunks: list[ChunkSummary]


class DocumentCatalog:
    """文档目录 sidecar: per-culture-domain document and chunk counts for browse/overview."""

    @classmethod
    def from_settings(cls, settings: Settings) -> DocumentCatalog:
        bindings = collection_storage_bindings(settings)
        return cls(bindings.catalog_read_path(), write_path=bindings.catalog_path)

    def __init__(self, path: Path, *, write_path: Path | None = None) -> None:
        self._path = path
        self._write_path = write_path or path
        self._documents: dict[str, dict[str, Any]] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        documents = raw.get("documents")
        self._documents = dict(documents) if isinstance(documents, dict) else {}

    def _save(self) -> None:
        payload = {"documents": self._documents}
        self._write_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._write_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self._write_path)

    def upsert_document(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot catalog empty chunk list")
        document_id = chunks[0].document_id
        sorted_chunks = sorted(chunks, key=lambda item: item.chunk_id)
        document_tags = _document_tags(sorted_chunks)
        self._documents[document_id] = {
            "culture_domain": culture_domain(sorted_chunks[0]),
            "title": title(sorted_chunks[0]),
            "summary": _document_summary(sorted_chunks),
            "tags": document_tags,
            "source_kind": source_kind(sorted_chunks[0]),
            "source_label": source_label(sorted_chunks[0]),
            "authors": authors(sorted_chunks[0]),
            "publication_year": publication_year(sorted_chunks[0]),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "preview": preview(chunk.text),
                    "审阅状态": review_status(chunk),
                    "chunk_title": chunk_title(chunk),
                    "summary": summary(chunk),
                    "tags": tags(chunk),
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

    def _parse_document(self, document_id: str, raw: dict[str, Any]) -> CatalogDocument:
        culture_domain = str(raw.get("culture_domain") or "其他")
        chunks_raw = raw.get("chunks") or []
        chunks = [
            ChunkSummary(
                chunk_id=str(item["chunk_id"]),
                document_id=str(item.get("document_id") or document_id),
                preview=str(item.get("preview") or ""),
                review_status=str(item.get("审阅状态") or "已通过"),
                chunk_title=str(item.get("chunk_title") or ""),
                summary=str(item.get("summary") or ""),
                tags=_coerce_tags(item.get("tags")),
            )
            for item in chunks_raw
            if isinstance(item, dict) and item.get("chunk_id")
        ]
        return CatalogDocument(
            document_id=document_id,
            culture_domain=culture_domain,
            title=str(raw.get("title") or document_id),
            chunk_count=len(chunks),
            summary=str(raw.get("summary") or ""),
            tags=_coerce_tags(raw.get("tags")),
            source_kind=str(raw.get("source_kind") or "group_doc"),
            source_label=str(raw.get("source_label") or "组内资料"),
            authors=str(raw.get("authors") or ""),
            publication_year=_coerce_publication_year(raw.get("publication_year")),
            chunks=chunks,
        )

    def get_document(self, document_id: str) -> CatalogDocument | None:
        raw = self._documents.get(document_id)
        if raw is None:
            return None
        return self._parse_document(document_id, raw)

    def browse_groups(self) -> list[CultureDomainGroup]:
        by_domain: dict[str, list[CatalogDocument]] = {}
        for document_id in sorted(self._documents):
            entry = self._parse_document(document_id, self._documents[document_id])
            by_domain.setdefault(entry.culture_domain, []).append(entry)

        groups: list[CultureDomainGroup] = []
        for domain in sorted(by_domain):
            documents = by_domain[domain]
            groups.append(
                CultureDomainGroup(
                    culture_domain=domain,
                    document_count=len(documents),
                    chunk_count=sum(doc.chunk_count for doc in documents),
                    documents=[
                        DocumentSummary(
                            document_id=doc.document_id,
                            title=doc.title,
                            chunk_count=doc.chunk_count,
                            summary=doc.summary,
                            tags=doc.tags,
                            chunks=doc.chunks,
                            source_kind=doc.source_kind,
                            source_label=doc.source_label,
                            authors=doc.authors,
                            publication_year=doc.publication_year,
                        )
                        for doc in documents
                    ],
                )
            )
        return groups

    def list_pending_documents(self) -> list[CatalogDocument]:
        """Return documents with at least one 待审 chunk (catalog-backed, no list_all)."""
        pending: list[CatalogDocument] = []
        for document_id in sorted(self._documents):
            entry = self._parse_document(document_id, self._documents[document_id])
            if any(chunk.review_status == REVIEW_PENDING for chunk in entry.chunks):
                pending.append(entry)
        return pending


__all__ = ["CatalogDocument", "DocumentCatalog"]


def _document_summary(chunks: list[Chunk]) -> str:
    summaries = [summary(chunk) for chunk in chunks if summary(chunk)]
    if not summaries:
        return ""
    return max(summaries, key=len)


def _document_tags(chunks: list[Chunk]) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        for tag in tags(chunk):
            if tag not in merged:
                merged.append(tag)
    return merged


def _coerce_tags(raw: object) -> list[str]:
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes, dict)):
        return []
    cleaned: list[str] = []
    for item in raw:
        tag = str(item).strip()
        if tag and tag not in cleaned:
            cleaned.append(tag)
    return cleaned


def _coerce_publication_year(raw: object) -> int | None:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None
