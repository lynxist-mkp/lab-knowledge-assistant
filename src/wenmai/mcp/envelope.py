from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from wenmai.config import Settings
from wenmai.knowledge.collections import resolve_collection_id
from wenmai.models import AskResult, Citation
from wenmai.storage.images import find_image_ids


@dataclass
class McpScope:
    collection_id: str
    culture_domain: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "collection_id": self.collection_id,
            "culture_domain": self.culture_domain,
        }


@dataclass
class McpRefs:
    document_ids: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "document_ids": self.document_ids,
            "chunk_ids": self.chunk_ids,
            "image_ids": self.image_ids,
        }


@dataclass
class McpMeta:
    count: int | None = None
    elapsed_ms: float | None = None
    mode: str | None = None

    def as_dict(self) -> dict[str, int | float | str | None]:
        return {
            "count": self.count,
            "elapsed_ms": self.elapsed_ms,
            "mode": self.mode,
        }


@dataclass
class McpEnvelope:
    data: Any
    scope: McpScope
    refs: McpRefs = field(default_factory=McpRefs)
    meta: McpMeta = field(default_factory=McpMeta)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "scope": self.scope.as_dict(),
            "refs": self.refs.as_dict(),
            "meta": self.meta.as_dict(),
            "warnings": self.warnings,
        }


def scope_for(
    settings: Settings,
    *,
    collection_id: str | None = None,
    culture_domain: str | None = None,
) -> McpScope:
    return McpScope(
        collection_id=resolve_collection_id(settings, collection_id),
        culture_domain=culture_domain,
    )


def refs_from_citations(citations: list[Citation]) -> McpRefs:
    document_ids: list[str] = []
    chunk_ids: list[str] = []
    image_ids: list[str] = []
    seen_docs: set[str] = set()
    seen_chunks: set[str] = set()
    seen_images: set[str] = set()

    for citation in citations:
        if citation.document_id not in seen_docs:
            document_ids.append(citation.document_id)
            seen_docs.add(citation.document_id)
        if citation.chunk_id not in seen_chunks:
            chunk_ids.append(citation.chunk_id)
            seen_chunks.add(citation.chunk_id)
        for image_id in find_image_ids(citation.excerpt):
            if image_id not in seen_images:
                image_ids.append(image_id)
                seen_images.add(image_id)

    return McpRefs(document_ids=document_ids, chunk_ids=chunk_ids, image_ids=image_ids)


def refs_from_ask_result(result: AskResult) -> McpRefs:
    refs = refs_from_citations(result.citations)
    cited_chunk_ids = {citation.chunk_id for citation in result.citations}
    if not cited_chunk_ids:
        cited_chunk_ids = {item.chunk.chunk_id for item in result.ranked_chunks}
    seen_images = set(refs.image_ids)
    for item in result.ranked_chunks:
        chunk = item.chunk
        if chunk.chunk_id not in cited_chunk_ids:
            continue
        raw_image_ids = chunk.metadata.get("image_ids")
        image_ids = raw_image_ids
        if isinstance(raw_image_ids, str):
            try:
                decoded = json.loads(raw_image_ids)
            except json.JSONDecodeError:
                decoded = None
            image_ids = decoded
        if not isinstance(image_ids, list):
            continue
        for image_id in image_ids:
            if not isinstance(image_id, str) or image_id in seen_images:
                continue
            refs.image_ids.append(image_id)
            seen_images.add(image_id)
    return refs


def envelope(
    *,
    data: Any,
    scope: McpScope,
    refs: McpRefs | None = None,
    meta: McpMeta | None = None,
    warnings: list[str] | None = None,
) -> McpEnvelope:
    return McpEnvelope(
        data=data,
        scope=scope,
        refs=refs or McpRefs(),
        meta=meta or McpMeta(),
        warnings=warnings or [],
    )


__all__ = [
    "McpEnvelope",
    "McpMeta",
    "McpRefs",
    "McpScope",
    "envelope",
    "refs_from_ask_result",
    "refs_from_citations",
    "scope_for",
]
