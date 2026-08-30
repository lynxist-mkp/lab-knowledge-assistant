from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    index: int
    chunk_id: str
    document_id: str
    title: str
    excerpt: str
    url: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "excerpt": self.excerpt,
        }
        if self.url:
            payload["url"] = self.url
        return payload


@dataclass
class AskResult:
    answer: str
    citations: list[Citation]
    trace_id: str
    refused: bool = False
    refusal_reason: str | None = None
    error: str | None = None
    ranked_chunks: list[ScoredChunk] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "answer": self.answer,
            "citations": [citation.as_dict() for citation in self.citations],
            "trace_id": self.trace_id,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class IngestResult:
    document_id: str
    chunk_count: int
    elapsed_ms: float
    trace_id: str
    status: str = "ingested"

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "elapsed_ms": self.elapsed_ms,
            "trace_id": self.trace_id,
            "status": self.status,
        }
