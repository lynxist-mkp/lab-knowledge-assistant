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
class IngestResult:
    document_id: str
    chunk_count: int
    elapsed_ms: float
    trace_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "elapsed_ms": self.elapsed_ms,
            "trace_id": self.trace_id,
        }
