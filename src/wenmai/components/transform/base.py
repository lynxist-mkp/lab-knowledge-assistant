from __future__ import annotations

from abc import ABC, abstractmethod

from wenmai.models import Chunk
from wenmai.tracing.context import TraceContext


class BaseTransform(ABC):
    """One step in the ingestion transform registry (refiner / enricher / captioner)."""

    name: str

    @abstractmethod
    def apply(self, chunks: list[Chunk], trace: TraceContext) -> list[Chunk]: ...
