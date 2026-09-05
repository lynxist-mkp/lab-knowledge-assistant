from __future__ import annotations

from abc import ABC, abstractmethod

from wenmai.ingestion.prepare import TransformTraceRecorder
from wenmai.models import Chunk


class BaseTransform(ABC):
    """One step in the ingestion transform registry (refiner / enricher / captioner)."""

    name: str

    @abstractmethod
    def apply(self, chunks: list[Chunk], recorder: TransformTraceRecorder) -> list[Chunk]: ...
