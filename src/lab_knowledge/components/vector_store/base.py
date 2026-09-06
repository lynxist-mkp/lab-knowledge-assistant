from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from lab_knowledge.models import Chunk, ScoredChunk


class BaseVectorStore(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def get_by_document_id(self, document_id: str) -> list[Chunk]: ...

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None: ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]: ...

    @abstractmethod
    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]: ...

    @abstractmethod
    def list_all(self) -> list[Chunk]: ...

    @abstractmethod
    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None: ...
