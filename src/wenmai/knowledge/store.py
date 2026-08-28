from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from wenmai.config import Settings
from wenmai.factories import bm25 as bm25_factory
from wenmai.factories import embedding as embedding_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.factories.loader import ensure_providers
from wenmai.knowledge.browse import (
    CultureDomainGroup,
    OverviewStats,
    browse_groups_from_chunks,
    chunk_detail_from_chunk,
    overview_from_chunks,
)
from wenmai.models import Chunk, ScoredChunk
from wenmai.storage.document_images import DocumentImages
from wenmai.storage.fingerprints import FingerprintStore

IngestStatus = Literal["ingested", "skipped", "rebuilt"]


@dataclass(frozen=True)
class UpsertResult:
    chunk_count: int
    embed_provider: str
    embed_dimension: int
    embed_elapsed_ms: float
    upsert_provider: str
    upsert_elapsed_ms: float


@dataclass(frozen=True)
class PrepareResult:
    """Read-only 入库幂等 plan: skip, rebuild, or new. Mutations happen in commit."""

    status: IngestStatus
    previous_document_id: str | None = None


class Knowledge:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedder = embedding_factory.create(settings)
        self._store = vector_store_factory.create(settings)
        self._bm25 = bm25_factory.create(settings)
        self._images = DocumentImages(settings)
        self._fingerprints = FingerprintStore.from_settings(settings)

    @property
    def images(self) -> DocumentImages:
        return self._images

    @property
    def dense_provider(self) -> str:
        return self._store.provider_name

    def plan_document(
        self,
        *,
        source_path: str,
        sha256: str,
        document_id: str,
    ) -> PrepareResult:
        """Decide skip/rebuild/new without mutating indexes or fingerprints."""
        _ = document_id
        previous = self._fingerprints.get_by_source_path(source_path)
        if previous and previous.sha256 == sha256:
            return PrepareResult(status="skipped", previous_document_id=previous.document_id)
        if previous:
            return PrepareResult(status="rebuilt", previous_document_id=previous.document_id)
        return PrepareResult(status="ingested", previous_document_id=None)

    def commit_document(
        self,
        *,
        source_path: str,
        sha256: str,
        document_id: str,
        status: IngestStatus,
        chunks: list[Chunk],
        previous_document_id: str | None = None,
    ) -> UpsertResult:
        """Atomically replace previous doc (if any), upsert chunks, record fingerprint."""
        if status == "skipped":
            raise ValueError("cannot commit a skipped ingest")
        if previous_document_id and previous_document_id != document_id:
            self.delete_document(previous_document_id)
        result = self.upsert(chunks)
        self._fingerprints.upsert(
            source_path=source_path,
            sha256=sha256,
            document_id=document_id,
            status=status,
        )
        return result

    def upsert(self, chunks: list[Chunk]) -> UpsertResult:
        started = time.perf_counter()
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        embed_elapsed_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        self._store.upsert(chunks)
        self._bm25.upsert(chunks)
        self._bm25.save()
        upsert_elapsed_ms = (time.perf_counter() - started) * 1000

        return UpsertResult(
            chunk_count=len(chunks),
            embed_provider=self._embedder.provider_name,
            embed_dimension=self._embedder.dimension,
            embed_elapsed_ms=embed_elapsed_ms,
            upsert_provider=self._store.provider_name,
            upsert_elapsed_ms=upsert_elapsed_ms,
        )

    def delete_document(self, document_id: str) -> None:
        self._store.delete_by_document_id(document_id)
        self._bm25.delete_by_document_id(document_id)
        self._bm25.save()
        self._images.delete_for_document(document_id)
        self._fingerprints.delete_by_document_id(document_id)

    def dense_search(
        self,
        query: str,
        *,
        top_k: int,
        culture_domain: str | None = None,
    ) -> list[ScoredChunk]:
        vector = self._embedder.embed_query(query)
        return self._store.query(
            vector,
            top_k=top_k,
            where=_metadata_filter(culture_domain),
        )

    def sparse_search(
        self,
        query: str,
        *,
        top_k: int,
        culture_domain: str | None = None,
    ) -> list[ScoredChunk]:
        hits = self._bm25.search(
            query,
            top_k=top_k,
            culture_domain=culture_domain,
        )
        chunk_ids = [hit.chunk_id for hit in hits]
        chunks = self._store.get_by_ids(chunk_ids)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        return [
            ScoredChunk(chunk=chunk_by_id[hit.chunk_id], score=hit.score)
            for hit in hits
            if hit.chunk_id in chunk_by_id
        ]

    def get_by_document_id(self, document_id: str) -> list[Chunk]:
        return self._store.get_by_document_id(document_id)

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        return self._store.get_by_chunk_id(chunk_id)

    def list_all(self) -> list[Chunk]:
        return self._store.list_all()

    def overview(self) -> OverviewStats:
        return overview_from_chunks(self.list_all(), self._settings)

    def browse_by_culture_domain(self) -> list[CultureDomainGroup]:
        return browse_groups_from_chunks(self.list_all())

    def chunk_detail(self, chunk_id: str) -> dict[str, Any] | None:
        chunk = self.get_by_chunk_id(chunk_id)
        if chunk is None:
            return None
        return chunk_detail_from_chunk(chunk)


def create_knowledge(settings: Settings) -> Knowledge:
    ensure_providers()
    return Knowledge(settings)


def _metadata_filter(culture_domain: str | None) -> dict[str, Any] | None:
    if culture_domain is None:
        return None
    return {"culture_domain": culture_domain}
