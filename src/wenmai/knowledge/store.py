from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from wenmai.config import Settings
from wenmai.factories import bm25 as bm25_factory
from wenmai.factories import embedding as embedding_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.factories.loader import ensure_providers
from wenmai.knowledge.browse import CultureDomainGroup, chunk_detail_from_chunk
from wenmai.knowledge.write import WritePath
from wenmai.models import Chunk, ScoredChunk
from wenmai.storage.catalog import DocumentCatalog
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
        self._catalog = DocumentCatalog.from_settings(settings)
        self._write = WritePath(
            embedder=self._embedder,
            store=self._store,
            bm25=self._bm25,
            images=self._images,
            fingerprints=self._fingerprints,
            catalog=self._catalog,
        )

    @property
    def images(self) -> DocumentImages:
        return self._images

    @property
    def dense_provider(self) -> str:
        return self._store.provider_name

    @property
    def catalog(self) -> DocumentCatalog:
        return self._catalog

    def plan_document(
        self,
        *,
        source_path: str,
        sha256: str,
        document_id: str,
    ) -> PrepareResult:
        """Decide skip/rebuild/new without mutating indexes or fingerprints."""
        return self._write.plan_document(
            source_path=source_path,
            sha256=sha256,
            document_id=document_id,
        )

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
        return self._write.commit_document(
            source_path=source_path,
            sha256=sha256,
            document_id=document_id,
            status=status,
            chunks=chunks,
            previous_document_id=previous_document_id,
        )

    def delete_document(self, document_id: str) -> None:
        self._write.delete_document(document_id)

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

    def browse_by_culture_domain(self) -> list[CultureDomainGroup]:
        return self._catalog.browse_groups()

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
