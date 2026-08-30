from __future__ import annotations

import time
from typing import TYPE_CHECKING

from wenmai.models import Chunk
from wenmai.storage.fingerprints import FingerprintRecord

if TYPE_CHECKING:
    from wenmai.components.bm25.index import Bm25Index
    from wenmai.components.embedding.base import BaseEmbedding
    from wenmai.components.vector_store.base import BaseVectorStore
    from wenmai.knowledge.store import IngestStatus, PrepareResult, UpsertResult
    from wenmai.storage.catalog import DocumentCatalog
    from wenmai.storage.document_images import DocumentImages
    from wenmai.storage.fingerprints import FingerprintStore


class WritePath:
    """入库 write seam: plan/commit/delete with compensation."""

    def __init__(
        self,
        *,
        embedder: BaseEmbedding,
        store: BaseVectorStore,
        bm25: Bm25Index,
        images: DocumentImages,
        fingerprints: FingerprintStore,
        catalog: DocumentCatalog,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._bm25 = bm25
        self._images = images
        self._fingerprints = fingerprints
        self._catalog = catalog

    def plan_document(
        self,
        *,
        source_path: str,
        sha256: str,
        document_id: str,
    ) -> PrepareResult:
        from wenmai.knowledge.store import PrepareResult

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
        if status == "skipped":
            raise ValueError("cannot commit a skipped ingest")
        previous_fingerprint = self._fingerprints.get_by_source_path(source_path)
        result = self._upsert_chunks(chunks)
        try:
            self._fingerprints.upsert(
                source_path=source_path,
                sha256=sha256,
                document_id=document_id,
                status=status,
            )
            if previous_document_id and previous_document_id != document_id:
                self.delete_document(previous_document_id)
            return result
        except Exception:
            self._compensate_document_write(document_id)
            self._restore_fingerprint(previous_fingerprint, source_path)
            raise

    def delete_document(self, document_id: str) -> None:
        backup = self._store.get_by_document_id(document_id)
        self._store.delete_by_document_id(document_id)
        try:
            self._bm25.delete_by_document_id(document_id)
            self._bm25.save()
        except Exception:
            if backup:
                self._store.upsert(backup)
            raise
        self._images.delete_for_document(document_id)
        self._fingerprints.delete_by_document_id(document_id)
        self._catalog.remove_document(document_id)

    def _upsert_chunks(self, chunks: list[Chunk]) -> UpsertResult:
        from wenmai.knowledge.store import UpsertResult

        if not chunks:
            raise ValueError("cannot upsert empty chunk list")

        document_id = chunks[0].document_id
        started = time.perf_counter()
        vectors = self._embedder.embed_documents([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.embedding = vector
        embed_elapsed_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        self._store.upsert(chunks)
        try:
            self._bm25.upsert(chunks)
            self._bm25.save()
        except Exception:
            self._store.delete_by_document_id(document_id)
            self._bm25.delete_by_document_id(document_id)
            try:
                self._bm25.save()
            except Exception:
                pass
            raise
        upsert_elapsed_ms = (time.perf_counter() - started) * 1000

        self._catalog.upsert_document(chunks)

        return UpsertResult(
            chunk_count=len(chunks),
            embed_provider=self._embedder.provider_name,
            embed_dimension=self._embedder.dimension,
            embed_elapsed_ms=embed_elapsed_ms,
            upsert_provider=self._store.provider_name,
            upsert_elapsed_ms=upsert_elapsed_ms,
        )

    def _compensate_document_write(self, document_id: str) -> None:
        self._store.delete_by_document_id(document_id)
        self._bm25.delete_by_document_id(document_id)
        try:
            self._bm25.save()
        except Exception:
            pass
        self._images.delete_for_document(document_id)
        self._catalog.remove_document(document_id)

    def _restore_fingerprint(
        self,
        previous: FingerprintRecord | None,
        source_path: str,
    ) -> None:
        if previous is not None:
            self._fingerprints.upsert(
                source_path=previous.source_path,
                sha256=previous.sha256,
                document_id=previous.document_id,
                status=previous.status,
            )
            return
        self._fingerprints.delete_by_source_path(source_path)
