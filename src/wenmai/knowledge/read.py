from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wenmai.knowledge.domain import is_searchable
from wenmai.models import Chunk, ScoredChunk

if TYPE_CHECKING:
    from wenmai.knowledge.review import PendingReviewDocument
    from wenmai.storage.catalog import DocumentCatalog


class ReadPath:
    """知识库 read seam: 检索与审阅状态读取。"""

    def __init__(
        self,
        *,
        embedder: object,
        store: object,
        bm25: object,
        catalog: DocumentCatalog,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._bm25 = bm25
        self._catalog = catalog

    def dense_search(
        self,
        query: str,
        *,
        top_k: int,
        culture_domain: str | None = None,
    ) -> list[ScoredChunk]:
        vector = self._embedder.embed_query(query)
        fetch_k = max(top_k * 3, top_k)
        hits = self._store.query(
            vector,
            top_k=fetch_k,
            where=_metadata_filter(culture_domain),
        )
        return _take_searchable(hits, top_k)

    def sparse_search(
        self,
        query: str,
        *,
        top_k: int,
        culture_domain: str | None = None,
    ) -> list[ScoredChunk]:
        fetch_k = max(top_k * 3, top_k)
        hits = self._bm25.search(
            query,
            top_k=fetch_k,
            culture_domain=culture_domain,
        )
        chunk_ids = [hit.chunk_id for hit in hits]
        chunks = self._store.get_by_ids(chunk_ids)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        scored = [
            ScoredChunk(chunk=chunk_by_id[hit.chunk_id], score=hit.score)
            for hit in hits
            if hit.chunk_id in chunk_by_id
        ]
        return _take_searchable(scored, top_k)

    def get_by_document_id(self, document_id: str) -> list[Chunk]:
        return self._store.get_by_document_id(document_id)

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        return self._store.get_by_chunk_id(chunk_id)

    def list_all(self) -> list[Chunk]:
        return self._store.list_all()

    def list_pending_review_documents(self) -> list[PendingReviewDocument]:
        from wenmai.knowledge.review import list_pending_from_catalog

        return list_pending_from_catalog(self._catalog)


def _metadata_filter(culture_domain: str | None) -> dict[str, Any] | None:
    if culture_domain is None:
        return None
    return {"culture_domain": culture_domain}


def _take_searchable(hits: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    filtered = [hit for hit in hits if is_searchable(hit.chunk)]
    return filtered[:top_k]
