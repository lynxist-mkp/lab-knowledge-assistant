from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wenmai.knowledge.browse import ChunkDetail, CultureDomainGroup, chunk_detail_from_chunk
from wenmai.knowledge.document_card import DocumentCard, DocumentNotFoundError
from wenmai.knowledge.domain import is_searchable
from wenmai.models import Chunk, ScoredChunk

if TYPE_CHECKING:
    from wenmai.knowledge.review import PendingReviewDocument
    from wenmai.storage.catalog import DocumentCatalog


class ReadPath:
    """知识库 read seam: 检索、目录浏览与审阅状态读取。"""

    def __init__(
        self,
        *,
        embedder: object | None,
        store: object | None,
        bm25: object | None,
        catalog: DocumentCatalog,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._bm25 = bm25
        self._catalog = catalog

    @classmethod
    def catalog_only(cls, catalog: DocumentCatalog) -> ReadPath:
        return cls(embedder=None, store=None, bm25=None, catalog=catalog)

    def dense_search(
        self,
        query: str,
        *,
        top_k: int,
        culture_domain: str | None = None,
    ) -> list[ScoredChunk]:
        embedder = self._require_embedder()
        store = self._require_store()
        vector = embedder.embed_query(query)
        fetch_k = max(top_k * 3, top_k)
        hits = store.query(
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
        bm25 = self._require_bm25()
        store = self._require_store()
        fetch_k = max(top_k * 3, top_k)
        hits = bm25.search(
            query,
            top_k=fetch_k,
            culture_domain=culture_domain,
        )
        chunk_ids = [hit.chunk_id for hit in hits]
        chunks = store.get_by_ids(chunk_ids)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        scored = [
            ScoredChunk(chunk=chunk_by_id[hit.chunk_id], score=hit.score)
            for hit in hits
            if hit.chunk_id in chunk_by_id
        ]
        return _take_searchable(scored, top_k)

    def get_by_document_id(self, document_id: str) -> list[Chunk]:
        return self._require_store().get_by_document_id(document_id)

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        return self._require_store().get_by_chunk_id(chunk_id)

    def list_all(self) -> list[Chunk]:
        return self._require_store().list_all()

    def list_pending_review_documents(self) -> list[PendingReviewDocument]:
        from wenmai.knowledge.review import list_pending_from_catalog

        return list_pending_from_catalog(self._catalog)

    @property
    def document_count(self) -> int:
        return self._catalog.document_count

    @property
    def chunk_count(self) -> int:
        return self._catalog.chunk_count

    def browse_by_culture_domain(self) -> list[CultureDomainGroup]:
        return self._catalog.browse_groups()

    def chunk_detail(self, chunk_id: str) -> ChunkDetail | None:
        chunk = self.get_by_chunk_id(chunk_id)
        if chunk is None:
            return None
        return chunk_detail_from_chunk(chunk)

    def document_card(self, document_id: str) -> DocumentCard:
        entry = self._catalog.get_document(document_id)
        if entry is None:
            raise DocumentNotFoundError(document_id)
        chunks = self.get_by_document_id(document_id)
        return DocumentCard(
            document_id=document_id,
            title=entry.title,
            culture_domain=entry.culture_domain,
            summary=_document_summary_from_chunks(chunks),
            chunk_count=entry.chunk_count,
        )

    def _require_embedder(self) -> Any:
        if self._embedder is None:
            raise RuntimeError("ReadPath embedder is not configured for retrieval")
        return self._embedder

    def _require_store(self) -> Any:
        if self._store is None:
            raise RuntimeError("ReadPath store is not configured for content reads")
        return self._store

    def _require_bm25(self) -> Any:
        if self._bm25 is None:
            raise RuntimeError("ReadPath bm25 is not configured for sparse retrieval")
        return self._bm25


def _metadata_filter(culture_domain: str | None) -> dict[str, Any] | None:
    if culture_domain is None:
        return None
    return {"culture_domain": culture_domain}


def _take_searchable(hits: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    filtered = [hit for hit in hits if is_searchable(hit.chunk)]
    return filtered[:top_k]


def _document_summary_from_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return ""
    summaries: list[str] = []
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        value = chunk.metadata.get("summary")
        if isinstance(value, str) and value.strip():
            summaries.append(value.strip())
    if not summaries:
        return ""
    return max(summaries, key=len)
