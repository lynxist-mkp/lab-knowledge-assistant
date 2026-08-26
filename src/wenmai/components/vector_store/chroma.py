from __future__ import annotations

import json
from typing import Any

import chromadb

from wenmai.components.fakes import apply_behavior
from wenmai.components.vector_store.base import BaseVectorStore
from wenmai.factories.vector_store import registry
from wenmai.models import Chunk, ScoredChunk


def _flatten(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    flat: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat


@registry.register("chroma")
class ChromaVectorStore(BaseVectorStore):
    def __init__(self, persist_path: str, collection: str, **kwargs: object) -> None:
        self.persist_path = persist_path
        self.collection_name = collection
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def provider_name(self) -> str:
        return "chroma"

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = [chunk.embedding for chunk in chunks]
        if any(item is None for item in embeddings):
            raise ValueError("every chunk must have an embedding before upsert")
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=[_flatten(chunk.metadata) for chunk in chunks],
        )

    def get_by_document_id(self, document_id: str) -> list[Chunk]:
        result = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        chunks: list[Chunk] = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True):
            meta = dict(metadata or {})
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id") or document_id),
                    text=text or "",
                    metadata=meta,
                )
            )
        chunks.sort(key=lambda chunk: chunk.chunk_id)
        return chunks

    def delete_by_document_id(self, document_id: str) -> None:
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)

    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = self._collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        chunks: list[Chunk] = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True):
            meta = dict(metadata or {})
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id") or ""),
                    text=text or "",
                    metadata=meta,
                )
            )
        order = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
        chunks.sort(key=lambda chunk: order.get(chunk.chunk_id, len(order)))
        return chunks

    def query(self, query_embedding: list[float], top_k: int) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        scored: list[ScoredChunk] = []
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            meta = dict(metadata or {})
            scored.append(
                ScoredChunk(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        document_id=str(meta.get("document_id") or ""),
                        text=text or "",
                        metadata=meta,
                    ),
                    score=1.0 - float(distance),
                )
            )
        return scored

    def list_all(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        chunks: list[Chunk] = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True):
            meta = dict(metadata or {})
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=str(meta.get("document_id") or ""),
                    text=text or "",
                    metadata=meta,
                )
            )
        chunks.sort(key=lambda chunk: chunk.chunk_id)
        return chunks

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = result.get("ids") or []
        if not ids:
            return None
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        text = documents[0] if documents else ""
        meta = dict(metadatas[0] or {}) if metadatas else {}
        return Chunk(
            chunk_id=chunk_id,
            document_id=str(meta.get("document_id") or ""),
            text=text or "",
            metadata=meta,
        )


@registry.register("fake")
class FakeVectorStore(BaseVectorStore):
    _memory: dict[str, dict[str, Chunk]] = {}

    def __init__(
        self,
        persist_path: str = "",
        collection: str = "fake",
        behavior: str = "ok",
        **kwargs: object,
    ) -> None:
        self.persist_path = persist_path
        self.collection_name = collection
        self.behavior = behavior
        self._memory.setdefault(collection, {})

    @property
    def provider_name(self) -> str:
        return "fake"

    def upsert(self, chunks: list[Chunk]) -> None:
        apply_behavior(self.behavior, "vector_store")
        if self.behavior == "garbage":
            return
        bucket = self._memory[self.collection_name]
        for chunk in chunks:
            bucket[chunk.chunk_id] = chunk

    def get_by_document_id(self, document_id: str) -> list[Chunk]:
        bucket = self._memory[self.collection_name]
        found = [chunk for chunk in bucket.values() if chunk.document_id == document_id]
        found.sort(key=lambda chunk: chunk.chunk_id)
        return found

    def delete_by_document_id(self, document_id: str) -> None:
        bucket = self._memory[self.collection_name]
        stale = [key for key, chunk in bucket.items() if chunk.document_id == document_id]
        for chunk_id in stale:
            del bucket[chunk_id]

    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        bucket = self._memory[self.collection_name]
        chunks = [bucket[chunk_id] for chunk_id in chunk_ids if chunk_id in bucket]
        order = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
        chunks.sort(key=lambda chunk: order.get(chunk.chunk_id, len(order)))
        return chunks

    def query(self, query_embedding: list[float], top_k: int) -> list[ScoredChunk]:
        apply_behavior(self.behavior, "vector_store")
        if top_k <= 0:
            return []
        bucket = self._memory[self.collection_name]

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = sum(x * x for x in a) ** 0.5 or 1.0
            norm_b = sum(x * x for x in b) ** 0.5 or 1.0
            return dot / (norm_a * norm_b)

        scored: list[ScoredChunk] = []
        for chunk in bucket.values():
            if chunk.embedding is None:
                continue
            scored.append(
                ScoredChunk(chunk=chunk, score=cosine(query_embedding, chunk.embedding))
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def list_all(self) -> list[Chunk]:
        bucket = self._memory[self.collection_name]
        found = list(bucket.values())
        found.sort(key=lambda chunk: chunk.chunk_id)
        return found

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        bucket = self._memory[self.collection_name]
        return bucket.get(chunk_id)
