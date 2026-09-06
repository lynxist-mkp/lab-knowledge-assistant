from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wenmai.components.bm25.tokenizer import ChineseTokenizer
from wenmai.models import Chunk


@dataclass
class Bm25Hit:
    chunk_id: str
    score: float


@dataclass
class _ChunkStats:
    document_id: str
    doc_length: int
    term_freqs: dict[str, int]
    culture_domain: str = ""


class Bm25Index:
    def __init__(
        self,
        collection: str,
        persist_path: Path,
        tokenizer: ChineseTokenizer,
        k1: float,
        b: float,
    ) -> None:
        self.collection = collection
        self.persist_path = persist_path
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, _ChunkStats] = {}
        self._index_path = persist_path / "index.json"
        if self._index_path.exists():
            self._load()

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._upsert_one(chunk)
        self._recompute_idf()

    def delete_by_document_id(self, document_id: str) -> None:
        stale = [
            chunk_id
            for chunk_id, stats in self._chunks.items()
            if stats.document_id == document_id
        ]
        for chunk_id in stale:
            del self._chunks[chunk_id]
        self._recompute_idf()

    def search(
        self,
        query: str,
        top_k: int,
        culture_domain: str | None = None,
    ) -> list[Bm25Hit]:
        if top_k <= 0 or not self._chunks:
            return []
        query_terms = self.tokenizer.tokenize(query)
        if not query_terms:
            return []

        avgdl = self._avg_doc_length()
        scores: dict[str, float] = {}
        inverted = self._build_inverted()
        for term in query_terms:
            entry = inverted.get(term)
            if entry is None:
                continue
            idf = entry["idf"]
            for posting in entry["postings"]:
                chunk_id = posting["chunk_id"]
                stats = self._chunks[chunk_id]
                if culture_domain is not None and stats.culture_domain != culture_domain:
                    continue
                tf = posting["tf"]
                doc_length = posting["doc_length"]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_length / avgdl)
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [Bm25Hit(chunk_id=chunk_id, score=score) for chunk_id, score in ranked[:top_k]]

    def save(self) -> None:
        payload = self._serialize()
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _upsert_one(self, chunk: Chunk) -> None:
        tokens = self.tokenizer.tokenize(chunk.text)
        term_freqs: dict[str, int] = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1
        culture_domain = str(chunk.metadata.get("culture_domain") or "")
        self._chunks[chunk.chunk_id] = _ChunkStats(
            document_id=chunk.document_id,
            doc_length=len(tokens),
            term_freqs=term_freqs,
            culture_domain=culture_domain,
        )

    def _avg_doc_length(self) -> float:
        if not self._chunks:
            return 0.0
        total = sum(stats.doc_length for stats in self._chunks.values())
        return total / len(self._chunks)

    def _recompute_idf(self) -> None:
        self._idf_cache = self._build_inverted()

    def _build_inverted(self) -> dict[str, dict[str, Any]]:
        num_docs = len(self._chunks)
        term_docs: dict[str, set[str]] = {}
        for chunk_id, stats in self._chunks.items():
            for term in stats.term_freqs:
                term_docs.setdefault(term, set()).add(chunk_id)

        inverted: dict[str, dict[str, Any]] = {}
        for term, chunk_ids in term_docs.items():
            df = len(chunk_ids)
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
            postings = []
            for chunk_id in sorted(chunk_ids):
                stats = self._chunks[chunk_id]
                postings.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": stats.document_id,
                        "tf": stats.term_freqs[term],
                        "doc_length": stats.doc_length,
                        "culture_domain": stats.culture_domain,
                    }
                )
            inverted[term] = {"idf": idf, "postings": postings}
        return inverted

    def _serialize(self) -> dict[str, Any]:
        inverted = self._build_inverted()
        return {
            "version": 1,
            "collection": self.collection,
            "num_docs": len(self._chunks),
            "avg_doc_length": self._avg_doc_length(),
            "terms": inverted,
        }

    def _load(self) -> None:
        raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        self._chunks = {}
        for term, entry in (raw.get("terms") or {}).items():
            for posting in entry.get("postings") or []:
                chunk_id = posting["chunk_id"]
                stats = self._chunks.setdefault(
                    chunk_id,
                    _ChunkStats(
                        document_id=posting["document_id"],
                        doc_length=posting["doc_length"],
                        term_freqs={},
                        culture_domain=str(posting.get("culture_domain") or ""),
                    ),
                )
                stats.term_freqs[term] = posting["tf"]
        self._recompute_idf()
