from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from wenmai.config import Settings
from wenmai.models import ScoredChunk
from wenmai.retrieval.rrf import reciprocal_rank_fusion

SearchMode = Literal["dense_only", "sparse_only", "rrf"]

_SEARCH_MODES = frozenset({"dense_only", "sparse_only", "rrf"})


@dataclass(frozen=True)
class SearchResult:
    """Unified retrieval result; sparse→dense hydration stays inside Knowledge."""

    chunks: list[ScoredChunk]
    mode: SearchMode
    dense_chunks: list[ScoredChunk] = field(default_factory=list)
    sparse_chunks: list[ScoredChunk] = field(default_factory=list)
    dense_elapsed_ms: float = 0.0
    sparse_elapsed_ms: float = 0.0
    fusion_elapsed_ms: float = 0.0


def validate_search_mode(mode: str) -> SearchMode:
    if mode not in _SEARCH_MODES:
        raise ValueError(f"unknown search mode: {mode!r}")
    return mode  # type: ignore[return-value]


def run_search(
  knowledge: object,
  query: str,
  *,
  mode: str,
  settings: Settings,
  culture_domain: str | None = None,
) -> SearchResult:
    """Dispatch dense-only, sparse-only, or RRF search on a Knowledge instance."""
    resolved = validate_search_mode(mode)
    if resolved == "dense_only":
        return _dense_only(knowledge, query, settings, culture_domain)
    if resolved == "sparse_only":
        return _sparse_only(knowledge, query, settings, culture_domain)
    return _rrf(knowledge, query, settings, culture_domain)


def _dense_only(
    knowledge: object,
    query: str,
    settings: Settings,
    culture_domain: str | None,
) -> SearchResult:
    started = time.perf_counter()
    chunks = knowledge.dense_search(  # type: ignore[attr-defined]
        query,
        top_k=settings.retrieval.dense_k,
        culture_domain=culture_domain,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return SearchResult(
        chunks=chunks,
        mode="dense_only",
        dense_chunks=chunks,
        dense_elapsed_ms=elapsed_ms,
    )


def _sparse_only(
    knowledge: object,
    query: str,
    settings: Settings,
    culture_domain: str | None,
) -> SearchResult:
    started = time.perf_counter()
    chunks = knowledge.sparse_search(  # type: ignore[attr-defined]
        query,
        top_k=settings.retrieval.sparse_k,
        culture_domain=culture_domain,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return SearchResult(
        chunks=chunks,
        mode="sparse_only",
        sparse_chunks=chunks,
        sparse_elapsed_ms=elapsed_ms,
    )


def _rrf(
    knowledge: object,
    query: str,
    settings: Settings,
    culture_domain: str | None,
) -> SearchResult:
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            _timed_dense, knowledge, query, settings, culture_domain
        )
        sparse_future = executor.submit(
            _timed_sparse, knowledge, query, settings, culture_domain
        )
        dense_chunks, dense_ms = dense_future.result()
        sparse_chunks, sparse_ms = sparse_future.result()

    chunk_by_id = {
        item.chunk.chunk_id: item.chunk for item in dense_chunks + sparse_chunks
    }
    started = time.perf_counter()
    fused_ids = reciprocal_rank_fusion(
        [
            [item.chunk.chunk_id for item in dense_chunks],
            [item.chunk.chunk_id for item in sparse_chunks],
        ],
        k=settings.retrieval.rrf_k,
        top_k=settings.retrieval.fused_k,
    )
    scored_chunks = [
        ScoredChunk(chunk=chunk_by_id[chunk_id], score=score)
        for chunk_id, score in fused_ids
        if chunk_id in chunk_by_id
    ]
    fusion_ms = (time.perf_counter() - started) * 1000
    return SearchResult(
        chunks=scored_chunks,
        mode="rrf",
        dense_chunks=dense_chunks,
        sparse_chunks=sparse_chunks,
        dense_elapsed_ms=dense_ms,
        sparse_elapsed_ms=sparse_ms,
        fusion_elapsed_ms=fusion_ms,
    )


def _timed_dense(
    knowledge: object,
    query: str,
    settings: Settings,
    culture_domain: str | None,
) -> tuple[list[ScoredChunk], float]:
    started = time.perf_counter()
    chunks = knowledge.dense_search(  # type: ignore[attr-defined]
        query,
        top_k=settings.retrieval.dense_k,
        culture_domain=culture_domain,
    )
    return chunks, (time.perf_counter() - started) * 1000


def _timed_sparse(
    knowledge: object,
    query: str,
    settings: Settings,
    culture_domain: str | None,
) -> tuple[list[ScoredChunk], float]:
    started = time.perf_counter()
    chunks = knowledge.sparse_search(  # type: ignore[attr-defined]
        query,
        top_k=settings.retrieval.sparse_k,
        culture_domain=culture_domain,
    )
    return chunks, (time.perf_counter() - started) * 1000
