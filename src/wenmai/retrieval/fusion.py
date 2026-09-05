from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from wenmai.config import Settings
from wenmai.models import Chunk, ScoredChunk
from wenmai.retrieval.rrf import reciprocal_rank_fusion
from wenmai.tracing.context import StageRecord

if TYPE_CHECKING:
    from wenmai.knowledge.store import Knowledge

RetrievalMode = Literal["dense_only", "sparse_only", "rrf"]

_RETRIEVAL_MODES = frozenset({"dense_only", "sparse_only", "rrf"})


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    mode: RetrievalMode
    stages: list[StageRecord] = field(default_factory=list)
    dense_chunks: list[ScoredChunk] = field(default_factory=list)
    sparse_chunks: list[ScoredChunk] = field(default_factory=list)
    dense_elapsed_ms: float = 0.0
    sparse_elapsed_ms: float = 0.0
    fusion_elapsed_ms: float = 0.0
    query_path_counts: list[dict[str, object]] = field(default_factory=list)


def validate_retrieval_mode(mode: str) -> RetrievalMode:
    if mode not in _RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval mode: {mode!r}")
    return mode  # type: ignore[return-value]


def unique_queries(primary: str, extra_queries: list[str] | None = None) -> list[str]:
    queries = [primary]
    seen = {primary}
    for query in extra_queries or []:
        if query in seen:
            continue
        seen.add(query)
        queries.append(query)
    return queries


def run_fusion(
    knowledge: Knowledge,
    query: str,
    *,
    mode: str,
    settings: Settings,
    culture_domain: str | None = None,
    extra_queries: list[str] | None = None,
) -> RetrievalResult:
    resolved = validate_retrieval_mode(mode)
    queries = unique_queries(query, extra_queries)
    if resolved == "dense_only":
        return _single_path_fusion(
            knowledge,
            queries,
            settings,
            culture_domain,
            mode="dense_only",
            search=knowledge.dense_search,
            top_k=settings.retrieval.dense_k,
            path_counter=lambda index, count: {
                "query_index": index,
                "dense_count": count,
                "sparse_count": 0,
            },
            primary_attr="dense_chunks",
        )
    if resolved == "sparse_only":
        return _single_path_fusion(
            knowledge,
            queries,
            settings,
            culture_domain,
            mode="sparse_only",
            search=knowledge.sparse_search,
            top_k=settings.retrieval.sparse_k,
            path_counter=lambda index, count: {
                "query_index": index,
                "dense_count": 0,
                "sparse_count": count,
            },
            primary_attr="sparse_chunks",
        )
    return _rrf(knowledge, queries, settings, culture_domain)


def _single_path_fusion(
    knowledge: Knowledge,
    queries: list[str],
    settings: Settings,
    culture_domain: str | None,
    *,
    mode: RetrievalMode,
    search: Callable[..., list[ScoredChunk]],
    top_k: int,
    path_counter: Callable[[int, int], dict[str, object]],
    primary_attr: Literal["dense_chunks", "sparse_chunks"],
) -> RetrievalResult:
    started = time.perf_counter()
    primary_chunks: list[ScoredChunk] = []
    ranked_lists: list[list[str]] = []
    chunk_by_id: dict[str, ScoredChunk] = {}
    query_path_counts: list[dict[str, object]] = []
    for index, query in enumerate(queries):
        chunks = search(query, top_k=top_k, culture_domain=culture_domain)
        if index == 0:
            primary_chunks = chunks
        ranked_lists.append([item.chunk.chunk_id for item in chunks])
        query_path_counts.append(path_counter(index, len(chunks)))
        for item in chunks:
            chunk_by_id[item.chunk.chunk_id] = item
    elapsed_ms = (time.perf_counter() - started) * 1000
    if len(ranked_lists) == 1:
        fused_chunks = primary_chunks
        fusion_ms = 0.0
    else:
        fusion_started = time.perf_counter()
        fused_ids = reciprocal_rank_fusion(
            ranked_lists,
            k=settings.retrieval.rrf_k,
            top_k=settings.retrieval.fused_k,
        )
        fused_chunks = [
            ScoredChunk(chunk=chunk_by_id[chunk_id].chunk, score=score)
            for chunk_id, score in fused_ids
            if chunk_id in chunk_by_id
        ]
        fusion_ms = (time.perf_counter() - fusion_started) * 1000
    if primary_attr == "dense_chunks":
        return RetrievalResult(
            chunks=fused_chunks,
            mode=mode,
            dense_chunks=primary_chunks,
            dense_elapsed_ms=elapsed_ms,
            fusion_elapsed_ms=fusion_ms,
            query_path_counts=query_path_counts,
        )
    return RetrievalResult(
        chunks=fused_chunks,
        mode=mode,
        sparse_chunks=primary_chunks,
        sparse_elapsed_ms=elapsed_ms,
        fusion_elapsed_ms=fusion_ms,
        query_path_counts=query_path_counts,
    )


def _rrf(
    knowledge: Knowledge,
    queries: list[str],
    settings: Settings,
    culture_domain: str | None,
) -> RetrievalResult:
    primary_dense: list[ScoredChunk] = []
    primary_sparse: list[ScoredChunk] = []
    dense_ms = 0.0
    sparse_ms = 0.0
    ranked_lists: list[list[str]] = []
    chunk_by_id: dict[str, Chunk] = {}
    query_path_counts: list[dict[str, object]] = []

    for index, query in enumerate(queries):
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(
                _timed_dense, knowledge, query, settings, culture_domain
            )
            sparse_future = executor.submit(
                _timed_sparse, knowledge, query, settings, culture_domain
            )
            dense_chunks, query_dense_ms = dense_future.result()
            sparse_chunks, query_sparse_ms = sparse_future.result()
        if index == 0:
            primary_dense = dense_chunks
            primary_sparse = sparse_chunks
            dense_ms = query_dense_ms
            sparse_ms = query_sparse_ms
        ranked_lists.append([item.chunk.chunk_id for item in dense_chunks])
        ranked_lists.append([item.chunk.chunk_id for item in sparse_chunks])
        query_path_counts.append(
            {
                "query_index": index,
                "dense_count": len(dense_chunks),
                "sparse_count": len(sparse_chunks),
            }
        )
        for item in dense_chunks + sparse_chunks:
            chunk_by_id[item.chunk.chunk_id] = item.chunk

    started = time.perf_counter()
    fused_ids = reciprocal_rank_fusion(
        ranked_lists,
        k=settings.retrieval.rrf_k,
        top_k=settings.retrieval.fused_k,
    )
    scored_chunks = [
        ScoredChunk(chunk=chunk_by_id[chunk_id], score=score)
        for chunk_id, score in fused_ids
        if chunk_id in chunk_by_id
    ]
    fusion_ms = (time.perf_counter() - started) * 1000
    return RetrievalResult(
        chunks=scored_chunks,
        mode="rrf",
        dense_chunks=primary_dense,
        sparse_chunks=primary_sparse,
        dense_elapsed_ms=dense_ms,
        sparse_elapsed_ms=sparse_ms,
        fusion_elapsed_ms=fusion_ms,
        query_path_counts=query_path_counts,
    )


def _timed_dense(
    knowledge: Knowledge,
    query: str,
    settings: Settings,
    culture_domain: str | None,
) -> tuple[list[ScoredChunk], float]:
    started = time.perf_counter()
    chunks = knowledge.dense_search(
        query,
        top_k=settings.retrieval.dense_k,
        culture_domain=culture_domain,
    )
    return chunks, (time.perf_counter() - started) * 1000


def _timed_sparse(
    knowledge: Knowledge,
    query: str,
    settings: Settings,
    culture_domain: str | None,
) -> tuple[list[ScoredChunk], float]:
    started = time.perf_counter()
    chunks = knowledge.sparse_search(
        query,
        top_k=settings.retrieval.sparse_k,
        culture_domain=culture_domain,
    )
    return chunks, (time.perf_counter() - started) * 1000
