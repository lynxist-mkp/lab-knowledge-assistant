from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from wenmai.config import Settings
from wenmai.factories import query_rewrite as query_rewrite_factory
from wenmai.factories import reranker as reranker_factory
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import ScoredChunk
from wenmai.retrieval.fusion import RetrievalResult, run_fusion
from wenmai.tracing.context import StageRecord
from wenmai.tracing.stages.query import QueryStage
from wenmai.tracing.stages.retrieval import stages_from_fusion

_RETRIEVAL_MODES = frozenset({"rrf", "dense_only", "sparse_only"})


def retrieve(
    question: str,
    settings: Settings,
    *,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    knowledge: Knowledge | None = None,
    extra_queries: list[str] | None = None,
    include_trace_stages: bool = True,
    rerank_enabled: bool | None = None,
) -> RetrievalResult:
    """Return fused chunks and timing metrics. Rerank runs in 提问编排 phase 3."""
    _ = rerank_enabled
    mode = _resolve_mode(settings, retrieval_mode)
    knowledge = knowledge or create_knowledge(settings)
    resolved_extras = extra_queries
    if resolved_extras is None:
        rewriter = query_rewrite_factory.create(settings)
        resolved_extras = rewriter.extra_queries(question)

    fusion_result = run_fusion(
        knowledge,
        question,
        mode=mode,
        settings=settings,
        culture_domain=culture_domain,
        extra_queries=resolved_extras,
    )
    stages: list[StageRecord] = []
    if include_trace_stages:
        stages = stages_from_fusion(
            knowledge, settings, fusion_result, culture_domain
        )

    return RetrievalResult(
        chunks=fusion_result.chunks,
        mode=fusion_result.mode,
        stages=stages,
        dense_chunks=fusion_result.dense_chunks,
        sparse_chunks=fusion_result.sparse_chunks,
        dense_elapsed_ms=fusion_result.dense_elapsed_ms,
        sparse_elapsed_ms=fusion_result.sparse_elapsed_ms,
        fusion_elapsed_ms=fusion_result.fusion_elapsed_ms,
        query_path_counts=fusion_result.query_path_counts,
    )


def _resolve_mode(settings: Settings, retrieval_mode: str | None) -> str:
    mode = settings.retrieval.mode if retrieval_mode is None else retrieval_mode
    if mode not in _RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval_mode: {mode!r}")
    return mode


def resolve_retrieval_mode(
    settings: Settings,
    retrieval_mode: str | None,
    rerank_enabled: bool | None,
) -> tuple[str, bool]:
    mode = _resolve_mode(settings, retrieval_mode)
    rerank = settings.retrieval.rerank_enabled if rerank_enabled is None else rerank_enabled
    if mode != "rrf":
        rerank = False
    return mode, bool(rerank)


def _rank_changes(
    pre_rerank: list[ScoredChunk],
    post_rerank: list[ScoredChunk],
) -> list[dict[str, object]]:
    pre_ids = [item.chunk.chunk_id for item in pre_rerank]
    changes: list[dict[str, object]] = []
    for new_rank, item in enumerate(post_rerank, start=1):
        old_rank = pre_ids.index(item.chunk.chunk_id) + 1
        if old_rank != new_rank:
            changes.append(
                {
                    "chunk_id": item.chunk.chunk_id,
                    "from": old_rank,
                    "to": new_rank,
                }
            )
    return changes


def _call_reranker(
    reranker: object,
    query: str,
    texts: list[str],
    timeout_seconds: float,
) -> list[tuple[int, float]]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(reranker.rerank, query, texts)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            raise TimeoutError(f"reranker timed out after {timeout_seconds}s") from exc


def _apply_rerank_rankings(
    pre_rerank: list[ScoredChunk],
    rankings: list[tuple[int, float]],
    rerank_top: int,
) -> list[ScoredChunk]:
    reranked: list[ScoredChunk] = []
    seen: set[str] = set()
    for index, score in rankings:
        if len(reranked) >= rerank_top:
            break
        if index < 0 or index >= len(pre_rerank):
            continue
        item = pre_rerank[index]
        chunk_id = item.chunk.chunk_id
        if chunk_id in seen:
            continue
        reranked.append(ScoredChunk(chunk=item.chunk, score=score))
        seen.add(chunk_id)
    for item in pre_rerank:
        if len(reranked) >= rerank_top:
            break
        chunk_id = item.chunk.chunk_id
        if chunk_id in seen:
            continue
        reranked.append(item)
        seen.add(chunk_id)
    return reranked


def rerank_chunks(
    settings: Settings,
    query: str,
    fused_chunks: list[ScoredChunk],
) -> tuple[list[ScoredChunk], StageRecord | None]:
    rerank_top = settings.retrieval.rerank_top
    if not fused_chunks or rerank_top <= 0:
        return fused_chunks, None

    pre_rerank = fused_chunks
    reranker = reranker_factory.create(settings)
    started = time.perf_counter()
    try:
        rankings = _call_reranker(
            reranker,
            query,
            [item.chunk.text for item in pre_rerank],
            settings.retrieval.rerank_timeout_seconds,
        )
        if not rankings:
            raise RuntimeError("reranker returned no scores")
        reranked = _apply_rerank_rankings(pre_rerank, rankings, rerank_top)
        if not reranked:
            raise RuntimeError("reranker produced no usable candidates")
        elapsed_ms = (time.perf_counter() - started) * 1000
        return reranked, QueryStage.rerank_success(
            provider=reranker.provider_name,
            elapsed_ms=elapsed_ms,
            pre_rerank=pre_rerank,
            reranked=reranked,
            rerank_top=rerank_top,
            rank_changes=_rank_changes(pre_rerank, reranked),
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        fallback = pre_rerank[:rerank_top]
        reason = f"{type(exc).__name__}: {exc}"
        return fallback, QueryStage.rerank_fallback(
            provider=reranker.provider_name,
            elapsed_ms=elapsed_ms,
            pre_rerank=pre_rerank,
            fallback=fallback,
            rerank_top=rerank_top,
            reason=reason,
        )


__all__ = ["rerank_chunks", "resolve_retrieval_mode", "retrieve"]
