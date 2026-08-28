from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

from wenmai.config import Settings
from wenmai.factories import reranker as reranker_factory
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import ScoredChunk
from wenmai.retrieval.rrf import reciprocal_rank_fusion

_RETRIEVAL_MODES = frozenset({"rrf", "dense_only", "sparse_only"})


@dataclass(frozen=True)
class RetrievalStage:
    name: str
    method: str
    provider: str
    elapsed_ms: float
    input_summary: str = ""
    output_summary: str = ""
    candidate_count: int | None = None
    error: str | None = None
    candidates: list[dict[str, Any]] | None = None
    dense_candidates: list[dict[str, Any]] | None = None
    sparse_candidates: list[dict[str, Any]] | None = None
    pre_rerank_candidates: list[dict[str, Any]] | None = None
    fallback_reason: str | None = None
    rank_changes: list[dict[str, Any]] | None = None

    def as_record_kwargs(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "provider": self.provider,
            "elapsed_ms": self.elapsed_ms,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "candidate_count": self.candidate_count,
            "error": self.error,
            "candidates": self.candidates,
            "dense_candidates": self.dense_candidates,
            "sparse_candidates": self.sparse_candidates,
            "pre_rerank_candidates": self.pre_rerank_candidates,
            "fallback_reason": self.fallback_reason,
            "rank_changes": self.rank_changes,
        }


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    stages: list[RetrievalStage] = field(default_factory=list)


def retrieve(
    question: str,
    settings: Settings,
    *,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
    knowledge: Knowledge | None = None,
) -> RetrievalResult:
    mode, do_rerank = _resolve_retrieval(settings, retrieval_mode, rerank_enabled)
    knowledge = knowledge or create_knowledge(settings)

    if mode == "sparse_only":
        chunks, stages = _sparse_only(knowledge, settings, question, culture_domain)
    elif mode == "dense_only":
        chunks, stages = _dense_only(knowledge, settings, question, culture_domain)
    else:
        chunks, stages = _fused(knowledge, settings, question, culture_domain)

    if do_rerank:
        chunks, rerank_stage = _rerank_chunks(settings, question, chunks)
        if rerank_stage is not None:
            stages = [*stages, rerank_stage]
    return RetrievalResult(chunks=chunks, stages=stages)


def _resolve_retrieval(
    settings: Settings,
    retrieval_mode: str | None,
    rerank_enabled: bool | None,
) -> tuple[str, bool]:
    mode = settings.retrieval.mode if retrieval_mode is None else retrieval_mode
    if mode not in _RETRIEVAL_MODES:
        raise ValueError(f"unknown retrieval_mode: {mode!r}")
    rerank = settings.retrieval.rerank_enabled if rerank_enabled is None else rerank_enabled
    if mode != "rrf":
        rerank = False
    return mode, bool(rerank)


def _candidate_records(scored_chunks: list[ScoredChunk]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": item.chunk.chunk_id,
            "score": round(item.score, 6),
        }
        for item in scored_chunks
    ]


def _dense_input_summary(settings: Settings, culture_domain: str | None) -> str:
    summary = f"k={settings.retrieval.dense_k}"
    if culture_domain is not None:
        summary += f" culture_domain={culture_domain}"
    return summary


def _sparse_input_summary(settings: Settings, culture_domain: str | None) -> str:
    summary = f"k={settings.retrieval.sparse_k}"
    if culture_domain is not None:
        summary += f" culture_domain={culture_domain}"
    return summary


def _timed_dense(
    knowledge: Knowledge,
    settings: Settings,
    question: str,
    culture_domain: str | None,
) -> tuple[list[ScoredChunk], float]:
    started = time.perf_counter()
    chunks = knowledge.dense_search(
        question,
        top_k=settings.retrieval.dense_k,
        culture_domain=culture_domain,
    )
    return chunks, (time.perf_counter() - started) * 1000


def _timed_sparse(
    knowledge: Knowledge,
    settings: Settings,
    question: str,
    culture_domain: str | None,
) -> tuple[list[ScoredChunk], float]:
    started = time.perf_counter()
    chunks = knowledge.sparse_search(
        question,
        top_k=settings.retrieval.sparse_k,
        culture_domain=culture_domain,
    )
    return chunks, (time.perf_counter() - started) * 1000


def _dense_stage(
    knowledge: Knowledge,
    settings: Settings,
    scored_chunks: list[ScoredChunk],
    elapsed_ms: float,
    culture_domain: str | None,
) -> RetrievalStage:
    return RetrievalStage(
        name="dense",
        method="vector_query",
        provider=knowledge.dense_provider,
        elapsed_ms=elapsed_ms,
        input_summary=_dense_input_summary(settings, culture_domain),
        output_summary=f"retrieved {len(scored_chunks)} chunks",
        candidate_count=len(scored_chunks),
        candidates=_candidate_records(scored_chunks),
    )


def _sparse_stage(
    settings: Settings,
    scored_chunks: list[ScoredChunk],
    elapsed_ms: float,
    culture_domain: str | None,
) -> RetrievalStage:
    return RetrievalStage(
        name="sparse",
        method="bm25",
        provider="local",
        elapsed_ms=elapsed_ms,
        input_summary=_sparse_input_summary(settings, culture_domain),
        output_summary=f"retrieved {len(scored_chunks)} chunks",
        candidate_count=len(scored_chunks),
        candidates=_candidate_records(scored_chunks),
    )


def _dense_only(
    knowledge: Knowledge,
    settings: Settings,
    question: str,
    culture_domain: str | None,
) -> tuple[list[ScoredChunk], list[RetrievalStage]]:
    chunks, elapsed_ms = _timed_dense(knowledge, settings, question, culture_domain)
    return chunks, [_dense_stage(knowledge, settings, chunks, elapsed_ms, culture_domain)]


def _sparse_only(
    knowledge: Knowledge,
    settings: Settings,
    question: str,
    culture_domain: str | None,
) -> tuple[list[ScoredChunk], list[RetrievalStage]]:
    chunks, elapsed_ms = _timed_sparse(knowledge, settings, question, culture_domain)
    return chunks, [_sparse_stage(settings, chunks, elapsed_ms, culture_domain)]


def _fused(
    knowledge: Knowledge,
    settings: Settings,
    question: str,
    culture_domain: str | None,
) -> tuple[list[ScoredChunk], list[RetrievalStage]]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            _timed_dense, knowledge, settings, question, culture_domain
        )
        sparse_future = executor.submit(
            _timed_sparse, knowledge, settings, question, culture_domain
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
    fusion = RetrievalStage(
        name="fusion",
        method="rrf",
        provider="local",
        elapsed_ms=fusion_ms,
        input_summary=(
            f"dense={len(dense_chunks)} sparse={len(sparse_chunks)} "
            f"k={settings.retrieval.rrf_k}"
        ),
        output_summary=f"fused {len(scored_chunks)} chunks",
        candidate_count=len(scored_chunks),
        candidates=_candidate_records(scored_chunks),
        dense_candidates=_candidate_records(dense_chunks),
        sparse_candidates=_candidate_records(sparse_chunks),
    )
    return scored_chunks, [
        _dense_stage(knowledge, settings, dense_chunks, dense_ms, culture_domain),
        _sparse_stage(settings, sparse_chunks, sparse_ms, culture_domain),
        fusion,
    ]


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


def _rerank_chunks(
    settings: Settings,
    query: str,
    fused_chunks: list[ScoredChunk],
) -> tuple[list[ScoredChunk], RetrievalStage | None]:
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
        return reranked, RetrievalStage(
            name="rerank",
            method="cross_encoder",
            provider=reranker.provider_name,
            elapsed_ms=elapsed_ms,
            input_summary=f"{len(pre_rerank)} candidates, top={rerank_top}",
            output_summary=f"reranked to {len(reranked)} chunks",
            candidate_count=len(reranked),
            candidates=_candidate_records(reranked),
            pre_rerank_candidates=_candidate_records(pre_rerank),
            rank_changes=_rank_changes(pre_rerank, reranked),
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        fallback = pre_rerank[:rerank_top]
        reason = f"{type(exc).__name__}: {exc}"
        return fallback, RetrievalStage(
            name="rerank",
            method="rrf_fallback",
            provider=reranker.provider_name,
            elapsed_ms=elapsed_ms,
            input_summary=f"{len(pre_rerank)} candidates, top={rerank_top}",
            output_summary=f"fallback to RRF top-{len(fallback)}",
            candidate_count=len(fallback),
            candidates=_candidate_records(fallback),
            pre_rerank_candidates=_candidate_records(pre_rerank),
            error=reason,
            fallback_reason=reason,
        )
