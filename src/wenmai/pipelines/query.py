from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

from wenmai.components.retrieval.rrf import reciprocal_rank_fusion
from wenmai.config import Settings
from wenmai.factories import bm25 as bm25_factory
from wenmai.factories import embedding as embedding_factory
from wenmai.factories import llm as llm_factory
from wenmai.factories import reranker as reranker_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.models import AskResult, Citation, ScoredChunk
from wenmai.storage.paths import store_path
from wenmai.tracing.context import TraceContext
from wenmai.tracing.writer import JsonlTraceWriter

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_REFUSAL_PREFIX = "拒答："


class QueryGenerationError(Exception):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


def _load_qa_prompt(settings: Settings) -> str:
    prompt_path = settings.root / settings.paths.prompts
    return prompt_path.read_text(encoding="utf-8")


def _normalize_question(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question.strip())
    return collapsed


def _build_context(scored_chunks: list[ScoredChunk]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(scored_chunks, start=1):
        chunk = item.chunk
        title = str(chunk.metadata.get("title") or chunk.metadata.get("chunk_title") or "")
        header = f"[{index}]"
        if title:
            header += f" {title}"
        blocks.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def _build_prompt(template: str, question: str, scored_chunks: list[ScoredChunk]) -> str:
    return template.format(question=question, context=_build_context(scored_chunks))


def _excerpt(text: str, limit: int = 160) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _citation_for_index(index: int, scored_chunks: list[ScoredChunk]) -> Citation | None:
    if index < 1 or index > len(scored_chunks):
        return None
    chunk = scored_chunks[index - 1].chunk
    return Citation(
        index=index,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=str(chunk.metadata.get("title") or chunk.metadata.get("chunk_title") or ""),
        excerpt=_excerpt(chunk.text),
        url=str(chunk.metadata.get("url") or ""),
    )


def _extract_citations(answer: str, scored_chunks: list[ScoredChunk]) -> list[Citation]:
    indices = sorted({int(match) for match in _CITATION_PATTERN.findall(answer)})
    citations: list[Citation] = []
    for index in indices:
        citation = _citation_for_index(index, scored_chunks)
        if citation is not None:
            citations.append(citation)
    return citations


def _citations_from_retrieved(scored_chunks: list[ScoredChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for index in range(1, len(scored_chunks) + 1):
        citation = _citation_for_index(index, scored_chunks)
        if citation is not None:
            citations.append(citation)
    return citations


def _is_refusal(answer: str) -> bool:
    return answer.strip().startswith(_REFUSAL_PREFIX)


def _resolve_response(answer: str, scored_chunks: list[ScoredChunk]) -> tuple[bool, list[Citation]]:
    refused = _is_refusal(answer)
    if refused:
        return True, _citations_from_retrieved(scored_chunks)
    return False, _extract_citations(answer, scored_chunks)


def _candidate_records(scored_chunks: list[ScoredChunk]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": item.chunk.chunk_id,
            "score": round(item.score, 6),
        }
        for item in scored_chunks
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


def _rerank_chunks(
    settings: Settings,
    query: str,
    fused_chunks: list[ScoredChunk],
    trace: TraceContext,
) -> list[ScoredChunk]:
    rerank_top = settings.retrieval.rerank_top
    if not fused_chunks or rerank_top <= 0:
        return fused_chunks

    pre_rerank = fused_chunks
    reranker = reranker_factory.create(settings)

    with trace.stage(
        "rerank",
        method="cross_encoder",
        provider=reranker.provider_name,
        input_summary=f"{len(pre_rerank)} candidates, top={rerank_top}",
    ) as rerank_info:
        texts = [item.chunk.text for item in pre_rerank]
        rerank_info["pre_rerank_candidates"] = _candidate_records(pre_rerank)
        try:
            rankings = reranker.rerank(query, texts)
            if not rankings:
                raise RuntimeError("reranker returned no scores")
            reranked: list[ScoredChunk] = []
            for index, score in rankings[:rerank_top]:
                if index < 0 or index >= len(pre_rerank):
                    continue
                chunk = pre_rerank[index]
                reranked.append(ScoredChunk(chunk=chunk.chunk, score=score))
            rerank_info["output_summary"] = f"reranked to {len(reranked)} chunks"
            rerank_info["candidate_count"] = len(reranked)
            rerank_info["candidates"] = _candidate_records(reranked)
            rerank_info["rank_changes"] = _rank_changes(pre_rerank, reranked)
            return reranked
        except Exception as exc:
            fallback = pre_rerank[:rerank_top]
            reason = f"{type(exc).__name__}: {exc}"
            rerank_info["method"] = "rrf_fallback"
            rerank_info["output_summary"] = f"fallback to RRF top-{len(fallback)}"
            rerank_info["error"] = reason
            rerank_info["fallback_reason"] = reason
            rerank_info["candidate_count"] = len(fallback)
            rerank_info["candidates"] = _candidate_records(fallback)
            return fallback


def _run_dense_search(settings: Settings, normalized: str) -> list[ScoredChunk]:
    store = vector_store_factory.create(settings)
    embedder = embedding_factory.create(settings)
    query_vector = embedder.embed_query(normalized)
    return store.query(query_vector, top_k=settings.retrieval.dense_k)


def _run_sparse_search(settings: Settings, normalized: str) -> list[ScoredChunk]:
    store = vector_store_factory.create(settings)
    bm25_index = bm25_factory.create(settings)
    hits = bm25_index.search(normalized, top_k=settings.retrieval.sparse_k)
    chunk_ids = [hit.chunk_id for hit in hits]
    chunks = store.get_by_ids(chunk_ids)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [
        ScoredChunk(chunk=chunk_by_id[hit.chunk_id], score=hit.score)
        for hit in hits
        if hit.chunk_id in chunk_by_id
    ]


def _timed_dense_search(settings: Settings, normalized: str) -> tuple[list[ScoredChunk], float]:
    started = time.perf_counter()
    chunks = _run_dense_search(settings, normalized)
    return chunks, (time.perf_counter() - started) * 1000


def _timed_sparse_search(settings: Settings, normalized: str) -> tuple[list[ScoredChunk], float]:
    started = time.perf_counter()
    chunks = _run_sparse_search(settings, normalized)
    return chunks, (time.perf_counter() - started) * 1000


def _record_dense_stage(
    trace: TraceContext,
    settings: Settings,
    scored_chunks: list[ScoredChunk],
    elapsed_ms: float,
) -> None:
    store = vector_store_factory.create(settings)
    trace.record_stage(
        name="dense",
        method="vector_query",
        provider=store.provider_name,
        elapsed_ms=elapsed_ms,
        input_summary=f"k={settings.retrieval.dense_k}",
        output_summary=f"retrieved {len(scored_chunks)} chunks",
        candidate_count=len(scored_chunks),
        candidates=_candidate_records(scored_chunks),
    )


def _record_sparse_stage(
    trace: TraceContext,
    settings: Settings,
    scored_chunks: list[ScoredChunk],
    elapsed_ms: float,
) -> None:
    trace.record_stage(
        name="sparse",
        method="bm25",
        provider="local",
        elapsed_ms=elapsed_ms,
        input_summary=f"k={settings.retrieval.sparse_k}",
        output_summary=f"retrieved {len(scored_chunks)} chunks",
        candidate_count=len(scored_chunks),
        candidates=_candidate_records(scored_chunks),
    )


def _dense_retrieve(
    settings: Settings,
    normalized: str,
    trace: TraceContext,
) -> list[ScoredChunk]:
    scored_chunks, elapsed_ms = _timed_dense_search(settings, normalized)
    _record_dense_stage(trace, settings, scored_chunks, elapsed_ms)
    return scored_chunks


def _sparse_retrieve(
    settings: Settings,
    normalized: str,
    trace: TraceContext,
) -> list[ScoredChunk]:
    scored_chunks, elapsed_ms = _timed_sparse_search(settings, normalized)
    _record_sparse_stage(trace, settings, scored_chunks, elapsed_ms)
    return scored_chunks


def _fuse_retrievals(
    settings: Settings,
    dense_chunks: list[ScoredChunk],
    sparse_chunks: list[ScoredChunk],
    trace: TraceContext,
) -> list[ScoredChunk]:
    chunk_by_id = {
        item.chunk.chunk_id: item.chunk
        for item in dense_chunks + sparse_chunks
    }
    fused_ids = reciprocal_rank_fusion(
        [
            [item.chunk.chunk_id for item in dense_chunks],
            [item.chunk.chunk_id for item in sparse_chunks],
        ],
        k=settings.retrieval.rrf_k,
        top_k=settings.retrieval.fused_k,
    )
    with trace.stage(
        "fusion",
        method="rrf",
        provider="local",
        input_summary=(
            f"dense={len(dense_chunks)} sparse={len(sparse_chunks)} "
            f"k={settings.retrieval.rrf_k}"
        ),
    ) as fusion_info:
        scored_chunks = [
            ScoredChunk(chunk=chunk_by_id[chunk_id], score=score)
            for chunk_id, score in fused_ids
            if chunk_id in chunk_by_id
        ]
        fusion_info["candidate_count"] = len(scored_chunks)
        fusion_info["output_summary"] = f"fused {len(scored_chunks)} chunks"
        fusion_info["dense_candidates"] = _candidate_records(dense_chunks)
        fusion_info["sparse_candidates"] = _candidate_records(sparse_chunks)
        fusion_info["candidates"] = _candidate_records(scored_chunks)
    return scored_chunks


def _retrieve_chunks(
    settings: Settings,
    normalized: str,
    trace: TraceContext,
) -> list[ScoredChunk]:
    mode = settings.retrieval.mode

    if mode == "sparse_only":
        return _sparse_retrieve(settings, normalized, trace)

    if mode == "dense_only":
        return _dense_retrieve(settings, normalized, trace)

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(_timed_dense_search, settings, normalized)
        sparse_future = executor.submit(_timed_sparse_search, settings, normalized)
        dense_chunks, dense_ms = dense_future.result()
        sparse_chunks, sparse_ms = sparse_future.result()
    _record_dense_stage(trace, settings, dense_chunks, dense_ms)
    _record_sparse_stage(trace, settings, sparse_chunks, sparse_ms)
    return _fuse_retrievals(settings, dense_chunks, sparse_chunks, trace)


def ask_question(question: str, settings: Settings) -> AskResult:
    trace = TraceContext(trace_type="query", metadata={"question": question})
    writer = JsonlTraceWriter(store_path(settings, "traces"))
    normalized = _normalize_question(question)

    try:
        with trace.stage(
            "query_processing",
            method="normalize",
            provider="local",
            input_summary=question,
        ) as stage_info:
            stage_info["output_summary"] = normalized
            stage_info["candidate_count"] = 1

        scored_chunks = _retrieve_chunks(settings, normalized, trace)
        if settings.retrieval.mode == "rrf":
            scored_chunks = _rerank_chunks(settings, normalized, scored_chunks, trace)

        llm = llm_factory.create(settings)
        template = _load_qa_prompt(settings)
        prompt = _build_prompt(template, normalized, scored_chunks)

        with trace.stage(
            "generation",
            method="llm",
            provider=llm.provider_name,
            input_summary=f"{len(scored_chunks)} chunks",
        ) as generation_info:
            try:
                answer = llm.generate(prompt)
            except Exception as exc:
                generation_info["output_summary"] = "generation failed"
                generation_info["error"] = f"{type(exc).__name__}: {exc}"
                trace.error = generation_info["error"]
                raise QueryGenerationError(str(exc), trace.trace_id) from exc
            refused, citations = _resolve_response(answer, scored_chunks)
            generation_info["output_summary"] = "refusal" if refused else f"{len(answer)} chars"
            generation_info["candidate_count"] = len(scored_chunks)

        return AskResult(
            answer=answer,
            citations=citations,
            trace_id=trace.trace_id,
            refused=refused,
        )
    finally:
        trace.close()
        writer.write(trace)
