from __future__ import annotations

import re
import time

from wenmai.config import Settings
from wenmai.factories import query_rewrite as query_rewrite_factory
from wenmai.generation import GenerationError, QueryGenerationError, generate
from wenmai.generation.expand import expand_for_generation
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import AskResult
from wenmai.retrieval import retrieve
from wenmai.tracing import TraceContext, save_trace
from wenmai.tracing.stages.query import QueryStage

__all__ = ["QueryGenerationError", "ask_question", "normalize_question"]


def normalize_question(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question.strip())
    return collapsed


def ask_question(
    question: str,
    settings: Settings,
    culture_domain: str | None = None,
    *,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
    knowledge: Knowledge | None = None,
    record_trace: bool = True,
) -> AskResult:
    trace = TraceContext(trace_type="query", metadata={"question": question})
    normalized = normalize_question(question)

    try:
        processing_started = time.perf_counter()
        rewriter = query_rewrite_factory.create(settings)
        extra_queries = rewriter.extra_queries(normalized)
        trace.append_stage(
            QueryStage.query_processing(
                question=question,
                normalized=normalized,
                elapsed_ms=(time.perf_counter() - processing_started) * 1000,
                culture_domain=culture_domain,
                extra_queries=extra_queries,
                rewriter=rewriter.provider_name,
            )
        )

        retrieved = retrieve(
            normalized,
            settings,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            extra_queries=extra_queries,
        )
        for stage in retrieved.stages:
            trace.append_stage(stage)
        scored_chunks = retrieved.chunks
        knowledge = knowledge or create_knowledge(settings)
        expanded_chunks, expanded_from, expanded_chunk_ids = expand_for_generation(
            scored_chunks,
            knowledge,
            settings.retrieval.adjacent_n,
        )

        generation_started = time.perf_counter()
        generation_input = f"{len(expanded_chunks)} chunks"
        try:
            gen_result = generate(normalized, expanded_chunks, settings)
        except GenerationError as exc:
            generation_error = f"{type(exc).__name__}: {exc}"
            trace.append_stage(
                QueryStage.generation(
                    provider=exc.provider_name,
                    elapsed_ms=(time.perf_counter() - generation_started) * 1000,
                    input_summary=generation_input,
                    output_summary="generation failed",
                    candidate_count=0,
                    error=generation_error,
                    expanded_from=expanded_from,
                    expanded_chunk_ids=expanded_chunk_ids,
                )
            )
            trace.error = generation_error
            raise QueryGenerationError(str(exc), trace.trace_id) from exc

        trace.append_stage(
            QueryStage.generation(
                provider=gen_result.provider_name,
                elapsed_ms=(time.perf_counter() - generation_started) * 1000,
                input_summary=generation_input,
                output_summary=gen_result.output_summary,
                candidate_count=gen_result.candidate_count,
                expanded_from=expanded_from,
                expanded_chunk_ids=expanded_chunk_ids,
            )
        )

        trace.metadata["outcome"] = {
            "refused": gen_result.refused,
            "refusal_reason": gen_result.refusal_reason,
            "citation_count": len(gen_result.citations),
        }

        return AskResult(
            answer=gen_result.answer,
            citations=gen_result.citations,
            trace_id=trace.trace_id,
            refused=gen_result.refused,
            refusal_reason=gen_result.refusal_reason,
            ranked_chunks=scored_chunks,
        )
    finally:
        if record_trace:
            save_trace(settings, trace)
