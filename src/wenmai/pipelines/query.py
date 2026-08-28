from __future__ import annotations

import re

from wenmai.config import Settings
from wenmai.generation import GenerationError, QueryGenerationError, generate
from wenmai.knowledge import Knowledge
from wenmai.models import AskResult
from wenmai.retrieval import retrieve
from wenmai.tracing import TraceContext, save_trace

__all__ = ["QueryGenerationError", "ask_question"]


def _normalize_question(question: str) -> str:
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
) -> AskResult:
    trace = TraceContext(trace_type="query", metadata={"question": question})
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
            if culture_domain is not None:
                stage_info["culture_domain"] = culture_domain

        retrieved = retrieve(
            normalized,
            settings,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
        )
        for stage in retrieved.stages:
            trace.record_stage(**stage.as_record_kwargs())
        scored_chunks = retrieved.chunks

        with trace.stage(
            "generation",
            method="llm",
            provider="unknown",
            input_summary=f"{len(scored_chunks)} chunks",
        ) as generation_info:
            try:
                gen_result = generate(normalized, scored_chunks, settings)
            except GenerationError as exc:
                generation_info["provider"] = exc.provider_name
                generation_info["output_summary"] = "generation failed"
                generation_info["error"] = f"{type(exc).__name__}: {exc}"
                trace.error = generation_info["error"]
                raise QueryGenerationError(str(exc), trace.trace_id) from exc
            generation_info["provider"] = gen_result.provider_name
            generation_info["output_summary"] = gen_result.output_summary
            generation_info["candidate_count"] = gen_result.candidate_count

        return AskResult(
            answer=gen_result.answer,
            citations=gen_result.citations,
            trace_id=trace.trace_id,
            refused=gen_result.refused,
            ranked_chunks=scored_chunks,
        )
    finally:
        save_trace(settings, trace)
