"""AskOutcome seam: 提问编排 -> outcome and QueryTrace <- outcome."""

from __future__ import annotations

import pytest

from wenmai.config import Settings
from wenmai.generation import GenerationError, GenerationResult, QueryGenerationError
from wenmai.models import Chunk, Citation, ScoredChunk
from wenmai.pipelines.query_orchestration import (
    AskWorkContext,
    OrchestrationWork,
    ask_outcome_from_work,
)
from wenmai.retrieval.fusion import RetrievalResult
from wenmai.tracing.ask_payload import AskOutcome, AskTracePayload
from wenmai.tracing.query_trace import QueryTrace
from wenmai.tracing.store import get_trace_record


def _scored_chunk(chunk_id: str = "doc:0001") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, document_id="doc", text="妈祖信仰发源地"),
        score=0.9,
    )


def _trace_payload(
    test_settings: Settings,
    *,
    generation: GenerationResult | None = None,
) -> AskTracePayload:
    chunk = _scored_chunk()
    return AskTracePayload(
        settings=test_settings,
        normalized="妈祖 信仰 发源地",
        retrieval_result=RetrievalResult(chunks=[chunk], mode="sparse_only"),
        chunks=[chunk],
        expanded_chunks=[chunk],
        expanded_from=["doc:0001"],
        expanded_chunk_ids=["doc:0001"],
        generation=generation,
    )


def _generation_result() -> GenerationResult:
    citation = Citation(
        index=1,
        chunk_id="doc:0001",
        document_id="doc",
        title="简介",
        excerpt="湄洲岛是妈祖信仰的发源地",
    )
    return GenerationResult(
        answer="湄洲岛是妈祖信仰的发源地。[1]",
        refused=False,
        citations=[citation],
        provider_name="fake",
        output_summary="answer text",
        candidate_count=1,
    )


def test_ask_outcome_from_work_success(test_settings: Settings) -> None:
    work = OrchestrationWork(
        normalized="妈祖 信仰 发源地",
        settings=test_settings,
        culture_domain="妈祖",
        retrieval_result=RetrievalResult(chunks=[_scored_chunk()], mode="sparse_only"),
        chunks=[_scored_chunk()],
        expanded_chunks=[_scored_chunk()],
        expanded_from=["doc:0001"],
        expanded_chunk_ids=["doc:0001"],
        generation=_generation_result(),
    )
    context = AskWorkContext(question="妈祖信仰的发源地在哪里？")

    outcome = ask_outcome_from_work(work, context)

    assert outcome.question == "妈祖信仰的发源地在哪里？"
    assert outcome.culture_domain == "妈祖"
    assert outcome.generation_error is None
    assert outcome.payload.normalized == "妈祖 信仰 发源地"
    assert outcome.payload.generation is not None


def test_ask_outcome_from_work_generation_failure(test_settings: Settings) -> None:
    gen_error = GenerationError("provider down", provider_name="fake")
    work = OrchestrationWork(
        normalized="妈祖 信仰 发源地",
        settings=test_settings,
        retrieval_result=RetrievalResult(chunks=[_scored_chunk()], mode="sparse_only"),
        chunks=[_scored_chunk()],
        expanded_chunks=[_scored_chunk()],
        generation_error=gen_error,
    )
    context = AskWorkContext(question="妈祖信仰的发源地在哪里？")

    outcome = ask_outcome_from_work(work, context)

    assert outcome.generation_error is gen_error
    assert outcome.payload.generation is None


def test_query_trace_finalize_success(test_settings: Settings) -> None:
    outcome = AskOutcome(
        question="妈祖信仰的发源地在哪里？",
        culture_domain="妈祖",
        payload=_trace_payload(test_settings, generation=_generation_result()),
    )
    trace = QueryTrace.begin(question=outcome.question)

    result = trace.finalize(outcome)

    assert result.answer.startswith("湄洲岛")
    assert result.trace_id == trace.trace_id
    assert trace._recorder.error is None


def test_query_trace_finalize_generation_failure(test_settings: Settings) -> None:
    gen_error = GenerationError("provider down", provider_name="fake")
    outcome = AskOutcome(
        question="妈祖信仰的发源地在哪里？",
        culture_domain="妈祖",
        payload=_trace_payload(test_settings),
        generation_error=gen_error,
    )
    trace = QueryTrace.begin(question=outcome.question)

    with pytest.raises(QueryGenerationError) as exc_info:
        trace.finalize(outcome)

    assert exc_info.value.trace_id == trace.trace_id
    assert trace._recorder.error is not None
    assert "provider down" in trace._recorder.error


def test_query_trace_finalize_generation_failure_persists_trace(
    test_settings: Settings,
) -> None:
    gen_error = GenerationError("provider down", provider_name="fake")
    outcome = AskOutcome(
        question="妈祖信仰的发源地在哪里？",
        culture_domain="妈祖",
        payload=_trace_payload(test_settings),
        generation_error=gen_error,
    )
    trace = QueryTrace.begin(question=outcome.question)

    with pytest.raises(QueryGenerationError):
        trace.finalize(outcome)
    trace.save(test_settings)

    record = get_trace_record(test_settings, trace.trace_id)
    assert record is not None
    assert record["error"] is not None
