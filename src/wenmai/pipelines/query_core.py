"""Shared ask pipeline with request-internal phase batching (Strategy A)."""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from wenmai.components.model_guard import ModelResource, begin_batch, end_batch
from wenmai.config import Settings
from wenmai.generation import GenerationError, QueryGenerationError, generate
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import AskResult
from wenmai.pipelines.query_orchestration import (
    OrchestrationWork,
    _run_phases,
    prepare_generation_context,
)
from wenmai.query_processing import multi_query
from wenmai.query_processing.extras import CollectedExtras
from wenmai.tracing import TraceRecorder

if TYPE_CHECKING:
    from wenmai.pipelines.query_batch import _AskJob


def normalize_question(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question.strip())
    return collapsed


@contextmanager
def _phase(resource: ModelResource, enabled: bool) -> Iterator[None]:
    if enabled:
        begin_batch(resource)
    try:
        yield
    finally:
        if enabled:
            end_batch()


def _collect_extras(
    question: str,
    settings: Settings,
    *,
    phase_batch: bool,
) -> CollectedExtras:
    from wenmai.factories import query_rewrite as query_rewrite_factory

    rewriter = query_rewrite_factory.create(settings)
    term_extras = rewriter.extra_queries(question)
    mq_extras: list[str] = []
    if settings.query_processing.multi_query:
        with _phase(ModelResource.MLX_VLM, phase_batch):
            mq_extras = multi_query.expand(question, settings)
    return CollectedExtras(term_extras=term_extras, multi_query_extras=mq_extras)


@dataclass
class AskPipelineInput:
    question: str
    settings: Settings
    culture_domain: str | None = None
    retrieval_mode: str | None = None
    rerank_enabled: bool | None = None
    knowledge: Knowledge | None = None
    record_trace: bool = True
    batch_id: str | None = None
    batch_size: int = 1
    batch_wait_ms: float = 0.0


def ask_pipeline_single(
    payload: AskPipelineInput,
    *,
    phase_batch: bool,
) -> AskResult:
    job_like = _SingleJob(payload)
    run_ask_pipeline([job_like], phase_batch=phase_batch, batch_meta=None)
    if job_like.error is not None:
        raise job_like.error
    assert job_like.result is not None
    return job_like.result


@dataclass
class _SingleJob:
    """Minimal job adapter for single-request pipeline runs."""

    payload: AskPipelineInput
    result: AskResult | None = None
    error: BaseException | None = None
    batch_id: str | None = None
    batch_size: int = 1
    batch_wait_ms: float = 0.0

    @property
    def question(self) -> str:
        return self.payload.question

    @property
    def settings(self) -> Settings:
        return self.payload.settings

    @property
    def culture_domain(self) -> str | None:
        return self.payload.culture_domain

    @property
    def retrieval_mode(self) -> str | None:
        return self.payload.retrieval_mode

    @property
    def rerank_enabled(self) -> bool | None:
        return self.payload.rerank_enabled

    @property
    def knowledge(self) -> Knowledge | None:
        return self.payload.knowledge

    @property
    def record_trace(self) -> bool:
        return self.payload.record_trace


def run_ask_pipeline(
    jobs: list[_AskJob | _SingleJob],
    *,
    phase_batch: bool,
    batch_meta: dict[str, object] | None,
) -> None:
    if not jobs:
        return

    if batch_meta:
        batch_id = str(batch_meta["batch_id"])
        batch_size = int(batch_meta["batch_size"])  # type: ignore[arg-type]
        for job in jobs:
            job.batch_id = batch_id
            job.batch_size = batch_size

    any_multi_query = any(job.settings.query_processing.multi_query for job in jobs)
    with _phase(ModelResource.MLX_VLM, phase_batch and any_multi_query):
        extras_by_job: list[CollectedExtras] = []
        for job in jobs:
            extras_by_job.append(
                _collect_extras(
                    normalize_question(job.question),
                    job.settings,
                    phase_batch=False,
                )
            )

    works: list[OrchestrationWork] = []
    for job, extras in zip(jobs, extras_by_job, strict=True):
        works.append(
            OrchestrationWork(
                normalized=normalize_question(job.question),
                settings=job.settings,
                culture_domain=job.culture_domain,
                retrieval_mode=job.retrieval_mode,
                rerank_enabled=job.rerank_enabled,
                knowledge=job.knowledge,
                extra_queries=extras.combined,
                term_extras=extras.term_extras,
                multi_query_extras=extras.multi_query_extras,
                collect_extras=False,
            )
        )

    _run_phases(works, phase_batch=phase_batch, tolerate_retrieval_errors=False)

    with _phase(ModelResource.MLX_VLM, phase_batch):
        for job, work in zip(jobs, works, strict=True):
            try:
                job.result = _finish_job(job, work=work)
            except BaseException as exc:
                job.error = exc


def _finish_job(
    job: _AskJob | _SingleJob,
    *,
    work: OrchestrationWork,
) -> AskResult:
    settings = job.settings
    question = job.question
    normalized = work.normalized
    trace = TraceRecorder(
        question=question,
        batch_id=job.batch_id,
        batch_size=job.batch_size,
        batch_wait_ms=job.batch_wait_ms,
    )

    try:
        from wenmai.factories import query_rewrite as query_rewrite_factory

        rewriter = query_rewrite_factory.create(settings)
        processing_started = time.perf_counter()
        mq_provider = (
            settings.providers.multimodal
            if settings.query_processing.multi_query
            else None
        )
        trace.record_query_processing(
            question=question,
            normalized=normalized,
            elapsed_ms=(time.perf_counter() - processing_started) * 1000,
            culture_domain=job.culture_domain,
            term_extras=work.term_extras,
            multi_query_extras=work.multi_query_extras,
            rewriter=rewriter.provider_name,
            multi_query_provider=mq_provider,
        )

        assert work.retrieval_result is not None
        trace.append_retrieval_stages(work.retrieval_result.stages)
        trace.append_rerank_stages(work.rerank_stages)

        scored_chunks = work.chunks or []
        knowledge = job.knowledge or create_knowledge(settings)
        expanded_chunks, expanded_from, expanded_chunk_ids = prepare_generation_context(
            scored_chunks,
            knowledge,
            settings,
        )

        generation_started = time.perf_counter()
        generation_input = f"{len(expanded_chunks)} chunks"
        try:
            gen_result = generate(normalized, expanded_chunks, settings)
        except GenerationError as exc:
            generation_error = f"{type(exc).__name__}: {exc}"
            trace.record_generation(
                provider=exc.provider_name,
                elapsed_ms=(time.perf_counter() - generation_started) * 1000,
                input_summary=generation_input,
                output_summary="generation failed",
                candidate_count=0,
                error=generation_error,
                expanded_from=expanded_from,
                expanded_chunk_ids=expanded_chunk_ids,
            )
            trace.error = generation_error
            raise QueryGenerationError(str(exc), trace.trace_id) from exc

        trace.record_generation(
            provider=gen_result.provider_name,
            elapsed_ms=(time.perf_counter() - generation_started) * 1000,
            input_summary=generation_input,
            output_summary=gen_result.output_summary,
            candidate_count=gen_result.candidate_count,
            expanded_from=expanded_from,
            expanded_chunk_ids=expanded_chunk_ids,
        )

        trace.set_outcome(
            refused=gen_result.refused,
            refusal_reason=gen_result.refusal_reason,
            citation_count=len(gen_result.citations),
        )

        return AskResult(
            answer=gen_result.answer,
            citations=gen_result.citations,
            trace_id=trace.trace_id,
            refused=gen_result.refused,
            refusal_reason=gen_result.refusal_reason,
            ranked_chunks=scored_chunks,
        )
    finally:
        if job.record_trace:
            trace.save(settings)


__all__ = [
    "AskPipelineInput",
    "ask_pipeline_single",
    "normalize_question",
    "run_ask_pipeline",
]
