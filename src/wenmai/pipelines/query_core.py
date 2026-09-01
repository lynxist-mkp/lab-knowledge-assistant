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
from wenmai.generation.expand import expand_for_generation
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import AskResult
from wenmai.query_processing import multi_query
from wenmai.query_processing.extras import CollectedExtras
from wenmai.retrieval import retrieve
from wenmai.retrieval.retrieve import rerank_chunks, resolve_retrieval_mode
from wenmai.tracing import TraceContext, save_trace
from wenmai.tracing.stages.query import QueryStage

if TYPE_CHECKING:
    from wenmai.pipelines.query_batch import _AskJob


_DENSE_MODES = frozenset({"dense_only", "rrf"})


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


def _needs_dense_embedding(mode: str) -> bool:
    return mode in _DENSE_MODES


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

    settings = jobs[0].settings
    if batch_meta:
        batch_id = str(batch_meta["batch_id"])
        batch_size = int(batch_meta["batch_size"])  # type: ignore[arg-type]
        for job in jobs:
            job.batch_id = batch_id
            job.batch_size = batch_size

    # Phase 1: multi-query (MLX)
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

    # Phase 2: retrieval embeddings (BGE when dense path is used)
    modes = [
        resolve_retrieval_mode(
            job.settings, job.retrieval_mode, job.rerank_enabled
        )
        for job in jobs
    ]
    needs_bge = any(_needs_dense_embedding(mode) for mode, _ in modes)
    retrieved_by_job: list[tuple[object, bool, CollectedExtras]] = []
    with _phase(ModelResource.BGE_M3, phase_batch and needs_bge):
        for job, (mode, do_rerank), extras in zip(jobs, modes, extras_by_job, strict=True):
            retrieved = retrieve(
                normalize_question(job.question),
                job.settings,
                culture_domain=job.culture_domain,
                retrieval_mode=mode,
                rerank_enabled=False,
                knowledge=job.knowledge,
                extra_queries=extras.combined,
            )
            retrieved_by_job.append((retrieved, do_rerank, extras))

    # Phase 3: rerank (CrossEncoder)
    any_rerank = any(do_rerank for _, do_rerank, _ in retrieved_by_job)
    reranked_by_job: list[tuple[list, list]] = []
    with _phase(ModelResource.CROSS_ENCODER, phase_batch and any_rerank):
        for job, (retrieved, do_rerank, _extras) in zip(
            jobs, retrieved_by_job, strict=True
        ):
            chunks = retrieved.chunks
            extra_stages: list = []
            if do_rerank:
                chunks, rerank_stage = rerank_chunks(
                    job.settings,
                    normalize_question(job.question),
                    chunks,
                )
                if rerank_stage is not None:
                    extra_stages.append(rerank_stage)
            reranked_by_job.append((chunks, extra_stages))

    # Phase 4: generation (MLX) — per job trace assembly
    with _phase(ModelResource.MLX_VLM, phase_batch):
        for job, (retrieved, _do_rerank, extras), (scored_chunks, rerank_stages) in zip(
            jobs, retrieved_by_job, reranked_by_job, strict=True
        ):
            try:
                job.result = _finish_job(
                    job,
                    extras=extras,
                    retrieved=retrieved,
                    rerank_stages=rerank_stages,
                    scored_chunks=scored_chunks,
                )
            except BaseException as exc:
                job.error = exc


def _finish_job(
    job: _AskJob | _SingleJob,
    *,
    extras: CollectedExtras,
    retrieved: object,
    rerank_stages: list,
    scored_chunks: list,
) -> AskResult:
    settings = job.settings
    question = job.question
    normalized = normalize_question(question)
    trace = TraceContext(trace_type="query", metadata={"question": question})
    if job.batch_id:
        trace.metadata["batch_id"] = job.batch_id
        trace.metadata["batch_size"] = job.batch_size
        trace.metadata["batch_wait_ms"] = round(job.batch_wait_ms, 1)

    try:
        from wenmai.factories import query_rewrite as query_rewrite_factory

        rewriter = query_rewrite_factory.create(settings)
        processing_started = time.perf_counter()
        mq_provider = (
            settings.providers.multimodal
            if settings.query_processing.multi_query
            else None
        )
        trace.append_stage(
            QueryStage.query_processing(
                question=question,
                normalized=normalized,
                elapsed_ms=(time.perf_counter() - processing_started) * 1000,
                culture_domain=job.culture_domain,
                term_extras=extras.term_extras,
                multi_query_extras=extras.multi_query_extras,
                rewriter=rewriter.provider_name,
                multi_query_provider=mq_provider,
            )
        )

        for stage in retrieved.stages:
            trace.append_stage(stage)
        for stage in rerank_stages:
            trace.append_stage(stage)

        knowledge = job.knowledge or create_knowledge(settings)
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
        if job.record_trace:
            save_trace(settings, trace)


__all__ = [
    "AskPipelineInput",
    "ask_pipeline_single",
    "normalize_question",
    "run_ask_pipeline",
]
