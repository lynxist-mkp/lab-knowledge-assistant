"""提问编排 — shared four-phase query orchestration for ask and eval."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lab_knowledge.components.model_guard import ModelResource
from lab_knowledge.components.model_guard import phase_batch as model_phase_batch
from lab_knowledge.config import Settings
from lab_knowledge.generation import (
    GenerationError,
    GenerationResult,
    QueryGenerationError,
    generate,
)
from lab_knowledge.generation.expand import expand_for_generation
from lab_knowledge.knowledge import Knowledge, create_knowledge
from lab_knowledge.models import AskResult, ScoredChunk
from lab_knowledge.query_processing.extras import prepare_query_extras
from lab_knowledge.retrieval import retrieve
from lab_knowledge.retrieval.fusion import RetrievalResult
from lab_knowledge.retrieval.retrieve import (
    attach_retrieval_trace_stages,
    rerank_chunks,
    resolve_retrieval_mode,
)
from lab_knowledge.tracing.ask_payload import AskOutcome, AskTracePayload
from lab_knowledge.tracing.query_trace import QueryTrace

if TYPE_CHECKING:
    from lab_knowledge.pipelines.query_batch import _AskJob

logger = logging.getLogger(__name__)

_DENSE_MODES = frozenset({"dense_only", "rrf"})

GenerateFn = Callable[[str, list[ScoredChunk], Settings], GenerationResult]


def normalize_question(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question.strip())
    return collapsed


def prepare_generation_context(
    scored_chunks: list[ScoredChunk],
    knowledge: Knowledge,
    settings: Settings,
) -> tuple[list[ScoredChunk], list[str], list[str]]:
    """Expand ranked hits with neighbor chunks before generation."""
    return expand_for_generation(
        scored_chunks,
        knowledge,
        settings.retrieval.adjacent_n,
    )


def _needs_dense_embedding(mode: str) -> bool:
    return mode in _DENSE_MODES


@dataclass
class OrchestrationWork:
    """One item flowing through the four-phase orchestration pipeline."""

    normalized: str
    settings: Settings
    culture_domain: str | None = None
    retrieval_mode: str | None = None
    rerank_enabled: bool | None = None
    knowledge: Knowledge | None = None
    extra_queries: list[str] = field(default_factory=list)
    term_extras: list[str] = field(default_factory=list)
    multi_query_extras: list[str] = field(default_factory=list)
    collect_extras: bool = False
    skip_retrieval: bool = False
    pre_chunks: list[ScoredChunk] | None = None
    rewriter_provider_name: str = "none"

    retrieval_result: RetrievalResult | None = None
    chunks: list[ScoredChunk] | None = None
    rerank_stages: list = field(default_factory=list)
    expanded_chunks: list[ScoredChunk] | None = None
    expanded_from: list[str] = field(default_factory=list)
    expanded_chunk_ids: list[str] = field(default_factory=list)
    generation: GenerationResult | None = None
    generation_error: object | None = None
    extras_elapsed_ms: float = 0.0

    _do_rerank: bool = field(default=False, repr=False)


@dataclass
class AskWorkContext:
    """Per-ask metadata for trace assembly."""

    question: str
    record_trace: bool = True
    batch_id: str | None = None
    batch_size: int = 1
    batch_wait_ms: float = 0.0


@dataclass
class AskWorkOutcome:
    context: AskWorkContext
    result: AskResult | None = None
    error: BaseException | None = None


def ask_outcome_from_work(
    work: OrchestrationWork,
    context: AskWorkContext,
) -> AskOutcome:
    """Build the stable trace seam outcome after orchestration phases."""
    assert work.retrieval_result is not None
    generation_error = (
        work.generation_error
        if isinstance(work.generation_error, GenerationError)
        else None
    )
    return AskOutcome(
        question=context.question,
        culture_domain=work.culture_domain,
        payload=AskTracePayload(
            settings=work.settings,
            normalized=work.normalized,
            retrieval_result=work.retrieval_result,
            extras_elapsed_ms=work.extras_elapsed_ms,
            term_extras=list(work.term_extras),
            multi_query_extras=list(work.multi_query_extras),
            rewriter_provider_name=work.rewriter_provider_name,
            rerank_stages=list(work.rerank_stages),
            chunks=list(work.chunks or []),
            expanded_chunks=list(work.expanded_chunks or []),
            expanded_from=list(work.expanded_from),
            expanded_chunk_ids=list(work.expanded_chunk_ids),
            generation=work.generation,
        ),
        generation_error=generation_error,
    )


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


def ask_work_from_job(job: _AskJob | _SingleJob) -> OrchestrationWork:
    """Build orchestration work for an ask job (internal work bag)."""
    return OrchestrationWork(
        normalized=normalize_question(job.question),
        settings=job.settings,
        culture_domain=job.culture_domain,
        retrieval_mode=job.retrieval_mode,
        rerank_enabled=job.rerank_enabled,
        knowledge=job.knowledge,
        collect_extras=True,
    )


def eval_work_from_item(
    *,
    question: str,
    settings: Settings,
    retrieval_mode: str,
    rerank_enabled: bool,
    knowledge: Knowledge,
    query_rewrite: bool = False,
    existing_chunks: list[ScoredChunk] | None = None,
) -> OrchestrationWork:
    """Build orchestration work for one eval item (optional gen-only retry)."""
    return OrchestrationWork(
        normalized=normalize_question(question),
        settings=settings,
        retrieval_mode=retrieval_mode,
        rerank_enabled=rerank_enabled,
        knowledge=knowledge,
        collect_extras=query_rewrite,
        skip_retrieval=existing_chunks is not None,
        pre_chunks=existing_chunks,
    )


def gen_retry_work(
    *,
    question: str,
    settings: Settings,
    retrieval_mode: str,
    rerank_enabled: bool,
    knowledge: Knowledge,
    pre_chunks: list[ScoredChunk],
    query_rewrite: bool = False,
) -> OrchestrationWork:
    """Build gen-only retry work from already-ranked chunks."""
    return eval_work_from_item(
        question=question,
        settings=settings,
        retrieval_mode=retrieval_mode,
        rerank_enabled=rerank_enabled,
        knowledge=knowledge,
        query_rewrite=query_rewrite,
        existing_chunks=pre_chunks,
    )


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


class ExtrasPhase:
    def run(
        self,
        works: list[OrchestrationWork],
        *,
        phase_batch: bool,
    ) -> None:
        retrieval_works = [work for work in works if not work.skip_retrieval]
        if not retrieval_works:
            return
        any_multi_query = any(
            work.collect_extras and work.settings.query_processing.multi_query
            for work in retrieval_works
        )
        with model_phase_batch(ModelResource.MLX_VLM, phase_batch and any_multi_query):
            for work in retrieval_works:
                if work.collect_extras:
                    extras = prepare_query_extras(work.normalized, work.settings)
                    work.term_extras = extras.term_extras
                    work.multi_query_extras = extras.multi_query_extras
                    work.extra_queries = extras.combined
                    work.extras_elapsed_ms = extras.extras_elapsed_ms
                    work.rewriter_provider_name = extras.rewriter_provider_name


class RetrievalPhase:
    def run(
        self,
        works: list[OrchestrationWork],
        *,
        phase_batch: bool,
        tolerate_errors: bool,
    ) -> None:
        retrieval_works = [work for work in works if not work.skip_retrieval]
        if not retrieval_works:
            return
        modes = [
            resolve_retrieval_mode(
                work.settings, work.retrieval_mode, work.rerank_enabled
            )
            for work in retrieval_works
        ]
        needs_bge = any(_needs_dense_embedding(mode) for mode, _ in modes)
        with model_phase_batch(ModelResource.BGE_M3, phase_batch and needs_bge):
            for work, (mode, do_rerank) in zip(
                retrieval_works, modes, strict=True
            ):
                work._do_rerank = do_rerank
                try:
                    knowledge = work.knowledge or create_knowledge(work.settings)
                    fusion = retrieve(
                        work.normalized,
                        work.settings,
                        culture_domain=work.culture_domain,
                        retrieval_mode=mode,
                        knowledge=knowledge,
                        extra_queries=work.extra_queries,
                    )
                    work.retrieval_result = attach_retrieval_trace_stages(
                        fusion,
                        work.settings,
                        knowledge=knowledge,
                        culture_domain=work.culture_domain,
                    )
                    work.chunks = work.retrieval_result.chunks
                except Exception:
                    if tolerate_errors:
                        logger.warning(
                            "orchestration retrieval failed question=%r mode=%s",
                            work.normalized,
                            mode,
                            exc_info=True,
                        )
                        work.chunks = None
                    else:
                        raise


class RerankPhase:
    def run(
        self,
        works: list[OrchestrationWork],
        *,
        phase_batch: bool,
        tolerate_errors: bool,
    ) -> None:
        retrieval_works = [work for work in works if not work.skip_retrieval]
        if not retrieval_works:
            return
        any_rerank = any(
            work._do_rerank and work.chunks is not None for work in retrieval_works
        )
        with model_phase_batch(ModelResource.CROSS_ENCODER, phase_batch and any_rerank):
            for work in retrieval_works:
                if work.chunks is None or not work._do_rerank:
                    continue
                try:
                    reranked, rerank_stage = rerank_chunks(
                        work.settings,
                        work.normalized,
                        work.chunks,
                    )
                    work.chunks = reranked
                    if rerank_stage is not None:
                        work.rerank_stages.append(rerank_stage)
                except Exception:
                    if tolerate_errors:
                        logger.warning(
                            "orchestration rerank failed question=%r",
                            work.normalized,
                            exc_info=True,
                        )
                        work.chunks = None
                    else:
                        raise


class GenerationPhase:
    def run(
        self,
        works: list[OrchestrationWork],
        *,
        phase_batch: bool,
        tolerate_errors: bool = False,
        generate_fn: GenerateFn | None = None,
    ) -> None:
        pending = [work for work in works if work.chunks is not None]
        if not pending:
            return

        gen = generate_fn or generate
        with model_phase_batch(ModelResource.MLX_VLM, phase_batch):
            for work in pending:
                knowledge = work.knowledge or create_knowledge(work.settings)
                expanded, expanded_from, expanded_chunk_ids = prepare_generation_context(
                    work.chunks,
                    knowledge,
                    work.settings,
                )
                work.expanded_chunks = expanded
                work.expanded_from = expanded_from
                work.expanded_chunk_ids = expanded_chunk_ids
                try:
                    work.generation = gen(
                        work.normalized,
                        expanded,
                        work.settings,
                    )
                except GenerationError as exc:
                    work.generation_error = exc
                    work.generation = None
                except Exception:
                    if tolerate_errors:
                        logger.warning(
                            "orchestration generation failed question=%r",
                            work.normalized,
                            exc_info=True,
                        )
                        work.generation = None
                    else:
                        raise


_EXTRAS_PHASE = ExtrasPhase()
_RETRIEVAL_PHASE = RetrievalPhase()
_RERANK_PHASE = RerankPhase()
_GENERATION_PHASE = GenerationPhase()


def _run_phases(
    works: list[OrchestrationWork],
    *,
    phase_batch: bool,
    tolerate_retrieval_errors: bool = False,
) -> None:
    if not works:
        return
    for work in works:
        if work.skip_retrieval:
            work.chunks = work.pre_chunks
    _EXTRAS_PHASE.run(works, phase_batch=phase_batch)
    _RETRIEVAL_PHASE.run(
        works,
        phase_batch=phase_batch,
        tolerate_errors=tolerate_retrieval_errors,
    )
    _RERANK_PHASE.run(
        works,
        phase_batch=phase_batch,
        tolerate_errors=tolerate_retrieval_errors,
    )


def _run_generation_phase(
    works: list[OrchestrationWork],
    *,
    phase_batch: bool,
    tolerate_generation_errors: bool = False,
    generate_fn: GenerateFn | None = None,
) -> None:
    _GENERATION_PHASE.run(
        works,
        phase_batch=phase_batch,
        tolerate_errors=tolerate_generation_errors,
        generate_fn=generate_fn,
    )


def run_eval_works(
    works: list[OrchestrationWork],
    *,
    phase_batch: bool,
    tolerate_errors: bool = True,
    generate_fn: GenerateFn | None = None,
) -> list[tuple[list[ScoredChunk] | None, GenerationResult | None]]:
    """评测入口：四阶段编排，不写 Trace。"""
    _run_phases(works, phase_batch=phase_batch, tolerate_retrieval_errors=tolerate_errors)
    _run_generation_phase(
        works,
        phase_batch=phase_batch,
        tolerate_generation_errors=tolerate_errors,
        generate_fn=generate_fn,
    )
    return [(work.chunks, work.generation) for work in works]


def run_ask_works(
    pairs: list[tuple[OrchestrationWork, AskWorkContext]],
    *,
    phase_batch: bool,
) -> list[AskWorkOutcome]:
    """提问入口：四阶段编排 + QueryTrace 组装 AskResult。"""
    works = [work for work, _ in pairs]
    _run_phases(works, phase_batch=phase_batch, tolerate_retrieval_errors=False)
    _run_generation_phase(works, phase_batch=phase_batch, tolerate_generation_errors=False)

    outcomes: list[AskWorkOutcome] = []
    for work, context in pairs:
        outcome = AskWorkOutcome(context=context)
        trace = QueryTrace.begin(
            question=context.question,
            batch_id=context.batch_id,
            batch_size=context.batch_size,
            batch_wait_ms=context.batch_wait_ms,
        )
        ask_outcome = ask_outcome_from_work(work, context)
        try:
            outcome.result = trace.finalize(ask_outcome)
        except QueryGenerationError as exc:
            outcome.error = exc
            if context.record_trace:
                trace.save(ask_outcome.payload.settings)
            outcomes.append(outcome)
            continue
        except BaseException as exc:
            outcome.error = exc
            if context.record_trace:
                trace.save(ask_outcome.payload.settings)
            outcomes.append(outcome)
            continue

        if context.record_trace:
            trace.save(ask_outcome.payload.settings)
        outcomes.append(outcome)
    return outcomes


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

    pairs: list[tuple[OrchestrationWork, AskWorkContext]] = []
    for job in jobs:
        work = ask_work_from_job(job)
        context = AskWorkContext(
            question=job.question,
            record_trace=job.record_trace,
            batch_id=job.batch_id,
            batch_size=job.batch_size,
            batch_wait_ms=job.batch_wait_ms,
        )
        pairs.append((work, context))

    outcomes = run_ask_works(pairs, phase_batch=phase_batch)
    for job, outcome in zip(jobs, outcomes, strict=True):
        job.result = outcome.result
        job.error = outcome.error


__all__ = [
    "AskPipelineInput",
    "AskWorkContext",
    "AskWorkOutcome",
    "ask_outcome_from_work",
    "ExtrasPhase",
    "GenerationPhase",
    "RerankPhase",
    "RetrievalPhase",
    "ask_pipeline_single",
    "ask_work_from_job",
    "eval_work_from_item",
    "gen_retry_work",
    "normalize_question",
    "prepare_generation_context",
    "run_ask_pipeline",
    "run_ask_works",
    "run_eval_works",
    "_run_generation_phase",
    "_run_phases",
]
