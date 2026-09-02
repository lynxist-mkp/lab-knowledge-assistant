"""提问编排 — shared four-phase query orchestration for ask and eval."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from wenmai.components.model_guard import ModelResource, begin_batch, end_batch
from wenmai.config import Settings
from wenmai.generation.expand import expand_for_generation
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import ScoredChunk
from wenmai.query_processing.extras import CollectedExtras, collect_extra_queries
from wenmai.retrieval import retrieve
from wenmai.retrieval.fusion import RetrievalResult
from wenmai.retrieval.retrieve import rerank_chunks, resolve_retrieval_mode

if TYPE_CHECKING:
    from wenmai.generation import GenerationResult

logger = logging.getLogger(__name__)

_DENSE_MODES = frozenset({"dense_only", "rrf"})


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

    retrieval_result: RetrievalResult | None = None
    chunks: list[ScoredChunk] | None = None
    rerank_stages: list = field(default_factory=list)
    expanded_chunks: list[ScoredChunk] | None = None
    expanded_from: list[str] = field(default_factory=list)
    expanded_chunk_ids: list[str] = field(default_factory=list)
    generation: GenerationResult | None = None

    _do_rerank: bool = field(default=False, repr=False)


def _collect_extras_for_work(work: OrchestrationWork, *, phase_batch: bool) -> None:
    extras = collect_extra_queries(work.normalized, work.settings)
    work.term_extras = extras.term_extras
    work.multi_query_extras = extras.multi_query_extras
    work.extra_queries = extras.combined


def _run_phases(
    works: list[OrchestrationWork],
    *,
    phase_batch: bool,
    tolerate_retrieval_errors: bool = False,
) -> None:
    if not works:
        return

    retrieval_works = [work for work in works if not work.skip_retrieval]
    for work in works:
        if work.skip_retrieval:
            work.chunks = work.pre_chunks

    if retrieval_works:
        any_collect = any(work.collect_extras for work in retrieval_works)
        any_multi_query = any(
            work.collect_extras and work.settings.query_processing.multi_query
            for work in retrieval_works
        )
        with _phase(ModelResource.MLX_VLM, phase_batch and any_multi_query):
            for work in retrieval_works:
                if work.collect_extras:
                    _collect_extras_for_work(work, phase_batch=False)

        modes = [
            resolve_retrieval_mode(
                work.settings, work.retrieval_mode, work.rerank_enabled
            )
            for work in retrieval_works
        ]
        needs_bge = any(_needs_dense_embedding(mode) for mode, _ in modes)
        with _phase(ModelResource.BGE_M3, phase_batch and needs_bge):
            for work, (mode, do_rerank) in zip(
                retrieval_works, modes, strict=True
            ):
                work._do_rerank = do_rerank
                try:
                    result = retrieve(
                        work.normalized,
                        work.settings,
                        culture_domain=work.culture_domain,
                        retrieval_mode=mode,
                        rerank_enabled=False,
                        knowledge=work.knowledge,
                        extra_queries=work.extra_queries,
                    )
                    work.retrieval_result = result
                    work.chunks = result.chunks
                except Exception:
                    if tolerate_retrieval_errors:
                        logger.warning(
                            "orchestration retrieval failed question=%r mode=%s",
                            work.normalized,
                            mode,
                            exc_info=True,
                        )
                        work.chunks = None
                    else:
                        raise

        any_rerank = any(
            work._do_rerank and work.chunks is not None for work in retrieval_works
        )
        with _phase(ModelResource.CROSS_ENCODER, phase_batch and any_rerank):
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
                    if tolerate_retrieval_errors:
                        logger.warning(
                            "orchestration rerank failed question=%r",
                            work.normalized,
                            exc_info=True,
                        )
                        work.chunks = None
                    else:
                        raise


def _run_generation_phase(
    works: list[OrchestrationWork],
    *,
    phase_batch: bool,
    tolerate_generation_errors: bool = False,
    generate_fn,
) -> None:
    pending = [work for work in works if work.chunks is not None]
    if not pending:
        return

    with _phase(ModelResource.MLX_VLM, phase_batch):
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
                work.generation = generate_fn(
                    work.normalized,
                    expanded,
                    work.settings,
                )
            except Exception:
                if tolerate_generation_errors:
                    logger.warning(
                        "orchestration generation failed question=%r",
                        work.normalized,
                        exc_info=True,
                    )
                    work.generation = None
                else:
                    raise


__all__ = [
    "OrchestrationWork",
    "prepare_generation_context",
    "_run_generation_phase",
    "_run_phases",
]
