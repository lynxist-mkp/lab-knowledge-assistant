from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from wenmai.components.model_guard import ModelResource, begin_batch, end_batch
from wenmai.config import Settings
from wenmai.eval.golden import GoldItem
from wenmai.eval.metrics import corpus_doc_ids_from_chunks, retrieval_item_snapshot
from wenmai.generation import GenerationError, GenerationResult, generate
from wenmai.knowledge import Knowledge
from wenmai.models import ScoredChunk
from wenmai.pipelines.query import normalize_question
from wenmai.query_processing.extras import collect_extra_queries
from wenmai.retrieval import retrieve
from wenmai.retrieval.retrieve import rerank_chunks, resolve_retrieval_mode

logger = logging.getLogger(__name__)

_DENSE_MODES = frozenset({"dense_only", "rrf"})


@dataclass(frozen=True)
class EvalItemResult:
    """Outcome of evaluating one golden item under one ablation group."""

    chunks: list[ScoredChunk] | None
    generation: GenerationResult | None

    @property
    def ok(self) -> bool:
        return self.chunks is not None and self.generation is not None

    @property
    def ranked_doc_ids(self) -> list[str]:
        if self.chunks is None:
            return []
        return corpus_doc_ids_from_chunks(self.chunks)

    def retrieval_snapshot(self) -> dict[str, Any]:
        if self.chunks is None:
            return {}
        return retrieval_item_snapshot(self.chunks)


def eval_item(
    item: GoldItem,
    settings: Settings,
    *,
    retrieval_mode: str,
    rerank_enabled: bool,
    knowledge: Knowledge,
    retrieved_chunks: list[ScoredChunk] | None = None,
    query_rewrite: bool = False,
) -> EvalItemResult:
    """Retrieve once (unless chunks supplied), then generate for one golden item."""
    normalized = normalize_question(item.question)

    if retrieved_chunks is not None:
        chunks = retrieved_chunks
    else:
        extra_queries: list[str]
        if query_rewrite:
            extra_queries = collect_extra_queries(normalized, settings).combined
        else:
            extra_queries = []
        try:
            result = retrieve(
                normalized,
                settings,
                retrieval_mode=retrieval_mode,
                rerank_enabled=rerank_enabled,
                knowledge=knowledge,
                extra_queries=extra_queries,
            )
            chunks = result.chunks
        except Exception:
            return EvalItemResult(chunks=None, generation=None)

    try:
        gen_result = generate(normalized, chunks, settings)
    except GenerationError:
        return EvalItemResult(chunks=chunks, generation=None)

    return EvalItemResult(chunks=chunks, generation=gen_result)


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


@dataclass
class EvalGroupItem:
    """One golden item, optionally with retrieval already done for gen-only retry."""

    item: GoldItem
    existing_chunks: list[ScoredChunk] | None = None


@dataclass
class _EvalWork:
    item: GoldItem
    normalized: str
    extras: list[str]
    existing_chunks: list[ScoredChunk] | None = None
    chunks: list[ScoredChunk] | None = field(default=None, init=False)
    generation: GenerationResult | None = field(default=None, init=False)


def run_eval_group_batched(
    group_items: list[EvalGroupItem],
    settings: Settings,
    *,
    retrieval_mode: str,
    rerank_enabled: bool,
    knowledge: Knowledge,
    query_rewrite: bool = False,
    phase_batch: bool | None = None,
) -> dict[str, tuple[list[ScoredChunk] | None, GenerationResult | None]]:
    """Evaluate all items in one ablation group with phase-level model batching."""
    if not group_items:
        return {}

    use_batch = (
        settings.resources.query_phase_batch
        if phase_batch is None
        else phase_batch
    ) and settings.resources.single_model_exclusive

    mode, do_rerank = resolve_retrieval_mode(settings, retrieval_mode, rerank_enabled)
    works: list[_EvalWork] = []
    for entry in group_items:
        normalized = normalize_question(entry.item.question)
        works.append(
            _EvalWork(
                item=entry.item,
                normalized=normalized,
                extras=[],
                existing_chunks=entry.existing_chunks,
            )
        )

    needs_retrieval = [work for work in works if work.existing_chunks is None]
    gen_only = [work for work in works if work.existing_chunks is not None]

    for work in gen_only:
        work.chunks = work.existing_chunks

    if needs_retrieval:
        any_multi_query = query_rewrite and settings.query_processing.multi_query
        with _phase(ModelResource.MLX_VLM, use_batch and any_multi_query):
            for work in needs_retrieval:
                if query_rewrite:
                    work.extras = collect_extra_queries(
                        work.normalized, settings
                    ).combined

        needs_bge = _needs_dense_embedding(mode)
        with _phase(ModelResource.BGE_M3, use_batch and needs_bge):
            for work in needs_retrieval:
                try:
                    result = retrieve(
                        work.normalized,
                        settings,
                        retrieval_mode=mode,
                        rerank_enabled=False,
                        knowledge=knowledge,
                        extra_queries=work.extras,
                    )
                    work.chunks = result.chunks
                except Exception:
                    logger.warning(
                        "eval retrieval failed item=%s mode=%s",
                        work.item.id,
                        mode,
                        exc_info=True,
                    )
                    work.chunks = None

        with _phase(ModelResource.CROSS_ENCODER, use_batch and do_rerank):
            for work in needs_retrieval:
                if work.chunks is None or not do_rerank:
                    continue
                try:
                    reranked, _stage = rerank_chunks(
                        settings, work.normalized, work.chunks
                    )
                    work.chunks = reranked
                except Exception:
                    logger.warning(
                        "eval rerank failed item=%s mode=%s",
                        work.item.id,
                        mode,
                        exc_info=True,
                    )
                    work.chunks = None

    pending_gen = [work for work in works if work.chunks is not None]
    with _phase(ModelResource.MLX_VLM, use_batch):
        for work in pending_gen:
            try:
                work.generation = generate(work.normalized, work.chunks, settings)
            except GenerationError:
                logger.warning(
                    "eval generation failed item=%s mode=%s",
                    work.item.id,
                    mode,
                    exc_info=True,
                )
                work.generation = None

    return {
        work.item.id: (work.chunks, work.generation)
        for work in works
    }
