from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from lab_knowledge.config import Settings
from lab_knowledge.eval.golden import GoldItem
from lab_knowledge.eval.metrics import corpus_doc_ids_from_chunks, retrieval_item_snapshot
from lab_knowledge.generation import GenerationResult, generate
from lab_knowledge.knowledge import Knowledge
from lab_knowledge.models import ScoredChunk
from lab_knowledge.pipelines.query_orchestration import (
    eval_work_from_item,
    gen_retry_work,
    run_eval_works,
)

logger = logging.getLogger(__name__)


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


@dataclass
class EvalGroupItem:
    """One golden item, optionally with retrieval already done for gen-only retry."""

    item: GoldItem
    existing_chunks: list[ScoredChunk] | None = None


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

    works = []
    for entry in group_items:
        if entry.existing_chunks is not None:
            works.append(
                gen_retry_work(
                    question=entry.item.question,
                    settings=settings,
                    retrieval_mode=retrieval_mode,
                    rerank_enabled=rerank_enabled,
                    knowledge=knowledge,
                    pre_chunks=entry.existing_chunks,
                    query_rewrite=query_rewrite,
                )
            )
        else:
            works.append(
                eval_work_from_item(
                    question=entry.item.question,
                    settings=settings,
                    retrieval_mode=retrieval_mode,
                    rerank_enabled=rerank_enabled,
                    knowledge=knowledge,
                    query_rewrite=query_rewrite,
                )
            )

    results = run_eval_works(
        works,
        phase_batch=use_batch,
        tolerate_errors=True,
        generate_fn=generate,
    )

    return {
        entry.item.id: result
        for entry, result in zip(group_items, results, strict=True)
    }


__all__ = [
    "EvalGroupItem",
    "EvalItemResult",
    "run_eval_group_batched",
]
