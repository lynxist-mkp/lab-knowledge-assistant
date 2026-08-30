from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wenmai.config import Settings
from wenmai.eval.golden import GoldItem
from wenmai.eval.metrics import corpus_doc_ids_from_chunks, retrieval_item_snapshot
from wenmai.generation import GenerationError, GenerationResult, generate
from wenmai.knowledge import Knowledge
from wenmai.models import ScoredChunk
from wenmai.pipelines.query import normalize_question
from wenmai.retrieval import retrieve


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
) -> EvalItemResult:
    """Retrieve once (unless chunks supplied), then generate for one golden item."""
    normalized = normalize_question(item.question)

    if retrieved_chunks is not None:
        chunks = retrieved_chunks
    else:
        try:
            result = retrieve(
                normalized,
                settings,
                retrieval_mode=retrieval_mode,
                rerank_enabled=rerank_enabled,
                knowledge=knowledge,
            )
            chunks = result.chunks
        except Exception:
            return EvalItemResult(chunks=None, generation=None)

    try:
        gen_result = generate(normalized, chunks, settings)
    except GenerationError:
        return EvalItemResult(chunks=chunks, generation=None)

    return EvalItemResult(chunks=chunks, generation=gen_result)
