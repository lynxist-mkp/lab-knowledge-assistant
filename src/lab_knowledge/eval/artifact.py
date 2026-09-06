"""评测 artifact 组装：run 结果 dict 的单一 seam。"""

from __future__ import annotations

from typing import Any

from lab_knowledge.eval.ablation import group_metrics_payload
from lab_knowledge.eval.golden import GoldItem
from lab_knowledge.eval.metrics import (
    aggregate_hit_at_5,
    aggregate_mrr,
    citation_coverage,
    corpus_doc_ids_from_chunks,
    refusal_accuracy,
    retrieval_item_snapshot,
)
from lab_knowledge.eval.views import FailedEvalItem
from lab_knowledge.generation import GenerationResult
from lab_knowledge.models import ScoredChunk


def _metrics_payload(
    items: list[GoldItem],
    ranked_by_id: dict[str, list[str]],
    generation_by_id: dict[str, GenerationResult],
) -> dict[str, Any]:
    hit_items: list[GoldItem] = []
    ranked_per_item: list[list[str]] = []
    for item in items:
        ranked = ranked_by_id.get(item.id)
        if ranked is None:
            continue
        hit_items.append(item)
        ranked_per_item.append(ranked)

    gen_items: list[GoldItem] = []
    refused_flags: list[bool] = []
    citation_counts: list[int] = []
    for item in items:
        result = generation_by_id.get(item.id)
        if result is None:
            continue
        gen_items.append(item)
        refused_flags.append(result.refused)
        citation_counts.append(len(result.citations))

    return group_metrics_payload(
        hit_at_5=aggregate_hit_at_5(hit_items, ranked_per_item),
        mrr=aggregate_mrr(hit_items, ranked_per_item),
        refusal_accuracy=refusal_accuracy(gen_items, refused_flags),
        citation_coverage=citation_coverage(gen_items, refused_flags, citation_counts),
        answerable_count=sum(1 for item in items if item.answerable),
        unanswerable_count=sum(1 for item in items if not item.answerable),
    )


def _item_snapshots(
    items: list[GoldItem],
    ranked_chunks_by_id: dict[str, list[ScoredChunk]],
    generation_by_id: dict[str, GenerationResult],
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for item in items:
        chunks = ranked_chunks_by_id.get(item.id)
        if chunks is None:
            continue
        payload: dict[str, Any] = {"retrieval": retrieval_item_snapshot(chunks)}
        result = generation_by_id.get(item.id)
        if result is not None:
            payload["refused"] = result.refused
            payload["citation_count"] = len(result.citations)
        snapshots[item.id] = payload
    return snapshots


def group_artifact_entry(
    items: list[GoldItem],
    ranked_chunks: dict[str, list[ScoredChunk]],
    generation_results: dict[str, GenerationResult],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config": config,
        "metrics": _metrics_payload(
            items,
            {
                item_id: corpus_doc_ids_from_chunks(chunks)
                for item_id, chunks in ranked_chunks.items()
            },
            generation_results,
        ),
        "items": _item_snapshots(items, ranked_chunks, generation_results),
    }


def build_ablation_eval_artifact(
    *,
    timestamp: str,
    golden_set: str,
    ablations: list[str],
    item_count: int,
    failures: list[FailedEvalItem],
    groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "golden_set": golden_set,
        "ablations": ablations,
        "item_count": item_count,
        "failures": [item.as_dict() for item in failures],
        "groups": groups,
    }


def build_rewrite_compare_eval_artifact(
    *,
    timestamp: str,
    golden_set: str,
    item_count: int,
    failures: list[FailedEvalItem],
    groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "golden_set": golden_set,
        "compare": "rewrite",
        "item_count": item_count,
        "failures": [item.as_dict() for item in failures],
        "groups": groups,
    }


__all__ = [
    "build_ablation_eval_artifact",
    "build_rewrite_compare_eval_artifact",
    "group_artifact_entry",
]
