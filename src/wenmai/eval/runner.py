from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.eval.ablation import (
    config_snapshot,
    group_metrics_payload,
    resolve_ablation,
    rewrite_compare_config_snapshot,
)
from wenmai.eval.golden import GoldItem, load_golden_set_from_settings
from wenmai.eval.metrics import (
    aggregate_hit_at_5,
    aggregate_mrr,
    citation_coverage,
    corpus_doc_ids_from_chunks,
    refusal_accuracy,
    retrieval_item_snapshot,
)
from wenmai.eval.pipeline import EvalGroupItem
from wenmai.eval import pipeline as eval_pipeline
from wenmai.eval.ragas_metrics import attach_ragas_to_artifact, should_run_ragas
from wenmai.eval.views import EvalRunView, FailedEvalItem, parse_eval_run
from wenmai.generation import GenerationResult
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import ScoredChunk
from wenmai.tracing.latency import query_latency_percentiles

logger = logging.getLogger(__name__)

FAILURE_RETRIES = 3

REWRITE_COMPARE_GROUPS = ("rewrite_off", "rewrite_on")
REWRITE_COMPARE_FLAGS: dict[str, bool] = {
    "rewrite_off": False,
    "rewrite_on": True,
}


def _runs_dir(settings: Settings) -> Path:
    raw = Path(settings.evaluation.runs)
    path = raw if raw.is_absolute() else settings.root / raw
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _run_group_batched(
    items: list[GoldItem],
    settings: Settings,
    group: str,
    knowledge: Knowledge,
    *,
    retrieval_mode: str,
    rerank_enabled: bool,
    query_rewrite: bool = False,
) -> tuple[
    dict[str, list[ScoredChunk]],
    dict[str, GenerationResult],
    list[FailedEvalItem],
]:
    """Run one ablation group with phase batching and generation retries."""
    ranked_chunks: dict[str, list[ScoredChunk]] = {}
    generation_results: dict[str, GenerationResult] = {}
    pending: list[GoldItem] = list(items)

    for attempt in range(FAILURE_RETRIES + 1):
        if not pending:
            break
        group_inputs: list[EvalGroupItem] = []
        for item in pending:
            existing = ranked_chunks.get(item.id)
            group_inputs.append(
                EvalGroupItem(item=item, existing_chunks=existing)
            )

        logger.info(
            "eval group=%s attempt=%d/%d items=%d",
            group,
            attempt + 1,
            FAILURE_RETRIES + 1,
            len(group_inputs),
        )
        outcomes = eval_pipeline.run_eval_group_batched(
            group_inputs,
            settings,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            query_rewrite=query_rewrite,
        )
        for item in pending:
            chunks, result = outcomes.get(item.id, (None, None))
            if chunks is not None:
                ranked_chunks[item.id] = chunks
            if result is not None:
                generation_results[item.id] = result

        pending = [
            item
            for item in pending
            if item.id not in ranked_chunks or item.id not in generation_results
        ]

    failures = [
        FailedEvalItem(group=group, item_id=item.id) for item in pending
    ]
    return ranked_chunks, generation_results, failures


def _run_grouped_eval(
    items: list[GoldItem],
    settings: Settings,
    groups: list[str],
    knowledge: Knowledge,
    *,
    query_rewrite_by_group: dict[str, bool] | None = None,
    spec_by_group: dict[str, tuple[str, bool]] | None = None,
) -> tuple[
    dict[str, dict[str, list[ScoredChunk]]],
    dict[str, dict[str, GenerationResult]],
    list[FailedEvalItem],
]:
    ranked_chunks: dict[str, dict[str, list[ScoredChunk]]] = {
        name: {} for name in groups
    }
    generation_results: dict[str, dict[str, GenerationResult]] = {
        name: {} for name in groups
    }
    failures: list[FailedEvalItem] = []
    total_ops = len(groups) * len(items)
    completed = 0

    for group_index, group in enumerate(groups, start=1):
        if spec_by_group is not None:
            retrieval_mode, rerank_enabled = spec_by_group[group]
        else:
            spec = resolve_ablation(group)
            retrieval_mode = spec.mode
            rerank_enabled = spec.rerank_enabled
        query_rewrite = (
            query_rewrite_by_group.get(group, False)
            if query_rewrite_by_group
            else False
        )

        logger.info(
            "eval starting group=%s (%d/%d) items=%d",
            group,
            group_index,
            len(groups),
            len(items),
        )
        group_ranked, group_generation, group_failures = _run_group_batched(
            items,
            settings,
            group,
            knowledge,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            query_rewrite=query_rewrite,
        )
        ranked_chunks[group] = group_ranked
        generation_results[group] = group_generation
        failures.extend(group_failures)
        completed += len(items)
        logger.info(
            "eval progress %d/%d item×group ops group=%s failed=%d",
            completed,
            total_ops,
            group,
            len(group_failures),
        )

    return ranked_chunks, generation_results, failures


def run_eval(
    settings: Settings,
    *,
    knowledge: Knowledge | None = None,
    query_rewrite: bool = False,
    ragas: bool | None = None,
) -> EvalRunView:
    items = load_golden_set_from_settings(settings)
    run_started = datetime.now(UTC).isoformat()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    groups = list(settings.evaluation.ablations)
    resolved_knowledge = knowledge or create_knowledge(settings)

    ranked_chunks, generation_results, failures = _run_grouped_eval(
        items,
        settings,
        groups,
        resolved_knowledge,
        query_rewrite_by_group={name: query_rewrite for name in groups},
    )
    latency_ms = query_latency_percentiles(settings, started_at_min=run_started)
    artifact = {
        "timestamp": timestamp,
        "golden_set": settings.evaluation.golden_set,
        "ablations": groups,
        "item_count": len(items),
        "failures": [item.as_dict() for item in failures],
        "latency_ms": latency_ms,
        "groups": {
            name: {
                "config": config_snapshot(settings, name),
                "metrics": _metrics_payload(
                    items,
                    {
                        item_id: corpus_doc_ids_from_chunks(chunks)
                        for item_id, chunks in ranked_chunks[name].items()
                    },
                    generation_results[name],
                ),
                "items": _item_snapshots(items, ranked_chunks[name], generation_results[name]),
            }
            for name in groups
        },
    }
    if should_run_ragas(settings, ragas):
        attach_ragas_to_artifact(
            artifact,
            items,
            generation_results["rrf_rerank"],
            ranked_chunks["rrf_rerank"],
            settings,
        )

    output_path = _runs_dir(settings) / f"{timestamp}.json"
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    parsed = parse_eval_run(artifact)
    if parsed is None:
        raise RuntimeError("eval run artifact could not be parsed")
    return parsed


def run_rewrite_compare(
    settings: Settings,
    *,
    knowledge: Knowledge | None = None,
) -> EvalRunView:
    items = load_golden_set_from_settings(settings)
    run_started = datetime.now(UTC).isoformat()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    groups = list(REWRITE_COMPARE_GROUPS)
    resolved_knowledge = knowledge or create_knowledge(settings)

    ranked_chunks, generation_results, failures = _run_grouped_eval(
        items,
        settings,
        groups,
        resolved_knowledge,
        query_rewrite_by_group=REWRITE_COMPARE_FLAGS,
        spec_by_group={name: ("rrf", True) for name in groups},
    )
    latency_ms = query_latency_percentiles(settings, started_at_min=run_started)
    artifact = {
        "timestamp": timestamp,
        "golden_set": settings.evaluation.golden_set,
        "compare": "rewrite",
        "item_count": len(items),
        "failures": [item.as_dict() for item in failures],
        "latency_ms": latency_ms,
        "groups": {
            name: {
                "config": rewrite_compare_config_snapshot(
                    settings,
                    group=name,
                    query_rewrite=REWRITE_COMPARE_FLAGS[name],
                ),
                "metrics": _metrics_payload(
                    items,
                    {
                        item_id: corpus_doc_ids_from_chunks(chunks)
                        for item_id, chunks in ranked_chunks[name].items()
                    },
                    generation_results[name],
                ),
                "items": _item_snapshots(items, ranked_chunks[name], generation_results[name]),
            }
            for name in groups
        },
    }

    output_path = _runs_dir(settings) / f"{timestamp}.json"
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    parsed = parse_eval_run(artifact)
    if parsed is None:
        raise RuntimeError("rewrite compare artifact could not be parsed")
    return parsed
