from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.eval.ablation import config_snapshot, group_metrics_payload, resolve_ablation
from wenmai.eval.golden import GoldItem, load_golden_set_from_settings
from wenmai.eval.metrics import (
    aggregate_hit_at_5,
    aggregate_mrr,
    citation_coverage,
    corpus_doc_ids_from_chunks,
    refusal_accuracy,
    retrieval_item_snapshot,
)
from wenmai.eval.pipeline import eval_item
from wenmai.eval.views import EvalRunView, FailedEvalItem, parse_eval_run
from wenmai.generation import GenerationResult
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import ScoredChunk
from wenmai.tracing.latency import query_latency_percentiles

FAILURE_RETRIES = 3


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


def _run_eval_item(
    item: GoldItem,
    settings: Settings,
    group: str,
    knowledge: Knowledge,
    existing_chunks: list[ScoredChunk] | None,
) -> tuple[list[ScoredChunk] | None, GenerationResult | None]:
    spec = resolve_ablation(group)
    outcome = eval_item(
        item,
        settings,
        retrieval_mode=spec.mode,
        rerank_enabled=spec.rerank_enabled,
        knowledge=knowledge,
        retrieved_chunks=existing_chunks,
    )
    return outcome.chunks, outcome.generation


def run_eval(
    settings: Settings,
    *,
    knowledge: Knowledge | None = None,
) -> EvalRunView:
    items = load_golden_set_from_settings(settings)
    run_started = datetime.now(UTC).isoformat()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    groups = list(settings.evaluation.ablations)
    ranked_chunks: dict[str, dict[str, list[ScoredChunk]]] = {name: {} for name in groups}
    generation_results: dict[str, dict[str, GenerationResult]] = {name: {} for name in groups}
    resolved_knowledge = knowledge or create_knowledge(settings)

    pending: list[tuple[str, GoldItem]] = [
        (group, item) for group in groups for item in items
    ]
    still: list[tuple[str, GoldItem]] = []
    for group, item in pending:
        chunks, result = _run_eval_item(item, settings, group, resolved_knowledge, None)
        if chunks is not None:
            ranked_chunks[group][item.id] = chunks
        if result is not None:
            generation_results[group][item.id] = result
        if chunks is None or result is None:
            still.append((group, item))

    for _ in range(FAILURE_RETRIES):
        if not still:
            break
        nxt: list[tuple[str, GoldItem]] = []
        for group, item in still:
            existing = ranked_chunks[group].get(item.id)
            chunks, result = _run_eval_item(
                item, settings, group, resolved_knowledge, existing
            )
            if chunks is not None:
                ranked_chunks[group][item.id] = chunks
            if result is not None:
                generation_results[group][item.id] = result
            if chunks is None or result is None:
                nxt.append((group, item))
        still = nxt

    failures = [
        FailedEvalItem(group=group, item_id=item.id) for group, item in still
    ]
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

    output_path = _runs_dir(settings) / f"{timestamp}.json"
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    parsed = parse_eval_run(artifact)
    if parsed is None:
        raise RuntimeError("eval run artifact could not be parsed")
    return parsed
