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
from wenmai.eval.views import EvalRunView, FailedEvalItem, parse_eval_run
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import AskResult, ScoredChunk
from wenmai.pipelines.query import ask_question, normalize_question
from wenmai.retrieval import retrieve

FAILURE_RETRIES = 3


def _runs_dir(settings: Settings) -> Path:
    raw = Path(settings.evaluation.runs)
    path = raw if raw.is_absolute() else settings.root / raw
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ablation_kwargs(group: str) -> dict[str, str | bool]:
    spec = resolve_ablation(group)
    return {
        "retrieval_mode": spec.mode,
        "rerank_enabled": spec.rerank_enabled,
    }


def _retrieve_item(
    item: GoldItem,
    settings: Settings,
    group: str,
    knowledge: Knowledge,
) -> list[ScoredChunk] | None:
    kwargs = _ablation_kwargs(group)
    try:
        result = retrieve(
            normalize_question(item.question),
            settings,
            retrieval_mode=str(kwargs["retrieval_mode"]),
            rerank_enabled=bool(kwargs["rerank_enabled"]),
            knowledge=knowledge,
        )
        return result.chunks
    except Exception:
        return None


def _ask_item(
    item: GoldItem,
    settings: Settings,
    group: str,
    knowledge: Knowledge,
) -> AskResult | None:
    kwargs = _ablation_kwargs(group)
    try:
        return ask_question(
            item.question,
            settings,
            retrieval_mode=str(kwargs["retrieval_mode"]),
            rerank_enabled=bool(kwargs["rerank_enabled"]),
            knowledge=knowledge,
            record_trace=False,
        )
    except Exception:
        return None


def _metrics_payload(
    items: list[GoldItem],
    ranked_by_id: dict[str, list[str]],
    generation_by_id: dict[str, AskResult],
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
    generation_by_id: dict[str, AskResult],
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


def run_eval(
    settings: Settings,
    *,
    knowledge: Knowledge | None = None,
) -> EvalRunView:
    items = load_golden_set_from_settings(settings)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    groups = list(settings.evaluation.ablations)
    ranked_chunks: dict[str, dict[str, list[ScoredChunk]]] = {name: {} for name in groups}
    generation_results: dict[str, dict[str, AskResult]] = {name: {} for name in groups}
    resolved_knowledge = knowledge or create_knowledge(settings)

    pending: list[tuple[str, GoldItem]] = [
        (group, item) for group in groups for item in items
    ]
    still: list[tuple[str, GoldItem]] = []
    for group, item in pending:
        chunks = _retrieve_item(item, settings, group, resolved_knowledge)
        result = _ask_item(item, settings, group, resolved_knowledge)
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
            chunks = ranked_chunks[group].get(item.id)
            if chunks is None:
                chunks = _retrieve_item(item, settings, group, resolved_knowledge)
                if chunks is not None:
                    ranked_chunks[group][item.id] = chunks
            result = generation_results[group].get(item.id)
            if result is None:
                result = _ask_item(item, settings, group, resolved_knowledge)
                if result is not None:
                    generation_results[group][item.id] = result
            if chunks is None or result is None:
                nxt.append((group, item))
        still = nxt

    failures = [
        FailedEvalItem(group=group, item_id=item.id) for group, item in still
    ]
    artifact = {
        "timestamp": timestamp,
        "golden_set": settings.evaluation.golden_set,
        "ablations": groups,
        "item_count": len(items),
        "failures": [item.as_dict() for item in failures],
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
