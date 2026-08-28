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
)
from wenmai.eval.views import EvalRunView, FailedEvalItem, parse_eval_run
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import AskResult
from wenmai.pipelines.query import ask_question

FAILURE_RETRIES = 3


def _runs_dir(settings: Settings) -> Path:
    raw = Path(settings.evaluation.runs)
    path = raw if raw.is_absolute() else settings.root / raw
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ask_item(
    item: GoldItem,
    settings: Settings,
    group: str,
    knowledge: Knowledge,
) -> AskResult:
    spec = resolve_ablation(group)
    return ask_question(
        item.question,
        settings,
        retrieval_mode=spec.mode,
        rerank_enabled=spec.rerank_enabled,
        knowledge=knowledge,
    )


def _try_ask(
    item: GoldItem,
    settings: Settings,
    group: str,
    knowledge: Knowledge,
) -> AskResult | None:
    try:
        return _ask_item(item, settings, group, knowledge)
    except Exception:
        return None


def _metrics_payload(
    items: list[GoldItem],
    results_by_id: dict[str, AskResult],
) -> dict[str, Any]:
    scored_items: list[GoldItem] = []
    ranked_per_item: list[list[str]] = []
    refused_flags: list[bool] = []
    citation_counts: list[int] = []
    for item in items:
        result = results_by_id.get(item.id)
        if result is None:
            continue
        scored_items.append(item)
        ranked_per_item.append(corpus_doc_ids_from_chunks(result.ranked_chunks))
        refused_flags.append(result.refused)
        citation_counts.append(len(result.citations))

    return group_metrics_payload(
        hit_at_5=aggregate_hit_at_5(scored_items, ranked_per_item),
        mrr=aggregate_mrr(scored_items, ranked_per_item),
        refusal_accuracy=refusal_accuracy(scored_items, refused_flags),
        citation_coverage=citation_coverage(scored_items, refused_flags, citation_counts),
        answerable_count=sum(1 for item in items if item.answerable),
        unanswerable_count=sum(1 for item in items if not item.answerable),
    )


def run_eval(
    settings: Settings,
    *,
    knowledge: Knowledge | None = None,
) -> EvalRunView:
    items = load_golden_set_from_settings(settings)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    groups = list(settings.evaluation.ablations)
    results: dict[str, dict[str, AskResult]] = {name: {} for name in groups}
    resolved_knowledge = knowledge or create_knowledge(settings)

    pending: list[tuple[str, GoldItem]] = [
        (group, item) for group in groups for item in items
    ]
    still: list[tuple[str, GoldItem]] = []
    for group, item in pending:
        result = _try_ask(item, settings, group, resolved_knowledge)
        if result is None:
            still.append((group, item))
        else:
            results[group][item.id] = result

    for _ in range(FAILURE_RETRIES):
        if not still:
            break
        nxt: list[tuple[str, GoldItem]] = []
        for group, item in still:
            result = _try_ask(item, settings, group, resolved_knowledge)
            if result is None:
                nxt.append((group, item))
            else:
                results[group][item.id] = result
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
                "metrics": _metrics_payload(items, results[name]),
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
