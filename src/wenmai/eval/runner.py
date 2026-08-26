from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.eval.ablation import apply_ablation, config_snapshot, group_metrics_payload
from wenmai.eval.golden import GoldItem, load_golden_set_from_settings
from wenmai.eval.metrics import (
    aggregate_hit_at_5,
    aggregate_mrr,
    corpus_doc_ids_from_chunks,
    refusal_accuracy,
)
from wenmai.pipelines.query import ask_question, retrieve_for_question


def _runs_dir(settings: Settings) -> Path:
    raw = Path(settings.evaluation.runs)
    path = raw if raw.is_absolute() else settings.root / raw
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_ablation_group(
    settings: Settings,
    items: list[GoldItem],
    group: str,
) -> dict[str, Any]:
    group_settings = apply_ablation(settings, group)
    ranked_per_item: list[list[str]] = []
    refused_flags: list[bool] = []

    for item in items:
        scored = retrieve_for_question(item.question, group_settings)
        ranked_per_item.append(corpus_doc_ids_from_chunks(scored))
        ask_result = ask_question(item.question, group_settings)
        refused_flags.append(ask_result.refused)

    answerable_count = sum(1 for item in items if item.answerable)
    unanswerable_count = sum(1 for item in items if not item.answerable)

    metrics = group_metrics_payload(
        hit_at_5=aggregate_hit_at_5(items, ranked_per_item),
        mrr=aggregate_mrr(items, ranked_per_item),
        refusal_accuracy=refusal_accuracy(items, refused_flags),
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
    )
    return {
        "config": config_snapshot(settings, group),
        "metrics": metrics,
    }


def run_ablation_batch(settings: Settings) -> Path:
    items = load_golden_set_from_settings(settings)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    groups: dict[str, Any] = {}
    for group in settings.evaluation.ablations:
        groups[group] = run_ablation_group(settings, items, group)

    artifact = {
        "timestamp": timestamp,
        "golden_set": settings.evaluation.golden_set,
        "ablations": list(settings.evaluation.ablations),
        "item_count": len(items),
        "groups": groups,
    }

    output_path = _runs_dir(settings) / f"{timestamp}.json"
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
