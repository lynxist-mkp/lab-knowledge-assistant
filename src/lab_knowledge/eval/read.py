from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lab_knowledge.components.evaluator.ragas_probe import probe_ragas_judge
from lab_knowledge.config import Settings
from lab_knowledge.eval.persist import eval_run_belongs_to_collection, runs_dir
from lab_knowledge.eval.views import (
    EvalDashboardView,
    EvalRunDetail,
    EvalRunSummary,
    RagasMetricStatus,
    RagasStatusView,
    parse_eval_run_detail,
    parse_eval_run_summary,
)


def _load_run_raw(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _load_run_summary(
    path: Path,
    *,
    settings: Settings | None = None,
    collection_id: str | None = None,
) -> EvalRunSummary | None:
    raw = _load_run_raw(path)
    if raw is None:
        return None
    if collection_id is not None and settings is not None:
        if not eval_run_belongs_to_collection(
            raw,
            collection_id,
            default_collection_id=settings.default_collection_id,
        ):
            return None
    return parse_eval_run_summary(raw)


def list_eval_run_summaries(
    settings: Settings, *, collection_id: str | None = None
) -> list[EvalRunSummary]:
    eval_runs_dir = runs_dir(settings)
    if not eval_runs_dir.is_dir():
        return []
    runs: list[EvalRunSummary] = []
    for path in eval_runs_dir.glob("*.json"):
        run = _load_run_summary(
            path,
            settings=settings,
            collection_id=collection_id,
        )
        if run is not None:
            runs.append(run)
    runs.sort(key=lambda item: item.timestamp, reverse=True)
    return runs


def get_eval_run_summary(
    settings: Settings,
    timestamp: str,
    *,
    collection_id: str | None = None,
) -> EvalRunSummary | None:
    if not timestamp:
        return None
    return _load_run_summary(
        runs_dir(settings) / f"{timestamp}.json",
        settings=settings,
        collection_id=collection_id,
    )


def get_eval_run_detail(
    settings: Settings,
    timestamp: str,
    *,
    collection_id: str | None = None,
) -> EvalRunDetail | None:
    if not timestamp:
        return None
    raw = _load_run_raw(runs_dir(settings) / f"{timestamp}.json")
    if raw is None:
        return None
    if collection_id is not None:
        if not eval_run_belongs_to_collection(
            raw,
            collection_id,
            default_collection_id=settings.default_collection_id,
        ):
            return None
    return parse_eval_run_detail(raw)


def get_ragas_status(settings: Settings) -> RagasStatusView:
    judge = settings.evaluation.ragas_judge
    provider, provider_label = judge.provider, judge.provider_label
    judge_available, init_error = probe_ragas_judge(settings)

    if judge_available:
        return RagasStatusView(
            configured=True,
            provider=provider,
            provider_label=provider_label,
            model=judge.model,
            faithfulness=RagasMetricStatus(
                name="faithfulness",
                status="ready",
                label="Judge 已就绪",
                value=None,
                reason="",
            ),
            context_precision=RagasMetricStatus(
                name="context_precision",
                status="ready",
                label="Judge 已就绪",
                value=None,
                reason="",
            ),
        )

    reason = init_error or "Ragas judge 未跑通"
    return RagasStatusView(
        configured=True,
        provider=provider,
        provider_label=provider_label,
        model=judge.model,
        faithfulness=RagasMetricStatus(
            name="faithfulness",
            status="unavailable",
            label="未跑通",
            value=None,
            reason=reason,
        ),
        context_precision=RagasMetricStatus(
            name="context_precision",
            status="unavailable",
            label="未跑通",
            value=None,
            reason=reason,
        ),
    )


def get_eval_dashboard(
    settings: Settings, *, collection_id: str | None = None
) -> EvalDashboardView:
    history = list_eval_run_summaries(settings, collection_id=collection_id)
    group_order = list(settings.evaluation.ablations)
    return EvalDashboardView(
        latest_run=history[0] if history else None,
        history=history,
        ragas=get_ragas_status(settings),
        group_order=group_order,
    )


def eval_chart_data(dashboard: EvalDashboardView) -> dict[str, object]:
    return {
        "group_order": list(dashboard.group_order),
        "latest_groups": dashboard.latest_run.as_dict()["groups"] if dashboard.latest_run else {},
        "history": [run.as_dict() for run in dashboard.history],
    }


__all__ = [
    "eval_chart_data",
    "get_eval_dashboard",
    "get_eval_run_detail",
    "get_eval_run_summary",
    "get_ragas_status",
    "list_eval_run_summaries",
]
