from __future__ import annotations

import json
from pathlib import Path

from wenmai.components.evaluator.ragas_probe import probe_ragas_judge
from wenmai.config import Settings
from wenmai.eval.persist import runs_dir
from wenmai.eval.views import (
    EvalDashboardView,
    EvalRunView,
    RagasMetricStatus,
    RagasStatusView,
    parse_eval_run,
)


def _load_run_file(path: Path) -> EvalRunView | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return parse_eval_run(raw)


def list_eval_runs(settings: Settings) -> list[EvalRunView]:
    eval_runs_dir = runs_dir(settings)
    if not eval_runs_dir.is_dir():
        return []
    runs: list[EvalRunView] = []
    for path in eval_runs_dir.glob("*.json"):
        run = _load_run_file(path)
        if run is not None:
            runs.append(run)
    runs.sort(key=lambda item: item.timestamp, reverse=True)
    return runs


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


def get_eval_dashboard(settings: Settings) -> EvalDashboardView:
    history = list_eval_runs(settings)
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
