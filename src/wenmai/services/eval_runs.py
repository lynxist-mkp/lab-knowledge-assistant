from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.factories import evaluator as evaluator_factory

ABLATION_GROUP_LABELS: dict[str, str] = {
    "dense_only": "Dense 单路",
    "sparse_only": "Sparse 单路",
    "rrf": "RRF 融合",
    "rrf_rerank": "RRF + Rerank",
}


@dataclass(frozen=True)
class GroupMetricsView:
    hit_at_5: float
    mrr: float
    refusal_accuracy: float
    citation_coverage: float | None
    answerable_count: int
    unanswerable_count: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GroupMetricsView:
        citation_raw = raw.get("citation_coverage")
        return cls(
            hit_at_5=float(raw["hit_at_5"]),
            mrr=float(raw["mrr"]),
            refusal_accuracy=float(raw["refusal_accuracy"]),
            citation_coverage=float(citation_raw) if citation_raw is not None else None,
            answerable_count=int(raw["answerable_count"]),
            unanswerable_count=int(raw["unanswerable_count"]),
        )

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class GroupRunView:
    name: str
    label: str
    metrics: GroupMetricsView

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class EvalRunView:
    timestamp: str
    item_count: int
    golden_set: str
    groups: dict[str, GroupRunView]

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "item_count": self.item_count,
            "golden_set": self.golden_set,
            "groups": {name: group.as_dict() for name, group in self.groups.items()},
        }


@dataclass(frozen=True)
class RagasMetricStatus:
    name: str
    status: str
    label: str
    value: float | None
    reason: str

    def as_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class RagasStatusView:
    configured: bool
    faithfulness: RagasMetricStatus
    context_precision: RagasMetricStatus

    def as_dict(self) -> dict[str, object]:
        return {
            "configured": self.configured,
            "faithfulness": self.faithfulness.as_dict(),
            "context_precision": self.context_precision.as_dict(),
        }


@dataclass(frozen=True)
class EvalDashboardView:
    latest_run: EvalRunView | None
    history: list[EvalRunView]
    ragas: RagasStatusView
    group_order: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "latest_run": self.latest_run.as_dict() if self.latest_run else None,
            "history": [run.as_dict() for run in self.history],
            "ragas": self.ragas.as_dict(),
            "group_order": list(self.group_order),
        }


def _runs_dir(settings: Settings) -> Path:
    raw = Path(settings.evaluation.runs)
    return raw if raw.is_absolute() else settings.root / raw


def _group_label(name: str) -> str:
    return ABLATION_GROUP_LABELS.get(name, name)


def _parse_run(raw: dict[str, Any]) -> EvalRunView | None:
    timestamp = str(raw.get("timestamp") or "")
    if not timestamp:
        return None
    groups_raw = raw.get("groups") or {}
    groups: dict[str, GroupRunView] = {}
    for name, payload in groups_raw.items():
        metrics_raw = payload.get("metrics")
        if not isinstance(metrics_raw, dict):
            continue
        groups[name] = GroupRunView(
            name=name,
            label=_group_label(name),
            metrics=GroupMetricsView.from_dict(metrics_raw),
        )
    if not groups:
        return None
    return EvalRunView(
        timestamp=timestamp,
        item_count=int(raw.get("item_count") or 0),
        golden_set=str(raw.get("golden_set") or ""),
        groups=groups,
    )


def _load_run_file(path: Path) -> EvalRunView | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return _parse_run(raw)


def list_eval_runs(settings: Settings) -> list[EvalRunView]:
    runs_dir = _runs_dir(settings)
    if not runs_dir.is_dir():
        return []
    runs: list[EvalRunView] = []
    for path in runs_dir.glob("*.json"):
        run = _load_run_file(path)
        if run is not None:
            runs.append(run)
    runs.sort(key=lambda item: item.timestamp, reverse=True)
    return runs


def get_ragas_status(settings: Settings) -> RagasStatusView:
    if settings.providers.evaluator != "ragas_collections":
        reason = f"当前 Evaluator 为 {settings.providers.evaluator!r}，未接入 Ragas judge"
        unavailable = RagasMetricStatus(
            name="faithfulness",
            status="not_configured",
            label="未接入",
            value=None,
            reason=reason,
        )
        precision = RagasMetricStatus(
            name="context_precision",
            status="not_configured",
            label="未接入",
            value=None,
            reason=reason,
        )
        return RagasStatusView(
            configured=False,
            faithfulness=unavailable,
            context_precision=precision,
        )

    evaluator = evaluator_factory.create(settings)
    judge_available = bool(getattr(evaluator, "judge_available", False))
    init_error = str(getattr(evaluator, "judge_unavailable_reason", "") or "")

    if judge_available:
        return RagasStatusView(
            configured=True,
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
    """JSON-serializable payload for Chart.js on the eval page."""
    return {
        "group_order": list(dashboard.group_order),
        "latest_groups": dashboard.latest_run.as_dict()["groups"] if dashboard.latest_run else {},
        "history": [run.as_dict() for run in dashboard.history],
    }
