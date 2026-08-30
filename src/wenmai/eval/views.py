from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ABLATION_GROUP_LABELS: dict[str, str] = {
    "dense_only": "Dense 单路",
    "sparse_only": "Sparse 单路",
    "rrf": "RRF 融合",
    "rrf_rerank": "RRF + Rerank",
}

REWRITE_COMPARE_GROUP_LABELS: dict[str, str] = {
    "rewrite_off": "无改写",
    "rewrite_on": "术语归一+Multi-Query",
}


def group_label(name: str) -> str:
    if name in ABLATION_GROUP_LABELS:
        return ABLATION_GROUP_LABELS[name]
    return REWRITE_COMPARE_GROUP_LABELS.get(name, name)


@dataclass(frozen=True)
class FailedEvalItem:
    group: str
    item_id: str

    @property
    def group_label(self) -> str:
        return group_label(self.group)

    def as_dict(self) -> dict[str, str]:
        return {
            "group": self.group,
            "group_label": self.group_label,
            "item_id": self.item_id,
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

    def as_dict(self) -> dict[str, float | int | None]:
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
    failed_count: int
    failures: list[FailedEvalItem]

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "item_count": self.item_count,
            "golden_set": self.golden_set,
            "failed_count": self.failed_count,
            "failures": [item.as_dict() for item in self.failures],
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
    provider: str
    provider_label: str
    model: str
    faithfulness: RagasMetricStatus
    context_precision: RagasMetricStatus

    def as_dict(self) -> dict[str, object]:
        return {
            "configured": self.configured,
            "provider": self.provider,
            "provider_label": self.provider_label,
            "model": self.model,
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


def parse_eval_run(raw: dict[str, Any]) -> EvalRunView | None:
    timestamp = str(raw.get("timestamp") or "")
    if not timestamp:
        return None
    groups_raw = raw.get("groups") or {}
    groups: dict[str, GroupRunView] = {}
    for name, payload in groups_raw.items():
        if not isinstance(payload, dict):
            continue
        metrics_raw = payload.get("metrics")
        if not isinstance(metrics_raw, dict):
            continue
        groups[name] = GroupRunView(
            name=name,
            label=group_label(name),
            metrics=GroupMetricsView.from_dict(metrics_raw),
        )
    if not groups:
        return None
    failures: list[FailedEvalItem] = []
    for item in raw.get("failures") or []:
        if not isinstance(item, dict):
            continue
        group = str(item.get("group") or "")
        item_id = str(item.get("item_id") or "")
        if group and item_id:
            failures.append(FailedEvalItem(group=group, item_id=item_id))
    return EvalRunView(
        timestamp=timestamp,
        item_count=int(raw.get("item_count") or 0),
        golden_set=str(raw.get("golden_set") or ""),
        groups=groups,
        failed_count=len(failures),
        failures=failures,
    )
