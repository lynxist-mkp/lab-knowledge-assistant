from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from wenmai.eval.golden import GoldItem
from wenmai.eval.metrics import hit_at_5

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
class EvalRunSummary:
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
class EvalRetrievalEvidenceView:
    ranked_doc_ids: list[str]
    ranked_chunks: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EvalRetrievalEvidenceView:
        retrieval = raw.get("retrieval") if "retrieval" in raw else raw
        if not isinstance(retrieval, dict):
            retrieval = {}
        ranked_doc_ids = [
            str(doc_id) for doc_id in (retrieval.get("ranked_doc_ids") or [])
        ]
        ranked_chunks = [
            dict(chunk)
            for chunk in (retrieval.get("ranked_chunks") or [])
            if isinstance(chunk, dict)
        ]
        return cls(ranked_doc_ids=ranked_doc_ids, ranked_chunks=ranked_chunks)

    def as_dict(self) -> dict[str, object]:
        return {
            "ranked_doc_ids": list(self.ranked_doc_ids),
            "ranked_chunks": [dict(chunk) for chunk in self.ranked_chunks],
        }


@dataclass(frozen=True)
class EvalItemEvidenceView:
    item_id: str
    retrieval: EvalRetrievalEvidenceView
    refused: bool | None = None
    citation_count: int | None = None

    @classmethod
    def from_dict(cls, item_id: str, raw: dict[str, Any]) -> EvalItemEvidenceView:
        refused_raw = raw.get("refused")
        citation_raw = raw.get("citation_count")
        return cls(
            item_id=item_id,
            retrieval=EvalRetrievalEvidenceView.from_dict(raw),
            refused=bool(refused_raw) if refused_raw is not None else None,
            citation_count=int(citation_raw) if citation_raw is not None else None,
        )

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "item_id": self.item_id,
            "retrieval": self.retrieval.as_dict(),
        }
        if self.refused is not None:
            payload["refused"] = self.refused
        if self.citation_count is not None:
            payload["citation_count"] = self.citation_count
        return payload


@dataclass(frozen=True)
class EvalGroupDetailView:
    name: str
    label: str
    metrics: GroupMetricsView
    config: dict[str, Any]
    items: dict[str, EvalItemEvidenceView]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "metrics": self.metrics.as_dict(),
            "config": dict(self.config),
            "items": {
                item_id: item.as_dict() for item_id, item in self.items.items()
            },
        }


@dataclass(frozen=True)
class RagasRunSummary:
    status: str
    faithfulness: float | None = None
    context_precision: float | None = None
    reason: str = ""
    scored_count: int | None = None
    skipped_count: int | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": self.status,
            "reason": self.reason,
        }
        if self.faithfulness is not None:
            payload["faithfulness"] = self.faithfulness
        if self.context_precision is not None:
            payload["context_precision"] = self.context_precision
        if self.scored_count is not None:
            payload["scored_count"] = self.scored_count
        if self.skipped_count is not None:
            payload["skipped_count"] = self.skipped_count
        return payload


@dataclass(frozen=True)
class HitAt5MissView:
    item_id: str
    question: str
    category: str
    evidence_doc_ids: str

    def as_dict(self) -> dict[str, str]:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "category": self.category,
            "evidence_doc_ids": self.evidence_doc_ids,
        }


@dataclass(frozen=True)
class EvalRunDetail:
    summary: EvalRunSummary
    groups: dict[str, EvalGroupDetailView]
    ragas: RagasRunSummary

    @property
    def timestamp(self) -> str:
        return self.summary.timestamp

    def as_dict(self) -> dict[str, object]:
        return {
            **self.summary.as_dict(),
            "ragas": self.ragas.as_dict(),
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
    latest_run: EvalRunSummary | None
    history: list[EvalRunSummary]
    ragas: RagasStatusView
    group_order: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "latest_run": self.latest_run.as_dict() if self.latest_run else None,
            "history": [run.as_dict() for run in self.history],
            "ragas": self.ragas.as_dict(),
            "group_order": list(self.group_order),
        }


def parse_eval_run_summary(raw: dict[str, Any]) -> EvalRunSummary | None:
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
    return EvalRunSummary(
        timestamp=timestamp,
        item_count=int(raw.get("item_count") or 0),
        golden_set=str(raw.get("golden_set") or ""),
        groups=groups,
        failed_count=len(failures),
        failures=failures,
    )

def parse_ragas_run_summary(raw: dict[str, Any]) -> RagasRunSummary:
    ragas_raw = raw.get("ragas")
    if ragas_raw is None:
        group_payload = (raw.get("groups") or {}).get("rrf_rerank") or {}
        metrics_raw = group_payload.get("metrics") or {}
        ragas_raw = metrics_raw.get("ragas")
    if not isinstance(ragas_raw, dict):
        return RagasRunSummary(status="unavailable", reason="未执行 Ragas")
    status = str(ragas_raw.get("status") or "unavailable")
    faithfulness = ragas_raw.get("faithfulness")
    context_precision = ragas_raw.get("context_precision")
    scored_count = ragas_raw.get("scored_count")
    skipped_count = ragas_raw.get("skipped_count")
    return RagasRunSummary(
        status=status,
        faithfulness=float(faithfulness) if faithfulness is not None else None,
        context_precision=float(context_precision)
        if context_precision is not None
        else None,
        reason=str(ragas_raw.get("reason") or ""),
        scored_count=int(scored_count) if scored_count is not None else None,
        skipped_count=int(skipped_count) if skipped_count is not None else None,
    )


def parse_eval_run_detail(raw: dict[str, Any]) -> EvalRunDetail | None:
    summary = parse_eval_run_summary(raw)
    if summary is None:
        return None
    groups_raw = raw.get("groups") or {}
    groups: dict[str, EvalGroupDetailView] = {}
    for name, payload in groups_raw.items():
        if not isinstance(payload, dict):
            continue
        metrics_raw = payload.get("metrics")
        if not isinstance(metrics_raw, dict):
            continue
        items_raw = payload.get("items") or {}
        items: dict[str, EvalItemEvidenceView] = {}
        if isinstance(items_raw, dict):
            for item_id, item_payload in items_raw.items():
                if not isinstance(item_payload, dict):
                    continue
                items[str(item_id)] = EvalItemEvidenceView.from_dict(
                    str(item_id),
                    item_payload,
                )
        groups[name] = EvalGroupDetailView(
            name=name,
            label=group_label(name),
            metrics=GroupMetricsView.from_dict(metrics_raw),
            config=dict(payload.get("config") or {}),
            items=items,
        )
    return EvalRunDetail(
        summary=summary,
        groups=groups,
        ragas=parse_ragas_run_summary(raw),
    )


def find_hit_at_5_misses(
    run: EvalRunDetail,
    items: list[GoldItem],
    *,
    group: str = "rrf_rerank",
) -> list[HitAt5MissView]:
    group_detail = run.groups.get(group)
    if group_detail is None:
        return []
    misses: list[HitAt5MissView] = []
    for item in items:
        if not item.answerable:
            continue
        evidence = group_detail.items.get(item.id)
        if evidence is None:
            continue
        score = hit_at_5(evidence.retrieval.ranked_doc_ids, item)
        if score == 0.0:
            misses.append(
                HitAt5MissView(
                    item_id=item.id,
                    question=item.question,
                    category=item.category,
                    evidence_doc_ids=",".join(item.evidence_doc_ids),
                )
            )
    return misses
