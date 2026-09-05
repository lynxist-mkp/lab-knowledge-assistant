"""运维观测：概览、任务进展、联查与 Trace 读的深 module。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wenmai.config import Settings
from wenmai.eval.read import get_eval_run
from wenmai.eval.views import EvalRunView
from wenmai.knowledge.browse import OverviewStats
from wenmai.storage.catalog import DocumentCatalog
from wenmai.task_progress import (
    TaskProgressDetail,
    TaskProgressSummary,
    get_task_progress,
    list_task_progress,
)
from wenmai.tracing.ingestion_views import (
    IngestionTraceDetail,
    IngestionTraceSummary,
    StageDegradation,
    ingestion_trace_detail,
    list_degradations,
    summarize_ingestion_trace,
)
from wenmai.tracing.latency import query_latency_percentiles
from wenmai.tracing.query_trace import (
    QueryTrace,
    QueryTraceDetail,
    QueryTraceSummary,
)
from wenmai.tracing.store import average_query_latency_ms, get_trace_record, read_trace_records

TraceSummary = QueryTraceSummary | IngestionTraceSummary
TraceDetail = QueryTraceDetail | IngestionTraceDetail


@dataclass(frozen=True)
class EvidenceLink:
    link_type: str
    target_id: str
    label: str
    status: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "link_type": self.link_type,
            "target_id": self.target_id,
            "label": self.label,
            "metadata": dict(self.metadata),
        }
        if self.status is not None:
            payload["status"] = self.status
        return payload


@dataclass(frozen=True)
class TaskInvestigationView:
    task: TaskProgressDetail
    trace_summaries: list[TraceSummary]
    eval_run: EvalRunView | None
    links: list[EvidenceLink]
    config_related_tasks: list[TaskProgressSummary]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.as_dict(),
            "trace_summaries": [item.as_dict() for item in self.trace_summaries],
            "eval_run": self.eval_run.as_dict() if self.eval_run is not None else None,
            "links": [item.as_dict() for item in self.links],
            "config_related_tasks": [item.as_dict() for item in self.config_related_tasks],
        }


@dataclass(frozen=True)
class HealthSignal:
    name: str
    count: int

    def as_dict(self) -> dict[str, int | str]:
        return {"name": self.name, "count": self.count}


@dataclass(frozen=True)
class ObservationHealthSnapshot:
    signals: list[HealthSignal]
    anomalies: list[TaskProgressSummary]

    def as_dict(self) -> dict[str, Any]:
        return {
            "signals": [item.as_dict() for item in self.signals],
            "anomalies": [item.as_dict() for item in self.anomalies],
        }


def load_overview_stats(settings: Settings) -> OverviewStats:
    """Settings → OverviewStats：目录计数 + 查询延迟分位（含 stage_latency）。"""
    catalog = DocumentCatalog.from_settings(settings)
    latency = query_latency_percentiles(
        settings,
        recent_n=settings.observability.query_latency_recent_n,
    )
    total = latency.get("total") or {}
    return OverviewStats(
        document_count=catalog.document_count,
        chunk_count=catalog.chunk_count,
        avg_query_latency_ms=average_query_latency_ms(settings),
        query_latency_p50_ms=total.get("p50"),
        query_latency_p95_ms=total.get("p95"),
        stage_latency=latency,
    )


def list_query_summaries(settings: Settings) -> list[QueryTraceSummary]:
    records = [
        record for record in read_trace_records(settings) if record.get("trace_type") == "query"
    ]
    return [QueryTrace.summarize(record) for record in reversed(records)]


def list_task_progress_summaries(
    settings: Settings,
    *,
    task_type: str | None = None,
    status: str | None = None,
    failure_kind: str | None = None,
    degraded: bool | None = None,
    has_trace: bool | None = None,
    config_fingerprint: str | None = None,
    needs_attention: bool | None = None,
) -> list[TaskProgressSummary]:
    summaries = list_task_progress(
        settings,
        task_type=task_type,
        status=status,
        failure_kind=failure_kind,
    )
    if degraded is not None:
        summaries = [item for item in summaries if item.degraded is degraded]
    if has_trace is not None:
        summaries = [_item for _item in summaries if _has_trace_link(_item) is has_trace]
    if config_fingerprint is not None:
        summaries = [
            item for item in summaries if item.config_fingerprint == config_fingerprint
        ]
    if needs_attention is not None:
        summaries = [
            item for item in summaries if _needs_attention(item) is needs_attention
        ]
    return summaries


def get_task_progress_detail(settings: Settings, task_id: str) -> TaskProgressDetail | None:
    return get_task_progress(settings, task_id)


def get_task_investigation(settings: Settings, task_id: str) -> TaskInvestigationView | None:
    detail = get_task_progress(settings, task_id)
    if detail is None:
        return None
    eval_run_id = detail.summary.links.get("eval_run")
    return TaskInvestigationView(
        task=detail,
        trace_summaries=[
            summary
            for trace_id in _trace_ids(detail)
            if (summary := get_trace_summary(settings, trace_id)) is not None
        ],
        eval_run=get_eval_run(settings, eval_run_id) if eval_run_id else None,
        links=_evidence_links(detail),
        config_related_tasks=[
            item
            for item in list_task_progress_summaries(
                settings,
                config_fingerprint=detail.summary.config_fingerprint,
            )
            if item.task_id != detail.summary.task_id
        ][:10],
    )


def load_health_snapshot(
    settings: Settings,
    *,
    task_type: str | None = None,
    failure_kind: str | None = None,
) -> ObservationHealthSnapshot:
    summaries = list_task_progress_summaries(
        settings,
        task_type=task_type,
        failure_kind=failure_kind,
    )
    signals = [
        HealthSignal(
            name="running",
            count=sum(1 for item in summaries if item.status == "running"),
        ),
        HealthSignal(
            name="failed",
            count=sum(1 for item in summaries if item.status == "failed"),
        ),
        HealthSignal(
            name="blocked",
            count=sum(1 for item in summaries if item.status == "blocked"),
        ),
        HealthSignal(
            name="partial_success",
            count=sum(1 for item in summaries if item.status == "partial_success"),
        ),
        HealthSignal(name="degraded", count=sum(1 for item in summaries if item.degraded)),
        HealthSignal(
            name="attention_needed",
            count=sum(1 for item in summaries if _needs_attention(item)),
        ),
    ]
    anomalies = [item for item in summaries if _needs_attention(item)]
    anomalies.sort(
        key=lambda item: (
            _severity_rank(item),
            item.last_progress_at or item.finished_at or item.started_at,
        ),
        reverse=True,
    )
    return ObservationHealthSnapshot(signals=signals, anomalies=anomalies[:10])


def list_ingestion_summaries(settings: Settings) -> list[IngestionTraceSummary]:
    records = [
        record
        for record in read_trace_records(settings)
        if record.get("trace_type") == "ingestion"
    ]
    return [summarize_ingestion_trace(record) for record in reversed(records)]


def get_trace_summary(settings: Settings, trace_id: str) -> TraceSummary | None:
    record = get_trace_record(settings, trace_id)
    if record is None:
        return None
    trace_type = record.get("trace_type")
    if trace_type == "query":
        return QueryTrace.summarize(record)
    if trace_type == "ingestion":
        return summarize_ingestion_trace(record)
    return None


def get_trace_detail(settings: Settings, trace_id: str) -> TraceDetail | None:
    record = get_trace_record(settings, trace_id)
    if record is None:
        return None
    trace_type = record.get("trace_type")
    if trace_type == "query":
        return QueryTrace.detail(record)
    if trace_type == "ingestion":
        return ingestion_trace_detail(record)
    return None


def get_query_detail(settings: Settings, trace_id: str) -> QueryTraceDetail | None:
    detail = get_trace_detail(settings, trace_id)
    return detail if isinstance(detail, QueryTraceDetail) else None


def get_ingestion_detail(settings: Settings, trace_id: str) -> IngestionTraceDetail | None:
    detail = get_trace_detail(settings, trace_id)
    return detail if isinstance(detail, IngestionTraceDetail) else None


def list_trace_degradations(
    settings: Settings, trace_id: str
) -> list[StageDegradation] | None:
    record = get_trace_record(settings, trace_id)
    if record is None:
        return None
    if record.get("trace_type") != "ingestion":
        return []
    return list_degradations(record)


def _has_trace_link(summary: TaskProgressSummary) -> bool:
    if summary.links.get("trace_id"):
        return True
    return False


def _needs_attention(summary: TaskProgressSummary) -> bool:
    return summary.status in {"failed", "blocked", "partial_success"} or summary.degraded


def _severity_rank(summary: TaskProgressSummary) -> int:
    if summary.status == "failed":
        return 4
    if summary.status == "blocked":
        return 3
    if summary.status == "partial_success":
        return 2
    if summary.degraded:
        return 1
    return 0


def _trace_ids(detail: TaskProgressDetail) -> list[str]:
    trace_ids: list[str] = []
    seen: set[str] = set()

    def append(trace_id: str | None) -> None:
        if not trace_id or trace_id in seen:
            return
        seen.add(trace_id)
        trace_ids.append(trace_id)

    append(detail.summary.links.get("trace_id"))
    for child in detail.children:
        append(child.trace_id)
        append(child.links.get("trace_id"))
    for stage in detail.stages:
        append(stage.links.get("trace_id"))
    return trace_ids


def _evidence_links(detail: TaskProgressDetail) -> list[EvidenceLink]:
    links: list[EvidenceLink] = []
    seen: set[tuple[str, str]] = set()

    def append(link: EvidenceLink) -> None:
        key = (link.link_type, link.target_id)
        if not link.target_id or key in seen:
            return
        seen.add(key)
        links.append(link)

    for trace_id in _trace_ids(detail):
        append(EvidenceLink("trace", trace_id, "关联 Trace", status=detail.summary.status))

    eval_run_id = detail.summary.links.get("eval_run")
    if eval_run_id:
        append(EvidenceLink("eval_run", eval_run_id, "评测运行", status=detail.summary.status))

    document_id = detail.summary.links.get("document_id")
    if document_id:
        append(EvidenceLink("document", document_id, "文档", status=detail.summary.status))

    append(
        EvidenceLink(
            "config_fingerprint",
            detail.summary.config_fingerprint,
            "配置指纹",
            metadata={"task_type": detail.summary.task_type},
        )
    )

    for child in detail.children:
        child_trace_id = child.trace_id or child.links.get("trace_id")
        if child_trace_id:
            append(
                EvidenceLink(
                    "trace",
                    child_trace_id,
                    child.label or "关联 Trace",
                    status=child.status,
                    metadata={"child_id": child.child_id},
                )
            )
        child_document_id = child.links.get("document_id")
        if child_document_id:
            append(
                EvidenceLink(
                    "document",
                    child_document_id,
                    child.label or "文档",
                    status=child.status,
                    metadata={"child_id": child.child_id},
                )
            )
        item_id = child.links.get("item_id")
        if item_id:
            append(
                EvidenceLink(
                    "eval_item",
                    item_id,
                    child.label or item_id,
                    status=child.status,
                    metadata={"group": child.links.get("group", "")},
                )
            )
        for evidence_doc_id in child.detail.get("evidence_doc_ids") or []:
            append(
                EvidenceLink(
                    "document",
                    str(evidence_doc_id),
                    f"{child.label} evidence",
                    status=child.status,
                    metadata={"source": "evidence_doc_ids"},
                )
            )
        for retrieval_doc_id in child.detail.get("retrieval_doc_ids") or []:
            append(
                EvidenceLink(
                    "document",
                    str(retrieval_doc_id),
                    f"{child.label} retrieved",
                    status=child.status,
                    metadata={"source": "retrieval_doc_ids"},
                )
            )
    return links


__all__ = [
    "EvidenceLink",
    "HealthSignal",
    "ObservationHealthSnapshot",
    "TaskProgressDetail",
    "TaskInvestigationView",
    "TaskProgressSummary",
    "get_task_investigation",
    "get_task_progress_detail",
    "TraceDetail",
    "TraceSummary",
    "get_ingestion_detail",
    "get_query_detail",
    "get_trace_detail",
    "get_trace_summary",
    "list_ingestion_summaries",
    "list_query_summaries",
    "list_task_progress_summaries",
    "list_trace_degradations",
    "load_health_snapshot",
    "load_overview_stats",
]
