"""运维观测：概览统计与 Trace 读的深 module。"""

from __future__ import annotations

from wenmai.config import Settings
from wenmai.knowledge.browse import OverviewStats
from wenmai.storage.catalog import DocumentCatalog
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


__all__ = [
    "TraceDetail",
    "TraceSummary",
    "get_ingestion_detail",
    "get_query_detail",
    "get_trace_detail",
    "get_trace_summary",
    "list_ingestion_summaries",
    "list_query_summaries",
    "list_trace_degradations",
    "load_overview_stats",
]
