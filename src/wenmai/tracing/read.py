"""Trace read shims — prefer wenmai.ops.observation (运维观测)."""

from wenmai.ops.observation import (
    TraceDetail,
    TraceSummary,
    get_ingestion_detail,
    get_query_detail,
    get_trace_detail,
    get_trace_summary,
    list_ingestion_summaries,
    list_query_summaries,
    list_trace_degradations,
)

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
]
