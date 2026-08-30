from __future__ import annotations

from typing import Any

from wenmai.tracing.views.query import (
    CandidateRow,
    QueryTraceDetail,
    QueryTraceSummary,
    RankChange,
    StageLatency,
    build_query_trace_detail,
    build_query_trace_summary,
    list_stage_latencies,
)

__all__ = [
    "CandidateRow",
    "QueryTraceDetail",
    "QueryTraceSummary",
    "RankChange",
    "StageLatency",
    "build_query_trace_detail",
    "build_query_trace_summary",
    "list_stage_latencies",
    "query_trace_detail",
    "summarize_query_trace",
]


def summarize_query_trace(record: dict[str, Any]) -> QueryTraceSummary:
    return build_query_trace_summary(record)


def query_trace_detail(record: dict[str, Any]) -> QueryTraceDetail:
    return build_query_trace_detail(record)
