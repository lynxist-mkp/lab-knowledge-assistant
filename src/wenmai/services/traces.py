from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.storage.paths import store_path


def trace_file_path(settings: Settings) -> Path:
    return store_path(settings, "traces")


def read_traces(settings: Settings) -> list[dict[str, Any]]:
    path = trace_file_path(settings)
    if not path.is_file():
        return []
    traces: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        traces.append(json.loads(line))
    return traces


def read_traces_by_type(settings: Settings, trace_type: str) -> list[dict[str, Any]]:
    return [trace for trace in read_traces(settings) if trace.get("trace_type") == trace_type]


def get_trace_by_id(settings: Settings, trace_id: str) -> dict[str, Any] | None:
    for trace in read_traces(settings):
        if trace.get("trace_id") == trace_id:
            return trace
    return None


def average_query_latency_ms(settings: Settings) -> float | None:
    query_elapsed = [
        float(trace["total_elapsed_ms"])
        for trace in read_traces(settings)
        if trace.get("trace_type") == "query" and trace.get("total_elapsed_ms") is not None
    ]
    if not query_elapsed:
        return None
    return sum(query_elapsed) / len(query_elapsed)
