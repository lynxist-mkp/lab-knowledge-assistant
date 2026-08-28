from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.storage.paths import store_path
from wenmai.tracing.context import TraceContext
from wenmai.tracing.writer import JsonlTraceWriter


def save_trace(settings: Settings, trace: TraceContext) -> None:
    if trace.finished_at is None:
        trace.close()
    JsonlTraceWriter(store_path(settings, "traces")).write(trace)


def _trace_file(settings: Settings) -> Path:
    return store_path(settings, "traces")


def read_trace_records(settings: Settings) -> list[dict[str, Any]]:
    path = _trace_file(settings)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def get_trace_record(settings: Settings, trace_id: str) -> dict[str, Any] | None:
    for record in read_trace_records(settings):
        if record.get("trace_id") == trace_id:
            return record
    return None


def average_query_latency_ms(settings: Settings) -> float | None:
    elapsed = [
        float(record["total_elapsed_ms"])
        for record in read_trace_records(settings)
        if record.get("trace_type") == "query" and record.get("total_elapsed_ms") is not None
    ]
    if not elapsed:
        return None
    return sum(elapsed) / len(elapsed)
