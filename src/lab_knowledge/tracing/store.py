from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lab_knowledge.config import Settings
from lab_knowledge.storage.paths import store_path
from lab_knowledge.tracing.context import TraceContext
from lab_knowledge.tracing.writer import JsonlTraceWriter


def stamp_trace_collection_id(payload: dict[str, Any], settings: Settings) -> dict[str, Any]:
    stamped = dict(payload)
    stamped["collection_id"] = settings.product.collection
    return stamped


def trace_belongs_to_collection(
    record: dict[str, Any],
    collection_id: str,
    *,
    default_collection_id: str,
) -> bool:
    record_collection = record.get("collection_id")
    if record_collection is None:
        return collection_id == default_collection_id
    return record_collection == collection_id


def filter_trace_records(
    records: list[dict[str, Any]],
    *,
    collection_id: str,
    default_collection_id: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if trace_belongs_to_collection(
            record,
            collection_id,
            default_collection_id=default_collection_id,
        )
    ]


def save_trace(settings: Settings, trace: TraceContext) -> None:
    if trace.finished_at is None:
        trace.close()
    payload = stamp_trace_collection_id(trace.to_dict(), settings)
    JsonlTraceWriter(store_path(settings, "traces")).write_payload(payload)


def _trace_file(settings: Settings) -> Path:
    return store_path(settings, "traces")


def _read_all_trace_records(settings: Settings) -> list[dict[str, Any]]:
    path = _trace_file(settings)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def read_trace_records(
    settings: Settings,
    *,
    collection_id: str | None = None,
) -> list[dict[str, Any]]:
    records = _read_all_trace_records(settings)
    if collection_id is None:
        return records
    return filter_trace_records(
        records,
        collection_id=collection_id,
        default_collection_id=settings.default_collection_id,
    )


def get_trace_record(
    settings: Settings,
    trace_id: str,
    *,
    collection_id: str | None = None,
) -> dict[str, Any] | None:
    for record in read_trace_records(settings, collection_id=collection_id):
        if record.get("trace_id") == trace_id:
            return record
    return None


def average_query_latency_ms(
    settings: Settings,
    *,
    collection_id: str | None = None,
) -> float | None:
    elapsed = [
        float(record["total_elapsed_ms"])
        for record in read_trace_records(settings, collection_id=collection_id)
        if record.get("trace_type") == "query" and record.get("total_elapsed_ms") is not None
    ]
    if not elapsed:
        return None
    return sum(elapsed) / len(elapsed)
