from __future__ import annotations

import math
from typing import Any

from wenmai.config import Settings
from wenmai.tracing.stage_result import stage_as_dict
from wenmai.tracing.store import read_trace_records

StageLatencyPercentiles = dict[str, float | None]


def percentile_nearest_rank(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: rank = ceil(p/100 * n), 1-indexed."""
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(p / 100.0 * len(ordered))
    return ordered[rank - 1]


def stage_percentiles(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": percentile_nearest_rank(values, 50),
        "p95": percentile_nearest_rank(values, 95),
    }


def _query_records(
    settings: Settings,
    *,
    collection_id: str | None = None,
    started_at_min: str | None = None,
    recent_n: int | None = None,
) -> list[dict[str, Any]]:
    records = [
        record
        for record in read_trace_records(settings, collection_id=collection_id)
        if record.get("trace_type") == "query"
    ]
    if started_at_min is not None:
        records = [
            record
            for record in records
            if str(record.get("started_at") or "") >= started_at_min
        ]
    if recent_n is not None and recent_n > 0 and len(records) > recent_n:
        records = records[-recent_n:]
    return records


def collect_stage_elapsed_ms(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    by_stage: dict[str, list[float]] = {}
    for record in records:
        for stage in record.get("stages") or []:
            if not isinstance(stage, dict):
                continue
            normalized = stage_as_dict(stage)
            name = normalized.get("name")
            elapsed = normalized.get("elapsed_ms")
            if not name or elapsed is None:
                continue
            by_stage.setdefault(str(name), []).append(float(elapsed))
    return by_stage


def latency_ms_payload(
    records: list[dict[str, Any]],
) -> dict[str, StageLatencyPercentiles]:
    totals = [
        float(record["total_elapsed_ms"])
        for record in records
        if record.get("total_elapsed_ms") is not None
    ]
    payload: dict[str, StageLatencyPercentiles] = {
        "total": stage_percentiles(totals),
    }
    for stage_name, values in collect_stage_elapsed_ms(records).items():
        payload[stage_name] = stage_percentiles(values)
    return payload


def query_latency_percentiles(
    settings: Settings,
    *,
    collection_id: str | None = None,
    started_at_min: str | None = None,
    recent_n: int | None = None,
) -> dict[str, StageLatencyPercentiles]:
    return latency_ms_payload(
        _query_records(
            settings,
            collection_id=collection_id,
            started_at_min=started_at_min,
            recent_n=recent_n,
        )
    )
