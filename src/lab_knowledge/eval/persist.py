"""评测 run 持久化：runs 目录与 artifact 写入的单一 seam。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lab_knowledge.config import Settings
from lab_knowledge.eval.views import EvalRunSummary, parse_eval_run_summary


def runs_dir(settings: Settings, *, mkdir: bool = False) -> Path:
    raw = Path(settings.evaluation.runs)
    path = raw if raw.is_absolute() else settings.root / raw
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def stamp_eval_run_collection_id(
    payload: dict[str, Any], settings: Settings
) -> dict[str, Any]:
    stamped = dict(payload)
    stamped["collection_id"] = settings.product.collection
    return stamped


def eval_run_belongs_to_collection(
    record: dict[str, Any],
    collection_id: str,
    *,
    default_collection_id: str,
) -> bool:
    record_collection = record.get("collection_id")
    if record_collection is None:
        return collection_id == default_collection_id
    return record_collection == collection_id


def filter_eval_run_records(
    records: list[dict[str, Any]],
    *,
    collection_id: str,
    default_collection_id: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if eval_run_belongs_to_collection(
            record,
            collection_id,
            default_collection_id=default_collection_id,
        )
    ]


def write_run_json(settings: Settings, filename: str, payload: dict[str, Any]) -> Path:
    """Write any JSON under the eval runs directory (EvalRun or Phase B summary)."""
    output_path = runs_dir(settings, mkdir=True) / filename
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def persist_eval_artifact(settings: Settings, artifact: dict[str, Any]) -> EvalRunSummary:
    """Write EvalRun artifact and return the typed view."""
    timestamp = artifact.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("eval artifact missing timestamp")
    stamped = stamp_eval_run_collection_id(artifact, settings)
    write_run_json(settings, f"{timestamp}.json", stamped)
    parsed = parse_eval_run_summary(stamped)
    if parsed is None:
        raise RuntimeError("eval run artifact could not be parsed")
    return parsed


__all__ = [
    "eval_run_belongs_to_collection",
    "filter_eval_run_records",
    "persist_eval_artifact",
    "runs_dir",
    "stamp_eval_run_collection_id",
    "write_run_json",
]
