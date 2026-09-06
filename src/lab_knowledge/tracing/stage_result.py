from __future__ import annotations

from typing import Any

from lab_knowledge.tracing.context import StageRecord

STAGE_SCHEMA_VERSION = 2

# Typed stage representation shared by query trace read/write paths.
StageResult = StageRecord

_STAGE_FIELDS = frozenset(StageRecord.__dataclass_fields__)


def stage_to_dict(stage: StageResult) -> dict[str, Any]:
    payload = stage.to_dict()
    payload["schema_version"] = STAGE_SCHEMA_VERSION
    return payload


def stage_from_dict(data: dict[str, Any]) -> StageResult:
    normalized = stage_as_dict(data)
    return StageRecord(
        **{key: normalized[key] for key in _STAGE_FIELDS if key in normalized}
    )


def stage_as_dict(data: dict[str, Any]) -> dict[str, Any]:
    if "stage" in data and isinstance(data["stage"], dict):
        return stage_as_dict(data["stage"])
    if data.get("schema_version") == STAGE_SCHEMA_VERSION:
        return {key: value for key, value in data.items() if key != "schema_version"}
    return data


def normalize_query_stages(record: dict[str, Any]) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for item in record.get("stages") or []:
        if not isinstance(item, dict):
            continue
        stages.append(stage_as_dict(item))
    return stages
