"""Ask-path saturation and governor evidence for 运维观测."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wenmai.config import Settings


@dataclass(frozen=True)
class AskEvidenceSummary:
    total_events: int
    saturation_total: int
    busy_total: int
    timeout_total: int
    long_task_total: int
    release_total: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total_events": self.total_events,
            "saturation_total": self.saturation_total,
            "busy_total": self.busy_total,
            "timeout_total": self.timeout_total,
            "long_task_total": self.long_task_total,
            "release_total": self.release_total,
        }


def ask_evidence_path(settings: Settings) -> Path:
    raw = Path(settings.observability.ask_evidence_file)
    return raw if raw.is_absolute() else settings.root / raw


def write_ask_evidence(settings: Settings, payload: dict[str, Any]) -> Path:
    path = ask_evidence_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def safe_record_ask_evidence(settings: Settings, payload: dict[str, Any]) -> Path | None:
    try:
        return write_ask_evidence(settings, payload)
    except OSError:
        return None


def read_ask_evidence_records(settings: Settings) -> list[dict[str, Any]]:
    path = ask_evidence_path(settings)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            records.append(raw)
    return records


def summarize_ask_evidence(
    settings: Settings,
    *,
    recent_n: int | None = None,
) -> AskEvidenceSummary:
    records = read_ask_evidence_records(settings)
    if recent_n is not None and recent_n > 0:
        records = records[-recent_n:]
    busy = timeout = long_task = release = saturation = 0
    for record in records:
        event = str(record.get("event") or "")
        if event == "saturation":
            saturation += 1
            code = str(record.get("code") or "")
            if code == "busy":
                busy += 1
            elif code == "timeout":
                timeout += 1
            elif code == "long_task_active":
                long_task += 1
        elif event == "release":
            release += 1
    return AskEvidenceSummary(
        total_events=len(records),
        saturation_total=saturation,
        busy_total=busy,
        timeout_total=timeout,
        long_task_total=long_task,
        release_total=release,
    )


__all__ = [
    "AskEvidenceSummary",
    "ask_evidence_path",
    "read_ask_evidence_records",
    "safe_record_ask_evidence",
    "summarize_ask_evidence",
    "write_ask_evidence",
]
