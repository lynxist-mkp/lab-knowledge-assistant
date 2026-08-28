from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wenmai.tracing.steps import (
    DEGRADATION_LABELS,
    INGESTION_LABELS,
    StepRow,
    stage_label,
    steps_from_record,
)


@dataclass
class IngestionTraceSummary:
    trace_id: str
    started_at: str
    finished_at: str | None
    total_elapsed_ms: float
    status: str
    chunk_count: int
    image_chunk_count: int
    source_path: str
    document_id: str
    skipped: bool
    error: str | None
    degradation_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_type": "ingestion",
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.total_elapsed_ms,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "image_chunk_count": self.image_chunk_count,
            "source_path": self.source_path,
            "document_id": self.document_id,
            "skipped": self.skipped,
            "error": self.error,
            "degradation_count": self.degradation_count,
        }


@dataclass
class StageDegradation:
    stage: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "reason": self.reason}


@dataclass
class IngestionTraceDetail:
    summary: IngestionTraceSummary
    degradations: list[StageDegradation]
    steps: list[StepRow]

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.summary.as_dict(),
            "degradations": [item.as_dict() for item in self.degradations],
            "steps": [item.as_dict() for item in self.steps],
        }


def _stage_by_name(record: dict[str, Any], name: str) -> dict[str, Any] | None:
    for stage in record.get("stages") or []:
        if stage.get("name") == name:
            return stage
    return None


def _integrity_status(record: dict[str, Any]) -> tuple[str, bool]:
    metadata = record.get("metadata") or {}
    status = metadata.get("status")
    if isinstance(status, str):
        return status, status == "skipped"

    integrity = _stage_by_name(record, "integrity")
    if integrity is None:
        return "unknown", False
    summary = str(integrity.get("output_summary") or "")
    if summary.startswith("skipped"):
        return "skipped", True
    if summary.startswith("rebuilt"):
        return "rebuilt", False
    return "ingested", False


def _chunk_counts(record: dict[str, Any]) -> tuple[int, int]:
    metadata = record.get("metadata") or {}
    chunk_count = metadata.get("chunk_count")
    image_chunk_count = metadata.get("chunks_with_images")
    if isinstance(chunk_count, int) and isinstance(image_chunk_count, int):
        return chunk_count, image_chunk_count

    split_stage = _stage_by_name(record, "split")
    upsert_stage = _stage_by_name(record, "upsert")
    resolved_chunk_count = 0
    if split_stage and split_stage.get("candidate_count") is not None:
        resolved_chunk_count = int(split_stage["candidate_count"])
    elif upsert_stage and upsert_stage.get("candidate_count") is not None:
        resolved_chunk_count = int(upsert_stage["candidate_count"])
    return resolved_chunk_count, 0


def _source_path(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    source_path = metadata.get("source_path")
    if isinstance(source_path, str) and source_path:
        return source_path
    load_stage = _stage_by_name(record, "load")
    if load_stage is None:
        return ""
    return str(load_stage.get("input_summary") or "")


def _document_id(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    document_id = metadata.get("document_id")
    if isinstance(document_id, str) and document_id:
        return document_id
    integrity = _stage_by_name(record, "integrity")
    if integrity is None:
        return ""
    return str(integrity.get("input_summary") or "")


def list_degradations(record: dict[str, Any]) -> list[StageDegradation]:
    degradations: list[StageDegradation] = []

    enricher = _stage_by_name(record, "enricher")
    if enricher and enricher.get("error"):
        degradations.append(
            StageDegradation(
                stage=stage_label("enricher", DEGRADATION_LABELS),
                reason=str(
                    enricher.get("error") or enricher.get("output_summary") or "Enricher 解析失败"
                ),
            )
        )

    captioner = _stage_by_name(record, "captioner")
    if captioner:
        if captioner.get("error"):
            degradations.append(
                StageDegradation(
                    stage=stage_label("captioner", DEGRADATION_LABELS),
                    reason=str(
                        captioner.get("error")
                        or captioner.get("output_summary")
                        or "Captioner 视觉失败"
                    ),
                )
            )
        elif "kept placeholder" in str(captioner.get("output_summary") or "").lower():
            degradations.append(
                StageDegradation(
                    stage=stage_label("captioner", DEGRADATION_LABELS),
                    reason=str(captioner.get("output_summary") or "Captioner 保留占位符"),
                )
            )

    discarded = record.get("metadata", {}).get("transform_discarded") or []
    if discarded:
        reasons = "; ".join(
            f"{item.get('chunk_id', '?')}: {item.get('reason', 'discarded')}"
            for item in discarded[:3]
        )
        if len(discarded) > 3:
            reasons += f" (+{len(discarded) - 3} more)"
        degradations.append(
            StageDegradation(stage=stage_label("refiner", DEGRADATION_LABELS), reason=reasons)
        )

    load_stage = _stage_by_name(record, "load")
    if load_stage and load_stage.get("error"):
        degradations.append(
            StageDegradation(
                stage=stage_label("load", DEGRADATION_LABELS),
                reason=str(load_stage["error"]),
            )
        )

    if record.get("error"):
        degradations.append(
            StageDegradation(
                stage=stage_label("pipeline", DEGRADATION_LABELS),
                reason=str(record["error"]),
            )
        )

    return degradations


def summarize_ingestion_trace(record: dict[str, Any]) -> IngestionTraceSummary:
    status, skipped = _integrity_status(record)
    if record.get("error") and status not in {"skipped"}:
        status = "failed"
    chunk_count, image_chunk_count = _chunk_counts(record)
    degradations = list_degradations(record)
    return IngestionTraceSummary(
        trace_id=str(record.get("trace_id") or ""),
        started_at=str(record.get("started_at") or ""),
        finished_at=record.get("finished_at"),
        total_elapsed_ms=float(record.get("total_elapsed_ms") or 0.0),
        status=status,
        chunk_count=chunk_count,
        image_chunk_count=image_chunk_count,
        source_path=_source_path(record),
        document_id=_document_id(record),
        skipped=skipped,
        error=record.get("error"),
        degradation_count=len(degradations),
    )


def ingestion_trace_detail(record: dict[str, Any]) -> IngestionTraceDetail:
    degradations = list_degradations(record)
    return IngestionTraceDetail(
        summary=summarize_ingestion_trace(record),
        degradations=degradations,
        steps=steps_from_record(record, INGESTION_LABELS),
    )
