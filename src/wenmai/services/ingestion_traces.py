from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def _stage_by_name(trace: dict[str, Any], name: str) -> dict[str, Any] | None:
    for stage in trace.get("stages") or []:
        if stage.get("name") == name:
            return stage
    return None


def _integrity_status(trace: dict[str, Any]) -> tuple[str, bool]:
    metadata = trace.get("metadata") or {}
    status = metadata.get("status")
    if isinstance(status, str):
        return status, status == "skipped"

    integrity = _stage_by_name(trace, "integrity")
    if integrity is None:
        return "unknown", False
    summary = str(integrity.get("output_summary") or "")
    if summary.startswith("skipped"):
        return "skipped", True
    if summary.startswith("rebuilt"):
        return "rebuilt", False
    return "ingested", False


def _chunk_counts(trace: dict[str, Any]) -> tuple[int, int]:
    metadata = trace.get("metadata") or {}
    chunk_count = metadata.get("chunk_count")
    image_chunk_count = metadata.get("chunks_with_images")
    if isinstance(chunk_count, int) and isinstance(image_chunk_count, int):
        return chunk_count, image_chunk_count

    split_stage = _stage_by_name(trace, "split")
    upsert_stage = _stage_by_name(trace, "upsert")
    resolved_chunk_count = 0
    if split_stage and split_stage.get("candidate_count") is not None:
        resolved_chunk_count = int(split_stage["candidate_count"])
    elif upsert_stage and upsert_stage.get("candidate_count") is not None:
        resolved_chunk_count = int(upsert_stage["candidate_count"])
    return resolved_chunk_count, 0


def _source_path(trace: dict[str, Any]) -> str:
    metadata = trace.get("metadata") or {}
    source_path = metadata.get("source_path")
    if isinstance(source_path, str) and source_path:
        return source_path
    load_stage = _stage_by_name(trace, "load")
    if load_stage is None:
        return ""
    return str(load_stage.get("input_summary") or "")


def _document_id(trace: dict[str, Any]) -> str:
    metadata = trace.get("metadata") or {}
    document_id = metadata.get("document_id")
    if isinstance(document_id, str) and document_id:
        return document_id
    integrity = _stage_by_name(trace, "integrity")
    if integrity is None:
        return ""
    return str(integrity.get("input_summary") or "")


def list_degradations(trace: dict[str, Any]) -> list[StageDegradation]:
    degradations: list[StageDegradation] = []

    enricher = _stage_by_name(trace, "enricher")
    if enricher and enricher.get("error"):
        degradations.append(
            StageDegradation(
                stage="enricher",
                reason=str(
                    enricher.get("error") or enricher.get("output_summary") or "Enricher 解析失败"
                ),
            )
        )

    captioner = _stage_by_name(trace, "captioner")
    if captioner:
        if captioner.get("error"):
            degradations.append(
                StageDegradation(
                    stage="captioner",
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
                    stage="captioner",
                    reason=str(captioner.get("output_summary") or "Captioner 保留占位符"),
                )
            )

    discarded = trace.get("metadata", {}).get("transform_discarded") or []
    if discarded:
        reasons = "; ".join(
            f"{item.get('chunk_id', '?')}: {item.get('reason', 'discarded')}"
            for item in discarded[:3]
        )
        if len(discarded) > 3:
            reasons += f" (+{len(discarded) - 3} more)"
        degradations.append(StageDegradation(stage="refiner", reason=reasons))

    load_stage = _stage_by_name(trace, "load")
    if load_stage and load_stage.get("error"):
        degradations.append(
            StageDegradation(stage="load", reason=str(load_stage["error"]))
        )

    if trace.get("error"):
        degradations.append(StageDegradation(stage="pipeline", reason=str(trace["error"])))

    return degradations


def summarize_ingestion_trace(trace: dict[str, Any]) -> IngestionTraceSummary:
    status, skipped = _integrity_status(trace)
    if trace.get("error") and status not in {"skipped"}:
        status = "failed"
    chunk_count, image_chunk_count = _chunk_counts(trace)
    degradations = list_degradations(trace)
    return IngestionTraceSummary(
        trace_id=str(trace.get("trace_id") or ""),
        started_at=str(trace.get("started_at") or ""),
        finished_at=trace.get("finished_at"),
        total_elapsed_ms=float(trace.get("total_elapsed_ms") or 0.0),
        status=status,
        chunk_count=chunk_count,
        image_chunk_count=image_chunk_count,
        source_path=_source_path(trace),
        document_id=_document_id(trace),
        skipped=skipped,
        error=trace.get("error"),
        degradation_count=len(degradations),
    )
