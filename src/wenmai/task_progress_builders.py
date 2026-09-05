"""Builders that adapt existing seams into task progress records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wenmai.generation import GenerationResult
from wenmai.models import ScoredChunk
from wenmai.task_progress import (
    ChildEvidence,
    FailureKind,
    StageEvent,
    TaskProgressOutcome,
    infer_failure_kind,
)
from wenmai.tracing.ingestion_views import list_degradations

if TYPE_CHECKING:
    from wenmai.eval.golden import GoldItem
    from wenmai.eval.views import FailedEvalItem


def build_ingestion_progress(
    record: dict[str, Any],
) -> TaskProgressOutcome:
    metadata = record.get("metadata") or {}
    trace_id = str(record.get("trace_id") or "")
    degradations = list_degradations(record)
    degraded_names = {item.stage for item in degradations}
    stages = [
        StageEvent(
            name=str(stage.get("name") or ""),
            status=_ingestion_stage_status(stage, degraded_names),
            started_at=None,
            finished_at=None,
            elapsed_ms=float(stage.get("elapsed_ms") or 0.0),
            failure_kind=infer_failure_kind(stage.get("error")),
            error=_maybe_str(stage.get("error")),
            upstream_summary="ingestion",
            input_summary=str(stage.get("input_summary") or ""),
            output_summary=str(stage.get("output_summary") or ""),
            degraded=_is_ingestion_stage_degraded(stage, degraded_names),
            links={"trace_id": trace_id},
        )
        for stage in record.get("stages") or []
        if isinstance(stage, dict)
    ]
    ingest_status = str(metadata.get("status") or "")
    task_failure_kind: FailureKind | None = None
    child_failure_kind = infer_failure_kind(record.get("error"))
    if record.get("error"):
        child_status = "failed"
    elif ingest_status == "rejected":
        child_status = "blocked"
        task_failure_kind = "input"
        child_failure_kind = "input"
    elif any(stage.degraded for stage in stages):
        child_status = "partial_success"
    else:
        child_status = "succeeded"
    child = ChildEvidence(
        child_id=str(metadata.get("document_id") or trace_id),
        child_type="document",
        label=str(metadata.get("title") or metadata.get("source_path") or "document"),
        status=child_status,
        failure_kind=child_failure_kind,
        summary=str(metadata.get("status") or ""),
        trace_id=trace_id or None,
        degraded=child_status == "partial_success",
        links={
            "trace_id": trace_id,
            "document_id": str(metadata.get("document_id") or ""),
            "source_path": str(metadata.get("source_path") or ""),
        },
        detail={
            "ingest_status": metadata.get("status"),
            "chunk_count": metadata.get("chunk_count"),
            "gray_review": _gray_review_seen(record),
        },
    )
    return TaskProgressOutcome(
        finished_at=_maybe_str(record.get("finished_at")),
        last_progress_at=(
            _maybe_str(record.get("finished_at")) or str(record.get("started_at") or "")
        ),
        stages=stages,
        children=[child],
        error=_maybe_str(record.get("error")),
        failure_kind_hint=task_failure_kind,
    )


def build_evaluation_progress(
    *,
    timestamp: str,
    items: list[GoldItem],
    groups: list[str],
    ranked_chunks: dict[str, dict[str, list[ScoredChunk]]],
    generation_results: dict[str, dict[str, GenerationResult]],
    failures: list[FailedEvalItem],
) -> TaskProgressOutcome:
    failure_pairs = {(item.group, item.item_id) for item in failures}
    stages: list[StageEvent] = []
    children: list[ChildEvidence] = []
    for group in groups:
        failed_in_group = [item_id for grp, item_id in failure_pairs if grp == group]
        degraded = bool(failed_in_group)
        stages.append(
            StageEvent(
                name=group,
                status="partial_success" if degraded else "succeeded",
                started_at=None,
                finished_at=None,
                elapsed_ms=0.0,
                failure_kind="unknown" if degraded else "none",
                error=f"{len(failed_in_group)} failed items" if degraded else None,
                upstream_summary="evaluation group",
                output_summary=(
                    f"{len(generation_results.get(group, {}))}/{len(items)} items generated"
                ),
                degraded=degraded,
                links={"eval_run": timestamp, "group": group},
            )
        )
        group_ranked = ranked_chunks.get(group, {})
        group_generated = generation_results.get(group, {})
        for gold in items:
            failed = (group, gold.id) in failure_pairs
            chunks = group_ranked.get(gold.id) or []
            generation = group_generated.get(gold.id)
            children.append(
                ChildEvidence(
                    child_id=f"{group}:{gold.id}",
                    child_type="eval_item",
                    label=f"{group}/{gold.id}",
                    status="failed" if failed else "succeeded",
                    failure_kind="unknown" if failed else "none",
                    summary=gold.question,
                    degraded=failed,
                    links={
                        "eval_run": timestamp,
                        "group": group,
                        "item_id": gold.id,
                    },
                    detail={
                        "question": gold.question,
                        "answerable": gold.answerable,
                        "evidence_doc_ids": list(gold.evidence_doc_ids),
                        "retrieval_doc_ids": sorted({chunk.chunk.document_id for chunk in chunks}),
                        "retrieval_chunk_count": len(chunks),
                        "generated": generation is not None,
                        "refused": generation.refused if generation is not None else None,
                        "citation_count": (
                            len(generation.citations) if generation is not None else None
                        ),
                    },
                )
            )
    return TaskProgressOutcome(
        finished_at=timestamp,
        last_progress_at=timestamp,
        stages=stages,
        children=children,
        error=None,
    )


def _ingestion_stage_status(stage: dict[str, Any], degraded_names: set[str]) -> str:
    if stage.get("error"):
        return "blocked" if str(stage.get("name") or "") == "quality_gate" else "failed"
    if _is_ingestion_stage_degraded(stage, degraded_names):
        return "partial_success"
    return "succeeded"


def _is_ingestion_stage_degraded(stage: dict[str, Any], degraded_names: set[str]) -> bool:
    name = str(stage.get("name") or "")
    label_map = {
        "enricher": "补元数据",
        "captioner": "图转文",
        "refiner": "清洗",
        "load": "读取",
        "pipeline": "整条流水线",
    }
    label = label_map.get(name, "")
    if label and label in degraded_names:
        return True
    return bool(stage.get("fallback_reason"))


def _gray_review_seen(record: dict[str, Any]) -> bool:
    for stage in record.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        if str(stage.get("name") or "") == "gray_review":
            return True
    return False


def _maybe_str(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw)
    return value if value else None

