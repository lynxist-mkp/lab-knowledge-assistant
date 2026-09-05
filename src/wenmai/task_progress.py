"""Task progress seam for long-running troubleshooting evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from wenmai.config import Settings

TaskStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "partial_success",
    "blocked",
    "cancelled",
]

FailureKind = Literal[
    "none",
    "input",
    "dependency",
    "model",
    "storage",
    "timeout",
    "config",
    "unknown",
]


@dataclass(frozen=True)
class TaskCounters:
    total: int = 0
    completed: int = 0
    failed: int = 0
    partial: int = 0
    blocked: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> TaskCounters:
        data = raw or {}
        return cls(
            total=int(data.get("total") or 0),
            completed=int(data.get("completed") or 0),
            failed=int(data.get("failed") or 0),
            partial=int(data.get("partial") or 0),
            blocked=int(data.get("blocked") or 0),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "partial": self.partial,
            "blocked": self.blocked,
        }


@dataclass(frozen=True)
class StageEvent:
    name: str
    status: TaskStatus
    started_at: str | None
    finished_at: str | None
    elapsed_ms: float
    failure_kind: FailureKind
    error: str | None = None
    upstream_summary: str = ""
    input_summary: str = ""
    output_summary: str = ""
    degraded: bool = False
    links: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StageEvent:
        return cls(
            name=str(raw.get("name") or ""),
            status=_coerce_status(raw.get("status")),
            started_at=_maybe_str(raw.get("started_at")),
            finished_at=_maybe_str(raw.get("finished_at")),
            elapsed_ms=float(raw.get("elapsed_ms") or 0.0),
            failure_kind=_coerce_failure_kind(raw.get("failure_kind")),
            error=_maybe_str(raw.get("error")),
            upstream_summary=str(raw.get("upstream_summary") or ""),
            input_summary=str(raw.get("input_summary") or ""),
            output_summary=str(raw.get("output_summary") or ""),
            degraded=bool(raw.get("degraded") or False),
            links=_string_dict(raw.get("links")),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_ms": self.elapsed_ms,
            "failure_kind": self.failure_kind,
            "upstream_summary": self.upstream_summary,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "degraded": self.degraded,
            "links": dict(self.links),
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class ChildEvidence:
    child_id: str
    child_type: str
    label: str
    status: TaskStatus
    failure_kind: FailureKind
    summary: str = ""
    trace_id: str | None = None
    degraded: bool = False
    links: dict[str, str] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ChildEvidence:
        return cls(
            child_id=str(raw.get("child_id") or ""),
            child_type=str(raw.get("child_type") or ""),
            label=str(raw.get("label") or ""),
            status=_coerce_status(raw.get("status")),
            failure_kind=_coerce_failure_kind(raw.get("failure_kind")),
            summary=str(raw.get("summary") or ""),
            trace_id=_maybe_str(raw.get("trace_id")),
            degraded=bool(raw.get("degraded") or False),
            links=_string_dict(raw.get("links")),
            detail=dict(raw.get("detail") or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "child_id": self.child_id,
            "child_type": self.child_type,
            "label": self.label,
            "status": self.status,
            "failure_kind": self.failure_kind,
            "summary": self.summary,
            "degraded": self.degraded,
            "links": dict(self.links),
            "detail": dict(self.detail),
        }
        if self.trace_id is not None:
            payload["trace_id"] = self.trace_id
        return payload


@dataclass(frozen=True)
class TaskProgressSummary:
    task_id: str
    task_type: str
    status: TaskStatus
    started_at: str
    finished_at: str | None
    last_progress_at: str | None
    trigger_source: str
    owner_surface: str
    config_fingerprint: str
    counters: TaskCounters
    failure_kind: FailureKind
    error: str | None
    stage_count: int
    child_count: int
    degraded: bool
    links: dict[str, str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskProgressSummary:
        stages = raw.get("stages") or []
        children = raw.get("children") or []
        return cls(
            task_id=str(raw.get("task_id") or ""),
            task_type=str(raw.get("task_type") or ""),
            status=_coerce_status(raw.get("status")),
            started_at=str(raw.get("started_at") or ""),
            finished_at=_maybe_str(raw.get("finished_at")),
            last_progress_at=_maybe_str(raw.get("last_progress_at")),
            trigger_source=str(raw.get("trigger_source") or ""),
            owner_surface=str(raw.get("owner_surface") or ""),
            config_fingerprint=str(raw.get("config_fingerprint") or ""),
            counters=TaskCounters.from_dict(raw.get("counters")),
            failure_kind=_coerce_failure_kind(raw.get("failure_kind")),
            error=_maybe_str(raw.get("error")),
            stage_count=len(stages) if isinstance(stages, list) else 0,
            child_count=len(children) if isinstance(children, list) else 0,
            degraded=bool(raw.get("degraded") or False),
            links=_string_dict(raw.get("links")),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_progress_at": self.last_progress_at,
            "trigger_source": self.trigger_source,
            "owner_surface": self.owner_surface,
            "config_fingerprint": self.config_fingerprint,
            "counters": self.counters.as_dict(),
            "failure_kind": self.failure_kind,
            "stage_count": self.stage_count,
            "child_count": self.child_count,
            "degraded": self.degraded,
            "links": dict(self.links),
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class TaskProgressDetail:
    summary: TaskProgressSummary
    stages: list[StageEvent]
    children: list[ChildEvidence]
    config_snapshot: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskProgressDetail:
        return cls(
            summary=TaskProgressSummary.from_dict(raw),
            stages=[
                StageEvent.from_dict(item)
                for item in raw.get("stages") or []
                if isinstance(item, dict)
            ],
            children=[
                ChildEvidence.from_dict(item)
                for item in raw.get("children") or []
                if isinstance(item, dict)
            ],
            config_snapshot=dict(raw.get("config_snapshot") or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.summary.as_dict(),
            "stages": [stage.as_dict() for stage in self.stages],
            "children": [child.as_dict() for child in self.children],
            "config_snapshot": dict(self.config_snapshot),
        }


@dataclass(frozen=True)
class TaskProgressRun:
    task_id: str
    task_type: str
    started_at: str
    trigger_source: str
    owner_surface: str
    links: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionTaskProgressConfig:
    pdf_load_mode: str | None


@dataclass(frozen=True)
class EvaluationTaskProgressConfig:
    groups: list[str]
    query_rewrite_by_group: dict[str, bool]


TaskProgressConfig = IngestionTaskProgressConfig | EvaluationTaskProgressConfig | dict[str, Any]


@dataclass(frozen=True)
class TaskProgressOutcome:
    finished_at: str | None
    last_progress_at: str | None
    stages: list[StageEvent] = field(default_factory=list)
    children: list[ChildEvidence] = field(default_factory=list)
    error: str | None = None
    failure_kind_hint: FailureKind | None = None


def task_progress_path(settings: Settings) -> Path:
    raw = Path(settings.observability.task_progress_file)
    return raw if raw.is_absolute() else settings.root / raw


def write_task_progress(settings: Settings, payload: dict[str, Any]) -> Path:
    path = task_progress_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def read_task_progress_records(settings: Settings) -> list[dict[str, Any]]:
    path = task_progress_path(settings)
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


def list_task_progress(
    settings: Settings,
    *,
    task_type: str | None = None,
    status: str | None = None,
    failure_kind: str | None = None,
) -> list[TaskProgressSummary]:
    latest_by_id: dict[str, TaskProgressSummary] = {}
    for item in reversed(read_task_progress_records(settings)):
        summary = TaskProgressSummary.from_dict(item)
        if summary.task_id and summary.task_id not in latest_by_id:
            latest_by_id[summary.task_id] = summary
    summaries = list(latest_by_id.values())
    if task_type is not None:
        summaries = [item for item in summaries if item.task_type == task_type]
    if status is not None:
        summaries = [item for item in summaries if item.status == status]
    if failure_kind is not None:
        summaries = [item for item in summaries if item.failure_kind == failure_kind]
    return summaries


def get_task_progress(settings: Settings, task_id: str) -> TaskProgressDetail | None:
    for record in reversed(read_task_progress_records(settings)):
        if str(record.get("task_id") or "") == task_id:
            return TaskProgressDetail.from_dict(record)
    return None


def persist_task_progress_running(
    settings: Settings,
    run: TaskProgressRun,
    *,
    total: int,
    config: TaskProgressConfig,
) -> Path:
    return persist_task_progress(
        settings,
        task_id=run.task_id,
        task_type=run.task_type,
        status="running",
        started_at=run.started_at,
        finished_at=None,
        last_progress_at=run.started_at,
        trigger_source=run.trigger_source,
        owner_surface=run.owner_surface,
        config_snapshot=resolve_task_progress_config(settings, config),
        counters=TaskCounters(total=total),
        links=run.links,
    )


def persist_task_progress_outcome(
    settings: Settings,
    run: TaskProgressRun,
    *,
    config: TaskProgressConfig,
    outcome: TaskProgressOutcome,
) -> Path:
    status = derive_task_status(outcome)
    return persist_task_progress(
        settings,
        task_id=run.task_id,
        task_type=run.task_type,
        status=status,
        started_at=run.started_at,
        finished_at=outcome.finished_at,
        last_progress_at=outcome.last_progress_at or outcome.finished_at or run.started_at,
        trigger_source=run.trigger_source,
        owner_surface=run.owner_surface,
        config_snapshot=resolve_task_progress_config(settings, config),
        links=run.links,
        stages=outcome.stages,
        children=outcome.children,
        counters=derive_task_counters(outcome, status=status),
        error=outcome.error,
        failure_kind=derive_failure_kind(outcome, status=status),
    )


def safe_persist_task_progress_running(
    settings: Settings,
    run: TaskProgressRun,
    *,
    total: int,
    config: TaskProgressConfig,
) -> Path | None:
    try:
        return persist_task_progress_running(settings, run, total=total, config=config)
    except OSError:
        return None


def safe_persist_task_progress_outcome(
    settings: Settings,
    run: TaskProgressRun,
    *,
    config: TaskProgressConfig,
    outcome: TaskProgressOutcome,
) -> Path | None:
    try:
        return persist_task_progress_outcome(settings, run, config=config, outcome=outcome)
    except OSError:
        return None


def persist_task_progress(
    settings: Settings,
    *,
    task_id: str,
    task_type: str,
    status: TaskStatus,
    started_at: str,
    finished_at: str | None,
    last_progress_at: str | None,
    trigger_source: str,
    owner_surface: str,
    config_snapshot: dict[str, Any],
    links: dict[str, str] | None = None,
    stages: list[StageEvent] | None = None,
    children: list[ChildEvidence] | None = None,
    counters: TaskCounters | None = None,
    error: str | None = None,
    failure_kind: FailureKind | None = None,
) -> Path:
    resolved_stages = stages or []
    resolved_children = children or []
    resolved_links = links or {}
    degraded = any(stage.degraded for stage in resolved_stages) or any(
        child.degraded for child in resolved_children
    )
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "last_progress_at": last_progress_at,
        "trigger_source": trigger_source,
        "owner_surface": owner_surface,
        "config_fingerprint": config_fingerprint(config_snapshot),
        "config_snapshot": config_snapshot,
        "counters": (counters or TaskCounters()).as_dict(),
        "failure_kind": failure_kind or infer_failure_kind(error),
        "error": error,
        "degraded": degraded,
        "links": resolved_links,
        "stages": [stage.as_dict() for stage in resolved_stages],
        "children": [child.as_dict() for child in resolved_children],
    }
    return write_task_progress(settings, payload)


def safe_persist_task_progress(settings: Settings, **kwargs: Any) -> Path | None:
    try:
        return persist_task_progress(settings, **kwargs)
    except OSError:
        return None


def resolve_task_progress_config(
    settings: Settings,
    config: TaskProgressConfig,
) -> dict[str, Any]:
    if isinstance(config, IngestionTaskProgressConfig):
        return {
            "pdf_load_mode": config.pdf_load_mode or settings.pdf_load.mode,
            "quality_gate": {
                "reject_below": settings.quality_gate.reject_below,
                "approve_above": settings.quality_gate.approve_above,
                "gray_review": settings.quality_gate.gray_review,
            },
            "transform": {
                "stages": list(settings.transform.stages),
                "refiner": settings.transform.refiner,
                "enricher": settings.transform.enricher,
                "captioner": settings.transform.captioner,
            },
            "providers": {
                "splitter": settings.providers.splitter,
                "embedding": settings.providers.embedding,
                "vector_store": settings.providers.vector_store,
            },
        }
    if isinstance(config, EvaluationTaskProgressConfig):
        return {
            "golden_set": settings.evaluation.golden_set,
            "groups": list(config.groups),
            "query_rewrite_by_group": dict(config.query_rewrite_by_group),
            "providers": {
                "embedding": settings.providers.embedding,
                "reranker": settings.providers.reranker,
                "multimodal": settings.providers.multimodal,
            },
            "retrieval": {
                "dense_k": settings.retrieval.dense_k,
                "sparse_k": settings.retrieval.sparse_k,
                "fused_k": settings.retrieval.fused_k,
                "rerank_top": settings.retrieval.rerank_top,
            },
        }
    return dict(config)


def derive_task_status(outcome: TaskProgressOutcome) -> TaskStatus:
    child_statuses = [child.status for child in outcome.children]
    success_count = sum(status in {"succeeded", "partial_success"} for status in child_statuses)
    failed_count = child_statuses.count("failed")
    blocked_count = child_statuses.count("blocked")
    partial_count = child_statuses.count("partial_success")
    degraded = any(stage.degraded for stage in outcome.stages) or any(
        child.degraded for child in outcome.children
    )
    if outcome.error and success_count == 0 and partial_count == 0 and blocked_count == 0:
        return "failed"
    if blocked_count and success_count == 0 and failed_count == 0 and partial_count == 0:
        return "blocked"
    if (
        partial_count > 0
        or degraded
        or (success_count > 0 and (failed_count > 0 or blocked_count > 0))
    ):
        return "partial_success"
    if failed_count and success_count == 0 and blocked_count == 0:
        return "failed"
    if blocked_count and success_count == 0:
        return "blocked"
    return "succeeded"


def derive_task_counters(
    outcome: TaskProgressOutcome,
    *,
    status: TaskStatus | None = None,
) -> TaskCounters:
    resolved_status = status or derive_task_status(outcome)
    total = len(outcome.children)
    completed = sum(child.status in {"succeeded", "partial_success"} for child in outcome.children)
    failed = sum(child.status == "failed" for child in outcome.children)
    blocked = sum(child.status == "blocked" for child in outcome.children)
    partial = sum(child.status == "partial_success" for child in outcome.children)
    if partial == 0 and resolved_status == "partial_success":
        partial = 1
    return TaskCounters(
        total=total,
        completed=completed,
        failed=failed,
        partial=partial,
        blocked=blocked,
    )


def derive_failure_kind(
    outcome: TaskProgressOutcome,
    *,
    status: TaskStatus | None = None,
) -> FailureKind:
    if outcome.failure_kind_hint and outcome.failure_kind_hint != "none":
        return outcome.failure_kind_hint
    inferred = infer_failure_kind(outcome.error)
    if inferred != "none":
        return inferred
    for child in outcome.children:
        if child.failure_kind != "none":
            return child.failure_kind
    for stage in outcome.stages:
        if stage.failure_kind != "none":
            return stage.failure_kind
    resolved_status = status or derive_task_status(outcome)
    return "unknown" if resolved_status in {"failed", "partial_success", "blocked"} else "none"


def config_fingerprint(payload: dict[str, Any]) -> str:
    material = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def infer_failure_kind(error: object) -> FailureKind:
    text = str(error or "").lower()
    if not text:
        return "none"
    if "timeout" in text:
        return "timeout"
    if "model" in text or "llm" in text or "rerank" in text or "embedding" in text:
        return "model"
    if "chroma" in text or "bm25" in text or "sqlite" in text or "upsert" in text:
        return "storage"
    if "file" in text or "path" in text or "pdf" in text or "markdown" in text:
        return "input"
    if "config" in text or "setting" in text:
        return "config"
    if "http" in text or "connection" in text or "provider" in text:
        return "dependency"
    return "unknown"


def _coerce_status(raw: object) -> TaskStatus:
    value = str(raw or "queued")
    if value in {
        "queued",
        "running",
        "succeeded",
        "failed",
        "partial_success",
        "blocked",
        "cancelled",
    }:
        return value
    return "queued"


def _coerce_failure_kind(raw: object) -> FailureKind:
    value = str(raw or "none")
    if value in {"none", "input", "dependency", "model", "storage", "timeout", "config", "unknown"}:
        return value
    return "unknown"


def _maybe_str(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw)
    return value if value else None


def _string_dict(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw.items()
        if value is not None and str(value) != ""
    }


__all__ = [
    "ChildEvidence",
    "EvaluationTaskProgressConfig",
    "FailureKind",
    "IngestionTaskProgressConfig",
    "StageEvent",
    "TaskCounters",
    "TaskProgressConfig",
    "TaskProgressDetail",
    "TaskProgressOutcome",
    "TaskProgressRun",
    "TaskProgressSummary",
    "TaskStatus",
    "config_fingerprint",
    "derive_failure_kind",
    "derive_task_counters",
    "derive_task_status",
    "get_task_progress",
    "infer_failure_kind",
    "list_task_progress",
    "persist_task_progress",
    "persist_task_progress_outcome",
    "persist_task_progress_running",
    "read_task_progress_records",
    "resolve_task_progress_config",
    "safe_persist_task_progress_outcome",
    "safe_persist_task_progress_running",
    "safe_persist_task_progress",
    "task_progress_path",
]
