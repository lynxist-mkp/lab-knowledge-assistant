"""Time-window batch coordinator for HTTP ingest (Strategy B)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from wenmai.components.model_guard import ModelResource, phase_batch as model_phase_batch
from wenmai.config import Settings
from wenmai.models import IngestResult
from wenmai.pipelines.ingestion import commit_prepared_ingest, prepare_ingest_source
from wenmai.pipelines.window_batch import WindowBatchCoordinator
from wenmai.tracing.context import StageRecord

if TYPE_CHECKING:
    from wenmai.knowledge import Knowledge

_COORDINATOR: IngestBatchCoordinator | None = None
_COORDINATOR_LOCK = threading.Lock()


@dataclass
class _IngestJob:
    source_path: Path
    settings: Settings
    pdf_load_mode: str | None
    on_stage: Callable[[StageRecord], None] | None
    knowledge: Knowledge | None
    enqueued_at: float = field(default_factory=time.monotonic)
    event: threading.Event = field(default_factory=threading.Event)
    result: IngestResult | None = None
    error: BaseException | None = None
    batch_id: str | None = None
    batch_size: int = 1
    batch_wait_ms: float = 0.0


def _process_ingest_batch(jobs: list[_IngestJob], batch_meta: dict[str, object]) -> None:
    batch_id = str(batch_meta["batch_id"])
    prepared_jobs: list[tuple[_IngestJob, object]] = []
    with model_phase_batch(ModelResource.MLX_VLM, True):
        for job in jobs:
            try:
                from wenmai.knowledge import create_knowledge

                knowledge = job.knowledge or create_knowledge(job.settings)
                prepared = prepare_ingest_source(
                    job.source_path,
                    job.settings,
                    pdf_load_mode=job.pdf_load_mode,
                    on_stage=job.on_stage,
                    knowledge=knowledge,
                )
                prepared_jobs.append((job, prepared))
            except BaseException as exc:
                job.error = exc

    commit_jobs: list[tuple[_IngestJob, object]] = []
    for job, prepared in prepared_jobs:
        if job.error is not None:
            continue
        if isinstance(prepared, IngestResult):
            job.result = prepared
        else:
            commit_jobs.append((job, prepared))

    if commit_jobs:
        with model_phase_batch(ModelResource.BGE_M3, True):
            for job, prepared in commit_jobs:
                try:
                    from wenmai.knowledge import create_knowledge

                    knowledge = job.knowledge or create_knowledge(job.settings)
                    job.result = commit_prepared_ingest(
                        prepared,
                        job.settings,
                        knowledge=knowledge,
                    )
                except BaseException as exc:
                    job.error = exc

    for job in jobs:
        if job.result is None and job.error is None:
            job.error = RuntimeError(f"ingest batch {batch_id} produced no result")


class IngestBatchCoordinator:
    def __init__(self, settings: Settings) -> None:
        resources = settings.resources
        self._inner = WindowBatchCoordinator(
            window_seconds=resources.batch_window_seconds,
            max_size=resources.batch_window_max_size,
            process_batch=_process_ingest_batch,
            worker_name="ingest-batch-worker",
        )

    def submit(
        self,
        source_path: Path,
        settings: Settings,
        *,
        pdf_load_mode: str | None = None,
        on_stage: Callable[[StageRecord], None] | None = None,
        knowledge: Knowledge | None = None,
    ) -> IngestResult:
        job = _IngestJob(
            source_path=source_path,
            settings=settings,
            pdf_load_mode=pdf_load_mode,
            on_stage=on_stage,
            knowledge=knowledge,
        )
        return self._inner.submit(job)


def ingest_window_batch_enabled(settings: Settings) -> bool:
    resources = settings.resources
    return resources.ingest_window_batch and resources.batch_window_seconds > 0


def get_ingest_coordinator(settings: Settings) -> IngestBatchCoordinator:
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        if _COORDINATOR is None:
            _COORDINATOR = IngestBatchCoordinator(settings)
        return _COORDINATOR


def reset_ingest_coordinator() -> None:
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        _COORDINATOR = None


__all__ = [
    "IngestBatchCoordinator",
    "get_ingest_coordinator",
    "ingest_window_batch_enabled",
    "reset_ingest_coordinator",
]
