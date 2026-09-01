"""Time-window batch coordinator for HTTP ingest (Strategy B)."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from wenmai.components.model_guard import ModelResource, begin_batch, end_batch
from wenmai.config import Settings
from wenmai.models import IngestResult
from wenmai.pipelines.ingestion import commit_prepared_ingest, prepare_ingest_source
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


class IngestBatchCoordinator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queue: deque[_IngestJob] = deque()
        self._worker: threading.Thread | None = None

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
        with self._cond:
            self._queue.append(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name="ingest-batch-worker",
                    daemon=True,
                )
                self._worker.start()
            self._cond.notify()
        job.event.wait()
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return job.result

    def _worker_loop(self) -> None:
        while True:
            batch = self._collect_window()
            if not batch:
                continue
            self._process_batch(batch)

    def _collect_window(self) -> list[_IngestJob]:
        resources = self._settings.resources
        window = resources.batch_window_seconds
        max_size = resources.batch_window_max_size
        with self._cond:
            while not self._queue:
                self._cond.wait()
            batch = [self._queue.popleft()]
            deadline = time.monotonic() + window
            while len(batch) < max_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not self._queue:
                    self._cond.wait(timeout=remaining)
                    if not self._queue:
                        break
                batch.append(self._queue.popleft())
            return batch

    def _process_batch(self, jobs: list[_IngestJob]) -> None:
        batch_id = str(uuid.uuid4())
        prepared_jobs: list[tuple[_IngestJob, object]] = []
        begin_batch(ModelResource.MLX_VLM)
        try:
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
        finally:
            end_batch()

        commit_jobs: list[tuple[_IngestJob, object]] = []
        for job, prepared in prepared_jobs:
            if job.error is not None:
                continue
            if isinstance(prepared, IngestResult):
                job.result = prepared
            else:
                commit_jobs.append((job, prepared))

        if commit_jobs:
            begin_batch(ModelResource.BGE_M3)
            try:
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
            finally:
                end_batch()

        for job in jobs:
            if job.result is None and job.error is None:
                job.error = RuntimeError(f"ingest batch {batch_id} produced no result")
            job.event.set()


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
