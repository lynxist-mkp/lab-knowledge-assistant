"""Time-window batch coordinator for /ask (Strategy B)."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wenmai.components.model_guard import ModelResource, begin_batch, end_batch
from wenmai.config import Settings
from wenmai.models import AskResult
from wenmai.pipelines.query_core import run_ask_pipeline

if TYPE_CHECKING:
    from wenmai.knowledge import Knowledge

_COORDINATOR: QueryBatchCoordinator | None = None
_COORDINATOR_LOCK = threading.Lock()


@dataclass
class _AskJob:
    question: str
    settings: Settings
    culture_domain: str | None
    retrieval_mode: str | None
    rerank_enabled: bool | None
    knowledge: Knowledge | None
    record_trace: bool
    enqueued_at: float = field(default_factory=time.monotonic)
    event: threading.Event = field(default_factory=threading.Event)
    result: AskResult | None = None
    error: BaseException | None = None
    batch_id: str | None = None
    batch_size: int = 1
    batch_wait_ms: float = 0.0


class QueryBatchCoordinator:
    """Collect ask jobs for batch_window_seconds, then run phase-level batches."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queue: deque[_AskJob] = deque()
        self._worker: threading.Thread | None = None

    def submit(
        self,
        question: str,
        settings: Settings,
        culture_domain: str | None = None,
        *,
        retrieval_mode: str | None = None,
        rerank_enabled: bool | None = None,
        knowledge: Knowledge | None = None,
        record_trace: bool = True,
    ) -> AskResult:
        job = _AskJob(
            question=question,
            settings=settings,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            record_trace=record_trace,
        )
        with self._cond:
            self._queue.append(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name="query-batch-worker",
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

    def _collect_window(self) -> list[_AskJob]:
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

    def _process_batch(self, jobs: list[_AskJob]) -> None:
        batch_id = str(uuid.uuid4())
        batch_size = len(jobs)
        started = time.monotonic()
        for job in jobs:
            job.batch_id = batch_id
            job.batch_size = batch_size
            job.batch_wait_ms = (started - job.enqueued_at) * 1000

        try:
            run_ask_pipeline(
                jobs,
                phase_batch=True,
                batch_meta={
                    "batch_id": batch_id,
                    "batch_size": batch_size,
                },
            )
        except BaseException as exc:
            for job in jobs:
                if job.error is None and job.result is None:
                    job.error = exc
        finally:
            for job in jobs:
                job.event.set()


def window_batch_enabled(settings: Settings) -> bool:
    resources = settings.resources
    return resources.query_window_batch and resources.batch_window_seconds > 0


def get_query_coordinator(settings: Settings) -> QueryBatchCoordinator:
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        if _COORDINATOR is None:
            _COORDINATOR = QueryBatchCoordinator(settings)
        return _COORDINATOR


def reset_query_coordinator() -> None:
    global _COORDINATOR
    with _COORDINATOR_LOCK:
        _COORDINATOR = None


__all__ = [
    "QueryBatchCoordinator",
    "get_query_coordinator",
    "reset_query_coordinator",
    "window_batch_enabled",
]
