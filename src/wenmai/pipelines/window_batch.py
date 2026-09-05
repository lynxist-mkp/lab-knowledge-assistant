"""Generic time-window batch coordinator shared by ask and ingest pipelines."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

JobT = TypeVar("JobT")
ResultT = TypeVar("ResultT")

ProcessBatch = Callable[[list[JobT], dict[str, object]], None]


@dataclass
class WindowBatchJob[ResultT]:
    """Base job state for window-batch coordinators."""

    enqueued_at: float = field(default_factory=time.monotonic)
    event: threading.Event = field(default_factory=threading.Event)
    result: ResultT | None = None
    error: BaseException | None = None


class WindowBatchCoordinator[JobT: WindowBatchJob[ResultT], ResultT]:
    """Collect jobs for batch_window_seconds, then invoke process_batch once."""

    def __init__(
        self,
        *,
        window_seconds: float,
        max_size: int,
        process_batch: ProcessBatch,
        worker_name: str,
    ) -> None:
        self._window_seconds = window_seconds
        self._max_size = max_size
        self._process_batch = process_batch
        self._worker_name = worker_name
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queue: deque[JobT] = deque()
        self._worker: threading.Thread | None = None

    def submit(self, job: JobT) -> ResultT:
        with self._cond:
            self._queue.append(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name=self._worker_name,
                    daemon=True,
                )
                self._worker.start()
            self._cond.notify()
        job.event.wait()  # type: ignore[attr-defined]
        if job.error is not None:  # type: ignore[attr-defined]
            raise job.error
        assert job.result is not None  # type: ignore[attr-defined]
        return job.result

    def _worker_loop(self) -> None:
        while True:
            batch = self._collect_window()
            if not batch:
                continue
            batch_id = str(uuid.uuid4())
            batch_size = len(batch)
            started = time.monotonic()
            for job in batch:
                job.batch_id = batch_id  # type: ignore[attr-defined]
                job.batch_size = batch_size  # type: ignore[attr-defined]
                job.batch_wait_ms = (started - job.enqueued_at) * 1000  # type: ignore[attr-defined]
            try:
                self._process_batch(
                    batch,
                    {"batch_id": batch_id, "batch_size": batch_size},
                )
            except BaseException as exc:
                for job in batch:
                    if job.error is None and job.result is None:  # type: ignore[attr-defined]
                        job.error = exc  # type: ignore[attr-defined]
            finally:
                for job in batch:
                    job.event.set()  # type: ignore[attr-defined]

    def _collect_window(self) -> list[JobT]:
        with self._cond:
            while not self._queue:
                self._cond.wait()
            batch = [self._queue.popleft()]
            deadline = time.monotonic() + self._window_seconds
            while len(batch) < self._max_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not self._queue:
                    self._cond.wait(timeout=remaining)
                    if not self._queue:
                        break
                batch.append(self._queue.popleft())
            return batch


__all__ = ["ProcessBatch", "WindowBatchCoordinator", "WindowBatchJob"]
