"""Time-window batch coordinator for /ask (Strategy B)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lab_knowledge.config import Settings
from lab_knowledge.models import AskResult
from lab_knowledge.pipelines.query_orchestration import run_ask_pipeline
from lab_knowledge.pipelines.window_batch import WindowBatchCoordinator

if TYPE_CHECKING:
    from lab_knowledge.knowledge import Knowledge

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


def _process_ask_batch(jobs: list[_AskJob], batch_meta: dict[str, object]) -> None:
    try:
        run_ask_pipeline(jobs, phase_batch=True, batch_meta=batch_meta)
    except BaseException as exc:
        for job in jobs:
            if job.error is None and job.result is None:
                job.error = exc


class QueryBatchCoordinator:
    """Collect ask jobs for batch_window_seconds, then run phase-level batches."""

    def __init__(self, settings: Settings) -> None:
        resources = settings.resources
        self._inner = WindowBatchCoordinator(
            window_seconds=resources.batch_window_seconds,
            max_size=resources.batch_window_max_size,
            process_batch=_process_ask_batch,
            worker_name="query-batch-worker",
        )

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
        return self._inner.submit(job)


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
