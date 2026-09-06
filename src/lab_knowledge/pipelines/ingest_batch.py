"""Time-window batch coordinator for HTTP ingest (Strategy B)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from lab_knowledge.config import Settings
from lab_knowledge.ingestion.source_metadata import (
    SOURCE_KIND_GROUP,
    LiteratureMetadataOverrides,
    SourceKind,
)
from lab_knowledge.models import IngestResult
from lab_knowledge.pipelines.ingestion import run_prepare_commit_batch
from lab_knowledge.pipelines.window_batch import WindowBatchCoordinator
from lab_knowledge.tracing.context import StageRecord

if TYPE_CHECKING:
    from lab_knowledge.knowledge import Knowledge

_COORDINATOR: IngestBatchCoordinator | None = None
_COORDINATOR_LOCK = threading.Lock()


@dataclass
class _IngestJob:
    source_path: Path
    settings: Settings
    pdf_load_mode: str | None
    source_kind: SourceKind
    literature: LiteratureMetadataOverrides | None
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
    run_prepare_commit_batch(jobs, batch_id=str(batch_meta["batch_id"]))


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
        source_kind: SourceKind = SOURCE_KIND_GROUP,
        literature: LiteratureMetadataOverrides | None = None,
        on_stage: Callable[[StageRecord], None] | None = None,
        knowledge: Knowledge | None = None,
    ) -> IngestResult:
        job = _IngestJob(
            source_path=source_path,
            settings=settings,
            pdf_load_mode=pdf_load_mode,
            source_kind=source_kind,
            literature=literature,
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
