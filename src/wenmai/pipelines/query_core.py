"""Shared ask pipeline with request-internal phase batching (Strategy A)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wenmai.config import Settings
from wenmai.knowledge import Knowledge
from wenmai.models import AskResult
from wenmai.pipelines.query_orchestration import (
    AskWorkContext,
    OrchestrationWork,
    run_ask_works,
)

if TYPE_CHECKING:
    from wenmai.pipelines.query_batch import _AskJob


def normalize_question(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question.strip())
    return collapsed


@dataclass
class AskPipelineInput:
    question: str
    settings: Settings
    culture_domain: str | None = None
    retrieval_mode: str | None = None
    rerank_enabled: bool | None = None
    knowledge: Knowledge | None = None
    record_trace: bool = True
    batch_id: str | None = None
    batch_size: int = 1
    batch_wait_ms: float = 0.0


def ask_pipeline_single(
    payload: AskPipelineInput,
    *,
    phase_batch: bool,
) -> AskResult:
    job_like = _SingleJob(payload)
    run_ask_pipeline([job_like], phase_batch=phase_batch, batch_meta=None)
    if job_like.error is not None:
        raise job_like.error
    assert job_like.result is not None
    return job_like.result


@dataclass
class _SingleJob:
    """Minimal job adapter for single-request pipeline runs."""

    payload: AskPipelineInput
    result: AskResult | None = None
    error: BaseException | None = None
    batch_id: str | None = None
    batch_size: int = 1
    batch_wait_ms: float = 0.0

    @property
    def question(self) -> str:
        return self.payload.question

    @property
    def settings(self) -> Settings:
        return self.payload.settings

    @property
    def culture_domain(self) -> str | None:
        return self.payload.culture_domain

    @property
    def retrieval_mode(self) -> str | None:
        return self.payload.retrieval_mode

    @property
    def rerank_enabled(self) -> bool | None:
        return self.payload.rerank_enabled

    @property
    def knowledge(self) -> Knowledge | None:
        return self.payload.knowledge

    @property
    def record_trace(self) -> bool:
        return self.payload.record_trace


def run_ask_pipeline(
    jobs: list[_AskJob | _SingleJob],
    *,
    phase_batch: bool,
    batch_meta: dict[str, object] | None,
) -> None:
    if not jobs:
        return

    if batch_meta:
        batch_id = str(batch_meta["batch_id"])
        batch_size = int(batch_meta["batch_size"])  # type: ignore[arg-type]
        for job in jobs:
            job.batch_id = batch_id
            job.batch_size = batch_size

    pairs: list[tuple[OrchestrationWork, AskWorkContext]] = []
    for job in jobs:
        normalized = normalize_question(job.question)
        work = OrchestrationWork(
            normalized=normalized,
            settings=job.settings,
            culture_domain=job.culture_domain,
            retrieval_mode=job.retrieval_mode,
            rerank_enabled=job.rerank_enabled,
            knowledge=job.knowledge,
            collect_extras=True,
        )
        context = AskWorkContext(
            question=job.question,
            record_trace=job.record_trace,
            batch_id=job.batch_id,
            batch_size=job.batch_size,
            batch_wait_ms=job.batch_wait_ms,
        )
        pairs.append((work, context))

    outcomes = run_ask_works(pairs, phase_batch=phase_batch)
    for job, outcome in zip(jobs, outcomes, strict=True):
        job.result = outcome.result
        job.error = outcome.error


__all__ = [
    "AskPipelineInput",
    "ask_pipeline_single",
    "normalize_question",
    "run_ask_pipeline",
]
