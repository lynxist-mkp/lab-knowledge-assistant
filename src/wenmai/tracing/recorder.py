from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from wenmai.config import Settings
from wenmai.tracing.context import TraceContext
from wenmai.tracing.stage_result import StageResult, stage_to_dict
from wenmai.tracing.stages.query import QueryStage
from wenmai.storage.paths import store_path
from wenmai.tracing.writer import JsonlTraceWriter

QUERY_TRACE_SCHEMA_VERSION = 2


class TraceRecorder:
    """Central owner for query trace assembly and typed serialization."""

    def __init__(
        self,
        *,
        question: str,
        batch_id: str | None = None,
        batch_size: int = 1,
        batch_wait_ms: float = 0.0,
    ) -> None:
        metadata: dict[str, Any] = {"question": question}
        if batch_id:
            metadata["batch_id"] = batch_id
            metadata["batch_size"] = batch_size
            metadata["batch_wait_ms"] = round(batch_wait_ms, 1)
        self._context = TraceContext(trace_type="query", metadata=metadata)

    @property
    def trace_id(self) -> str:
        return self._context.trace_id

    @property
    def error(self) -> str | None:
        return self._context.error

    @error.setter
    def error(self, value: str | None) -> None:
        self._context.error = value

    def append_stage(self, stage: StageResult) -> None:
        self._context.append_stage(stage)

    def record_query_processing(
        self,
        *,
        question: str,
        normalized: str,
        elapsed_ms: float,
        culture_domain: str | None = None,
        term_extras: list[str] | None = None,
        multi_query_extras: list[str] | None = None,
        rewriter: str = "local",
        multi_query_provider: str | None = None,
    ) -> None:
        self.append_stage(
            QueryStage.query_processing(
                question=question,
                normalized=normalized,
                elapsed_ms=elapsed_ms,
                culture_domain=culture_domain,
                term_extras=term_extras,
                multi_query_extras=multi_query_extras,
                rewriter=rewriter,
                multi_query_provider=multi_query_provider,
            )
        )

    def append_retrieval_stages(self, stages: Iterable[StageResult]) -> None:
        for stage in stages:
            self.append_stage(stage)

    def append_rerank_stages(self, stages: Iterable[StageResult]) -> None:
        for stage in stages:
            self.append_stage(stage)

    def record_generation(
        self,
        *,
        provider: str,
        elapsed_ms: float,
        input_summary: str,
        output_summary: str,
        candidate_count: int,
        error: str | None = None,
        expanded_from: list[str] | None = None,
        expanded_chunk_ids: list[str] | None = None,
    ) -> None:
        self.append_stage(
            QueryStage.generation(
                provider=provider,
                elapsed_ms=elapsed_ms,
                input_summary=input_summary,
                output_summary=output_summary,
                candidate_count=candidate_count,
                error=error,
                expanded_from=expanded_from,
                expanded_chunk_ids=expanded_chunk_ids,
            )
        )

    def set_outcome(
        self,
        *,
        refused: bool,
        refusal_reason: str | None,
        citation_count: int,
    ) -> None:
        self._context.metadata["outcome"] = {
            "refused": refused,
            "refusal_reason": refusal_reason,
            "citation_count": citation_count,
        }

    def to_dict(self) -> dict[str, Any]:
        if self._context.finished_at is None:
            self._context.close()
        payload = self._context.to_dict()
        payload["schema_version"] = QUERY_TRACE_SCHEMA_VERSION
        payload["stages"] = [stage_to_dict(stage) for stage in self._context.stages]
        return payload

    def save(self, settings: Settings) -> None:
        JsonlTraceWriter(store_path(settings, "traces")).write_payload(self.to_dict())
