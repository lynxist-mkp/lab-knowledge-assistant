"""QueryTrace — unified read/write module for 提问 Trace."""

from __future__ import annotations

from typing import Any

from wenmai.config import Settings
from wenmai.tracing.ask_payload import AskTracePayload
from wenmai.tracing.recorder import TraceRecorder
from wenmai.tracing.views.query import (
    CandidateRow,
    QueryTraceDetail,
    QueryTraceSummary,
    RankChange,
    StageLatency,
    build_query_trace_detail,
    build_query_trace_summary,
    list_stage_latencies,
)

__all__ = [
    "AskTracePayload",
    "CandidateRow",
    "QueryTrace",
    "QueryTraceDetail",
    "QueryTraceSummary",
    "RankChange",
    "StageLatency",
]


class QueryTrace:
    """Owns query trace write (AskTracePayload) and read (summary/detail)."""

    def __init__(self, recorder: TraceRecorder) -> None:
        self._recorder = recorder

    @property
    def trace_id(self) -> str:
        return self._recorder.trace_id

    @classmethod
    def begin(
        cls,
        *,
        question: str,
        batch_id: str | None = None,
        batch_size: int = 1,
        batch_wait_ms: float = 0.0,
    ) -> QueryTrace:
        return cls(
            TraceRecorder(
                question=question,
                batch_id=batch_id,
                batch_size=batch_size,
                batch_wait_ms=batch_wait_ms,
            )
        )

    def finalize_ask_work(
        self,
        *,
        payload: AskTracePayload,
        question: str,
        culture_domain: str | None,
    ) -> object:
        return self._recorder.finalize_ask_work(
            payload=payload,
            question=question,
            culture_domain=culture_domain,
        )

    def finalize_generation_error(
        self,
        *,
        payload: AskTracePayload,
        question: str,
        culture_domain: str | None,
        error: object,
    ) -> None:
        self._recorder.finalize_ask_generation_error(
            payload=payload,
            question=question,
            culture_domain=culture_domain,
            error=error,
        )

    def save(self, settings: Settings) -> None:
        self._recorder.save(settings)

    @staticmethod
    def summarize(record: dict[str, Any]) -> QueryTraceSummary:
        return build_query_trace_summary(record)

    @staticmethod
    def detail(record: dict[str, Any]) -> QueryTraceDetail:
        return build_query_trace_detail(record)

    @staticmethod
    def stage_latencies(record: dict[str, Any]) -> list[StageLatency]:
        return list_stage_latencies(record)
