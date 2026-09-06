"""QueryTrace — unified read/write module for 提问 Trace."""

from __future__ import annotations

from typing import Any

from lab_knowledge.config import Settings
from lab_knowledge.tracing.ask_payload import AskOutcome, AskTracePayload
from lab_knowledge.tracing.recorder import TraceRecorder
from lab_knowledge.tracing.views.query import (
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
    "AskOutcome",
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

    def finalize(self, outcome: AskOutcome) -> object:
        return self._recorder.finalize_ask(outcome)

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
