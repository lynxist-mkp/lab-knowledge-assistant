from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class StageRecord:
    name: str
    method: str
    provider: str
    elapsed_ms: float
    input_summary: str = ""
    output_summary: str = ""
    candidate_count: int | None = None
    error: str | None = None
    candidates: list[dict[str, Any]] | None = None
    dense_candidates: list[dict[str, Any]] | None = None
    sparse_candidates: list[dict[str, Any]] | None = None
    pre_rerank_candidates: list[dict[str, Any]] | None = None
    fallback_reason: str | None = None
    rank_changes: list[dict[str, Any]] | None = None
    culture_domain: str | None = None
    expanded_from: list[str] | None = None
    expanded_chunk_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "method": self.method,
            "provider": self.provider,
            "elapsed_ms": self.elapsed_ms,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
        }
        if self.candidate_count is not None:
            payload["candidate_count"] = self.candidate_count
        if self.error is not None:
            payload["error"] = self.error
        if self.candidates is not None:
            payload["candidates"] = self.candidates
        if self.dense_candidates is not None:
            payload["dense_candidates"] = self.dense_candidates
        if self.sparse_candidates is not None:
            payload["sparse_candidates"] = self.sparse_candidates
        if self.pre_rerank_candidates is not None:
            payload["pre_rerank_candidates"] = self.pre_rerank_candidates
        if self.fallback_reason is not None:
            payload["fallback_reason"] = self.fallback_reason
        if self.rank_changes is not None:
            payload["rank_changes"] = self.rank_changes
        if self.culture_domain is not None:
            payload["culture_domain"] = self.culture_domain
        if self.expanded_from is not None:
            payload["expanded_from"] = self.expanded_from
        if self.expanded_chunk_ids is not None:
            payload["expanded_chunk_ids"] = self.expanded_chunk_ids
        return payload


@dataclass
class TraceContext:
    trace_type: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    stages: list[StageRecord] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _wall_start: float = field(default_factory=time.perf_counter, repr=False)
    _on_stage: Callable[[StageRecord], None] | None = field(default=None, repr=False)

    def record_stage(
        self,
        name: str,
        method: str,
        provider: str,
        elapsed_ms: float,
        input_summary: str = "",
        output_summary: str = "",
        candidate_count: int | None = None,
        error: str | None = None,
        candidates: list[dict[str, Any]] | None = None,
        dense_candidates: list[dict[str, Any]] | None = None,
        sparse_candidates: list[dict[str, Any]] | None = None,
        pre_rerank_candidates: list[dict[str, Any]] | None = None,
        fallback_reason: str | None = None,
        rank_changes: list[dict[str, Any]] | None = None,
        culture_domain: str | None = None,
        expanded_from: list[str] | None = None,
        expanded_chunk_ids: list[str] | None = None,
    ) -> None:
        record = StageRecord(
            name=name,
            method=method,
            provider=provider,
            elapsed_ms=elapsed_ms,
            input_summary=input_summary,
            output_summary=output_summary,
            candidate_count=candidate_count,
            error=error,
            candidates=candidates,
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            pre_rerank_candidates=pre_rerank_candidates,
            fallback_reason=fallback_reason,
            rank_changes=rank_changes,
            culture_domain=culture_domain,
            expanded_from=expanded_from,
            expanded_chunk_ids=expanded_chunk_ids,
        )
        self.stages.append(record)
        if self._on_stage is not None:
            self._on_stage(record)

    def append_stage(self, record: StageRecord) -> None:
        self.stages.append(record)
        if self._on_stage is not None:
            self._on_stage(record)

    def close(self) -> None:
        self.finished_at = _now()
        self.total_elapsed_ms = (time.perf_counter() - self._wall_start) * 1000

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.total_elapsed_ms,
            "stages": [stage.to_dict() for stage in self.stages],
            "error": self.error,
            "metadata": self.metadata,
        }
        return payload

    @contextmanager
    def stage(
        self,
        name: str,
        method: str,
        provider: str,
        input_summary: str = "",
    ) -> Iterator[dict[str, Any]]:
        extras: dict[str, Any] = {
            "output_summary": "",
            "candidate_count": None,
        }
        started = time.perf_counter()
        error: str | None = None
        try:
            yield extras
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.error = error
            raise
        finally:
            stage_method = str(extras.get("method") or method)
            stage_provider = str(extras.get("provider") or provider)
            stage_error = extras.get("error") or error
            self.record_stage(
                name=name,
                method=stage_method,
                provider=stage_provider,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                input_summary=input_summary,
                output_summary=str(extras.get("output_summary") or ""),
                candidate_count=extras.get("candidate_count"),
                error=stage_error,
                candidates=extras.get("candidates"),
                dense_candidates=extras.get("dense_candidates"),
                sparse_candidates=extras.get("sparse_candidates"),
                pre_rerank_candidates=extras.get("pre_rerank_candidates"),
                fallback_reason=extras.get("fallback_reason"),
                rank_changes=extras.get("rank_changes"),
                culture_domain=extras.get("culture_domain"),
            )
