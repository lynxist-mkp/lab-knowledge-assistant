from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wenmai.tracing.steps import QUERY_LABELS, QUERY_STAGE_ORDER, stage_label


@dataclass
class QueryTraceSummary:
    trace_id: str
    started_at: str
    finished_at: str | None
    total_elapsed_ms: float
    question: str
    culture_domain: str | None
    status: str
    refused: bool
    rerank_fallback: bool
    error: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_type": "query",
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.total_elapsed_ms,
            "question": self.question,
            "culture_domain": self.culture_domain,
            "status": self.status,
            "refused": self.refused,
            "rerank_fallback": self.rerank_fallback,
            "error": self.error,
        }


@dataclass
class StageLatency:
    label: str
    elapsed_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "elapsed_ms": self.elapsed_ms}


@dataclass
class CandidateRow:
    rank: int
    chunk_id: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "chunk_id": self.chunk_id, "score": self.score}


@dataclass
class RankChange:
    chunk_id: str
    from_rank: int
    to_rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "from_rank": self.from_rank,
            "to_rank": self.to_rank,
        }


@dataclass
class QueryTraceDetail:
    summary: QueryTraceSummary
    stage_latencies: list[StageLatency]
    dense_candidates: list[CandidateRow]
    sparse_candidates: list[CandidateRow]
    fusion_candidates: list[CandidateRow]
    rerank_candidates: list[CandidateRow]
    rank_changes: list[RankChange]
    rerank_fallback_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.summary.as_dict(),
            "stage_latencies": [item.as_dict() for item in self.stage_latencies],
            "dense_candidates": [item.as_dict() for item in self.dense_candidates],
            "sparse_candidates": [item.as_dict() for item in self.sparse_candidates],
            "fusion_candidates": [item.as_dict() for item in self.fusion_candidates],
            "rerank_candidates": [item.as_dict() for item in self.rerank_candidates],
            "rank_changes": [item.as_dict() for item in self.rank_changes],
            "rerank_fallback_reason": self.rerank_fallback_reason,
        }


def _stage_by_name(record: dict[str, Any], name: str) -> dict[str, Any] | None:
    for stage in record.get("stages") or []:
        if stage.get("name") == name:
            return stage
    return None


def _question(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    question = metadata.get("question")
    if isinstance(question, str) and question:
        return question
    processing = _stage_by_name(record, "query_processing")
    if processing is None:
        return ""
    return str(processing.get("input_summary") or "")


def _culture_domain(record: dict[str, Any]) -> str | None:
    processing = _stage_by_name(record, "query_processing")
    if processing is None:
        return None
    culture_domain = processing.get("culture_domain")
    if isinstance(culture_domain, str) and culture_domain:
        return culture_domain
    return None


def _is_refusal(record: dict[str, Any]) -> bool:
    generation = _stage_by_name(record, "generation")
    if generation is None:
        return False
    return generation.get("output_summary") == "refusal"


def _is_rerank_fallback(record: dict[str, Any]) -> bool:
    rerank = _stage_by_name(record, "rerank")
    if rerank is None:
        return False
    return rerank.get("method") == "rrf_fallback"


def _rerank_fallback_reason(record: dict[str, Any]) -> str | None:
    rerank = _stage_by_name(record, "rerank")
    if rerank is None:
        return None
    reason = rerank.get("fallback_reason") or rerank.get("error")
    return str(reason) if reason else None


def _candidate_rows(stage: dict[str, Any] | None, key: str = "candidates") -> list[CandidateRow]:
    if stage is None:
        return []
    rows: list[CandidateRow] = []
    for index, item in enumerate(stage.get(key) or [], start=1):
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue
        score = item.get("score")
        rows.append(
            CandidateRow(
                rank=index,
                chunk_id=str(chunk_id),
                score=float(score if score is not None else 0.0),
            )
        )
    return rows


def _rank_changes(record: dict[str, Any]) -> list[RankChange]:
    rerank = _stage_by_name(record, "rerank")
    if rerank is None:
        return []
    changes: list[RankChange] = []
    for item in rerank.get("rank_changes") or []:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        from_rank = item.get("from")
        to_rank = item.get("to")
        if chunk_id is None or from_rank is None or to_rank is None:
            continue
        changes.append(
            RankChange(
                chunk_id=str(chunk_id),
                from_rank=int(from_rank),
                to_rank=int(to_rank),
            )
        )
    return changes


def list_stage_latencies(record: dict[str, Any]) -> list[StageLatency]:
    stage_by_name = {
        stage.get("name"): stage
        for stage in record.get("stages") or []
        if isinstance(stage.get("name"), str)
    }
    latencies: list[StageLatency] = []
    for name in QUERY_STAGE_ORDER:
        stage = stage_by_name.get(name)
        elapsed_ms = float(stage.get("elapsed_ms") or 0.0) if stage else 0.0
        latencies.append(
            StageLatency(label=stage_label(name, QUERY_LABELS), elapsed_ms=elapsed_ms)
        )
    return latencies


def summarize_query_trace(record: dict[str, Any]) -> QueryTraceSummary:
    refused = _is_refusal(record)
    rerank_fallback = _is_rerank_fallback(record)
    error = record.get("error")
    if error:
        status = "failed"
    elif refused:
        status = "refused"
    else:
        status = "ok"
    return QueryTraceSummary(
        trace_id=str(record.get("trace_id") or ""),
        started_at=str(record.get("started_at") or ""),
        finished_at=record.get("finished_at"),
        total_elapsed_ms=float(record.get("total_elapsed_ms") or 0.0),
        question=_question(record),
        culture_domain=_culture_domain(record),
        status=status,
        refused=refused,
        rerank_fallback=rerank_fallback,
        error=error if isinstance(error, str) else None,
    )


def query_trace_detail(record: dict[str, Any]) -> QueryTraceDetail:
    return QueryTraceDetail(
        summary=summarize_query_trace(record),
        stage_latencies=list_stage_latencies(record),
        dense_candidates=_candidate_rows(_stage_by_name(record, "dense")),
        sparse_candidates=_candidate_rows(_stage_by_name(record, "sparse")),
        fusion_candidates=_candidate_rows(_stage_by_name(record, "fusion")),
        rerank_candidates=_candidate_rows(_stage_by_name(record, "rerank")),
        rank_changes=_rank_changes(record),
        rerank_fallback_reason=_rerank_fallback_reason(record),
    )
