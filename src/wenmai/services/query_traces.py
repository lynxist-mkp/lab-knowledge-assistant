from __future__ import annotations

from dataclasses import dataclass
from typing import Any

QUERY_STAGE_NAMES = (
    "query_processing",
    "dense",
    "sparse",
    "fusion",
    "rerank",
    "generation",
)

STAGE_LABELS: dict[str, str] = {
    "query_processing": "查询处理",
    "dense": "嵌入检索",
    "sparse": "稀疏检索",
    "fusion": "融合",
    "rerank": "精排",
    "generation": "生成",
}


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
    name: str
    label: str
    elapsed_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class CandidateRow:
    rank: int
    chunk_id: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "score": self.score,
        }


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


def _stage_by_name(trace: dict[str, Any], name: str) -> dict[str, Any] | None:
    for stage in trace.get("stages") or []:
        if stage.get("name") == name:
            return stage
    return None


def _question(trace: dict[str, Any]) -> str:
    metadata = trace.get("metadata") or {}
    question = metadata.get("question")
    if isinstance(question, str) and question:
        return question
    processing = _stage_by_name(trace, "query_processing")
    if processing is None:
        return ""
    return str(processing.get("input_summary") or "")


def _culture_domain(trace: dict[str, Any]) -> str | None:
    processing = _stage_by_name(trace, "query_processing")
    if processing is None:
        return None
    culture_domain = processing.get("culture_domain")
    if isinstance(culture_domain, str) and culture_domain:
        return culture_domain
    return None


def is_refusal(trace: dict[str, Any]) -> bool:
    generation = _stage_by_name(trace, "generation")
    if generation is None:
        return False
    return generation.get("output_summary") == "refusal"


def is_rerank_fallback(trace: dict[str, Any]) -> bool:
    rerank = _stage_by_name(trace, "rerank")
    if rerank is None:
        return False
    return rerank.get("method") == "rrf_fallback"


def _candidate_rows(stage: dict[str, Any] | None, key: str = "candidates") -> list[CandidateRow]:
    if stage is None:
        return []
    candidates = stage.get(key) or []
    rows: list[CandidateRow] = []
    for index, item in enumerate(candidates, start=1):
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


def get_dense_candidates(trace: dict[str, Any]) -> list[CandidateRow]:
    return _candidate_rows(_stage_by_name(trace, "dense"))


def get_sparse_candidates(trace: dict[str, Any]) -> list[CandidateRow]:
    return _candidate_rows(_stage_by_name(trace, "sparse"))


def get_fusion_candidates(trace: dict[str, Any]) -> list[CandidateRow]:
    return _candidate_rows(_stage_by_name(trace, "fusion"))


def get_fusion_dense_candidates(trace: dict[str, Any]) -> list[CandidateRow]:
    return _candidate_rows(_stage_by_name(trace, "fusion"), key="dense_candidates")


def get_fusion_sparse_candidates(trace: dict[str, Any]) -> list[CandidateRow]:
    return _candidate_rows(_stage_by_name(trace, "fusion"), key="sparse_candidates")


def get_rerank_candidates(trace: dict[str, Any]) -> list[CandidateRow]:
    return _candidate_rows(_stage_by_name(trace, "rerank"))


def get_rank_changes(trace: dict[str, Any]) -> list[RankChange]:
    rerank = _stage_by_name(trace, "rerank")
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


def list_stage_latencies(trace: dict[str, Any]) -> list[StageLatency]:
    stage_by_name = {
        stage.get("name"): stage
        for stage in trace.get("stages") or []
        if isinstance(stage.get("name"), str)
    }
    latencies: list[StageLatency] = []
    for name in QUERY_STAGE_NAMES:
        stage = stage_by_name.get(name)
        elapsed_ms = float(stage.get("elapsed_ms") or 0.0) if stage else 0.0
        latencies.append(
            StageLatency(
                name=name,
                label=STAGE_LABELS.get(name, name),
                elapsed_ms=elapsed_ms,
            )
        )
    return latencies


def summarize_query_trace(trace: dict[str, Any]) -> QueryTraceSummary:
    refused = is_refusal(trace)
    rerank_fallback = is_rerank_fallback(trace)
    error = trace.get("error")
    if error:
        status = "failed"
    elif refused:
        status = "refused"
    else:
        status = "ok"
    return QueryTraceSummary(
        trace_id=str(trace.get("trace_id") or ""),
        started_at=str(trace.get("started_at") or ""),
        finished_at=trace.get("finished_at"),
        total_elapsed_ms=float(trace.get("total_elapsed_ms") or 0.0),
        question=_question(trace),
        culture_domain=_culture_domain(trace),
        status=status,
        refused=refused,
        rerank_fallback=rerank_fallback,
        error=error if isinstance(error, str) else None,
    )
