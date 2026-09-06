from __future__ import annotations

from typing import Any

from lab_knowledge.components.evaluator.ragas_probe import probe_ragas_judge
from lab_knowledge.config import Settings
from lab_knowledge.eval.golden import GoldItem
from lab_knowledge.factories import evaluator as evaluator_factory
from lab_knowledge.generation import GenerationResult
from lab_knowledge.models import ScoredChunk

_STATUS_OK = "ok"
_STATUS_UNAVAILABLE = "unavailable"
_RAGAS_GROUP = "rrf_rerank"


def should_run_ragas(settings: Settings, ragas: bool | None) -> bool:
    if ragas is False:
        return False
    if ragas is True:
        return True
    available, _ = probe_ragas_judge(settings)
    return available


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _metric_value(payload: dict[str, Any]) -> float | None:
    if payload.get("status") != _STATUS_OK:
        return None
    raw = payload.get("value")
    if raw is None:
        return None
    return float(raw)


def compute_group_ragas_metrics(
    items: list[GoldItem],
    generation_by_id: dict[str, GenerationResult],
    chunks_by_id: dict[str, list[ScoredChunk]],
    settings: Settings,
) -> dict[str, Any]:
    """Aggregate Ragas faithfulness and context_precision for one ablation group."""
    available, reason = probe_ragas_judge(settings)
    if not available:
        return {"status": _STATUS_UNAVAILABLE, "reason": reason or "Ragas judge 未就绪"}

    try:
        evaluator = evaluator_factory.create(settings)
    except Exception as exc:
        return {"status": _STATUS_UNAVAILABLE, "reason": str(exc)}
    faithfulness_scores: list[float] = []
    context_precision_scores: list[float] = []
    scored_count = 0
    skipped_count = 0

    for item in items:
        if not item.answerable:
            continue
        generation = generation_by_id.get(item.id)
        chunks = chunks_by_id.get(item.id)
        if generation is None or chunks is None:
            skipped_count += 1
            continue
        if generation.refused:
            skipped_count += 1
            continue

        contexts = [scored.chunk.text for scored in chunks]
        result = evaluator.score(
            user_input=item.question,
            response=generation.answer,
            retrieved_contexts=contexts,
            reference=item.reference_answer,
        )
        faith = _metric_value(result.get("faithfulness") or {})
        ctx = _metric_value(result.get("context_precision") or {})
        if faith is None and ctx is None:
            skipped_count += 1
            continue
        scored_count += 1
        if faith is not None:
            faithfulness_scores.append(faith)
        if ctx is not None:
            context_precision_scores.append(ctx)

    faith_avg = _average(faithfulness_scores)
    ctx_avg = _average(context_precision_scores)
    if scored_count == 0:
        return {
            "status": _STATUS_UNAVAILABLE,
            "reason": "无可用评分样本（生成失败或拒答）",
            "scored_count": 0,
            "skipped_count": skipped_count,
        }

    payload: dict[str, Any] = {
        "status": _STATUS_OK,
        "faithfulness": faith_avg,
        "context_precision": ctx_avg,
        "scored_count": scored_count,
        "skipped_count": skipped_count,
        "group": _RAGAS_GROUP,
    }
    return payload


def attach_ragas_to_artifact(
    artifact: dict[str, Any],
    items: list[GoldItem],
    generation_by_id: dict[str, GenerationResult],
    chunks_by_id: dict[str, list[ScoredChunk]],
    settings: Settings,
) -> dict[str, Any]:
    """Return ragas payload and optionally merge into group metrics."""
    ragas = compute_group_ragas_metrics(
        items,
        generation_by_id,
        chunks_by_id,
        settings,
    )
    group = artifact.get("groups", {}).get(_RAGAS_GROUP)
    if isinstance(group, dict) and isinstance(group.get("metrics"), dict):
        group["metrics"]["ragas"] = ragas
    artifact["ragas"] = ragas
    return ragas
