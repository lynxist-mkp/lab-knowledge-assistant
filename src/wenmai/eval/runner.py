from __future__ import annotations

import logging
from datetime import UTC, datetime

from wenmai.config import Settings
from wenmai.eval import pipeline as eval_pipeline
from wenmai.eval.ablation import (
    config_snapshot,
    resolve_ablation,
    rewrite_compare_config_snapshot,
)
from wenmai.eval.artifact import (
    build_ablation_eval_artifact,
    build_rewrite_compare_eval_artifact,
    group_artifact_entry,
)
from wenmai.eval.golden import GoldItem, load_golden_set_from_settings
from wenmai.eval.persist import persist_eval_artifact
from wenmai.eval.pipeline import EvalGroupItem
from wenmai.eval.ragas_metrics import attach_ragas_to_artifact, should_run_ragas
from wenmai.eval.views import EvalRunView, FailedEvalItem
from wenmai.generation import GenerationResult
from wenmai.knowledge import Knowledge, create_knowledge
from wenmai.models import ScoredChunk
from wenmai.task_progress import (
    EvaluationTaskProgressConfig,
    TaskProgressRun,
    safe_persist_task_progress_outcome,
    safe_persist_task_progress_running,
)
from wenmai.task_progress_builders import build_evaluation_progress

logger = logging.getLogger(__name__)

FAILURE_RETRIES = 3

REWRITE_COMPARE_GROUPS = ("rewrite_off", "rewrite_on")
REWRITE_COMPARE_FLAGS: dict[str, bool] = {
    "rewrite_off": False,
    "rewrite_on": True,
}


def _run_group_batched(
    items: list[GoldItem],
    settings: Settings,
    group: str,
    knowledge: Knowledge,
    *,
    retrieval_mode: str,
    rerank_enabled: bool,
    query_rewrite: bool = False,
) -> tuple[
    dict[str, list[ScoredChunk]],
    dict[str, GenerationResult],
    list[FailedEvalItem],
]:
    """Run one ablation group with phase batching and generation retries."""
    ranked_chunks: dict[str, list[ScoredChunk]] = {}
    generation_results: dict[str, GenerationResult] = {}
    pending: list[GoldItem] = list(items)

    for attempt in range(FAILURE_RETRIES + 1):
        if not pending:
            break
        group_inputs: list[EvalGroupItem] = []
        for item in pending:
            existing = ranked_chunks.get(item.id)
            group_inputs.append(
                EvalGroupItem(item=item, existing_chunks=existing)
            )

        logger.info(
            "eval group=%s attempt=%d/%d items=%d",
            group,
            attempt + 1,
            FAILURE_RETRIES + 1,
            len(group_inputs),
        )
        outcomes = eval_pipeline.run_eval_group_batched(
            group_inputs,
            settings,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            query_rewrite=query_rewrite,
        )
        for item in pending:
            chunks, result = outcomes.get(item.id, (None, None))
            if chunks is not None:
                ranked_chunks[item.id] = chunks
            if result is not None:
                generation_results[item.id] = result

        pending = [
            item
            for item in pending
            if item.id not in ranked_chunks or item.id not in generation_results
        ]

    failures = [
        FailedEvalItem(group=group, item_id=item.id) for item in pending
    ]
    return ranked_chunks, generation_results, failures


def _run_grouped_eval(
    items: list[GoldItem],
    settings: Settings,
    groups: list[str],
    knowledge: Knowledge,
    *,
    query_rewrite_by_group: dict[str, bool] | None = None,
    spec_by_group: dict[str, tuple[str, bool]] | None = None,
) -> tuple[
    dict[str, dict[str, list[ScoredChunk]]],
    dict[str, dict[str, GenerationResult]],
    list[FailedEvalItem],
]:
    ranked_chunks: dict[str, dict[str, list[ScoredChunk]]] = {
        name: {} for name in groups
    }
    generation_results: dict[str, dict[str, GenerationResult]] = {
        name: {} for name in groups
    }
    failures: list[FailedEvalItem] = []
    total_ops = len(groups) * len(items)
    completed = 0

    for group_index, group in enumerate(groups, start=1):
        if spec_by_group is not None:
            retrieval_mode, rerank_enabled = spec_by_group[group]
        else:
            spec = resolve_ablation(group)
            retrieval_mode = spec.mode
            rerank_enabled = spec.rerank_enabled
        query_rewrite = (
            query_rewrite_by_group.get(group, False)
            if query_rewrite_by_group
            else False
        )

        logger.info(
            "eval starting group=%s (%d/%d) items=%d",
            group,
            group_index,
            len(groups),
            len(items),
        )
        group_ranked, group_generation, group_failures = _run_group_batched(
            items,
            settings,
            group,
            knowledge,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            query_rewrite=query_rewrite,
        )
        ranked_chunks[group] = group_ranked
        generation_results[group] = group_generation
        failures.extend(group_failures)
        completed += len(items)
        logger.info(
            "eval progress %d/%d item×group ops group=%s failed=%d",
            completed,
            total_ops,
            group,
            len(group_failures),
        )

    return ranked_chunks, generation_results, failures


def run_eval(
    settings: Settings,
    *,
    knowledge: Knowledge | None = None,
    query_rewrite: bool = False,
    ragas: bool | None = None,
    item_limit: int | None = None,
    groups: list[str] | None = None,
) -> EvalRunView:
    items = load_golden_set_from_settings(settings)
    if item_limit is not None:
        items = items[:item_limit]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    resolved_groups = list(groups or settings.evaluation.ablations)
    resolved_knowledge = knowledge or create_knowledge(settings)
    progress_run = TaskProgressRun(
        task_id=f"evaluation:{timestamp}",
        task_type="evaluation",
        started_at=timestamp,
        trigger_source="eval_runner",
        owner_surface="ops",
        links={"eval_run": timestamp},
    )
    config = EvaluationTaskProgressConfig(
        groups=resolved_groups,
        query_rewrite_by_group={name: query_rewrite for name in resolved_groups},
    )
    safe_persist_task_progress_running(
        settings,
        progress_run,
        total=len(items) * len(resolved_groups),
        config=config,
    )

    ranked_chunks, generation_results, failures = _run_grouped_eval(
        items,
        settings,
        resolved_groups,
        resolved_knowledge,
        query_rewrite_by_group={name: query_rewrite for name in resolved_groups},
    )
    artifact = build_ablation_eval_artifact(
        timestamp=timestamp,
        golden_set=settings.evaluation.golden_set,
        ablations=resolved_groups,
        item_count=len(items),
        failures=failures,
        groups={
            name: group_artifact_entry(
                items,
                ranked_chunks[name],
                generation_results[name],
                config=config_snapshot(settings, name),
            )
            for name in resolved_groups
        },
    )
    if should_run_ragas(settings, ragas) and "rrf_rerank" in resolved_groups:
        attach_ragas_to_artifact(
            artifact,
            items,
            generation_results["rrf_rerank"],
            ranked_chunks["rrf_rerank"],
            settings,
        )
    run = persist_eval_artifact(settings, artifact)
    safe_persist_task_progress_outcome(
        settings,
        progress_run,
        config=config,
        outcome=build_evaluation_progress(
            timestamp=timestamp,
            items=items,
            groups=resolved_groups,
            ranked_chunks=ranked_chunks,
            generation_results=generation_results,
            failures=failures,
        ),
    )
    return run


def run_rewrite_compare(
    settings: Settings,
    *,
    knowledge: Knowledge | None = None,
) -> EvalRunView:
    items = load_golden_set_from_settings(settings)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    groups = list(REWRITE_COMPARE_GROUPS)
    resolved_knowledge = knowledge or create_knowledge(settings)
    progress_run = TaskProgressRun(
        task_id=f"evaluation:{timestamp}",
        task_type="evaluation",
        started_at=timestamp,
        trigger_source="eval_runner",
        owner_surface="ops",
        links={"eval_run": timestamp},
    )
    config = EvaluationTaskProgressConfig(
        groups=groups,
        query_rewrite_by_group=REWRITE_COMPARE_FLAGS,
    )
    safe_persist_task_progress_running(
        settings,
        progress_run,
        total=len(items) * len(groups),
        config=config,
    )

    ranked_chunks, generation_results, failures = _run_grouped_eval(
        items,
        settings,
        groups,
        resolved_knowledge,
        query_rewrite_by_group=REWRITE_COMPARE_FLAGS,
        spec_by_group={name: ("rrf", True) for name in groups},
    )
    artifact = build_rewrite_compare_eval_artifact(
        timestamp=timestamp,
        golden_set=settings.evaluation.golden_set,
        item_count=len(items),
        failures=failures,
        groups={
            name: group_artifact_entry(
                items,
                ranked_chunks[name],
                generation_results[name],
                config=rewrite_compare_config_snapshot(
                    settings,
                    group=name,
                    query_rewrite=REWRITE_COMPARE_FLAGS[name],
                ),
            )
            for name in groups
        },
    )

    run = persist_eval_artifact(settings, artifact)
    safe_persist_task_progress_outcome(
        settings,
        progress_run,
        config=config,
        outcome=build_evaluation_progress(
            timestamp=timestamp,
            items=items,
            groups=groups,
            ranked_chunks=ranked_chunks,
            generation_results=generation_results,
            failures=failures,
        ),
    )
    return run
