from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wenmai.config import Settings
from wenmai.eval.corpus_ingest import (
    DEFAULT_ITEMS_DIR,
    DEFAULT_MANIFEST,
    IngestManifestResult,
    ingest_corpus_manifest,
)
from wenmai.eval.golden import GoldItem, load_golden_set_from_settings
from wenmai.eval.metrics import hit_at_5
from wenmai.eval.runner import run_eval, run_rewrite_compare
from wenmai.eval.views import EvalRunView

DEFAULT_BAD_CASES_PATH = Path(".scratch/fuyun-wenmai/phase-b-bad-cases.md")
PHASE_B_GROUP = "rrf_rerank"


@dataclass(frozen=True)
class PhaseBRunResult:
    timestamp: str
    summary_path: Path
    ablation_run: EvalRunView
    rewrite_compare_run: EvalRunView
    ingest_result: IngestManifestResult | None
    bad_cases_path: Path | None


def _runs_dir(settings: Settings) -> Path:
    raw = Path(settings.evaluation.runs)
    path = raw if raw.is_absolute() else settings.root / raw
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_repo_path(settings: Settings, path: Path) -> Path:
    return path if path.is_absolute() else settings.root / path


def _resolve_bad_cases_path(settings: Settings, path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.as_posix() == DEFAULT_BAD_CASES_PATH.as_posix():
        return settings.root.parent / path
    return settings.root / path


def find_hit_at_5_misses(
    artifact: dict[str, Any],
    items: list[GoldItem],
    *,
    group: str = PHASE_B_GROUP,
) -> list[dict[str, str]]:
    groups = artifact.get("groups") or {}
    group_payload = groups.get(group) or {}
    item_snapshots = group_payload.get("items") or {}
    misses: list[dict[str, str]] = []
    for item in items:
        if not item.answerable:
            continue
        snapshot = item_snapshots.get(item.id)
        if not isinstance(snapshot, dict):
            continue
        retrieval = snapshot.get("retrieval") or {}
        ranked_doc_ids = retrieval.get("ranked_doc_ids") or []
        score = hit_at_5(ranked_doc_ids, item)
        if score == 0.0:
            misses.append(
                {
                    "item_id": item.id,
                    "question": item.question,
                    "category": item.category,
                    "evidence_doc_ids": ",".join(item.evidence_doc_ids),
                }
            )
    return misses


def ensure_bad_cases_template(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Phase B Bad Cases

记录评测中 Hit@5=0 的可答题，供讲述五个阶段（检索、融合、重排、生成、拒答）时对照 Trace。

## 填写说明

| 字段 | 说明 |
|------|------|
| item id | 黄金集条目 id |
| question | 原题 |
| what went wrong | 简述失败现象（人工填写） |
| trace stage | 怀疑出问题的阶段（人工填写） |
| trace_id | 对应 query trace（人工填写） |

## 待分析（脚本自动生成）

""",
        encoding="utf-8",
    )


def append_bad_case_stubs(path: Path, misses: list[dict[str, str]], *, run_timestamp: str) -> None:
    ensure_bad_cases_template(path)
    if not misses:
        return
    lines = [f"\n### Run {run_timestamp}\n"]
    for miss in misses:
        lines.extend(
            [
                f"#### {miss['item_id']}\n",
                f"- **question**: {miss['question']}\n",
                f"- **category**: {miss['category']}\n",
                f"- **evidence_doc_ids**: {miss['evidence_doc_ids']}\n",
                "- **what went wrong**: （待填写）\n",
                "- **trace stage**: （待填写）\n",
                "- **trace_id**: （待填写）\n",
                "\n",
            ]
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.writelines(lines)


def build_phase_b_summary(
    settings: Settings,
    *,
    timestamp: str,
    ablation_artifact_path: Path,
    rewrite_compare_artifact_path: Path,
    ablation_run: EvalRunView,
    rewrite_compare_run: EvalRunView,
    ingest_result: IngestManifestResult | None,
) -> dict[str, Any]:
    ablation_artifact = json.loads(ablation_artifact_path.read_text(encoding="utf-8"))
    rewrite_artifact = json.loads(rewrite_compare_artifact_path.read_text(encoding="utf-8"))
    rrf_metrics = ablation_run.groups[PHASE_B_GROUP].metrics
    rewrite_off = rewrite_compare_run.groups["rewrite_off"].metrics
    rewrite_on = rewrite_compare_run.groups["rewrite_on"].metrics
    ragas = ablation_artifact.get("ragas") or (
        (ablation_artifact.get("groups", {}).get(PHASE_B_GROUP, {}).get("metrics") or {}).get(
            "ragas"
        )
    )
    if ragas is None:
        ragas = {"status": "unavailable", "reason": "未执行 Ragas"}
    return {
        "timestamp": timestamp,
        "phase": "B",
        "golden_set": settings.evaluation.golden_set,
        "ingest": None if ingest_result is None else {
            "total": ingest_result.total,
            "attempted": ingest_result.attempted,
            "ingested": ingest_result.ingested,
            "skipped": ingest_result.skipped,
            "missing": ingest_result.missing,
            "failed": ingest_result.failed,
        },
        "artifacts": {
            "ablation": str(ablation_artifact_path),
            "rewrite_compare": str(rewrite_compare_artifact_path),
        },
        "metrics": {
            "ablation": {
                group: {
                    "hit_at_5": ablation_run.groups[group].metrics.hit_at_5,
                    "mrr": ablation_run.groups[group].metrics.mrr,
                    "refusal_accuracy": ablation_run.groups[group].metrics.refusal_accuracy,
                    "citation_coverage": ablation_run.groups[group].metrics.citation_coverage,
                }
                for group in ablation_run.groups
            },
            "rrf_rerank": {
                "hit_at_5": rrf_metrics.hit_at_5,
                "mrr": rrf_metrics.mrr,
                "refusal_accuracy": rrf_metrics.refusal_accuracy,
                "citation_coverage": rrf_metrics.citation_coverage,
            },
            "rewrite_compare": {
                "rewrite_off": {
                    "hit_at_5": rewrite_off.hit_at_5,
                    "mrr": rewrite_off.mrr,
                },
                "rewrite_on": {
                    "hit_at_5": rewrite_on.hit_at_5,
                    "mrr": rewrite_on.mrr,
                },
            },
        },
        "ragas": ragas,
        "rewrite_compare_item_count": rewrite_artifact.get("item_count"),
        "ablation_item_count": ablation_artifact.get("item_count"),
    }


def format_metrics_summary(summary: dict[str, Any]) -> str:
    rrf = summary["metrics"]["rrf_rerank"]
    ragas = summary.get("ragas") or {}
    ragas_part = "ragas=unavailable"
    if isinstance(ragas, dict) and ragas.get("status") == "ok":
        faith = ragas.get("faithfulness")
        ctx = ragas.get("context_precision")
        ragas_part = f"faithfulness={faith:.3f} context_precision={ctx:.3f}"
    return (
        f"rrf_rerank Hit@5={rrf['hit_at_5']:.3f} MRR={rrf['mrr']:.3f} "
        f"refusal={rrf['refusal_accuracy']:.3f} {ragas_part}"
    )


def run_phase_b_batch(
    settings: Settings,
    *,
    skip_ingest: bool = False,
    manifest_path: Path = DEFAULT_MANIFEST,
    items_dir: Path = DEFAULT_ITEMS_DIR,
    bad_cases_path: Path = DEFAULT_BAD_CASES_PATH,
    ragas: bool | None = None,
    ingest_limit: int | None = None,
    ingest_dry_run: bool = False,
) -> PhaseBRunResult:
    resolved_manifest = _resolve_repo_path(settings, manifest_path)
    resolved_items = _resolve_repo_path(settings, items_dir)
    resolved_bad_cases = _resolve_bad_cases_path(settings, bad_cases_path)

    ingest_result: IngestManifestResult | None = None
    if not skip_ingest:
        ingest_result = ingest_corpus_manifest(
            settings,
            resolved_manifest,
            resolved_items,
            dry_run=ingest_dry_run,
            limit=ingest_limit,
        )
        if not ingest_result.success:
            raise RuntimeError(
                "语料入库全部失败: " + "; ".join(ingest_result.errors[:5])
            )

    ablation_run = run_eval(settings, query_rewrite=False, ragas=ragas)
    rewrite_compare_run = run_rewrite_compare(settings)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ablation_path = _runs_dir(settings) / f"{ablation_run.timestamp}.json"
    rewrite_path = _runs_dir(settings) / f"{rewrite_compare_run.timestamp}.json"

    summary = build_phase_b_summary(
        settings,
        timestamp=timestamp,
        ablation_artifact_path=ablation_path,
        rewrite_compare_artifact_path=rewrite_path,
        ablation_run=ablation_run,
        rewrite_compare_run=rewrite_compare_run,
        ingest_result=ingest_result,
    )
    summary_path = _runs_dir(settings) / f"phase_b_{timestamp}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ablation_artifact = json.loads(ablation_path.read_text(encoding="utf-8"))
    golden_items = load_golden_set_from_settings(settings)
    misses = find_hit_at_5_misses(ablation_artifact, golden_items)
    append_bad_case_stubs(resolved_bad_cases, misses, run_timestamp=timestamp)

    return PhaseBRunResult(
        timestamp=timestamp,
        summary_path=summary_path,
        ablation_run=ablation_run,
        rewrite_compare_run=rewrite_compare_run,
        ingest_result=ingest_result,
        bad_cases_path=resolved_bad_cases if misses else None,
    )
