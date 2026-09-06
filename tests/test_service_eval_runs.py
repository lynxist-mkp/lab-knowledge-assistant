"""Eval dashboard reads the same typed runs that run_eval writes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from lab_knowledge.app import create_app
from lab_knowledge.config import Settings
from lab_knowledge.eval import get_eval_dashboard
from lab_knowledge.eval.read import (
    get_eval_run_detail,
    get_eval_run_summary,
    get_ragas_status,
    list_eval_run_summaries,
)
from lab_knowledge.eval.views import GroupMetricsView, parse_eval_run_summary


def _sample_metrics(hit: float, mrr: float, refusal: float) -> dict[str, float | int]:
    return GroupMetricsView(
        hit_at_5=hit,
        mrr=mrr,
        refusal_accuracy=refusal,
        citation_coverage=0.8,
        answerable_count=40,
        unanswerable_count=10,
    ).as_dict()


def _write_run(
    runs_dir: Path,
    timestamp: str,
    *,
    dense_hit: float,
    rrf_rerank_hit: float,
    failures: list[dict[str, str]] | None = None,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "dense_only": _sample_metrics(dense_hit, 0.2, 0.9),
        "sparse_only": _sample_metrics(0.3, 0.25, 0.85),
        "rrf": _sample_metrics(0.5, 0.4, 0.88),
        "rrf_rerank": _sample_metrics(rrf_rerank_hit, 0.55, 0.92),
    }
    artifact = {
        "timestamp": timestamp,
        "golden_set": "data/eval/golden.jsonl",
        "ablations": list(groups.keys()),
        "item_count": 50,
        "failures": failures or [],
        "groups": {
            name: {"config": {"ablation_group": name}, "metrics": metrics}
            for name, metrics in groups.items()
        },
    }
    parsed = parse_eval_run_summary(artifact)
    assert parsed is not None
    path = runs_dir / f"{timestamp}.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_list_eval_runs_empty(test_settings: Settings, tmp_path: Path) -> None:
    test_settings.evaluation.runs = str(tmp_path / "runs")

    assert list_eval_run_summaries(test_settings) == []


def test_list_eval_runs_newest_first(test_settings: Settings, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    test_settings.evaluation.runs = str(runs_dir)
    _write_run(runs_dir, "20260101T100000Z", dense_hit=0.1, rrf_rerank_hit=0.2)
    _write_run(runs_dir, "20260102T100000Z", dense_hit=0.3, rrf_rerank_hit=0.4)

    runs = list_eval_run_summaries(test_settings)

    assert [run.timestamp for run in runs] == ["20260102T100000Z", "20260101T100000Z"]
    assert runs[0].groups["rrf_rerank"].metrics.hit_at_5 == 0.4
    assert runs[0].failed_count == 0


def test_dashboard_exposes_failures(test_settings: Settings, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    test_settings.evaluation.runs = str(runs_dir)
    _write_run(
        runs_dir,
        "20260102T100000Z",
        dense_hit=0.3,
        rrf_rerank_hit=0.4,
        failures=[{"group": "dense_only", "item_id": "g001"}],
    )

    dashboard = get_eval_dashboard(test_settings)

    assert dashboard.latest_run is not None
    assert dashboard.latest_run.failed_count == 1
    assert dashboard.latest_run.failures[0].item_id == "g001"
    assert dashboard.latest_run.failures[0].group_label == "Dense 单路"


def test_get_eval_dashboard_exposes_latest_and_history(
    test_settings: Settings, tmp_path: Path
) -> None:
    runs_dir = tmp_path / "runs"
    test_settings.evaluation.runs = str(runs_dir)
    _write_run(runs_dir, "20260101T100000Z", dense_hit=0.1, rrf_rerank_hit=0.2)
    _write_run(runs_dir, "20260102T100000Z", dense_hit=0.3, rrf_rerank_hit=0.45)

    dashboard = get_eval_dashboard(test_settings)

    assert dashboard.latest_run is not None
    assert dashboard.latest_run.timestamp == "20260102T100000Z"
    assert len(dashboard.history) == 2
    assert set(dashboard.latest_run.groups) == {
        "dense_only",
        "sparse_only",
        "rrf",
        "rrf_rerank",
    }


def test_get_ragas_status_marks_unavailable_without_api_key(
    test_settings: Settings,
    without_ragas_judge_key: None,
) -> None:
    status = get_ragas_status(test_settings)

    assert status.configured is True
    assert status.faithfulness.status == "unavailable"
    assert status.context_precision.status == "unavailable"
    assert "未跑通" in status.faithfulness.label
    assert "ZHIPUAI_API_KEY" in status.faithfulness.reason


def test_api_eval_runs_endpoint(test_settings: Settings, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    test_settings.evaluation.runs = str(runs_dir)
    _write_run(runs_dir, "20260102T100000Z", dense_hit=0.25, rrf_rerank_hit=0.62)

    client = TestClient(create_app(test_settings))
    response = client.get("/api/eval/runs")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["timestamp"] == "20260102T100000Z"
    assert payload[0]["groups"]["rrf_rerank"]["metrics"]["hit_at_5"] == 0.62
    assert payload[0]["failed_count"] == 0


def _write_run_with_items(
    runs_dir: Path,
    timestamp: str,
    *,
    ragas: dict[str, object] | None = None,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics = _sample_metrics(0.5, 0.4, 0.9)
    if ragas is not None:
        metrics = {**metrics, "ragas": ragas}
    artifact = {
        "timestamp": timestamp,
        "golden_set": "data/eval/golden.jsonl",
        "ablations": ["rrf_rerank"],
        "item_count": 2,
        "failures": [],
        "ragas": ragas,
        "groups": {
            "rrf_rerank": {
                "config": {"ablation_group": "rrf_rerank"},
                "metrics": metrics,
                "items": {
                    "g001": {
                        "retrieval": {
                            "ranked_doc_ids": ["doc-a"],
                            "ranked_chunks": [
                                {
                                    "chunk_id": "c1",
                                    "document_id": "d1",
                                    "score": 0.9,
                                }
                            ],
                        },
                        "refused": False,
                        "citation_count": 1,
                    }
                },
            }
        },
    }
    path = runs_dir / f"{timestamp}.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_eval_run_summary_and_detail_seams(test_settings: Settings, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    test_settings.evaluation.runs = str(runs_dir)
    _write_run_with_items(
        runs_dir,
        "20260103T100000Z",
        ragas={
            "status": "ok",
            "faithfulness": 0.91,
            "context_precision": 0.82,
            "scored_count": 1,
            "skipped_count": 0,
        },
    )

    summaries = list_eval_run_summaries(test_settings)
    assert len(summaries) == 1
    assert summaries[0].timestamp == "20260103T100000Z"
    assert "g001" not in summaries[0].as_dict()

    summary = get_eval_run_summary(test_settings, "20260103T100000Z")
    assert summary is not None
    assert summary.groups["rrf_rerank"].metrics.hit_at_5 == 0.5

    detail = get_eval_run_detail(test_settings, "20260103T100000Z")
    assert detail is not None
    assert detail.summary.timestamp == "20260103T100000Z"
    assert detail.groups["rrf_rerank"].items["g001"].retrieval.ranked_doc_ids == ["doc-a"]
    assert detail.ragas.status == "ok"
    assert detail.ragas.faithfulness == 0.91
