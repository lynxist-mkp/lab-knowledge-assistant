"""Eval dashboard reads the same typed runs that run_eval writes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.eval import get_eval_dashboard, list_eval_runs
from wenmai.eval.read import get_ragas_status
from wenmai.eval.views import GroupMetricsView, parse_eval_run


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
    parsed = parse_eval_run(artifact)
    assert parsed is not None
    path = runs_dir / f"{timestamp}.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_list_eval_runs_empty(test_settings: Settings, tmp_path: Path) -> None:
    test_settings.evaluation.runs = str(tmp_path / "runs")

    assert list_eval_runs(test_settings) == []


def test_list_eval_runs_newest_first(test_settings: Settings, tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    test_settings.evaluation.runs = str(runs_dir)
    _write_run(runs_dir, "20260101T100000Z", dense_hit=0.1, rrf_rerank_hit=0.2)
    _write_run(runs_dir, "20260102T100000Z", dense_hit=0.3, rrf_rerank_hit=0.4)

    runs = list_eval_runs(test_settings)

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
