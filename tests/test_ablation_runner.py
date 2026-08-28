"""Golden-set eval: run, list, dashboard through one interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.eval import get_eval_dashboard, list_eval_runs, run_eval
from wenmai.generation import GenerationError
from wenmai.pipelines.query import ask_question
from wenmai.tracing.store import read_trace_records


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_minpai_markdown(path: Path, title: str, body: str, domain: str = "妈祖") -> Path:
    path.write_text(
        f"""---
source_url: https://example.com/{path.stem}
culture_domain: {domain}
space: minpai_culture
title: {title}
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _prepare_eval(test_settings: Settings, tmp_path: Path) -> None:
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            '{"id":"g001","question":"妈祖信仰的发源地在哪里？","evidence_doc_ids":["matsu-intro"],'
            '"answerable":true,"reference_answer":"湄洲岛。","category":"单跳事实"}',
            '{"id":"g002","question":"船政学堂是什么时候创办的？","evidence_doc_ids":[],'
            '"answerable":false,"reference_answer":"","category":"明确不可答"}',
        ],
    )
    source = _write_minpai_markdown(
        tmp_path / "matsu-intro.md",
        "湄洲妈祖祖庙简介",
        "湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上，是信俗活动的中心场所。",
    )
    test_settings.evaluation.golden_set = str(golden)
    test_settings.evaluation.runs = str(tmp_path / "runs")
    test_settings.evaluation.ablations = ["dense_only", "sparse_only", "rrf", "rrf_rerank"]
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200


def test_run_eval_round_trip_typed_views(
    test_settings: Settings,
    tmp_path: Path,
    without_ragas_judge_key: None,
) -> None:
    _prepare_eval(test_settings, tmp_path)

    run = run_eval(test_settings)
    listed = list_eval_runs(test_settings)

    assert listed[0].timestamp == run.timestamp
    assert listed[0].item_count == 2
    assert listed[0].failed_count == 0
    assert listed[0].failures == []
    labels = [listed[0].groups[name].label for name in test_settings.evaluation.ablations]
    assert labels == ["Dense 单路", "Sparse 单路", "RRF 融合", "RRF + Rerank"]
    metrics = listed[0].groups["rrf_rerank"].metrics
    assert metrics.answerable_count == 1
    assert metrics.unanswerable_count == 1

    dashboard = get_eval_dashboard(test_settings)
    assert dashboard.latest_run is not None
    assert dashboard.latest_run.timestamp == run.timestamp
    assert dashboard.ragas.faithfulness.status == "unavailable"


def test_eval_retries_failed_item_then_succeeds(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_eval(test_settings, tmp_path)
    real_ask = ask_question
    seen = {"n": 0}

    def flaky(question: str, settings: Settings, **kwargs: object):
        if "发源地" in question and kwargs.get("retrieval_mode") == "dense_only":
            seen["n"] += 1
            if seen["n"] == 1:
                raise RuntimeError("generation failed")
        return real_ask(question, settings, **kwargs)

    monkeypatch.setattr("wenmai.eval.runner.ask_question", flaky)

    run = run_eval(test_settings)

    assert seen["n"] == 2
    assert run.failed_count == 0
    assert list_eval_runs(test_settings)[0].failed_count == 0


def test_eval_keeps_failure_after_retries(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_eval(test_settings, tmp_path)
    real_ask = ask_question
    seen = {"n": 0}

    def always_fail_dense(question: str, settings: Settings, **kwargs: object):
        if "发源地" in question and kwargs.get("retrieval_mode") == "dense_only":
            seen["n"] += 1
            raise RuntimeError("generation failed")
        return real_ask(question, settings, **kwargs)

    monkeypatch.setattr("wenmai.eval.runner.ask_question", always_fail_dense)

    run = run_eval(test_settings)

    assert seen["n"] == 4
    assert run.failed_count == 1
    assert run.failures[0].item_id == "g001"
    assert run.failures[0].group == "dense_only"
    assert run.failures[0].group_label == "Dense 单路"
    listed = list_eval_runs(test_settings)[0]
    assert listed.failed_count == 1
    assert listed.failures[0].item_id == "g001"


def test_post_eval_runs_returns_typed_run(test_settings: Settings, tmp_path: Path) -> None:
    _prepare_eval(test_settings, tmp_path)
    client = TestClient(create_app(test_settings))

    response = client.post("/api/eval/runs")

    assert response.status_code == 200
    body = response.json()
    assert body["item_count"] == 2
    assert body["failed_count"] == 0
    assert body["groups"]["rrf_rerank"]["label"] == "RRF + Rerank"
    assert "hit_at_5" in body["groups"]["rrf_rerank"]["metrics"]
    assert "stages" not in body


def test_run_eval_does_not_write_query_traces(
    test_settings: Settings,
    tmp_path: Path,
    without_ragas_judge_key: None,
) -> None:
    _prepare_eval(test_settings, tmp_path)
    before = read_trace_records(test_settings)
    query_before = [record for record in before if record.get("trace_type") == "query"]

    run_eval(test_settings)

    after = read_trace_records(test_settings)
    query_after = [record for record in after if record.get("trace_type") == "query"]
    assert query_after == query_before


def test_run_eval_artifact_contains_retrieval_snapshots(
    test_settings: Settings,
    tmp_path: Path,
    without_ragas_judge_key: None,
) -> None:
    _prepare_eval(test_settings, tmp_path)

    run = run_eval(test_settings)
    artifact_path = Path(test_settings.evaluation.runs) / f"{run.timestamp}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    items = artifact["groups"]["rrf_rerank"]["items"]

    assert "g001" in items
    assert "matsu-intro" in items["g001"]["retrieval"]["ranked_doc_ids"]
    assert items["g001"]["retrieval"]["ranked_chunks"]
    assert items["g001"]["retrieval"]["ranked_chunks"][0]["chunk_id"]


def test_run_eval_hit_at_5_survives_generation_failure(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_eval(test_settings, tmp_path)

    def fail_generate(*_args: object, **_kwargs: object) -> object:
        raise GenerationError("fake provider failed", provider_name="fake")

    monkeypatch.setattr("wenmai.pipelines.query.generate", fail_generate)

    run = run_eval(test_settings)

    assert run.failed_count > 0
    artifact_path = Path(test_settings.evaluation.runs) / f"{run.timestamp}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metrics = artifact["groups"]["rrf_rerank"]["metrics"]
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr"] == 1.0
