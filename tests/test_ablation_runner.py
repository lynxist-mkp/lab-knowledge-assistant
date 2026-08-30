"""Golden-set eval: run, list, dashboard through one interface."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.eval import get_eval_dashboard, list_eval_runs, run_eval
from wenmai.eval import pipeline as eval_pipeline
from wenmai.generation import GenerationError, generate
from wenmai.retrieval import retrieve
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
    real_generate = generate
    seen = {"n": 0}

    def flaky(question: str, scored_chunks: list[object], settings: Settings):
        if "发源地" in question and seen["n"] == 0:
            seen["n"] += 1
            raise GenerationError("generation failed", provider_name="fake")
        return real_generate(question, scored_chunks, settings)

    monkeypatch.setattr("wenmai.eval.pipeline.generate", flaky)

    run = run_eval(test_settings)

    assert seen["n"] == 1
    assert run.failed_count == 0
    assert list_eval_runs(test_settings)[0].failed_count == 0


def test_eval_keeps_failure_after_retries(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_eval(test_settings, tmp_path)
    real_eval_item = eval_pipeline.eval_item
    real_generate = generate
    seen = {"n": 0}
    current_mode: list[str] = []

    def tracking_eval_item(
        item: object,
        settings: Settings,
        *,
        retrieval_mode: str,
        rerank_enabled: bool,
        knowledge: object,
        retrieved_chunks: list[object] | None = None,
        query_rewrite: bool = False,
    ):
        current_mode.clear()
        current_mode.append(retrieval_mode)
        return real_eval_item(
            item,
            settings,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            retrieved_chunks=retrieved_chunks,
            query_rewrite=query_rewrite,
        )

    def always_fail_dense(question: str, scored_chunks: list[object], settings: Settings):
        if (
            "发源地" in question
            and current_mode
            and current_mode[0] == "dense_only"
        ):
            seen["n"] += 1
            raise GenerationError("generation failed", provider_name="fake")
        return real_generate(question, scored_chunks, settings)

    monkeypatch.setattr("wenmai.eval.runner.eval_item", tracking_eval_item)
    monkeypatch.setattr("wenmai.eval.pipeline.generate", always_fail_dense)

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


def test_run_eval_artifact_contains_latency_ms(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    without_ragas_judge_key: None,
) -> None:
    _prepare_eval(test_settings, tmp_path)
    run_started = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        "wenmai.eval.runner.datetime",
        SimpleNamespace(
            now=lambda tz=None: run_started,
            UTC=UTC,
        ),
    )
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    generation_stage = {
        "name": "generation",
        "method": "llm",
        "provider": "fake",
        "elapsed_ms": 0.0,
    }
    for trace_id, generation_ms in [("eval-q1", 100.0), ("eval-q2", 300.0)]:
        trace = {
            "trace_id": trace_id,
            "trace_type": "query",
            "started_at": run_started.isoformat(),
            "finished_at": "2026-06-01T12:00:01+00:00",
            "total_elapsed_ms": generation_ms,
            "stages": [{**generation_stage, "elapsed_ms": generation_ms}],
            "error": None,
            "metadata": {},
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

    run = run_eval(test_settings)
    artifact_path = Path(test_settings.evaluation.runs) / f"{run.timestamp}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["latency_ms"]["generation"]["p50"] == 100.0
    assert artifact["latency_ms"]["generation"]["p95"] == 300.0
    assert artifact["latency_ms"]["total"]["p50"] == 100.0
    assert artifact["latency_ms"]["total"]["p95"] == 300.0


def test_run_eval_artifact_latency_empty_without_traces(
    test_settings: Settings,
    tmp_path: Path,
    without_ragas_judge_key: None,
) -> None:
    _prepare_eval(test_settings, tmp_path)

    run = run_eval(test_settings)
    artifact_path = Path(test_settings.evaluation.runs) / f"{run.timestamp}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["latency_ms"]["total"]["p50"] is None
    assert artifact["latency_ms"]["total"]["p95"] is None


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

    monkeypatch.setattr("wenmai.eval.pipeline.generate", fail_generate)

    run = run_eval(test_settings)

    assert run.failed_count > 0
    artifact_path = Path(test_settings.evaluation.runs) / f"{run.timestamp}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metrics = artifact["groups"]["rrf_rerank"]["metrics"]
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr"] == 1.0


def test_eval_single_retrieve_per_item(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_eval(test_settings, tmp_path)
    real_retrieve = retrieve
    real_eval_item = eval_pipeline.eval_item
    real_generate = generate
    retrieve_calls: dict[tuple[str, str], int] = {}
    generate_calls: dict[tuple[str, str], int] = {}
    current_mode: list[str] = []

    def counting_eval_item(
        item: object,
        settings: Settings,
        *,
        retrieval_mode: str,
        rerank_enabled: bool,
        knowledge: object,
        retrieved_chunks: list[object] | None = None,
        query_rewrite: bool = False,
    ):
        current_mode.clear()
        current_mode.append(retrieval_mode)
        return real_eval_item(
            item,
            settings,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            retrieved_chunks=retrieved_chunks,
            query_rewrite=query_rewrite,
        )

    def counting_retrieve(
        question: str,
        settings: Settings,
        *,
        retrieval_mode: str | None = None,
        rerank_enabled: bool | None = None,
        **kwargs: object,
    ):
        key = (question, str(retrieval_mode))
        retrieve_calls[key] = retrieve_calls.get(key, 0) + 1
        return real_retrieve(
            question,
            settings,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            **kwargs,
        )

    def flaky_generate(question: str, scored_chunks: list[object], settings: Settings):
        mode = current_mode[0] if current_mode else ""
        key = (question, mode)
        generate_calls[key] = generate_calls.get(key, 0) + 1
        if (
            "发源地" in question
            and mode == "dense_only"
            and generate_calls[key] == 1
        ):
            raise GenerationError("generation failed", provider_name="fake")
        return real_generate(question, scored_chunks, settings)

    monkeypatch.setattr("wenmai.eval.runner.eval_item", counting_eval_item)
    monkeypatch.setattr("wenmai.eval.pipeline.retrieve", counting_retrieve)
    monkeypatch.setattr("wenmai.eval.pipeline.generate", flaky_generate)

    run = run_eval(test_settings)

    assert run.failed_count == 0
    dense_key = ("妈祖信仰的发源地在哪里？", "dense_only")
    assert retrieve_calls[dense_key] == 1
    assert generate_calls[dense_key] == 2
