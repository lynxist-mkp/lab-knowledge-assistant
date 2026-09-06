from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import register_collection
from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.eval import run_eval
from wenmai.ops.observation import get_task_progress_detail, list_task_progress_summaries
from wenmai.pipelines.ingestion import run_prepare_commit
from wenmai.task_progress import (
    ChildEvidence,
    EvaluationTaskProgressConfig,
    IngestionTaskProgressConfig,
    StageEvent,
    TaskCounters,
    TaskProgressOutcome,
    TaskProgressRun,
    get_task_progress,
    list_task_progress,
    persist_task_progress,
    persist_task_progress_outcome,
    persist_task_progress_running,
    read_task_progress_records,
    task_progress_path,
)
from wenmai.tracing.store import read_trace_records


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    registered = register_collection(test_settings, other_id)
    return replace(
        registered,
        product=replace(test_settings.product, collection=other_id),
    )


def _write_markdown(path: Path, title: str = "测试文档", body: str = "闽派文化材料。") -> Path:
    path.write_text(
        f"""---
title: {title}
culture_domain: 妈祖
space: minpai_culture
source_url: https://example.com/{path.stem}
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _prepare_eval(test_settings: Settings, tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [
            '{"id":"g001","question":"妈祖信仰的发源地在哪里？","evidence_doc_ids":["matsu-intro"],'
            '"answerable":true,"reference_answer":"湄洲岛。","category":"单跳事实"}',
            '{"id":"g002","question":"船政学堂是什么时候创办的？","evidence_doc_ids":[],'
            '"answerable":false,"reference_answer":"","category":"明确不可答"}',
        ],
    )
    source = _write_markdown(
        tmp_path / "matsu-intro.md",
        title="湄洲妈祖祖庙简介",
        body="湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上，是信俗活动的中心场所。",
    )
    test_settings.evaluation.golden_set = str(golden)
    test_settings.evaluation.runs = str(tmp_path / "runs")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source), "pdf_load_mode": "ocr"})
    assert ingest.status_code == 200


def test_task_progress_round_trip_via_observation_seam(test_settings: Settings) -> None:
    persist_task_progress(
        test_settings,
        task_id="evaluation:demo",
        task_type="evaluation",
        status="partial_success",
        started_at="2026-09-05T00:00:00+00:00",
        finished_at="2026-09-05T00:01:00+00:00",
        last_progress_at="2026-09-05T00:00:55+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf_rerank"]},
        links={"eval_run": "demo"},
        counters=TaskCounters(total=2, completed=1, failed=1, partial=1),
        error=None,
        failure_kind="unknown",
        stages=[
            StageEvent(
                name="rrf_rerank",
                status="partial_success",
                started_at=None,
                finished_at=None,
                elapsed_ms=0.0,
                failure_kind="unknown",
                error="1 failed items",
                degraded=True,
                links={"eval_run": "demo"},
            )
        ],
        children=[
            ChildEvidence(
                child_id="rrf_rerank:g001",
                child_type="eval_item",
                label="rrf_rerank/g001",
                status="failed",
                failure_kind="unknown",
                summary="测试问题",
                degraded=True,
                links={"eval_run": "demo", "item_id": "g001"},
            )
        ],
    )

    summaries = list_task_progress_summaries(test_settings, task_type="evaluation")
    assert len(summaries) == 1
    assert summaries[0].task_id == "evaluation:demo"
    assert summaries[0].status == "partial_success"
    assert summaries[0].counters.failed == 1

    detail = get_task_progress_detail(test_settings, "evaluation:demo")
    assert detail is not None
    assert detail.summary.links["eval_run"] == "demo"
    assert detail.children[0].links["item_id"] == "g001"


def test_api_task_progress_detail(test_settings: Settings) -> None:
    persist_task_progress(
        test_settings,
        task_id="ingestion:demo",
        task_type="ingestion",
        status="succeeded",
        started_at="2026-09-05T00:00:00+00:00",
        finished_at="2026-09-05T00:00:10+00:00",
        last_progress_at="2026-09-05T00:00:10+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        config_snapshot={"pdf_load_mode": "auto"},
        links={"trace_id": "trace-demo"},
        counters=TaskCounters(total=1, completed=1),
    )

    client = TestClient(create_app(test_settings))
    list_resp = client.get("/api/tasks/progress", params={"task_type": "ingestion"})
    assert list_resp.status_code == 200
    assert list_resp.json()[0]["task_id"] == "ingestion:demo"

    detail_resp = client.get("/api/tasks/progress/ingestion:demo")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["links"]["trace_id"] == "trace-demo"

    filtered = client.get(
        "/api/tasks/progress",
        params={"task_type": "ingestion", "has_trace": "true", "needs_attention": "false"},
    )
    assert filtered.status_code == 200
    assert filtered.json()[0]["task_id"] == "ingestion:demo"

    investigation = client.get("/api/tasks/progress/ingestion:demo/investigation")
    assert investigation.status_code == 200
    payload = investigation.json()
    assert payload["task"]["task_id"] == "ingestion:demo"
    assert ("trace", "trace-demo") in {
        (item["link_type"], item["target_id"]) for item in payload["links"]
    }


def test_task_progress_builders_import_cleanly() -> None:
    module = import_module("wenmai.task_progress_builders")
    assert hasattr(module, "build_ingestion_progress")


def test_task_progress_running_and_outcome_seam_owns_config_and_failed_status(
    test_settings: Settings,
) -> None:
    run = TaskProgressRun(
        task_id="ingestion:failed-demo",
        task_type="ingestion",
        started_at="2026-09-05T00:00:00+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        links={"trace_id": "trace-failed"},
    )

    persist_task_progress_running(
        test_settings,
        run,
        total=1,
        config=IngestionTaskProgressConfig(pdf_load_mode="manual"),
    )
    persist_task_progress_outcome(
        test_settings,
        run,
        config=IngestionTaskProgressConfig(pdf_load_mode="manual"),
        outcome=TaskProgressOutcome(
            finished_at="2026-09-05T00:00:02+00:00",
            last_progress_at="2026-09-05T00:00:02+00:00",
            stages=[
                StageEvent(
                    name="load",
                    status="failed",
                    started_at=None,
                    finished_at=None,
                    elapsed_ms=12.0,
                    failure_kind="input",
                    error="pdf parse failed",
                )
            ],
            children=[
                ChildEvidence(
                    child_id="doc-1",
                    child_type="document",
                    label="坏文档",
                    status="failed",
                    failure_kind="input",
                )
            ],
            error="pdf parse failed",
        ),
    )

    detail = get_task_progress_detail(test_settings, "ingestion:failed-demo")
    assert detail is not None
    assert detail.summary.status == "failed"
    assert detail.summary.counters.failed == 1
    assert detail.summary.config_fingerprint
    assert detail.config_snapshot["pdf_load_mode"] == "manual"
    assert "quality_gate" in detail.config_snapshot


def test_task_progress_outcome_seam_derives_eval_partial_and_blocked_status(
    test_settings: Settings,
) -> None:
    blocked_run = TaskProgressRun(
        task_id="ingestion:blocked-demo",
        task_type="ingestion",
        started_at="2026-09-05T00:00:00+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
    )
    persist_task_progress_outcome(
        test_settings,
        blocked_run,
        config=IngestionTaskProgressConfig(pdf_load_mode=None),
        outcome=TaskProgressOutcome(
            finished_at="2026-09-05T00:00:01+00:00",
            last_progress_at="2026-09-05T00:00:01+00:00",
            children=[
                ChildEvidence(
                    child_id="doc-2",
                    child_type="document",
                    label="待拦截文档",
                    status="blocked",
                    failure_kind="input",
                )
            ],
        ),
    )
    blocked_detail = get_task_progress_detail(test_settings, "ingestion:blocked-demo")
    assert blocked_detail is not None
    assert blocked_detail.summary.status == "blocked"
    assert blocked_detail.summary.failure_kind == "input"

    eval_run = TaskProgressRun(
        task_id="evaluation:partial-demo",
        task_type="evaluation",
        started_at="2026-09-05T00:10:00+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        links={"eval_run": "partial-demo"},
    )
    persist_task_progress_outcome(
        test_settings,
        eval_run,
        config=EvaluationTaskProgressConfig(
            groups=["dense_only", "rrf"],
            query_rewrite_by_group={"dense_only": False, "rrf": False},
        ),
        outcome=TaskProgressOutcome(
            finished_at="2026-09-05T00:11:00+00:00",
            last_progress_at="2026-09-05T00:11:00+00:00",
            children=[
                ChildEvidence(
                    child_id="dense_only:g001",
                    child_type="eval_item",
                    label="dense_only/g001",
                    status="failed",
                    failure_kind="unknown",
                    degraded=True,
                ),
                ChildEvidence(
                    child_id="rrf:g001",
                    child_type="eval_item",
                    label="rrf/g001",
                    status="succeeded",
                    failure_kind="none",
                ),
            ],
        ),
    )
    eval_detail = get_task_progress_detail(test_settings, "evaluation:partial-demo")
    assert eval_detail is not None
    assert eval_detail.summary.status == "partial_success"
    assert eval_detail.summary.counters.total == 2
    assert eval_detail.summary.counters.completed == 1
    assert eval_detail.summary.counters.failed == 1
    assert eval_detail.summary.counters.partial == 1
    assert eval_detail.config_snapshot["query_rewrite_by_group"]["rrf"] is False


def test_ingest_writes_task_progress(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_markdown(tmp_path / "doc.md")
    client = TestClient(create_app(test_settings))

    ingest = client.post("/ingest", json={"source_path": str(source), "pdf_load_mode": "ocr"})
    assert ingest.status_code == 200
    trace_id = ingest.json()["trace_id"]

    summaries = list_task_progress_summaries(test_settings, task_type="ingestion")
    assert len(summaries) == 1
    assert summaries[0].task_id == f"ingestion:{trace_id}"
    assert summaries[0].links["trace_id"] == trace_id

    detail = get_task_progress_detail(test_settings, f"ingestion:{trace_id}")
    assert detail is not None
    assert detail.children[0].trace_id == trace_id
    assert detail.children[0].detail["ingest_status"] in {
        "ingested",
        "rebuilt",
        "rejected",
        "skipped",
    }


def test_rejected_ingest_maps_to_blocked_task_progress(
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    test_settings.quality_gate.reject_below = 1.1
    source = _write_markdown(tmp_path / "bad.md", body="12345\n67890\n")
    client = TestClient(create_app(test_settings))

    ingest = client.post("/ingest", json={"source_path": str(source), "pdf_load_mode": "ocr"})
    assert ingest.status_code == 200
    assert ingest.json()["status"] == "rejected"

    summaries = list_task_progress_summaries(test_settings, task_type="ingestion")
    assert summaries[0].status == "blocked"
    assert summaries[0].failure_kind == "input"
    detail = get_task_progress_detail(test_settings, summaries[0].task_id)
    assert detail is not None
    assert detail.children[0].status == "blocked"
    assert detail.children[0].failure_kind == "input"
    assert detail.children[0].detail["ingest_status"] == "rejected"
    assert detail.config_snapshot["pdf_load_mode"] == "ocr"


def test_failed_commit_keeps_document_link_in_task_progress(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _write_markdown(tmp_path / "doc.md")

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("upsert failed")

    monkeypatch.setattr("wenmai.knowledge.store.Knowledge.commit_document", boom)

    with pytest.raises(RuntimeError, match="upsert failed"):
        run_prepare_commit(source, test_settings)

    summaries = list_task_progress_summaries(test_settings, task_type="ingestion")
    assert len(summaries) == 1
    detail = get_task_progress_detail(
        test_settings,
        summaries[0].task_id,
    )
    assert detail is not None
    assert detail.summary.status == "failed"
    assert detail.summary.links["document_id"]


def test_eval_writes_task_progress_without_query_trace_pollution(
    test_settings: Settings,
    tmp_path: Path,
    without_ragas_judge_key: None,
) -> None:
    _prepare_eval(test_settings, tmp_path)
    before = read_trace_records(test_settings)
    query_before = [record for record in before if record.get("trace_type") == "query"]

    run = run_eval(test_settings)

    after = read_trace_records(test_settings)
    query_after = [record for record in after if record.get("trace_type") == "query"]
    assert query_after == query_before

    summaries = list_task_progress_summaries(test_settings, task_type="evaluation")
    assert summaries
    assert summaries[0].task_id == f"evaluation:{run.timestamp}"
    assert summaries[0].links["eval_run"] == run.timestamp

    detail = get_task_progress_detail(test_settings, f"evaluation:{run.timestamp}")
    assert detail is not None
    assert detail.summary.counters.total == run.item_count * len(run.groups)
    assert any(child.child_type == "eval_item" for child in detail.children)
    assert any(stage.links.get("group") for stage in detail.stages)

    artifact_path = Path(test_settings.evaluation.runs) / f"{run.timestamp}.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["timestamp"] == run.timestamp

    client = TestClient(create_app(test_settings))
    investigation = client.get(f"/api/tasks/progress/evaluation:{run.timestamp}/investigation")
    assert investigation.status_code == 200
    payload = investigation.json()
    assert payload["eval_run"]["timestamp"] == run.timestamp
    link_types = {(item["link_type"], item["target_id"]) for item in payload["links"]}
    assert ("eval_run", run.timestamp) in link_types
    assert ("document", "matsu-intro") in link_types
    assert ("eval_item", "g001") in link_types

    health = client.get("/api/stats/health", params={"task_type": "evaluation"})
    assert health.status_code == 200
    signals = {item["name"]: item["count"] for item in health.json()["signals"]}
    assert signals["attention_needed"] == 0


def test_task_progress_write_failure_does_not_break_eval_or_ingest(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch,
    without_ragas_judge_key: None,
) -> None:
    from wenmai import task_progress as task_progress_module

    source = _write_markdown(tmp_path / "doc.md")
    client = TestClient(create_app(test_settings))

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(task_progress_module, "write_task_progress", fail_write)

    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    _prepare_eval(test_settings, tmp_path / "eval")
    run = run_eval(test_settings)
    assert run.item_count == 2


def test_persist_task_progress_stamps_collection_id(test_settings: Settings) -> None:
    persist_task_progress(
        test_settings,
        task_id="evaluation:stamp",
        task_type="evaluation",
        status="succeeded",
        started_at="2026-09-05T00:00:00+00:00",
        finished_at="2026-09-05T00:00:01+00:00",
        last_progress_at="2026-09-05T00:00:01+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf"]},
        counters=TaskCounters(total=1, completed=1),
    )
    records = read_task_progress_records(test_settings)
    assert records[-1]["collection_id"] == test_settings.product.collection


def test_task_progress_collection_filter_and_legacy_visibility(
    test_settings: Settings,
) -> None:
    other_id = "other-collection"
    register_collection(test_settings, other_id)
    other_settings = _other_collection_settings(test_settings, other_id)

    # Legacy record: write directly without collection_id field.
    legacy_path = task_progress_path(test_settings)
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "task_id": "ingestion:legacy",
                "task_type": "ingestion",
                "status": "succeeded",
                "started_at": "2026-06-01T12:00:00+00:00",
                "finished_at": "2026-06-01T12:00:01+00:00",
                "last_progress_at": "2026-06-01T12:00:01+00:00",
                "trigger_source": "ingest_api",
                "owner_surface": "ops",
                "config_fingerprint": "legacy",
                "config_snapshot": {},
                "counters": TaskCounters(total=1, completed=1).as_dict(),
                "failure_kind": "none",
                "degraded": False,
                "links": {},
                "stages": [],
                "children": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    persist_task_progress(
        other_settings,
        task_id="ingestion:scoped",
        task_type="ingestion",
        status="succeeded",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at="2026-06-01T12:00:01+00:00",
        last_progress_at="2026-06-01T12:00:01+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        config_snapshot={"mode": "full"},
        counters=TaskCounters(total=1, completed=1),
    )

    default_ids = {
        item.task_id
        for item in list_task_progress(
            test_settings, collection_id=test_settings.default_collection_id
        )
    }
    other_ids = {
        item.task_id
        for item in list_task_progress(test_settings, collection_id=other_id)
    }
    assert default_ids == {"ingestion:legacy"}
    assert other_ids == {"ingestion:scoped"}

    assert (
        get_task_progress(
            test_settings,
            "ingestion:legacy",
            collection_id=test_settings.default_collection_id,
        )
        is not None
    )
    assert (
        get_task_progress(
            test_settings,
            "ingestion:scoped",
            collection_id=test_settings.default_collection_id,
        )
        is None
    )
    assert get_task_progress(test_settings, "ingestion:legacy", collection_id=other_id) is None
    assert get_task_progress(test_settings, "ingestion:scoped", collection_id=other_id) is not None
