"""Ingest and eval HTTP entrypoints: collection_id query routing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import register_collection

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.eval import list_eval_run_summaries, run_eval
from wenmai.eval.persist import persist_eval_artifact, stamp_eval_run_collection_id
from wenmai.eval.views import GroupMetricsView, parse_eval_run_summary
from wenmai.knowledge import create_knowledge


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    registered = register_collection(test_settings, other_id)
    return replace(
        registered,
        product=replace(test_settings.product, collection=other_id),
    )


def _write_minpai_markdown(path: Path) -> Path:
    path.write_text(
        """---
source_url: https://www.mzmz.org.cn/introduction.html
culture_domain: 妈祖
title: 湄洲妈祖祖庙简介
---

湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上，是信俗活动的中心场所。
""",
        encoding="utf-8",
    )
    return path


def _sample_eval_artifact(timestamp: str) -> dict[str, object]:
    metrics = GroupMetricsView(
        hit_at_5=0.5,
        mrr=0.4,
        refusal_accuracy=0.9,
        citation_coverage=0.8,
        answerable_count=1,
        unanswerable_count=0,
    ).as_dict()
    return {
        "timestamp": timestamp,
        "golden_set": "data/eval/golden.jsonl",
        "ablations": ["dense_only"],
        "item_count": 1,
        "failures": [],
        "groups": {
            "dense_only": {
                "config": {"ablation_group": "dense_only"},
                "metrics": metrics,
            }
        },
    }


def test_ingest_explicit_collection_id_writes_target_collection(
    test_settings: Settings, tmp_path: Path
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    source = _write_minpai_markdown(tmp_path / "scoped-ingest.md")
    client = TestClient(create_app(settings))

    response = client.post(
        "/ingest",
        json={"source_path": str(source)},
        params={"collection_id": other_id},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"

    default_knowledge = create_knowledge(test_settings)
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    assert default_knowledge.get_by_document_id(body["document_id"]) == []
    assert len(other_knowledge.get_by_document_id(body["document_id"])) >= 1


def test_api_ingestion_run_explicit_collection_id_writes_target_collection(
    test_settings: Settings, tmp_path: Path
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    source = _write_minpai_markdown(tmp_path / "scoped-sse-ingest.md")
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/ingestion/run",
        json={"source_path": str(source)},
        params={"collection_id": other_id},
    )
    assert response.status_code == 200

    done_event = None
    for line in response.iter_lines():
        if line.startswith("data: "):
            payload = json.loads(line[len("data: ") :])
            if payload.get("event") == "done":
                done_event = payload
                break
    assert done_event is not None
    result = done_event["result"]
    assert result["status"] == "ingested"

    default_knowledge = create_knowledge(test_settings)
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
    assert default_knowledge.get_by_document_id(result["document_id"]) == []
    assert len(other_knowledge.get_by_document_id(result["document_id"])) >= 1


def test_ingest_unknown_collection_returns_not_found(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "missing-collection.md")
    client = TestClient(create_app(test_settings))

    response = client.post(
        "/ingest",
        json={"source_path": str(source)},
        params={"collection_id": "missing-collection"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "collection not found"


def test_ingest_missing_source_still_returns_source_not_found(
    test_settings: Settings,
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    client = TestClient(create_app(settings))

    response = client.post(
        "/ingest",
        json={"source_path": "/tmp/does-not-exist.md"},
        params={"collection_id": other_id},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "source file not found"


def test_eval_runs_list_filters_by_collection_id(
    test_settings: Settings, tmp_path: Path
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    runs_dir = tmp_path / "runs"
    settings.evaluation.runs = str(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    legacy_artifact = _sample_eval_artifact("20260101T100000Z")
    scoped_artifact = stamp_eval_run_collection_id(
        _sample_eval_artifact("20260102T100000Z"),
        _other_collection_settings(settings, other_id),
    )
    (runs_dir / "20260101T100000Z.json").write_text(
        json.dumps(legacy_artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (runs_dir / "20260102T100000Z.json").write_text(
        json.dumps(scoped_artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    default_runs = list_eval_run_summaries(settings)
    other_runs = list_eval_run_summaries(settings, collection_id=other_id)
    default_scoped_runs = list_eval_run_summaries(
        settings, collection_id=settings.default_collection_id
    )

    assert [run.timestamp for run in default_runs] == [
        "20260102T100000Z",
        "20260101T100000Z",
    ]
    assert [run.timestamp for run in other_runs] == ["20260102T100000Z"]
    assert [run.timestamp for run in default_scoped_runs] == ["20260101T100000Z"]

    client = TestClient(create_app(settings))
    default_response = client.get("/api/eval/runs")
    other_response = client.get("/api/eval/runs", params={"collection_id": other_id})

    assert default_response.status_code == 200
    assert other_response.status_code == 200
    assert [item["timestamp"] for item in default_response.json()] == [
        "20260101T100000Z",
    ]
    assert [item["timestamp"] for item in other_response.json()] == [
        "20260102T100000Z",
    ]


def test_persist_eval_artifact_stamps_collection_id(
    test_settings: Settings, tmp_path: Path
) -> None:
    artifact = _sample_eval_artifact("20260103T100000Z")
    scoped_settings = _other_collection_settings(test_settings, "other-collection")
    scoped_settings.evaluation.runs = str(tmp_path / "runs")

    run = persist_eval_artifact(scoped_settings, artifact)

    assert run.timestamp == "20260103T100000Z"
    raw = json.loads(
        (tmp_path / "runs" / "20260103T100000Z.json").read_text(encoding="utf-8")
    )
    assert raw["collection_id"] == "other-collection"


def test_post_eval_runs_uses_scoped_settings_and_knowledge(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    settings.evaluation.runs = str(tmp_path / "runs")
    client = TestClient(create_app(settings))

    captured: dict[str, object] = {}

    def fake_run_eval(
        scoped_settings: Settings,
        *,
        knowledge=None,
        query_rewrite: bool = False,
        ragas: bool | None = None,
        item_limit: int | None = None,
        groups: list[str] | None = None,
    ):
        captured["settings"] = scoped_settings
        captured["knowledge"] = knowledge
        artifact = stamp_eval_run_collection_id(
            _sample_eval_artifact("20260104T100000Z"),
            scoped_settings,
        )
        return parse_eval_run_summary(artifact)

    monkeypatch.setattr("wenmai.http.eval.run_eval", fake_run_eval)

    response = client.post("/api/eval/runs", params={"collection_id": other_id})

    assert response.status_code == 200
    assert captured["settings"].product.collection == other_id
    assert captured["knowledge"] is not None
    assert captured["knowledge"] is not client.app.state.knowledge


def test_post_eval_runs_unknown_collection_returns_not_found(
    test_settings: Settings,
) -> None:
    client = TestClient(create_app(test_settings))

    response = client.post(
        "/api/eval/runs",
        params={"collection_id": "missing-collection"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "collection not found"


def test_run_eval_stamps_collection_id_on_persist(
    test_settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_id = "other-collection"
    scoped_settings = _other_collection_settings(
        register_collection(test_settings, other_id),
        other_id,
    )
    scoped_settings.evaluation.runs = str(tmp_path / "runs")
    scoped_settings.evaluation.ablations = ["dense_only"]
    scoped_settings.evaluation.golden_set = str(
        _write_golden_jsonl(tmp_path / "golden.jsonl")
    )

    def mock_grouped(items, settings, groups, knowledge, **kwargs):
        return (
            {name: {} for name in groups},
            {name: {} for name in groups},
            [],
        )

    monkeypatch.setattr(
        "wenmai.eval.runner._run_grouped_eval",
        mock_grouped,
    )

    run = run_eval(scoped_settings)

    raw = json.loads(
        (tmp_path / "runs" / f"{run.timestamp}.json").read_text(encoding="utf-8")
    )
    assert raw["collection_id"] == other_id


def _write_golden_jsonl(path: Path) -> Path:
    path.write_text(
        '{"id":"g001","question":"妈祖信仰的发源地在哪里？","evidence_doc_ids":["doc-a"],'
        '"answerable":true,"reference_answer":"湄洲岛。","category":"单跳事实"}\n',
        encoding="utf-8",
    )
    return path
