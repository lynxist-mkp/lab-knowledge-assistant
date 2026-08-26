"""Query seam: Cross-Encoder rerank with RRF fallback."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings


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


def test_rerank_stage_recorded_in_query_trace(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})
    assert response.status_code == 200
    body = response.json()

    trace_path = Path(test_settings.paths.traces)
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    query_trace = next(trace for trace in traces if trace["trace_id"] == body["trace_id"])
    stage_names = [stage["name"] for stage in query_trace["stages"]]
    assert stage_names == [
        "query_processing",
        "dense",
        "sparse",
        "fusion",
        "rerank",
        "generation",
    ]

    rerank_stage = next(stage for stage in query_trace["stages"] if stage["name"] == "rerank")
    assert rerank_stage["candidates"]
    assert rerank_stage.get("pre_rerank_candidates")
    assert len(rerank_stage["candidates"]) <= test_settings.retrieval.rerank_top


def test_rerank_failure_falls_back_to_rrf_and_query_succeeds(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["reranker"] = "error"
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"]

    trace_path = Path(test_settings.paths.traces)
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    query_trace = next(trace for trace in traces if trace["trace_id"] == body["trace_id"])
    rerank_stage = next(stage for stage in query_trace["stages"] if stage["name"] == "rerank")
    assert rerank_stage.get("error")
    assert "fallback" in rerank_stage["output_summary"].lower()
    assert rerank_stage["method"] == "rrf_fallback"
    assert rerank_stage["candidates"]
    assert rerank_stage["candidates"][0]["chunk_id"] == body["citations"][0]["chunk_id"]
