"""Query seam: ask with citations and trace stages."""

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
source_org: 湄洲妈祖祖庙
license_note: 政府网站公开信息，引用时保留 URL
culture_domain: 妈祖
space: minpai_culture
title: 湄洲妈祖祖庙简介
---

湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上，是信俗活动的中心场所。
每年农历三月二十三，信众会到祖庙参加祭典。
""",
        encoding="utf-8",
    )
    return path


def test_ask_returns_answer_with_matching_citations_and_query_trace(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})

    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"]
    assert "[1]" in body["answer"]
    assert body["citations"]
    first = body["citations"][0]
    assert first["index"] == 1
    assert first["chunk_id"]
    assert first["document_id"] == ingest.json()["document_id"]
    assert "妈祖" in first["excerpt"]

    trace_path = Path(test_settings.paths.traces)
    raw_lines = trace_path.read_text(encoding="utf-8").splitlines()
    traces = [json.loads(line) for line in raw_lines if line]
    query_trace = next(trace for trace in traces if trace["trace_id"] == body["trace_id"])
    assert query_trace["trace_type"] == "query"
    assert [stage["name"] for stage in query_trace["stages"]] == [
        "query_processing",
        "dense",
        "generation",
    ]
    dense_stage = next(stage for stage in query_trace["stages"] if stage["name"] == "dense")
    assert dense_stage["candidates"]
    assert dense_stage["candidates"][0]["chunk_id"] == first["chunk_id"]
    assert "score" in dense_stage["candidates"][0]


def test_ask_records_generation_failure_in_trace_and_returns_error(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["llm"] = "error"
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["trace_id"]
    assert "error" in detail["message"].lower() or "fake" in detail["message"].lower()

    trace_path = Path(test_settings.paths.traces)
    raw_lines = trace_path.read_text(encoding="utf-8").splitlines()
    traces = [json.loads(line) for line in raw_lines if line]
    query_trace = next(trace for trace in traces if trace["trace_id"] == detail["trace_id"])
    generation_stage = next(
        stage for stage in query_trace["stages"] if stage["name"] == "generation"
    )
    assert generation_stage.get("error")
    assert query_trace.get("error")
