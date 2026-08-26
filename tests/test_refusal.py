"""Query seam: prompt-driven refusal with retrieved source listing."""

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


def test_ask_refuses_unanswerable_question_and_lists_retrieved_sources(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    response = client.post(
        "/ask", json={"question": "船政学堂是什么时候创办的？"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["answer"].startswith("拒答：")
    assert body["trace_id"]
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
    generation_stage = next(
        stage for stage in query_trace["stages"] if stage["name"] == "generation"
    )
    assert generation_stage["output_summary"] == "refusal"
