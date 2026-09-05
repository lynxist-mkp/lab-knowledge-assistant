"""Trace module: ingestion summaries, details, and HTTP."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.ops.observation import list_ingestion_summaries, list_trace_degradations
from wenmai.tracing import TraceContext, save_trace


def _write_markdown(path: Path, body: str = "闽派文化材料。") -> Path:
    path.write_text(
        f"""---
title: 测试文档
---

{body}
""",
        encoding="utf-8",
    )
    return path


def test_ingestion_summary_uses_metadata(test_settings: Settings) -> None:
    trace = TraceContext(trace_type="ingestion")
    trace.metadata.update(
        {
            "source_path": "/tmp/doc.md",
            "document_id": "abc123",
            "status": "ingested",
            "chunk_count": 3,
            "chunks_with_images": 1,
        }
    )
    trace.record_stage(
        "load",
        method="markdown",
        provider="local",
        elapsed_ms=5.0,
        input_summary="/tmp/doc.md",
        output_summary="测试文档",
    )
    save_trace(test_settings, trace)
    summary = list_ingestion_summaries(test_settings)[0]
    assert summary.chunk_count == 3
    assert summary.image_chunk_count == 1
    assert summary.source_path == "/tmp/doc.md"
    assert summary.document_id == "abc123"
    assert summary.status == "ingested"
    assert summary.skipped is False


def test_ingestion_summary_skipped(test_settings: Settings) -> None:
    trace = TraceContext(trace_type="ingestion")
    trace.metadata.update(
        {
            "source_path": "/tmp/doc.md",
            "document_id": "abc123",
            "status": "skipped",
            "chunk_count": 0,
            "chunks_with_images": 0,
        }
    )
    save_trace(test_settings, trace)
    summary = list_ingestion_summaries(test_settings)[0]
    assert summary.status == "skipped"
    assert summary.skipped is True
    assert summary.chunk_count == 0


def test_degradations_use_chinese_labels(test_settings: Settings) -> None:
    trace = TraceContext(trace_type="ingestion")
    trace.record_stage(
        "enricher",
        method="llm",
        provider="fake",
        elapsed_ms=1.0,
        error="ValueError: bad json",
    )
    trace.record_stage(
        "captioner",
        method="vision",
        provider="fake",
        elapsed_ms=1.0,
        output_summary="captioned 0 images; 2 kept placeholder",
    )
    trace.metadata["transform_discarded"] = [
        {"chunk_id": "doc:0001", "reason": "effective_char_ratio 0.10 below 0.20"}
    ]
    save_trace(test_settings, trace)
    degradations = list_trace_degradations(test_settings, trace.trace_id)
    assert degradations is not None
    stages = {item.stage for item in degradations}
    assert stages == {"补元数据", "图转文", "清洗"}


def test_api_ingestion_traces_and_detail_after_ingest(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_markdown(tmp_path / "doc.md")
    client = TestClient(create_app(test_settings))

    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200
    trace_id = ingest.json()["trace_id"]

    list_resp = client.get("/api/traces/ingestion")
    assert list_resp.status_code == 200
    summaries = list_resp.json()
    assert summaries[0]["trace_id"] == trace_id
    assert summaries[0]["trace_type"] == "ingestion"
    assert summaries[0]["chunk_count"] >= 1

    detail = client.get(f"/api/traces/{trace_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["trace_type"] == "ingestion"
    assert "stages" not in body
    assert any(step["label"] == "读取" for step in body["steps"])

    summary = client.get(f"/api/traces/{trace_id}/summary")
    assert summary.status_code == 200
    assert summary.json()["status"] == "ingested"
    assert summary.json()["trace_type"] == "ingestion"


def test_api_ingestion_run_streams_stage_events(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_markdown(tmp_path / "doc.md")
    client = TestClient(create_app(test_settings))

    with client.stream(
        "POST",
        "/api/ingestion/run",
        json={"source_path": str(source)},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
        assert "event" in body
        assert '"stage"' in body or '"done"' in body

    events = []
    for block in body.split("\n\n"):
        line = block.strip()
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[6:]))
    stage_names = [
        event["stage"]["name"] for event in events if event.get("event") == "stage"
    ]
    assert "load" in stage_names
    assert "integrity" in stage_names
    assert any(event.get("event") == "done" for event in events)
