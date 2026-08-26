"""Service layer for ingestion trace summaries and dashboard APIs."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.services.ingestion_traces import list_degradations, summarize_ingestion_trace


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


def test_summarize_ingestion_trace_uses_metadata_counts() -> None:
    trace = {
        "trace_id": "t1",
        "trace_type": "ingestion",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "total_elapsed_ms": 1200.0,
        "error": None,
        "metadata": {
            "source_path": "/tmp/doc.md",
            "document_id": "abc123",
            "status": "ingested",
            "chunk_count": 3,
            "chunks_with_images": 1,
        },
        "stages": [
            {"name": "load", "input_summary": "/tmp/doc.md", "output_summary": "测试文档"},
            {"name": "integrity", "output_summary": "new file"},
        ],
    }
    summary = summarize_ingestion_trace(trace)
    assert summary.chunk_count == 3
    assert summary.image_chunk_count == 1
    assert summary.source_path == "/tmp/doc.md"
    assert summary.document_id == "abc123"
    assert summary.status == "ingested"
    assert summary.skipped is False


def test_summarize_skipped_trace() -> None:
    trace = {
        "trace_id": "t2",
        "trace_type": "ingestion",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:00+00:00",
        "total_elapsed_ms": 10.0,
        "error": None,
        "metadata": {
            "source_path": "/tmp/doc.md",
            "document_id": "abc123",
            "status": "skipped",
            "chunk_count": 0,
            "chunks_with_images": 0,
        },
        "stages": [
            {"name": "load", "input_summary": "/tmp/doc.md"},
            {"name": "integrity", "output_summary": "skipped: unchanged sha256"},
        ],
    }
    summary = summarize_ingestion_trace(trace)
    assert summary.status == "skipped"
    assert summary.skipped is True
    assert summary.chunk_count == 0


def test_list_degradations_detects_enricher_captioner_and_refiner() -> None:
    trace = {
        "stages": [
            {"name": "enricher", "error": "ValueError: bad json"},
            {"name": "captioner", "output_summary": "captioned 0 images; 2 kept placeholder"},
            {"name": "transform", "output_summary": "1 kept, 1 discarded"},
        ],
        "metadata": {
            "transform_discarded": [
                {"chunk_id": "doc:0001", "reason": "effective_char_ratio 0.10 below 0.20"}
            ]
        },
    }
    degradations = list_degradations(trace)
    stages = {item.stage for item in degradations}
    assert "enricher" in stages
    assert "captioner" in stages
    assert "refiner" in stages


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
    assert summaries[0]["chunk_count"] >= 1

    detail = client.get(f"/api/traces/{trace_id}")
    assert detail.status_code == 200
    assert detail.json()["trace_type"] == "ingestion"

    summary = client.get(f"/api/traces/{trace_id}/summary")
    assert summary.status_code == 200
    assert summary.json()["status"] == "ingested"


def test_ingestion_pages_render(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_markdown(tmp_path / "doc.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    trace_id = ingest.json()["trace_id"]

    manage = client.get("/ingestion")
    assert manage.status_code == 200
    assert "Ingestion 管理" in manage.text

    listing = client.get("/ingestion/traces")
    assert listing.status_code == 200
    assert trace_id in listing.text

    detail_page = client.get(f"/ingestion/traces/{trace_id}")
    assert detail_page.status_code == 200
    assert "stages[]" in detail_page.text


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
