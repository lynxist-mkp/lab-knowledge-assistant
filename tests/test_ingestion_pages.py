"""Ingestion management API and retired HTML paths."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings


def _write_markdown(path: Path) -> Path:
    path.write_text(
        """---
title: 湄洲妈祖祖庙简介
---

湄洲岛是妈祖信仰的发源地。
""",
        encoding="utf-8",
    )
    return path


def test_ingestion_trace_list_and_summary_after_ingest(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200
    trace_id = ingest.json()["trace_id"]

    listing = client.get("/api/traces/ingestion")
    assert listing.status_code == 200
    items = listing.json()
    assert len(items) == 1
    assert items[0]["trace_id"] == trace_id
    assert items[0]["status"] == "ingested"
    assert items[0]["chunk_count"] >= 1

    summary = client.get(f"/api/traces/{trace_id}/summary")
    assert summary.status_code == 200
    assert summary.json()["source_path"] == str(source)


def test_ingestion_trace_shows_skip_status(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    first = client.post("/ingest", json={"source_path": str(source)})
    second = client.post("/ingest", json={"source_path": str(source)})
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "skipped"

    listing = client.get("/api/traces/ingestion")
    skipped = next(item for item in listing.json() if item["status"] == "skipped")
    assert skipped["skipped"] is True


def test_ingestion_degradations_api_for_enricher_garbage(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["multimodal"] = "garbage"
    source = _write_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    trace_id = ingest.json()["trace_id"]

    degradations = client.get(f"/api/traces/{trace_id}/degradations")
    assert degradations.status_code == 200
    body = degradations.json()
    assert any(item["stage"] == "补元数据" for item in body)
