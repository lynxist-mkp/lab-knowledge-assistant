"""Ingestion seam: one markdown in, chunks retrievable, trace has pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.factories import vector_store as vector_store_factory


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


def _ingest_trace(test_settings: Settings, source: Path) -> tuple[dict, dict]:
    client = TestClient(create_app(test_settings))
    response = client.post("/ingest", json={"source_path": str(source)})
    assert response.status_code == 200
    body = response.json()

    trace_path = Path(test_settings.paths.traces)
    lines = [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    trace = json.loads(lines[-1])
    return body, trace


def test_ingesting_markdown_makes_chunks_retrievable_and_records_pipeline_stages(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))

    response = client.post("/ingest", json={"source_path": str(source)})

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] >= 1
    assert body["document_id"]
    assert body["elapsed_ms"] >= 0
    assert body["trace_id"]

    store = vector_store_factory.create(test_settings)
    chunks = store.get_by_document_id(body["document_id"])
    assert len(chunks) == body["chunk_count"]
    first = chunks[0]
    assert "妈祖" in first.text
    assert first.metadata["document_id"] == body["document_id"]
    assert first.metadata["title"] == "湄洲妈祖祖庙简介"
    assert first.metadata["url"] == "https://www.mzmz.org.cn/introduction.html"
    assert first.metadata["page"] == 1

    trace_path = Path(test_settings.paths.traces)
    raw_lines = trace_path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in raw_lines if line]
    assert len(lines) == 1
    trace = json.loads(lines[0])
    assert trace["trace_id"] == body["trace_id"]
    assert trace["trace_type"] == "ingestion"
    assert trace["started_at"]
    assert trace["finished_at"]
    assert trace["total_elapsed_ms"] >= 0
    assert "error" in trace
    assert [stage["name"] for stage in trace["stages"]] == [
        "load",
        "split",
        "transform",
        "enricher",
        "embed",
        "upsert",
    ]
    for stage in trace["stages"]:
        assert "method" in stage
        assert "provider" in stage
        assert "elapsed_ms" in stage
        assert stage["input_summary"]
        assert stage["output_summary"]


def _write_dirty_markdown(path: Path) -> Path:
    good_body = "湄洲岛是妈祖信仰的发源地，祖庙是信俗活动的中心场所。" * 30
    garbage = "@#" * 40
    path.write_text(
        f"""---
title: 带噪声的妈祖材料
---

第 1 页

{good_body}

{garbage}

版权所有 2024 湄洲妈祖祖庙
""",
        encoding="utf-8",
    )
    return path


def test_refiner_cleans_headers_footers_and_drops_garbage_chunks(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_dirty_markdown(tmp_path / "dirty_matsu.md")
    client = TestClient(create_app(test_settings))

    response = client.post("/ingest", json={"source_path": str(source)})

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] >= 1

    store = vector_store_factory.create(test_settings)
    chunks = store.get_by_document_id(body["document_id"])
    assert len(chunks) == body["chunk_count"]
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "妈祖" in combined
    assert "第 1 页" not in combined
    assert "版权所有" not in combined
    assert "@#" not in combined

    trace_path = Path(test_settings.paths.traces)
    trace = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    stage_names = [stage["name"] for stage in trace["stages"]]
    assert "transform" in stage_names
    transform_stage = next(stage for stage in trace["stages"] if stage["name"] == "transform")
    assert transform_stage["method"] == "rule"
    assert transform_stage["provider"] == "refiner"
    assert "discarded" in transform_stage["output_summary"].lower()

    discarded = trace["metadata"].get("transform_discarded", [])
    assert discarded
    assert any("effective_char_ratio" in item["reason"] for item in discarded)


def test_enricher_garbage_llm_still_ingests_with_empty_enrichment_and_trace_note(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["llm"] = "garbage"
    source = _write_minpai_markdown(tmp_path / "matsu.md")

    body, trace = _ingest_trace(test_settings, source)

    store = vector_store_factory.create(test_settings)
    chunks = store.get_by_document_id(body["document_id"])
    assert len(chunks) >= 1
    first = chunks[0]
    assert "chunk_title" not in first.metadata
    assert "summary" not in first.metadata
    assert "tags" not in first.metadata

    enricher_stage = next(stage for stage in trace["stages"] if stage["name"] == "enricher")
    assert enricher_stage["output_summary"]
    assert enricher_stage.get("error")


def test_enricher_writes_culture_domain_to_chunk_metadata(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")

    body, _trace = _ingest_trace(test_settings, source)

    store = vector_store_factory.create(test_settings)
    chunks = store.get_by_document_id(body["document_id"])
    assert chunks[0].metadata["culture_domain"] == "妈祖"
    assert chunks[0].metadata["chunk_title"] == "妈祖祖庙"
