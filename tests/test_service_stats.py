"""Service layer for dashboard: overview stats and culture-domain browse."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import Chunk
from wenmai.ops.overview import build_overview_stats
from wenmai.storage.catalog import DocumentCatalog
from wenmai.tracing.store import average_query_latency_ms


def _overview_stats(settings: Settings) -> object:
    catalog = DocumentCatalog.from_settings(settings)
    return build_overview_stats(
        settings,
        catalog,
        avg_query_latency_ms=average_query_latency_ms(settings),
    )


def _write_markdown(path: Path, *, culture_domain: str, title: str, body: str) -> Path:
    path.write_text(
        f"""---
source_url: https://example.com/{path.stem}
culture_domain: {culture_domain}
space: minpai_culture
title: {title}
---

{body}
""",
        encoding="utf-8",
    )
    return path


def _append_query_trace(path: Path, *, trace_id: str, elapsed_ms: float) -> None:
    trace = {
        "trace_id": trace_id,
        "trace_type": "query",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "total_elapsed_ms": elapsed_ms,
        "stages": [],
        "error": None,
        "metadata": {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")


def test_overview_stats_empty_store(test_settings: Settings) -> None:
    stats = _overview_stats(test_settings)

    assert stats.document_count == 0
    assert stats.chunk_count == 0
    assert stats.avg_query_latency_ms is None


def test_overview_stats_after_ingest(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_markdown(
        tmp_path / "matsu.md",
        culture_domain="妈祖",
        title="湄洲妈祖祖庙简介",
        body="湄洲岛是妈祖信仰的发源地。",
    )
    client = TestClient(create_app(test_settings))
    response = client.post("/ingest", json={"source_path": str(source)})
    assert response.status_code == 200

    stats = _overview_stats(test_settings)

    assert stats.document_count == 1
    assert stats.chunk_count == response.json()["chunk_count"]
    assert stats.avg_query_latency_ms is None


def test_overview_avg_query_latency_from_traces(test_settings: Settings) -> None:
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    for trace_id, elapsed_ms in [("q1", 100.0), ("q2", 300.0)]:
        _append_query_trace(trace_path, trace_id=trace_id, elapsed_ms=elapsed_ms)
    ingestion_trace = {
        "trace_id": "ing1",
        "trace_type": "ingestion",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "total_elapsed_ms": 999.0,
        "stages": [],
        "error": None,
        "metadata": {},
    }
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ingestion_trace, ensure_ascii=False) + "\n")

    stats = _overview_stats(test_settings)

    assert stats.avg_query_latency_ms == pytest.approx(200.0)


def test_browse_groups_documents_by_culture_domain(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    knowledge.commit_document(
        source_path="/tmp/doc-matsu.md",
        sha256="doc-matsu",
        document_id="doc-matsu",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="matsu-001",
                document_id="doc-matsu",
                text="妈祖信仰发源于湄洲岛。",
                metadata={"title": "妈祖简介", "culture_domain": "妈祖", "document_id": "doc-matsu"},
            ),
        ],
    )
    knowledge.commit_document(
        source_path="/tmp/doc-zhuzi.md",
        sha256="doc-zhuzi",
        document_id="doc-zhuzi",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="zhuzi-001",
                document_id="doc-zhuzi",
                text="朱熹是理学集大成者。",
                metadata={"title": "朱子理学", "culture_domain": "朱子", "document_id": "doc-zhuzi"},
            ),
        ],
    )

    groups = knowledge.browse_by_culture_domain()
    by_domain = {group.culture_domain: group for group in groups}

    assert set(by_domain) == {"妈祖", "朱子"}
    assert by_domain["妈祖"].document_count == 1
    assert by_domain["妈祖"].chunk_count == 1
    assert by_domain["妈祖"].documents[0].title == "妈祖简介"
    assert by_domain["朱子"].documents[0].title == "朱子理学"


def test_get_chunk_detail_returns_final_text(test_settings: Settings, tmp_path: Path) -> None:
    caption_text = "图中为湄洲妈祖祖庙大殿。"
    source = _write_markdown(
        tmp_path / "caption.md",
        culture_domain="妈祖",
        title="带图说明",
        body=f"正文段落。\n\n[IMAGE: img-001]\n\n{caption_text}",
    )
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    groups = create_knowledge(test_settings).browse_by_culture_domain()
    chunk_id = groups[0].documents[0].chunks[0].chunk_id
    detail = create_knowledge(test_settings).chunk_detail(chunk_id)

    assert detail is not None
    assert detail["chunk_id"] == chunk_id
    assert detail["culture_domain"] == "妈祖"
    assert "正文段落" in detail["text"]
    assert caption_text in detail["text"]


def test_api_overview_and_browse_endpoints(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_markdown(
        tmp_path / "matsu.md",
        culture_domain="妈祖",
        title="妈祖简介",
        body="妈祖信仰发源于湄洲岛。",
    )
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    overview = client.get("/api/stats/overview")
    assert overview.status_code == 200
    body = overview.json()
    assert body["document_count"] == 1
    assert body["chunk_count"] >= 1

    browse = client.get("/api/browse")
    assert browse.status_code == 200
    assert browse.json()[0]["culture_domain"] == "妈祖"
