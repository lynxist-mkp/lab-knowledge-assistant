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
from wenmai.ops.observation import load_overview_stats
from tests.conftest import register_collection
from wenmai.tracing.latency import query_latency_percentiles


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
    _append_query_trace_with_stages(
        path,
        trace_id=trace_id,
        total_elapsed_ms=elapsed_ms,
        stages=[],
    )


def _append_query_trace_with_stages(
    path: Path,
    *,
    trace_id: str,
    total_elapsed_ms: float,
    stages: list[dict[str, object]],
    started_at: str = "2026-01-01T00:00:00+00:00",
) -> None:
    trace = {
        "trace_id": trace_id,
        "trace_type": "query",
        "started_at": started_at,
        "finished_at": "2026-01-01T00:00:01+00:00",
        "total_elapsed_ms": total_elapsed_ms,
        "stages": stages,
        "error": None,
        "metadata": {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace, ensure_ascii=False) + "\n")


def test_overview_stats_empty_store(test_settings: Settings) -> None:
    stats = load_overview_stats(test_settings)

    assert stats.document_count == 0
    assert stats.chunk_count == 0
    assert stats.avg_query_latency_ms is None
    assert stats.query_latency_p50_ms is None
    assert stats.query_latency_p95_ms is None


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

    stats = load_overview_stats(test_settings)

    assert stats.document_count == 1
    assert stats.chunk_count == response.json()["chunk_count"]
    assert stats.avg_query_latency_ms is None
    assert stats.query_latency_p50_ms is None
    assert stats.query_latency_p95_ms is None


def test_overview_query_latency_percentiles_from_traces(test_settings: Settings) -> None:
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    generation_stage = {
        "name": "generation",
        "method": "llm",
        "provider": "fake",
        "elapsed_ms": 0.0,
    }
    for trace_id, generation_ms, total_ms in [
        ("q1", 100.0, 150.0),
        ("q2", 300.0, 350.0),
    ]:
        _append_query_trace_with_stages(
            trace_path,
            trace_id=trace_id,
            total_elapsed_ms=total_ms,
            stages=[{**generation_stage, "elapsed_ms": generation_ms}],
        )

    stats = load_overview_stats(test_settings)
    latency = query_latency_percentiles(test_settings)

    assert stats.query_latency_p50_ms == pytest.approx(150.0)
    assert stats.query_latency_p95_ms == pytest.approx(350.0)
    assert latency["generation"]["p50"] == pytest.approx(100.0)
    assert latency["generation"]["p95"] == pytest.approx(300.0)
    assert latency["total"]["p50"] == pytest.approx(150.0)
    assert latency["total"]["p95"] == pytest.approx(350.0)


def test_query_latency_percentiles_use_recent_n_only(test_settings: Settings) -> None:
    trace_path = Path(test_settings.paths.traces)
    for index in range(5):
        _append_query_trace_with_stages(
            trace_path,
            trace_id=f"old-{index}",
            total_elapsed_ms=10.0,
            stages=[{"name": "generation", "elapsed_ms": 10.0}],
            started_at=f"2026-01-01T00:00:0{index}+00:00",
        )
    _append_query_trace_with_stages(
        trace_path,
        trace_id="recent-slow",
        total_elapsed_ms=300.0,
        stages=[{"name": "generation", "elapsed_ms": 300.0}],
        started_at="2026-01-02T00:00:00+00:00",
    )

    all_latency = query_latency_percentiles(test_settings, recent_n=None)
    recent_latency = query_latency_percentiles(test_settings, recent_n=1)

    assert all_latency["total"]["p50"] == pytest.approx(10.0)
    assert recent_latency["total"]["p50"] == pytest.approx(300.0)


def test_query_latency_percentiles_empty_traces(test_settings: Settings) -> None:
    latency = query_latency_percentiles(test_settings)

    assert latency == {"total": {"p50": None, "p95": None}}


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    from dataclasses import replace

    registered = register_collection(test_settings, other_id)
    return replace(
        registered,
        product=replace(test_settings.product, collection=other_id),
    )


def test_overview_latency_isolated_by_collection(test_settings: Settings) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    create_knowledge(_other_collection_settings(test_settings, other_id)).commit_document(
        source_path="/tmp/latency-other.md",
        sha256="latency-other",
        document_id="latency-other",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="latency-other:0000",
                document_id="latency-other",
                text="其他集合延迟",
                metadata={
                    "document_id": "latency-other",
                    "title": "latency-other",
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            )
        ],
    )

    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    for trace_id, elapsed_ms, collection_id in [
        ("default-fast", 100.0, None),
        ("default-slow", 300.0, None),
        ("other-only", 999.0, other_id),
    ]:
        payload: dict[str, object] = {
            "trace_id": trace_id,
            "trace_type": "query",
            "started_at": "2026-06-01T12:00:00+00:00",
            "finished_at": "2026-06-01T12:00:01+00:00",
            "total_elapsed_ms": elapsed_ms,
            "stages": [],
            "error": None,
            "metadata": {},
        }
        if collection_id is not None:
            payload["collection_id"] = collection_id
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    default_stats = load_overview_stats(settings)
    other_stats = load_overview_stats(settings, collection_id=other_id)

    assert default_stats.avg_query_latency_ms == pytest.approx(200.0)
    assert default_stats.query_latency_p50_ms == pytest.approx(100.0)
    assert default_stats.query_latency_p95_ms == pytest.approx(300.0)
    assert other_stats.avg_query_latency_ms == pytest.approx(999.0)
    assert other_stats.query_latency_p50_ms == pytest.approx(999.0)
    assert other_stats.query_latency_p95_ms == pytest.approx(999.0)


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

    stats = load_overview_stats(test_settings)

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
                metadata={
                    "title": "妈祖简介",
                    "culture_domain": "妈祖",
                    "document_id": "doc-matsu",
                },
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
                metadata={
                    "title": "朱子理学",
                    "culture_domain": "朱子",
                    "document_id": "doc-zhuzi",
                },
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
