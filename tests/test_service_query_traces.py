"""Service layer for query trace summaries and dashboard APIs."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.services.query_traces import (
    get_fusion_candidates,
    get_rank_changes,
    is_refusal,
    is_rerank_fallback,
    list_stage_latencies,
    summarize_query_trace,
)


def _sample_query_trace(*, refused: bool = False, rerank_fallback: bool = False) -> dict:
    rerank_stage: dict = {
        "name": "rerank",
        "method": "cross_encoder",
        "provider": "fake",
        "elapsed_ms": 45.0,
        "candidates": [{"chunk_id": "doc:0002", "score": 0.91}],
        "pre_rerank_candidates": [
            {"chunk_id": "doc:0001", "score": 0.5},
            {"chunk_id": "doc:0002", "score": 0.4},
        ],
        "rank_changes": [{"chunk_id": "doc:0002", "from": 2, "to": 1}],
    }
    if rerank_fallback:
        rerank_stage["method"] = "rrf_fallback"
        rerank_stage["fallback_reason"] = "RuntimeError: reranker failed"
        rerank_stage["error"] = rerank_stage["fallback_reason"]

    generation_output = "refusal" if refused else "42 chars"

    return {
        "trace_id": "q-trace-1",
        "trace_type": "query",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "total_elapsed_ms": 320.5,
        "error": None,
        "metadata": {"question": "妈祖信仰的发源地在哪里？"},
        "stages": [
            {
                "name": "query_processing",
                "method": "normalize",
                "elapsed_ms": 2.0,
                "input_summary": "妈祖信仰的发源地在哪里？",
                "output_summary": "妈祖 信仰 发源地",
                "culture_domain": "妈祖",
            },
            {
                "name": "dense",
                "method": "vector_query",
                "elapsed_ms": 80.0,
                "candidates": [{"chunk_id": "doc:0001", "score": 0.88}],
            },
            {
                "name": "sparse",
                "method": "bm25",
                "elapsed_ms": 12.0,
                "candidates": [{"chunk_id": "doc:0002", "score": 3.5}],
            },
            {
                "name": "fusion",
                "method": "rrf",
                "elapsed_ms": 1.0,
                "dense_candidates": [{"chunk_id": "doc:0001", "score": 0.88}],
                "sparse_candidates": [{"chunk_id": "doc:0002", "score": 3.5}],
                "candidates": [
                    {"chunk_id": "doc:0001", "score": 0.016},
                    {"chunk_id": "doc:0002", "score": 0.015},
                ],
            },
            rerank_stage,
            {
                "name": "generation",
                "method": "llm",
                "elapsed_ms": 180.0,
                "output_summary": generation_output,
            },
        ],
    }


def test_summarize_query_trace_extracts_question_and_flags() -> None:
    trace = _sample_query_trace()
    summary = summarize_query_trace(trace)
    assert summary.trace_id == "q-trace-1"
    assert summary.question == "妈祖信仰的发源地在哪里？"
    assert summary.culture_domain == "妈祖"
    assert summary.status == "ok"
    assert summary.refused is False
    assert summary.rerank_fallback is False
    assert summary.total_elapsed_ms == 320.5


def test_summarize_query_trace_marks_refusal_and_fallback() -> None:
    trace = _sample_query_trace(refused=True, rerank_fallback=True)
    summary = summarize_query_trace(trace)
    assert summary.status == "refused"
    assert summary.refused is True
    assert summary.rerank_fallback is True


def test_summarize_query_trace_marks_failed_on_error() -> None:
    trace = _sample_query_trace()
    trace["error"] = "RuntimeError: llm down"
    summary = summarize_query_trace(trace)
    assert summary.status == "failed"
    assert summary.error == "RuntimeError: llm down"


def test_list_stage_latencies_includes_all_query_stages() -> None:
    trace = _sample_query_trace()
    latencies = list_stage_latencies(trace)
    names = [item.name for item in latencies]
    assert names == [
        "query_processing",
        "dense",
        "sparse",
        "fusion",
        "rerank",
        "generation",
    ]
    assert latencies[1].elapsed_ms == 80.0
    assert latencies[1].label == "嵌入检索"


def test_get_fusion_candidates_and_rank_changes() -> None:
    trace = _sample_query_trace()
    fusion = get_fusion_candidates(trace)
    assert [row.chunk_id for row in fusion] == ["doc:0001", "doc:0002"]
    assert fusion[0].rank == 1

    changes = get_rank_changes(trace)
    assert len(changes) == 1
    assert changes[0].chunk_id == "doc:0002"
    assert changes[0].from_rank == 2
    assert changes[0].to_rank == 1


def test_refusal_and_fallback_helpers() -> None:
    trace = _sample_query_trace(refused=True, rerank_fallback=True)
    assert is_refusal(trace) is True
    assert is_rerank_fallback(trace) is True


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


def test_api_query_traces_listing(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    ask = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})
    assert ask.status_code == 200
    body = ask.json()
    trace_id = body["trace_id"]

    list_resp = client.get("/api/traces/query")
    assert list_resp.status_code == 200
    summaries = list_resp.json()
    assert summaries[0]["trace_id"] == trace_id
    assert summaries[0]["question"] == "妈祖信仰的发源地在哪里？"
    assert summaries[0]["refused"] is body["refused"]
    expected_status = "refused" if body["refused"] else "ok"
    assert summaries[0]["status"] == expected_status
