"""Trace module: query summaries, details, and HTTP."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lab_knowledge.app import create_app
from lab_knowledge.config import Settings
from lab_knowledge.ops.observation import get_query_detail, list_query_summaries
from lab_knowledge.tracing import TraceContext, save_trace


def _save_sample_query(
    settings: Settings,
    *,
    refused: bool = False,
    rerank_fallback: bool = False,
    error: str | None = None,
) -> str:
    trace = TraceContext(
        trace_type="query",
        metadata={"question": "妈祖信仰的发源地在哪里？"},
    )
    trace.record_stage(
        "query_processing",
        method="normalize",
        provider="local",
        elapsed_ms=2.0,
        input_summary="妈祖信仰的发源地在哪里？",
        output_summary="妈祖 信仰 发源地",
        culture_domain="妈祖",
    )
    trace.record_stage(
        "dense",
        method="vector_query",
        provider="fake",
        elapsed_ms=80.0,
        candidates=[{"chunk_id": "doc:0001", "score": 0.88}],
    )
    trace.record_stage(
        "sparse",
        method="bm25",
        provider="fake",
        elapsed_ms=12.0,
        candidates=[{"chunk_id": "doc:0002", "score": 3.5}],
    )
    trace.record_stage(
        "fusion",
        method="rrf",
        provider="local",
        elapsed_ms=1.0,
        candidates=[
            {"chunk_id": "doc:0001", "score": 0.016},
            {"chunk_id": "doc:0002", "score": 0.015},
        ],
    )
    rerank_method = "rrf_fallback" if rerank_fallback else "cross_encoder"
    trace.record_stage(
        "rerank",
        method=rerank_method,
        provider="fake",
        elapsed_ms=45.0,
        candidates=[{"chunk_id": "doc:0002", "score": 0.91}],
        rank_changes=[{"chunk_id": "doc:0002", "from": 2, "to": 1}],
        fallback_reason="RuntimeError: reranker failed" if rerank_fallback else None,
        error="RuntimeError: reranker failed" if rerank_fallback else None,
    )
    trace.record_stage(
        "generation",
        method="llm",
        provider="fake",
        elapsed_ms=180.0,
        output_summary="unrelated summary text",
    )
    if refused:
        trace.metadata["outcome"] = {
            "refused": True,
            "refusal_reason": "model_refused",
            "citation_count": 2,
        }
    else:
        trace.metadata["outcome"] = {
            "refused": False,
            "refusal_reason": None,
            "citation_count": 1,
        }
    if error:
        trace.error = error
    save_trace(settings, trace)
    return trace.trace_id


def test_query_summary_and_detail_round_trip(test_settings: Settings) -> None:
    trace_id = _save_sample_query(test_settings)
    summaries = list_query_summaries(test_settings)
    assert summaries[0].trace_id == trace_id
    assert summaries[0].question == "妈祖信仰的发源地在哪里？"
    assert summaries[0].culture_domain == "妈祖"
    assert summaries[0].status == "ok"
    assert summaries[0].refused is False
    assert summaries[0].rerank_fallback is False

    detail = get_query_detail(test_settings, trace_id)
    assert detail is not None
    labels = [item.label for item in detail.stage_latencies]
    assert labels == ["查询处理", "嵌入检索", "稀疏检索", "融合", "精排", "生成"]
    assert detail.stage_latencies[1].elapsed_ms == 80.0
    assert [row.chunk_id for row in detail.fusion_candidates] == ["doc:0001", "doc:0002"]
    assert detail.rank_changes[0].chunk_id == "doc:0002"
    assert detail.rank_changes[0].from_rank == 2
    assert detail.rank_changes[0].to_rank == 1


def test_query_summary_marks_refusal_and_fallback(test_settings: Settings) -> None:
    _save_sample_query(test_settings, refused=True, rerank_fallback=True)
    summary = list_query_summaries(test_settings)[0]
    assert summary.status == "refused"
    assert summary.refused is True
    assert summary.refusal_reason == "model_refused"
    assert summary.rerank_fallback is True


def test_query_summary_ignores_generation_output_summary_for_refusal(
    test_settings: Settings,
) -> None:
    trace = TraceContext(
        trace_type="query",
        metadata={
            "question": "测试问题",
            "outcome": {
                "refused": True,
                "refusal_reason": "insufficient_evidence",
                "citation_count": 0,
            },
        },
    )
    trace.record_stage(
        "generation",
        method="llm",
        provider="fake",
        elapsed_ms=1.0,
        output_summary="42 chars",
    )
    save_trace(test_settings, trace)

    summary = list_query_summaries(test_settings)[0]
    assert summary.refused is True
    assert summary.refusal_reason == "insufficient_evidence"


def test_query_summary_marks_failed_on_error(test_settings: Settings) -> None:
    _save_sample_query(test_settings, error="RuntimeError: llm down")
    summary = list_query_summaries(test_settings)[0]
    assert summary.status == "failed"
    assert summary.error == "RuntimeError: llm down"


def _write_minpai_markdown(path: Path) -> Path:
    path.write_text(
        """---
source_url: https://www.mzmz.org.cn/introduction.html
source_org: 湄洲妈祖祖庙
license_note: 政府网站公开信息，引用时保留 URL
culture_domain: 妈祖
space: lab_knowledge
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
    assert summaries[0]["trace_type"] == "query"
    assert summaries[0]["question"] == "妈祖信仰的发源地在哪里？"
    assert summaries[0]["refused"] is body["refused"]
    expected_status = "refused" if body["refused"] else "ok"
    assert summaries[0]["status"] == expected_status


def test_api_query_summary_and_detail_are_typed(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})
    ask = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})
    trace_id = ask.json()["trace_id"]

    summary = client.get(f"/api/traces/{trace_id}/summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert payload["trace_type"] == "query"
    assert payload["question"] == "妈祖信仰的发源地在哪里？"
    assert "chunk_count" not in payload

    detail = client.get(f"/api/traces/{trace_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["trace_type"] == "query"
    assert "stages" not in body
    labels = [item["label"] for item in body["stage_latencies"]]
    assert "嵌入检索" in labels
    assert "dense" not in labels
