"""提问：回答带出处；生成失败返回 502。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.pipelines.query import ask_question
from wenmai.tracing.store import get_trace_record


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
    assert "ranked_chunks" not in body


def test_ask_question_includes_ranked_chunks_for_eval(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    result = ask_question("妈祖信仰的发源地在哪里？", test_settings)

    assert result.ranked_chunks
    assert result.ranked_chunks[0].chunk.chunk_id == result.citations[0].chunk_id
    assert "ranked_chunks" not in result.as_dict()


def test_ask_accepts_retrieval_mode_and_keeps_public_result(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    response = client.post(
        "/ask",
        json={
            "question": "妈祖信仰的发源地在哪里？",
            "retrieval_mode": "dense_only",
            "rerank_enabled": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert "ranked_chunks" not in body
    assert set(body) <= {"answer", "citations", "trace_id", "refused", "error"}


def test_ask_records_generation_failure_in_trace_and_returns_error(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["multimodal"] = "error"
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["trace_id"]
    assert "error" in detail["message"].lower() or "fake" in detail["message"].lower()


def test_ask_empty_kb_refuses_without_calling_llm(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm_called = False

    def _fail_if_called(settings: Settings) -> object:
        nonlocal llm_called
        llm_called = True
        raise AssertionError("LLM should not be called for zero chunks")

    monkeypatch.setattr("wenmai.factories.multimodal.create", _fail_if_called)

    result = ask_question("妈祖信仰的发源地在哪里？", test_settings)

    assert llm_called is False
    assert result.refused is True
    assert result.answer.startswith("拒答：")
    assert result.citations == []

    record = get_trace_record(test_settings, result.trace_id)
    assert record is not None
    outcome = record["metadata"]["outcome"]
    assert outcome["refused"] is True
    assert outcome["refusal_reason"] == "insufficient_evidence"
    assert outcome["citation_count"] == 0


def test_ask_writes_trace_outcome_metadata(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    test_settings.fakes["multimodal"] = "refuse"
    response = client.post(
        "/ask", json={"question": "船政学堂是什么时候创办的？"}
    )
    body = response.json()
    record = get_trace_record(test_settings, body["trace_id"])
    assert record is not None
    outcome = record["metadata"]["outcome"]
    assert outcome["refused"] is body["refused"]
    assert outcome["refusal_reason"] == "model_refused"
    assert outcome["citation_count"] == len(body["citations"])
