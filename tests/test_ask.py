"""提问：回答带出处；生成失败返回 502。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wenmai.ask_surface import AskSurfaceError, AskSurfaceResult
from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.http.ask_service import run_ask
from wenmai.knowledge import create_knowledge
from wenmai.models import AskResult
import wenmai.pipelines.query as query_compat
from wenmai.pipelines.query_orchestration import AskPipelineInput, ask_pipeline_single
from wenmai.tracing.store import get_trace_record, read_trace_records


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


def _run_query_pipeline(
    question: str,
    settings: Settings,
    *,
    culture_domain: str | None = None,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
    knowledge=None,
    record_trace: bool = True,
) -> AskResult:
    return ask_pipeline_single(
        AskPipelineInput(
            question=question,
            settings=settings,
            culture_domain=culture_domain,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            record_trace=record_trace,
        ),
        phase_batch=settings.resources.query_phase_batch,
    )


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


def test_query_compat_default_path_delegates_to_run_ask(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def _fake_run_ask(question, settings, **kwargs):
        captured["question"] = question
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return AskResult(answer="ok", citations=[], trace_id="trace-compat")

    monkeypatch.setattr("wenmai.http.ask_service.run_ask", _fake_run_ask)

    result = query_compat.ask_question(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        culture_domain="妈祖",
        retrieval_mode="dense_only",
        rerank_enabled=False,
    )

    assert result.trace_id == "trace-compat"
    assert captured["question"] == "妈祖信仰的发源地在哪里？"
    assert captured["settings"] is test_settings
    assert captured["kwargs"]["entrypoint"] == "query-compat"
    assert captured["kwargs"]["culture_domain"] == "妈祖"
    assert captured["kwargs"]["retrieval_mode"] == "dense_only"
    assert captured["kwargs"]["rerank_enabled"] is False


def test_query_compat_record_trace_false_stays_on_deeper_orchestration(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def _unexpected_run_ask(question, settings, **kwargs):
        raise AssertionError("compat helper should not use run_ask here")

    def _fake_ask_pipeline_single(payload, *, phase_batch):
        captured["payload"] = payload
        captured["phase_batch"] = phase_batch
        return AskResult(answer="ok", citations=[], trace_id="trace-deep")

    monkeypatch.setattr("wenmai.http.ask_service.run_ask", _unexpected_run_ask)
    monkeypatch.setattr(
        "wenmai.pipelines.query.ask_pipeline_single",
        _fake_ask_pipeline_single,
    )

    result = query_compat.ask_question(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        record_trace=False,
    )

    assert result.trace_id == "trace-deep"
    assert captured["payload"].question == "妈祖信仰的发源地在哪里？"
    assert captured["payload"].record_trace is False
    assert captured["phase_batch"] == test_settings.resources.query_phase_batch


def test_ask_question_includes_ranked_chunks_for_eval(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    result = _run_query_pipeline("妈祖信仰的发源地在哪里？", test_settings)

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
    assert set(body) <= {
        "answer",
        "citations",
        "trace_id",
        "refused",
        "refusal_reason",
        "error",
    }


def test_http_ask_endpoint_uses_shared_surface(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    captured = {}

    def _fake_ask_surface(question, settings, **kwargs):
        captured["question"] = question
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return AskSurfaceResult(
            result=AskResult(answer="ok", citations=[], trace_id="trace-http"),
            elapsed_ms=7.5,
        )

    monkeypatch.setattr("wenmai.http.workbench.ask_surface", _fake_ask_surface)

    response = client.post(
        "/ask",
        json={
            "question": "妈祖信仰的发源地在哪里？",
            "culture_domain": "妈祖",
            "retrieval_mode": "dense_only",
            "rerank_enabled": False,
        },
    )

    assert response.status_code == 200
    assert response.json()["trace_id"] == "trace-http"
    assert captured["question"] == "妈祖信仰的发源地在哪里？"
    assert captured["settings"] is test_settings
    assert captured["kwargs"]["entrypoint"] == "http"
    assert captured["kwargs"]["knowledge"] is app.state.knowledge
    assert captured["kwargs"]["culture_domain"] == "妈祖"
    assert captured["kwargs"]["retrieval_mode"] == "dense_only"
    assert captured["kwargs"]["rerank_enabled"] is False


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


def test_http_ask_surface_error_returns_502(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TestClient(create_app(test_settings))

    def _fail_ask_surface(question, settings, **kwargs):
        raise AskSurfaceError("surface failed", "trace-surface")

    monkeypatch.setattr("wenmai.http.workbench.ask_surface", _fail_ask_surface)

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})

    assert response.status_code == 502
    assert response.json()["detail"] == {
        "message": "surface failed",
        "trace_id": "trace-surface",
    }


def test_ask_empty_kb_refuses_without_calling_llm(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm_called = False

    def _fail_if_called(settings: Settings) -> object:
        nonlocal llm_called
        llm_called = True
        raise AssertionError("LLM should not be called for zero chunks")

    monkeypatch.setattr("wenmai.factories.multimodal.create", _fail_if_called)

    result = run_ask("妈祖信仰的发源地在哪里？", test_settings)

    assert llm_called is False
    assert result.refused is True
    assert result.refusal_reason == "insufficient_evidence"
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
    assert body["refusal_reason"] == "model_refused"
    record = get_trace_record(test_settings, body["trace_id"])
    assert record is not None
    outcome = record["metadata"]["outcome"]
    assert outcome["refused"] is body["refused"]
    assert outcome["refusal_reason"] == "model_refused"
    assert outcome["citation_count"] == len(body["citations"])


def test_ask_question_record_trace_false_skips_trace_write(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    query_traces = [
        record
        for record in read_trace_records(test_settings)
        if record.get("trace_type") == "query"
    ]
    before = len(query_traces)
    result = _run_query_pipeline(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        record_trace=False,
    )
    after = len(
        [
            record
            for record in read_trace_records(test_settings)
            if record.get("trace_type") == "query"
        ]
    )

    assert result.trace_id
    assert after == before


def test_ask_excludes_pending_chunks_until_approved(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200
    document_id = ingest.json()["document_id"]

    knowledge = create_knowledge(test_settings)
    knowledge.set_review_status(document_id, "待审")

    pending = run_ask("妈祖信仰的发源地在哪里？", test_settings, knowledge=knowledge)
    assert pending.refused is True
    assert pending.citations == []

    knowledge.set_review_status(document_id, "已通过")
    approved = run_ask("妈祖信仰的发源地在哪里？", test_settings, knowledge=knowledge)
    assert approved.citations
    assert approved.citations[0].document_id == document_id
