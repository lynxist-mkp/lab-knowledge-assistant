"""MCP ask contracts: shared ask seam plus compatibility alias."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from wenmai.ask_surface import AskSurfaceResult
from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.http.ask_service import run_ask
from wenmai.models import AskResult
from wenmai.mcp.ask import AskWenmaiError, ask_wenmai
from wenmai.mcp.server import create_mcp_server
from wenmai.mcp.tools.ask import AskAnswerError, ask_answer


def _seed_doc(client: TestClient, tmp_path, text: str = "湄洲岛是妈祖信仰的发源地。") -> None:
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    response = client.post("/ingest", json={"source_path": str(path)})
    assert response.status_code == 200


def test_ask_service_returns_answer(
    test_settings: Settings, tmp_path
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    _seed_doc(client, tmp_path)

    result = run_ask(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        knowledge=app.state.knowledge,
    ).as_dict()

    assert result["answer"]
    assert result["citations"]
    assert result["trace_id"]
    assert result["refused"] is False


def test_ask_service_delegates_to_query_orchestration_entry(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def _fake_ask_pipeline_single(payload, *, phase_batch):
        captured["payload"] = payload
        captured["phase_batch"] = phase_batch
        return AskResult(answer="ok", citations=[], trace_id="trace-123")

    monkeypatch.setattr(
        "wenmai.http.ask_service.ask_pipeline_single",
        _fake_ask_pipeline_single,
    )

    result = run_ask(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        culture_domain="妈祖",
        retrieval_mode="dense_only",
        rerank_enabled=False,
    )

    assert result.trace_id == "trace-123"
    assert captured["payload"].question == "妈祖信仰的发源地在哪里？"
    assert captured["payload"].settings is test_settings
    assert captured["payload"].culture_domain == "妈祖"
    assert captured["payload"].retrieval_mode == "dense_only"
    assert captured["payload"].rerank_enabled is False
    assert captured["phase_batch"] == test_settings.resources.query_phase_batch


def test_ask_answer_envelope_contains_trace_id(
    test_settings: Settings, tmp_path
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    _seed_doc(client, tmp_path)

    result = ask_answer(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        collection_id=test_settings.product.collection,
        knowledge=app.state.knowledge,
    )

    assert result["data"]["answer"]
    assert result["data"]["citations"]
    assert result["data"]["trace_id"]
    assert result["scope"]["collection_id"] == test_settings.product.collection


def test_ask_answer_uses_shared_surface(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def _fake_ask_surface(question, settings, **kwargs):
        captured["question"] = question
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return AskSurfaceResult(
            result=AskResult(answer="ok", citations=[], trace_id="trace-123"),
            elapsed_ms=12.5,
        )

    monkeypatch.setattr("wenmai.mcp.tools.ask.ask_surface", _fake_ask_surface)

    result = ask_answer(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        collection_id=test_settings.product.collection,
        culture_domain="妈祖",
        retrieval_mode="dense_only",
        rerank_enabled=False,
    )

    assert result["data"]["trace_id"] == "trace-123"
    assert result["meta"]["elapsed_ms"] == 12.5
    assert captured["question"] == "妈祖信仰的发源地在哪里？"
    assert captured["settings"] is test_settings
    assert captured["kwargs"]["entrypoint"] == "mcp"
    assert captured["kwargs"]["culture_domain"] == "妈祖"
    assert captured["kwargs"]["retrieval_mode"] == "dense_only"
    assert captured["kwargs"]["rerank_enabled"] is False


def test_legacy_ask_wenmai_returns_plain_contract(
    test_settings: Settings, tmp_path
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    _seed_doc(client, tmp_path)

    result = ask_wenmai(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        culture_domain="妈祖",
    )

    assert result["answer"]
    assert result["citations"]
    assert result["trace_id"]
    assert "scope" not in result


def test_legacy_ask_wenmai_uses_shared_surface(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def _fake_ask_surface(question, settings, **kwargs):
        captured["question"] = question
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return AskSurfaceResult(
            result=AskResult(answer="ok", citations=[], trace_id="trace-legacy"),
            elapsed_ms=8.0,
        )

    monkeypatch.setattr("wenmai.mcp.ask.ask_surface", _fake_ask_surface)

    result = ask_wenmai(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        culture_domain="妈祖",
        retrieval_mode="dense_only",
        rerank_enabled=False,
    )

    assert result["trace_id"] == "trace-legacy"
    assert captured["question"] == "妈祖信仰的发源地在哪里？"
    assert captured["settings"] is test_settings
    assert captured["kwargs"]["entrypoint"] == "mcp-legacy"
    assert captured["kwargs"]["culture_domain"] == "妈祖"
    assert captured["kwargs"]["retrieval_mode"] == "dense_only"
    assert captured["kwargs"]["rerank_enabled"] is False


def test_mcp_server_registers_new_and_legacy_ask_tools(test_settings: Settings) -> None:
    server = create_mcp_server(test_settings)
    tool_names = {tool.name for tool in asyncio.run(server.list_tools())}

    assert "ask.answer" in tool_names
    assert "ask_wenmai" in tool_names


def test_ask_surfaces_generation_failure_with_trace_id(
    test_settings: Settings, tmp_path
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    _seed_doc(client, tmp_path)
    test_settings.fakes["multimodal"] = "error"

    with pytest.raises(AskAnswerError) as exc_info:
        ask_answer(
            "妈祖信仰的发源地在哪里？",
            test_settings,
            collection_id=test_settings.product.collection,
            knowledge=app.state.knowledge,
        )

    assert exc_info.value.trace_id

    with pytest.raises(AskWenmaiError) as legacy_exc_info:
        ask_wenmai("妈祖信仰的发源地在哪里？", test_settings)

    assert legacy_exc_info.value.trace_id
