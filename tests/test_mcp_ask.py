"""MCP ask contracts: shared ask seam plus compatibility alias."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.generation import QueryGenerationError
from wenmai.http.ask_service import run_ask
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


def test_legacy_ask_wenmai_unknown_collection_raises(test_settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown collection"):
        ask_wenmai("问题", test_settings, collection_id="missing")


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


def test_ask_answer_wraps_query_generation_error(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_run_ask(question, settings, **kwargs):
        raise QueryGenerationError("generation failed", "trace-ask")

    monkeypatch.setattr("wenmai.mcp.tools.ask.run_ask", _fail_run_ask)

    with pytest.raises(AskAnswerError) as exc_info:
        ask_answer("问题", test_settings)

    assert exc_info.value.trace_id == "trace-ask"
