"""MCP ask tool: 提问 without MCP wire protocol."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.pipelines.query import QueryGenerationError, ask_question


def _seed_doc(client: TestClient, tmp_path, text: str = "湄洲岛是妈祖信仰的发源地。") -> None:
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    response = client.post("/ingest", json={"source_path": str(path)})
    assert response.status_code == 200


def test_ask_via_shared_knowledge_returns_answer(
    test_settings: Settings, tmp_path
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    _seed_doc(client, tmp_path)

    result = ask_question(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        knowledge=app.state.knowledge,
    ).as_dict()

    assert result["answer"]
    assert result["citations"]
    assert result["trace_id"]
    assert result["refused"] is False


def test_ask_surfaces_generation_failure_with_trace_id(
    test_settings: Settings, tmp_path
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    _seed_doc(client, tmp_path)
    test_settings.fakes["multimodal"] = "error"

    with pytest.raises(QueryGenerationError) as exc_info:
        ask_question(
            "妈祖信仰的发源地在哪里？",
            test_settings,
            knowledge=app.state.knowledge,
        )

    assert exc_info.value.trace_id
