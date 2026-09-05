"""MCP ask saturation responses."""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError

from wenmai.app import create_app
from wenmai.config import AskConcurrencyConfig
from wenmai.http.ask_governor import (
    AskSaturationError,
    get_ask_governor,
    reset_ask_governor,
)
from wenmai.mcp.errors import (
    AskSaturationToolError,
    saturation_detail_from_tool_error,
)
from wenmai.mcp.server import _ask_tool_error, create_mcp_server
from wenmai.mcp.tools.ask import ask_answer
from wenmai.task_progress import TaskCounters, persist_task_progress


@pytest.fixture(autouse=True)
def _reset_governor() -> None:
    reset_ask_governor()


def _seed_doc(client: TestClient, tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("湄洲岛是妈祖信仰的发源地。", encoding="utf-8")
    response = client.post("/ingest", json={"source_path": str(path)})
    assert response.status_code == 200


def test_ask_saturation_tool_error_preserves_structured_detail() -> None:
    exc = AskSaturationError(
        "busy",
        "提问服务繁忙：当前并发已满，请稍后重试。",
        in_flight=1,
        max_in_flight=2,
        wait_ms=0.0,
        entrypoint="mcp",
    )
    wrapped = _ask_tool_error(exc)

    assert isinstance(wrapped, AskSaturationToolError)
    detail = wrapped.as_dict()
    assert detail["code"] == "busy"
    assert detail["in_flight"] == 1
    assert detail["max_in_flight"] == 2
    assert detail["wait_ms"] == 0.0
    assert detail["entrypoint"] == "mcp"
    assert "提问服务繁忙" in str(detail["message"])


def test_saturation_detail_from_tool_error_parses_mcp_wrapper_message() -> None:
    exc = AskSaturationError(
        "timeout",
        "提问服务等待超时：队列等待超过上限。",
        in_flight=1,
        max_in_flight=1,
        wait_ms=152.3,
        entrypoint="mcp",
    )
    wrapped = AskSaturationToolError(exc)
    outer = ToolError(f"Error executing tool ask.answer: {wrapped}")

    detail = saturation_detail_from_tool_error(outer)
    assert detail is not None
    assert detail["code"] == "timeout"
    assert detail["wait_ms"] == 152.3


def test_mcp_ask_answer_reports_busy_with_structured_detail(
    test_settings, tmp_path
) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=1,
        saturation_policy="busy",
    )
    app = create_app(test_settings)
    client = TestClient(app)
    _seed_doc(client, tmp_path)
    gate = threading.Event()
    holder_started = threading.Event()
    governor = get_ask_governor(test_settings)

    def hold_governor() -> None:
        with governor.acquire(test_settings, entrypoint="holder"):
            holder_started.set()
            gate.wait(timeout=2)

    thread = threading.Thread(target=hold_governor)
    thread.start()
    assert holder_started.wait(timeout=2)

    with pytest.raises(AskSaturationError) as exc_info:
        ask_answer(
            "妈祖信仰的发源地在哪里？",
            test_settings,
            collection_id=test_settings.product.collection,
            knowledge=app.state.knowledge,
        )

    wrapped = _ask_tool_error(exc_info.value)
    assert isinstance(wrapped, AskSaturationToolError)
    detail = wrapped.as_dict()
    assert detail["code"] == "busy"
    gate.set()
    thread.join(timeout=2)


def test_mcp_server_call_tool_reports_structured_busy(
    test_settings, tmp_path
) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=1,
        saturation_policy="busy",
    )
    app = create_app(test_settings)
    client = TestClient(app)
    _seed_doc(client, tmp_path)
    server = create_mcp_server(test_settings)
    gate = threading.Event()
    holder_started = threading.Event()
    governor = get_ask_governor(test_settings)

    def hold_governor() -> None:
        with governor.acquire(test_settings, entrypoint="holder"):
            holder_started.set()
            gate.wait(timeout=2)

    thread = threading.Thread(target=hold_governor)
    thread.start()
    assert holder_started.wait(timeout=2)

    with pytest.raises(ToolError) as exc_info:
        asyncio.run(
            server.call_tool(
                "ask.answer",
                {"question": "妈祖信仰的发源地在哪里？"},
            )
        )

    detail = saturation_detail_from_tool_error(exc_info.value)
    assert detail is not None
    assert detail["code"] == "busy"
    assert detail["in_flight"] == 1
    assert detail["max_in_flight"] == 1
    gate.set()
    thread.join(timeout=2)


def test_mcp_server_call_tool_reports_structured_long_task_active(
    test_settings, tmp_path
) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=3,
        long_task_guard=True,
        long_task_max_in_flight=0,
        saturation_policy="busy",
    )
    persist_task_progress(
        test_settings,
        task_id="ingest-running",
        task_type="ingestion",
        status="running",
        started_at="2026-01-01T00:00:00Z",
        finished_at=None,
        last_progress_at="2026-01-01T00:00:00Z",
        trigger_source="ops",
        owner_surface="ops",
        config_snapshot={"pdf_load_mode": "auto"},
        counters=TaskCounters(total=1),
    )
    app = create_app(test_settings)
    client = TestClient(app)
    _seed_doc(client, tmp_path)
    server = create_mcp_server(test_settings)

    with pytest.raises(ToolError) as exc_info:
        asyncio.run(
            server.call_tool(
                "ask.answer",
                {"question": "妈祖信仰的发源地在哪里？"},
            )
        )

    detail = saturation_detail_from_tool_error(exc_info.value)
    assert detail is not None
    assert detail["code"] == "long_task_active"
    assert detail["max_in_flight"] == 0
