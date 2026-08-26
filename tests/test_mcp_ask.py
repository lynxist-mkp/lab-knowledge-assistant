"""MCP ask_wenmai handler: service-layer ask without MCP wire protocol."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.mcp.ask import AskWenmaiError, ask_wenmai


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


def test_ask_wenmai_returns_answer_citations_and_trace_id(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    result = ask_wenmai("妈祖信仰的发源地在哪里？", settings=test_settings)

    assert result["trace_id"]
    assert "[1]" in result["answer"]
    assert result["citations"]
    first = result["citations"][0]
    assert first["index"] == 1
    assert first["chunk_id"]
    assert first["document_id"] == ingest.json()["document_id"]
    assert "妈祖" in first["excerpt"]


def test_ask_wenmai_surfaces_generation_failure_with_trace_id(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["llm"] = "error"
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    try:
        ask_wenmai("妈祖信仰的发源地在哪里？", settings=test_settings)
    except AskWenmaiError as exc:
        assert exc.trace_id
        assert "error" in str(exc).lower() or "fake" in str(exc).lower()
    else:
        raise AssertionError("expected AskWenmaiError")
