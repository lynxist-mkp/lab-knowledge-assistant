"""MCP get_document_summary: document card without MCP wire protocol."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.mcp.summary import GetDocumentSummaryError, get_document_summary


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


def test_get_document_summary_returns_card(
    test_settings: Settings, tmp_path: Path
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200
    document_id = ingest.json()["document_id"]

    result = get_document_summary(
        document_id,
        test_settings,
        knowledge=app.state.knowledge,
    )

    assert result["document_id"] == document_id
    assert result["title"] == "湄洲妈祖祖庙简介"
    assert result["culture_domain"] == "妈祖"
    assert result["chunk_count"] >= 1
    assert result["summary"] == "湄洲岛妈祖信仰中心"


def test_get_document_summary_unknown_id_raises(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with pytest.raises(GetDocumentSummaryError, match="document not found"):
        get_document_summary(
            "no-such-document",
            test_settings,
            knowledge=app.state.knowledge,
        )


def test_api_document_card_matches_mcp_handler(
    test_settings: Settings, tmp_path: Path
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200
    document_id = ingest.json()["document_id"]

    response = client.get(f"/api/documents/{document_id}")
    assert response.status_code == 200
    assert response.json() == get_document_summary(
        document_id,
        test_settings,
        knowledge=app.state.knowledge,
    )


def test_api_document_card_returns_404_for_unknown(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    response = client.get("/api/documents/no-such-document")
    assert response.status_code == 404
    assert "document not found" in response.json()["detail"]
