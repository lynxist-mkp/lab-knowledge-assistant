"""MCP documents.get: document card via document-management facade."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.knowledge import create_document_management
from wenmai.mcp.tools.documents import documents_get


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


def test_documents_get_returns_enveloped_card(
    test_settings: Settings, tmp_path: Path
) -> None:
    app = create_app(test_settings)
    client = TestClient(app)
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200
    document_id = ingest.json()["document_id"]

    mgmt = create_document_management(test_settings, knowledge=app.state.knowledge)
    result = documents_get(
        mgmt,
        document_id,
        collection_id=test_settings.product.collection,
    )

    assert result["data"]["document_id"] == document_id
    assert result["data"]["title"] == "湄洲妈祖祖庙简介"
    assert result["data"]["culture_domain"] == "妈祖"
    assert result["data"]["chunk_count"] >= 1
    assert result["scope"]["collection_id"] == test_settings.product.collection
    assert "data" in result and "refs" in result


def test_documents_get_unknown_id_raises(test_settings: Settings) -> None:
    app = create_app(test_settings)
    mgmt = create_document_management(test_settings, knowledge=app.state.knowledge)
    with pytest.raises(ValueError, match="document not found"):
        documents_get(
            mgmt,
            "no-such-document",
            collection_id=test_settings.product.collection,
        )


def test_api_document_card_matches_facade_detail(
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

    mgmt = create_document_management(test_settings, knowledge=app.state.knowledge)
    facade = documents_get(
        mgmt,
        document_id,
        collection_id=test_settings.product.collection,
    )["data"]
    for key in ("document_id", "title", "culture_domain", "summary", "chunk_count"):
        assert response.json()[key] == facade[key]


def test_api_document_card_returns_404_for_unknown(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    response = client.get("/api/documents/no-such-document")
    assert response.status_code == 404
    assert "document not found" in response.json()["detail"]
