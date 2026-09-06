"""#80: source semantics and literature metadata in user-visible views."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings


def _personal_literature_markdown(path: Path) -> Path:
    path.write_text(
        """---
source_kind: personal_literature
culture_domain: 检索增强
author: Vaswani et al.
year: 2017
title: Attention Is All You Need
---

Transformer 架构通过自注意力机制建模序列依赖。
""",
        encoding="utf-8",
    )
    return path


def test_browse_api_exposes_source_and_literature_fields(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _personal_literature_markdown(tmp_path / "attention.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post(
        "/ingest",
        json={
            "source_path": str(source),
            "source_kind": "personal_literature",
            "literature": {
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani",
                "year": 2017,
            },
        },
    )
    assert ingest.status_code == 200

    browse = client.get("/api/browse")
    assert browse.status_code == 200
    doc = browse.json()[0]["documents"][0]
    assert doc["source_kind"] == "personal_literature"
    assert doc["source_label"] == "个人文献库"
    assert doc["authors"] == "Ashish Vaswani"
    assert doc["publication_year"] == 2017
    assert doc["title"] == "Attention Is All You Need"


def test_chunk_api_exposes_source_and_literature_fields(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _personal_literature_markdown(tmp_path / "attention-chunk.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post(
        "/ingest",
        json={
            "source_path": str(source),
            "source_kind": "personal_literature",
            "literature": {
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani",
                "year": 2017,
            },
        },
    )
    assert ingest.status_code == 200
    chunk_id = client.get("/api/browse").json()[0]["documents"][0]["chunks"][0]["chunk_id"]

    detail = client.get(f"/api/chunks/{chunk_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["source_kind"] == "personal_literature"
    assert body["source_label"] == "个人文献库"
    assert body["authors"] == "Ashish Vaswani"
    assert body["publication_year"] == 2017


def test_document_card_api_exposes_source_and_literature_fields(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _personal_literature_markdown(tmp_path / "attention-card.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post(
        "/ingest",
        json={
            "source_path": str(source),
            "source_kind": "personal_literature",
            "literature": {
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani",
                "year": 2017,
            },
        },
    )
    assert ingest.status_code == 200
    document_id = ingest.json()["document_id"]

    card = client.get(f"/api/documents/{document_id}")
    assert card.status_code == 200
    body = card.json()
    assert body["source_kind"] == "personal_literature"
    assert body["source_label"] == "个人文献库"
    assert body["authors"] == "Ashish Vaswani"
    assert body["publication_year"] == 2017


def test_ask_citations_expose_source_semantics(test_settings: Settings, tmp_path: Path) -> None:
    source = _personal_literature_markdown(tmp_path / "attention-ask.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post(
        "/ingest",
        json={
            "source_path": str(source),
            "source_kind": "personal_literature",
            "literature": {
                "title": "Attention Is All You Need",
                "authors": "Ashish Vaswani",
                "year": 2017,
            },
        },
    )
    assert ingest.status_code == 200

    response = client.post(
        "/ask",
        json={"question": "Transformer 架构如何建模序列依赖？"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    first = body["citations"][0]
    assert first["source_kind"] == "personal_literature"
    assert first["source_label"] == "个人文献库"
    assert first["authors"] == "Ashish Vaswani"
    assert first["publication_year"] == 2017


def test_ops_browse_template_surfaces_source_semantics(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    html = client.get("/ops").text

    assert "renderSourceKindBadge" in html
    assert "formatLiteratureMeta" in html
    assert "source-kind-badge" in html
    assert "doc-literature-meta" in html
    assert "资料来源:" in html


def test_workbench_template_surfaces_source_in_citation_views(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    html = client.get("/").text

    assert "citation-drawer-source-label" in html
    assert "citation-drawer-literature" in html
    assert "renderSourceKindBadge" in html
    assert "formatLiteratureMeta" in html
    assert "citation-literature-meta" in html
    assert "data-source-label" in html
