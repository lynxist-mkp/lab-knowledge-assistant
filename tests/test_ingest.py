"""Ingestion seam: one source in, chunks retrievable."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.pipelines.ingestion import ingest_source


def _write_minpai_markdown(path: Path) -> Path:
    path.write_text(
        """---
source_url: https://www.mzmz.org.cn/introduction.html
source_org: 湄洲妈祖祖庙
license_note: 政府网站公开信息，引用时保留 URL
culture_domain: 检索增强
space: lab_knowledge
title: 湄洲妈祖祖庙简介
---

湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上，是信俗活动的中心场所。
每年农历三月二十三，信众会到祖庙参加祭典。
""",
        encoding="utf-8",
    )
    return path


def test_ingesting_markdown_makes_chunks_retrievable(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))

    response = client.post("/ingest", json={"source_path": str(source)})

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] >= 1
    assert body["document_id"]
    assert body["elapsed_ms"] >= 0
    assert body["trace_id"]
    assert body["status"] == "ingested"

    chunks = create_knowledge(test_settings).get_by_document_id(body["document_id"])
    assert len(chunks) == body["chunk_count"]
    first = chunks[0]
    assert "妈祖" in first.text
    assert first.metadata["document_id"] == body["document_id"]
    assert first.metadata["title"] == "湄洲妈祖祖庙简介"
    assert first.metadata["url"] == "https://www.mzmz.org.cn/introduction.html"
    assert first.metadata["page"] == 1

    python_result = ingest_source(source, test_settings)
    assert python_result.status == "skipped"


def _write_dirty_markdown(path: Path) -> Path:
    good_body = "湄洲岛是妈祖信仰的发源地，祖庙是信俗活动的中心场所。" * 30
    garbage = "@#" * 40
    path.write_text(
        f"""---
title: 带噪声的妈祖材料
---

第 1 页

{good_body}

{garbage}

版权所有 2024 湄洲妈祖祖庙
""",
        encoding="utf-8",
    )
    return path


def test_refiner_cleans_headers_footers_and_drops_garbage_chunks(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_dirty_markdown(tmp_path / "dirty_matsu.md")
    client = TestClient(create_app(test_settings))

    response = client.post("/ingest", json={"source_path": str(source)})

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] >= 1

    chunks = create_knowledge(test_settings).get_by_document_id(body["document_id"])
    assert len(chunks) == body["chunk_count"]
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "妈祖" in combined
    assert "第 1 页" not in combined
    assert "版权所有" not in combined
    assert "@#" not in combined


def test_enricher_garbage_llm_still_ingests_with_empty_enrichment(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["multimodal"] = "garbage"
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    response = client.post("/ingest", json={"source_path": str(source)})
    assert response.status_code == 200
    body = response.json()

    chunks = create_knowledge(test_settings).get_by_document_id(body["document_id"])
    assert len(chunks) >= 1
    first = chunks[0]
    assert "chunk_title" not in first.metadata
    assert "summary" not in first.metadata
    assert "tags" not in first.metadata


def test_enricher_writes_culture_domain_to_chunk_metadata(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    response = client.post("/ingest", json={"source_path": str(source)})
    assert response.status_code == 200
    body = response.json()

    chunks = create_knowledge(test_settings).get_by_document_id(body["document_id"])
    assert chunks[0].metadata["culture_domain"] == "检索增强"
    assert chunks[0].metadata["chunk_title"] == "妈祖祖庙"
    assert chunks[0].metadata["summary"] == "湄洲岛妈祖信仰中心"
    assert chunks[0].metadata["tags"] == ["妈祖"]


def test_ingest_stamps_review_status_approved(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    response = client.post("/ingest", json={"source_path": str(source)})
    assert response.status_code == 200
    body = response.json()

    chunks = create_knowledge(test_settings).get_by_document_id(body["document_id"])
    assert chunks
    assert all(chunk.metadata.get("审阅状态") == "已通过" for chunk in chunks)


def test_default_ingest_stamps_group_doc_source_metadata(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    response = client.post("/ingest", json={"source_path": str(source)})
    assert response.status_code == 200
    body = response.json()

    chunks = create_knowledge(test_settings).get_by_document_id(body["document_id"])
    assert chunks[0].metadata["source_kind"] == "group_doc"
    assert chunks[0].metadata["source_label"] == "组内资料"
    assert "authors" not in chunks[0].metadata

    card = create_knowledge(test_settings).document_card(body["document_id"])
    assert card.source_kind == "group_doc"
    assert card.source_label == "组内资料"


def test_personal_literature_ingest_persists_metadata_and_trace_fields(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = tmp_path / "Vaswani et al - Attention Is All You Need (2017).md"
    source.write_text(
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
    client = TestClient(create_app(test_settings))
    response = client.post(
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
    assert response.status_code == 200
    body = response.json()

    chunks = create_knowledge(test_settings).get_by_document_id(body["document_id"])
    first = chunks[0]
    assert first.metadata["source_kind"] == "personal_literature"
    assert first.metadata["source_label"] == "个人文献库"
    assert first.metadata["authors"] == "Ashish Vaswani"
    assert first.metadata["publication_year"] == 2017
    assert first.metadata["title"] == "Attention Is All You Need"

    card = create_knowledge(test_settings).document_card(body["document_id"])
    assert card.source_kind == "personal_literature"
    assert card.authors == "Ashish Vaswani"
    assert card.publication_year == 2017


def test_unsupported_source_type_returns_400(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = tmp_path / "clip.mp3"
    source.write_bytes(b"not-audio")
    client = TestClient(create_app(test_settings))
    response = client.post("/ingest", json={"source_path": str(source)})
    assert response.status_code == 400
    assert "unsupported source type" in response.json()["detail"]


def test_illegal_pdf_load_mode_returns_400(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    response = client.post(
        "/ingest",
        json={"source_path": str(source), "pdf_load_mode": "clip"},
    )
    assert response.status_code == 400
    assert "unsupported pdf load mode" in response.json()["detail"]
