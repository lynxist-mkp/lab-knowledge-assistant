"""Ingestion seam: fingerprint skip and rebuild."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lab_knowledge.app import create_app
from lab_knowledge.config import Settings
from lab_knowledge.knowledge import create_knowledge


def _write_markdown(path: Path, body: str) -> Path:
    path.write_text(
        f"""---
source_url: https://example.com/doc
title: 测试文档
---

{body}
""",
        encoding="utf-8",
    )
    return path


def test_same_file_twice_does_not_double_chunk_count(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_markdown(tmp_path / "doc.md", "闽派文化材料第一段。")
    client = TestClient(create_app(test_settings))

    first = client.post("/ingest", json={"source_path": str(source)})
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["status"] == "ingested"
    assert first_body["chunk_count"] >= 1

    second = client.post("/ingest", json={"source_path": str(source)})
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["status"] == "skipped"
    assert second_body["chunk_count"] == 0
    assert second_body["document_id"] == first_body["document_id"]

    knowledge = create_knowledge(test_settings)
    chunks = knowledge.get_by_document_id(first_body["document_id"])
    assert len(chunks) == first_body["chunk_count"]


def test_changed_content_replaces_old_chunks(test_settings: Settings, tmp_path: Path) -> None:
    source = _write_markdown(tmp_path / "doc.md", "旧版闽派文化内容。")
    client = TestClient(create_app(test_settings))

    first = client.post("/ingest", json={"source_path": str(source)})
    assert first.status_code == 200
    first_body = first.json()
    old_document_id = first_body["document_id"]

    source.write_text(
        """---
source_url: https://example.com/doc
title: 测试文档
---

新版闽派文化内容，与旧版不同。
""",
        encoding="utf-8",
    )

    third = client.post("/ingest", json={"source_path": str(source)})
    assert third.status_code == 200
    third_body = third.json()
    assert third_body["status"] == "rebuilt"
    assert third_body["document_id"] != old_document_id
    assert third_body["chunk_count"] >= 1

    knowledge = create_knowledge(test_settings)
    assert knowledge.get_by_document_id(old_document_id) == []
    new_chunks = knowledge.get_by_document_id(third_body["document_id"])
    assert len(new_chunks) == third_body["chunk_count"]
    assert any("新版" in chunk.text for chunk in new_chunks)
