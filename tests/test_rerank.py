"""提问：精排失败时仍返回回答和出处。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings


def _write_minpai_markdown(path: Path) -> Path:
    path.write_text(
        """---
source_url: https://www.mzmz.org.cn/introduction.html
culture_domain: 妈祖
title: 湄洲妈祖祖庙简介
---

湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上，是信俗活动的中心场所。
""",
        encoding="utf-8",
    )
    return path


def test_ask_returns_citations_when_rerank_enabled(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    response = client.post(
        "/ask",
        json={"question": "妈祖信仰的发源地在哪里？", "rerank_enabled": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert body["citations"][0]["document_id"] == ingest.json()["document_id"]


def test_rerank_failure_still_returns_answer_and_citations(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["reranker"] = "error"
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["citations"]
    assert body["citations"][0]["document_id"] == ingest.json()["document_id"]


def test_rerank_timeout_still_returns_answer(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["reranker"] = "timeout"
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})

    assert response.status_code == 200
    assert response.json()["citations"]


def test_rerank_empty_rankings_still_returns_answer(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["reranker"] = "garbage"
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    client = TestClient(create_app(test_settings))
    client.post("/ingest", json={"source_path": str(source)})

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})

    assert response.status_code == 200
    assert response.json()["citations"]
