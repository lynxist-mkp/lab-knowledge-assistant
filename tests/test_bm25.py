"""Tokenizer seam + sparse-only 提问 still returns citations."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.components.bm25.tokenizer import ChineseTokenizer
from wenmai.config import Settings


def _tokenizer(settings: Settings) -> ChineseTokenizer:
    return ChineseTokenizer.from_settings(settings)


def test_index_and_query_use_same_tokenizer(test_settings: Settings) -> None:
    tokenizer = _tokenizer(test_settings)
    text = "船政学堂与湄洲祖庙都是福建文化地标"
    index_tokens = tokenizer.tokenize(text)
    query_tokens = tokenizer.tokenize("湄洲祖庙 船政学堂")
    assert {"船政", "学堂"}.issubset(index_tokens)
    assert {"湄洲", "祖庙"}.issubset(index_tokens)
    assert {"船政", "学堂"}.issubset(query_tokens)
    assert {"湄洲", "祖庙"}.issubset(query_tokens)
    assert "的" not in query_tokens


def test_domain_dict_keeps_compound_terms(test_settings: Settings) -> None:
    tokenizer = _tokenizer(test_settings)
    domain_path = Path(test_settings.root) / test_settings.bm25.domain_dict
    for term in domain_path.read_text(encoding="utf-8").splitlines():
        term = term.strip()
        if not term:
            continue
        tokens = tokenizer.tokenize(f"介绍{term}的历史")
        parts = [part for part in term.split() if part]
        assert parts
        assert set(parts).issubset(tokens)


def test_stopwords_are_removed(test_settings: Settings) -> None:
    tokenizer = _tokenizer(test_settings)
    tokens = tokenizer.tokenize("这是一个关于妈祖信仰的简介")
    assert "的" not in tokens
    assert "是" not in tokens
    assert "一个" not in tokens
    assert {"妈祖", "信仰"}.issubset(tokens)


def test_sparse_only_ask_returns_citations(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = tmp_path / "ship.md"
    source.write_text(
        """---
title: 船政学堂史料
---

船政学堂创办于马尾，是中国近代第一所海军学堂。
""",
        encoding="utf-8",
    )
    client = TestClient(create_app(test_settings))
    ingest = client.post("/ingest", json={"source_path": str(source)})
    assert ingest.status_code == 200

    response = client.post(
        "/ask",
        json={"question": "船政学堂在哪里创办？", "retrieval_mode": "sparse_only"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"]
    assert body["citations"]
    assert body["citations"][0]["document_id"] == ingest.json()["document_id"]
