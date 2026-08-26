"""BM25 narrow seam + sparse-only query trace."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from wenmai.components.bm25.index import Bm25Index
from wenmai.components.bm25.tokenizer import ChineseTokenizer
from wenmai.config import Settings
from wenmai.factories import bm25 as bm25_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.models import Chunk


def _tokenizer(settings: Settings) -> ChineseTokenizer:
    return ChineseTokenizer.from_settings(settings)


def _index_path(settings: Settings) -> Path:
    return Path(settings.paths.bm25) / settings.product.collection / "index.json"


def test_build_records_tf_doc_length_and_idf(test_settings: Settings) -> None:
    tokenizer = _tokenizer(test_settings)
    index = Bm25Index(
        collection=test_settings.product.collection,
        persist_path=Path(test_settings.paths.bm25),
        tokenizer=tokenizer,
        k1=test_settings.bm25.k1,
        b=test_settings.bm25.b,
    )
    chunks = [
        Chunk(
            chunk_id="doc-a:0000",
            document_id="doc-a",
            text="船政学堂培养了许多海军人才。",
        ),
        Chunk(
            chunk_id="doc-b:0000",
            document_id="doc-b",
            text="湄洲祖庙是妈祖信仰的中心。",
        ),
    ]
    index.upsert(chunks)
    index.save()

    payload = json.loads(_index_path(test_settings).read_text(encoding="utf-8"))
    assert "text" not in json.dumps(payload, ensure_ascii=False)
    assert payload["num_docs"] == 2

    ship_entry = payload["terms"]["船政学堂"]
    assert ship_entry["idf"] > 0
    posting = ship_entry["postings"][0]
    assert posting["chunk_id"] == "doc-a:0000"
    assert posting["tf"] == 1
    assert posting["doc_length"] == len(tokenizer.tokenize(chunks[0].text))

    temple_entry = payload["terms"]["湄洲祖庙"]
    assert len(temple_entry["postings"]) == 1
    assert temple_entry["postings"][0]["chunk_id"] == "doc-b:0000"

    # idf for a term in 1 of 2 docs: log((2 - 1 + 0.5) / (1 + 0.5) + 1)
    expected_idf = math.log((2 - 1 + 0.5) / (1 + 0.5) + 1)
    assert ship_entry["idf"] == pytest.approx(expected_idf)


def test_search_ranks_more_relevant_chunk_higher(test_settings: Settings) -> None:
    tokenizer = _tokenizer(test_settings)
    index = Bm25Index(
        collection=test_settings.product.collection,
        persist_path=Path(test_settings.paths.bm25),
        tokenizer=tokenizer,
        k1=test_settings.bm25.k1,
        b=test_settings.bm25.b,
    )
    chunks = [
        Chunk(
            chunk_id="doc-a:0000",
            document_id="doc-a",
            text="船政学堂船政学堂是近代海军摇篮。",
        ),
        Chunk(
            chunk_id="doc-b:0000",
            document_id="doc-b",
            text="船政学堂曾在马尾设立分校，后来迁往别处。",
        ),
    ]
    index.upsert(chunks)

    hits = index.search("船政学堂", top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "doc-a:0000"
    assert hits[0].score > hits[1].score


def test_index_and_query_use_same_tokenizer(test_settings: Settings) -> None:
    tokenizer = _tokenizer(test_settings)
    text = "船政学堂与湄洲祖庙都是福建文化地标"
    index_tokens = tokenizer.tokenize(text)
    query_tokens = tokenizer.tokenize("湄洲祖庙 船政学堂")
    assert "船政学堂" in index_tokens
    assert "湄洲祖庙" in index_tokens
    assert "船政学堂" in query_tokens
    assert "湄洲祖庙" in query_tokens
    assert "的" not in query_tokens


def test_domain_dict_keeps_compound_terms(test_settings: Settings) -> None:
    tokenizer = _tokenizer(test_settings)
    domain_path = Path(test_settings.root) / test_settings.bm25.domain_dict
    for term in domain_path.read_text(encoding="utf-8").splitlines():
        term = term.strip()
        if not term:
            continue
        tokens = tokenizer.tokenize(f"介绍{term}的历史")
        assert term in tokens


def test_stopwords_are_removed(test_settings: Settings) -> None:
    tokenizer = _tokenizer(test_settings)
    tokens = tokenizer.tokenize("这是一个关于妈祖信仰的简介")
    assert "的" not in tokens
    assert "是" not in tokens
    assert "一个" not in tokens
    assert "妈祖信仰" in tokens


def test_delete_document_removes_postings(test_settings: Settings) -> None:
    index = bm25_factory.create(test_settings)
    chunks = [
        Chunk(chunk_id="doc-a:0000", document_id="doc-a", text="船政学堂简介"),
        Chunk(chunk_id="doc-b:0000", document_id="doc-b", text="湄洲祖庙简介"),
    ]
    index.upsert(chunks)
    index.delete_by_document_id("doc-a")
    index.save()

    payload = json.loads(_index_path(test_settings).read_text(encoding="utf-8"))
    for term_entry in payload["terms"].values():
        for posting in term_entry["postings"]:
            assert posting["document_id"] != "doc-a"


def test_search_results_fetch_text_from_chroma_not_index(
    test_settings: Settings,
) -> None:
    store = vector_store_factory.create(test_settings)
    index = bm25_factory.create(test_settings)
    chunk = Chunk(
        chunk_id="doc-a:0000",
        document_id="doc-a",
        text="船政学堂培养了近代海军将领。",
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        metadata={"document_id": "doc-a", "title": "船政"},
    )
    store.upsert([chunk])
    index.upsert([chunk])
    index.save()

    hits = index.search("船政学堂", top_k=1)
    assert hits
    fetched = store.get_by_ids([hits[0].chunk_id])
    assert fetched[0].text == chunk.text
    assert hits[0].score > 0


def test_sparse_only_ask_records_sparse_stage_in_trace(
    test_settings: Settings, tmp_path: Path
) -> None:
    import json

    from fastapi.testclient import TestClient

    from wenmai.app import create_app

    test_settings.retrieval.mode = "sparse_only"
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

    response = client.post("/ask", json={"question": "船政学堂在哪里创办？"})
    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"]

    trace_path = Path(test_settings.paths.traces)
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    query_trace = next(trace for trace in traces if trace["trace_id"] == body["trace_id"])
    assert [stage["name"] for stage in query_trace["stages"]] == [
        "query_processing",
        "sparse",
        "generation",
    ]
    sparse_stage = next(stage for stage in query_trace["stages"] if stage["name"] == "sparse")
    assert sparse_stage["method"] == "bm25"
    assert sparse_stage["candidates"]
