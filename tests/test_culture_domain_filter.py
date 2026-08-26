"""Culture domain filter: cross-domain same-term chunks must not leak."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import Settings
from wenmai.factories import bm25 as bm25_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.models import Chunk
from wenmai.pipelines.query import ask_question

_SHARED_TERM = "通商口岸"
_HAISI_CHUNK_ID = "doc-haisi:0000"
_SHIP_CHUNK_ID = "doc-ship:0000"
_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _seed_cross_domain_chunks(settings: Settings) -> None:
    store = vector_store_factory.create(settings)
    index = bm25_factory.create(settings)
    chunks = [
        Chunk(
            chunk_id=_HAISI_CHUNK_ID,
            document_id="doc-haisi",
            text=f"{_SHARED_TERM}是海丝贸易的重要节点，泉州港曾接待外国商船。",
            embedding=_EMBEDDING,
            metadata={
                "document_id": "doc-haisi",
                "title": "海丝通商",
                "culture_domain": "海丝",
            },
        ),
        Chunk(
            chunk_id=_SHIP_CHUNK_ID,
            document_id="doc-ship",
            text=f"{_SHARED_TERM}支撑了船政物资进口，马尾口岸与船政建设密切相关。",
            embedding=_EMBEDDING,
            metadata={
                "document_id": "doc-ship",
                "title": "船政通商",
                "culture_domain": "船政",
            },
        ),
    ]
    store.upsert(chunks)
    index.upsert(chunks)
    index.save()


def _candidate_chunk_ids(trace: dict[str, object], stage_name: str) -> list[str]:
    stage = next(item for item in trace["stages"] if item["name"] == stage_name)
    return [item["chunk_id"] for item in stage.get("candidates") or []]


def test_ask_with_culture_domain_filters_dense_and_sparse_paths(
    test_settings: Settings,
) -> None:
    _seed_cross_domain_chunks(test_settings)
    client = TestClient(create_app(test_settings))

    response = client.post(
        "/ask",
        json={"question": f"{_SHARED_TERM}的历史意义", "culture_domain": "海丝"},
    )
    assert response.status_code == 200
    body = response.json()

    trace_path = Path(test_settings.paths.traces)
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    query_trace = next(trace for trace in traces if trace["trace_id"] == body["trace_id"])

    processing = next(
        stage for stage in query_trace["stages"] if stage["name"] == "query_processing"
    )
    assert processing["culture_domain"] == "海丝"

    dense_ids = _candidate_chunk_ids(query_trace, "dense")
    sparse_ids = _candidate_chunk_ids(query_trace, "sparse")
    fused_ids = _candidate_chunk_ids(query_trace, "fusion")

    assert _HAISI_CHUNK_ID in dense_ids
    assert _SHIP_CHUNK_ID not in dense_ids
    assert _HAISI_CHUNK_ID in sparse_ids
    assert _SHIP_CHUNK_ID not in sparse_ids
    assert fused_ids == [_HAISI_CHUNK_ID]


def test_ask_without_culture_domain_keeps_cross_domain_retrieval(
    test_settings: Settings,
) -> None:
    _seed_cross_domain_chunks(test_settings)

    result = ask_question(f"{_SHARED_TERM}的历史意义", test_settings)

    trace_path = Path(test_settings.paths.traces)
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    query_trace = next(trace for trace in traces if trace["trace_id"] == result.trace_id)

    processing = next(
        stage for stage in query_trace["stages"] if stage["name"] == "query_processing"
    )
    assert "culture_domain" not in processing

    fused_ids = _candidate_chunk_ids(query_trace, "fusion")
    assert _HAISI_CHUNK_ID in fused_ids
    assert _SHIP_CHUNK_ID in fused_ids
