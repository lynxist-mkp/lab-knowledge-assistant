"""Culture domain filter: cross-domain same-term chunks must not leak."""

from __future__ import annotations

from fastapi.testclient import TestClient

from lab_knowledge.app import create_app
from lab_knowledge.config import Settings
from lab_knowledge.http.ask_service import run_ask
from lab_knowledge.knowledge import create_knowledge
from lab_knowledge.models import Chunk

_SHARED_TERM = "通商口岸"
_HAISI_CHUNK_ID = "doc-haisi:0000"
_SHIP_CHUNK_ID = "doc-ship:0000"


def _seed_cross_domain_chunks(settings: Settings) -> None:
    knowledge = create_knowledge(settings)
    knowledge.commit_document(
        source_path="/tmp/doc-haisi.md",
        sha256="doc-haisi",
        document_id="doc-haisi",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id=_HAISI_CHUNK_ID,
                document_id="doc-haisi",
                text=f"{_SHARED_TERM}是海丝贸易的重要节点，泉州港曾接待外国商船。",
                metadata={
                    "document_id": "doc-haisi",
                    "title": "海丝通商",
                    "culture_domain": "海丝",
                },
            ),
        ],
    )
    knowledge.commit_document(
        source_path="/tmp/doc-ship.md",
        sha256="doc-ship",
        document_id="doc-ship",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id=_SHIP_CHUNK_ID,
                document_id="doc-ship",
                text=f"{_SHARED_TERM}支撑了船政物资进口，马尾口岸与船政建设密切相关。",
                metadata={
                    "document_id": "doc-ship",
                    "title": "船政通商",
                    "culture_domain": "船政",
                },
            ),
        ],
    )


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
    citation_ids = {item["chunk_id"] for item in body["citations"]}
    assert _HAISI_CHUNK_ID in citation_ids
    assert _SHIP_CHUNK_ID not in citation_ids


def test_ask_without_culture_domain_keeps_cross_domain_retrieval(
    test_settings: Settings,
) -> None:
    _seed_cross_domain_chunks(test_settings)

    result = run_ask(f"{_SHARED_TERM}的历史意义", test_settings)
    ranked_ids = [item.chunk.chunk_id for item in result.ranked_chunks]
    assert _HAISI_CHUNK_ID in ranked_ids
    assert _SHIP_CHUNK_ID in ranked_ids
