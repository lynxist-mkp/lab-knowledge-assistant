"""Fusion module: dense/sparse/RRF modes and timing fields."""

from __future__ import annotations

import pytest

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import Chunk
from wenmai.retrieval.fusion import run_fusion, validate_retrieval_mode


def _chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        metadata={"document_id": document_id, "title": document_id},
    )


def _commit(knowledge, document_id: str, text: str) -> None:
    knowledge.commit_document(
        source_path=f"/tmp/{document_id}.md",
        sha256=document_id,
        document_id=document_id,
        status="ingested",
        chunks=[_chunk(f"{document_id}:0000", document_id, text)],
    )


def test_dense_only_populates_dense_timing(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "船政学堂创办于马尾，是近代海军摇篮。")
    result = run_fusion(
        knowledge, "船政学堂在哪里", mode="dense_only", settings=test_settings
    )
    assert result.mode == "dense_only"
    assert result.chunks
    assert result.dense_chunks == result.chunks
    assert result.dense_elapsed_ms > 0
    assert result.sparse_elapsed_ms == 0.0
    assert result.fusion_elapsed_ms == 0.0
    assert not result.sparse_chunks


def test_sparse_only_populates_sparse_timing(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "船政学堂创办于马尾，是近代海军摇篮。")
    result = run_fusion(
        knowledge, "船政学堂", mode="sparse_only", settings=test_settings
    )
    assert result.mode == "sparse_only"
    assert result.chunks[0].chunk.chunk_id == "doc-a:0000"
    assert result.sparse_chunks == result.chunks
    assert result.sparse_elapsed_ms > 0
    assert result.dense_elapsed_ms == 0.0
    assert result.fusion_elapsed_ms == 0.0
    assert not result.dense_chunks


def test_rrf_mode_fuses_dense_and_sparse(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "船政学堂创办于马尾，是近代海军摇篮。")
    _commit(knowledge, "doc-b", "湄洲祖庙是妈祖信仰的中心。")
    result = run_fusion(knowledge, "船政学堂", mode="rrf", settings=test_settings)
    assert result.mode == "rrf"
    assert result.chunks
    assert result.dense_chunks
    assert result.sparse_chunks
    assert result.dense_elapsed_ms > 0
    assert result.sparse_elapsed_ms > 0
    assert result.fusion_elapsed_ms >= 0


def test_unknown_mode_raises(test_settings: Settings) -> None:
    knowledge = create_knowledge(test_settings)
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        run_fusion(knowledge, "妈祖", mode="clip", settings=test_settings)


def test_validate_retrieval_mode_accepts_known_modes() -> None:
    assert validate_retrieval_mode("rrf") == "rrf"
    assert validate_retrieval_mode("dense_only") == "dense_only"
    assert validate_retrieval_mode("sparse_only") == "sparse_only"
