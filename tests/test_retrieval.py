"""检索 interface：按检索方式返回排好的片段和 Trace 阶段载荷。"""

from __future__ import annotations

from pathlib import Path

import pytest

from wenmai.config import Settings
from wenmai.pipelines.ingestion import ingest_source
from wenmai.retrieval import retrieve


def _write_minpai_markdown(path: Path) -> Path:
    path.write_text(
        """---
title: 湄洲妈祖祖庙简介
culture_domain: 妈祖
---

湄洲岛是妈祖信仰的发源地。祖庙坐落在湄洲岛上，是信俗活动的中心场所。
""",
        encoding="utf-8",
    )
    return path


def test_rrf_retrieve_returns_chunks_and_fusion_stage(
    test_settings: Settings, tmp_path: Path
) -> None:
    ingest_source(_write_minpai_markdown(tmp_path / "matsu.md"), test_settings)
    result = retrieve("妈祖信仰的发源地在哪里？", test_settings, retrieval_mode="rrf")
    assert result.chunks
    names = [stage.name for stage in result.stages]
    assert names[:3] == ["dense", "sparse", "fusion"]
    assert "妈祖" in result.chunks[0].chunk.text


def test_sparse_only_omits_dense_stage(
    test_settings: Settings, tmp_path: Path
) -> None:
    ingest_source(_write_minpai_markdown(tmp_path / "matsu.md"), test_settings)
    result = retrieve(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        retrieval_mode="sparse_only",
    )
    assert result.chunks
    assert [stage.name for stage in result.stages] == ["sparse"]


def test_dense_only_omits_sparse_stage(
    test_settings: Settings, tmp_path: Path
) -> None:
    ingest_source(_write_minpai_markdown(tmp_path / "matsu.md"), test_settings)
    result = retrieve(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        retrieval_mode="dense_only",
    )
    assert result.chunks
    assert [stage.name for stage in result.stages] == ["dense"]


def test_unknown_retrieval_mode_raises(test_settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown retrieval_mode"):
        retrieve("妈祖", test_settings, retrieval_mode="clip")


def test_rerank_failure_still_returns_fused_chunks(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["reranker"] = "error"
    ingest_source(_write_minpai_markdown(tmp_path / "matsu.md"), test_settings)
    result = retrieve("妈祖信仰的发源地在哪里？", test_settings, rerank_enabled=True)
    assert result.chunks
    rerank = next(stage for stage in result.stages if stage.name == "rerank")
    assert rerank.method == "rrf_fallback"
    assert rerank.fallback_reason


def test_retrieve_excludes_pending_chunks_until_approved(
    test_settings: Settings, tmp_path: Path
) -> None:
    from wenmai.knowledge import create_knowledge

    source = _write_minpai_markdown(tmp_path / "matsu.md")
    result = ingest_source(source, test_settings)
    knowledge = create_knowledge(test_settings)
    knowledge.set_review_status(result.document_id, "待审")

    pending = retrieve("妈祖信仰的发源地在哪里？", test_settings, knowledge=knowledge)
    assert not pending.chunks

    knowledge.set_review_status(result.document_id, "已通过")
    approved = retrieve("妈祖信仰的发源地在哪里？", test_settings, knowledge=knowledge)
    assert approved.chunks
    assert "妈祖" in approved.chunks[0].chunk.text
