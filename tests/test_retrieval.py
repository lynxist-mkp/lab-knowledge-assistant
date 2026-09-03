"""检索 interface：run_fusion 返回排好的片段；Trace 在编排 seam 附着。"""

from __future__ import annotations

from pathlib import Path

import pytest

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.pipelines.ingestion import ingest_source
from wenmai.pipelines.rerank import rerank_chunks
from wenmai.retrieval import attach_retrieval_trace_stages, resolve_retrieval_mode, run_fusion


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


def _retrieve_with_stages(
    question: str,
    settings: Settings,
    *,
    retrieval_mode: str | None = None,
    knowledge=None,
    extra_queries: list[str] | None = None,
):
    knowledge = knowledge or create_knowledge(settings)
    mode, _ = resolve_retrieval_mode(settings, retrieval_mode, None)
    fusion = run_fusion(
        knowledge,
        question,
        mode=mode,
        settings=settings,
        extra_queries=extra_queries or [],
    )
    return attach_retrieval_trace_stages(
        fusion,
        settings,
        knowledge=knowledge,
    )


def test_rrf_retrieve_returns_chunks_and_fusion_stage(
    test_settings: Settings, tmp_path: Path
) -> None:
    ingest_source(_write_minpai_markdown(tmp_path / "matsu.md"), test_settings)
    result = _retrieve_with_stages("妈祖信仰的发源地在哪里？", test_settings, retrieval_mode="rrf")
    assert result.chunks
    names = [stage.name for stage in result.stages]
    assert names[:3] == ["dense", "sparse", "fusion"]
    assert "妈祖" in result.chunks[0].chunk.text


def test_sparse_only_omits_dense_stage(
    test_settings: Settings, tmp_path: Path
) -> None:
    ingest_source(_write_minpai_markdown(tmp_path / "matsu.md"), test_settings)
    result = _retrieve_with_stages(
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
    result = _retrieve_with_stages(
        "妈祖信仰的发源地在哪里？",
        test_settings,
        retrieval_mode="dense_only",
    )
    assert result.chunks
    assert [stage.name for stage in result.stages] == ["dense"]


def test_unknown_retrieval_mode_raises(test_settings: Settings) -> None:
    with pytest.raises(ValueError, match="unknown retrieval_mode"):
        resolve_retrieval_mode(test_settings, "clip", None)


def test_rerank_failure_still_returns_fused_chunks(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.fakes["reranker"] = "error"
    ingest_source(_write_minpai_markdown(tmp_path / "matsu.md"), test_settings)
    result = _retrieve_with_stages("妈祖信仰的发源地在哪里？", test_settings)
    assert result.chunks
    assert not any(stage.name == "rerank" for stage in result.stages)

    reranked, rerank_stage = rerank_chunks(
        test_settings,
        "妈祖信仰的发源地在哪里？",
        result.chunks,
    )
    assert reranked
    assert rerank_stage is not None
    assert rerank_stage.method == "rrf_fallback"
    assert rerank_stage.fallback_reason


def test_retrieve_excludes_pending_chunks_until_approved(
    test_settings: Settings, tmp_path: Path
) -> None:
    source = _write_minpai_markdown(tmp_path / "matsu.md")
    result = ingest_source(source, test_settings)
    knowledge = create_knowledge(test_settings)
    knowledge.set_review_status(result.document_id, "待审")

    mode, _ = resolve_retrieval_mode(test_settings, None, None)
    pending = run_fusion(
        knowledge,
        "妈祖信仰的发源地在哪里？",
        mode=mode,
        settings=test_settings,
    )
    assert not pending.chunks

    knowledge.set_review_status(result.document_id, "已通过")
    approved = run_fusion(
        knowledge,
        "妈祖信仰的发源地在哪里？",
        mode=mode,
        settings=test_settings,
    )
    assert approved.chunks
    assert "妈祖" in approved.chunks[0].chunk.text
