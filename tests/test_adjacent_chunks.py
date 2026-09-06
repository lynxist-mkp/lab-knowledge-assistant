"""Adjacent chunk expansion at generation time."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab_knowledge.config import Settings
from lab_knowledge.knowledge import create_knowledge
from lab_knowledge.models import AskResult
from lab_knowledge.pipelines.ingestion import ingest_source
from lab_knowledge.pipelines.query_orchestration import AskPipelineInput, ask_pipeline_single
from lab_knowledge.tracing.store import get_trace_record


def _write_three_paragraph_markdown(path: Path) -> Path:
    path.write_text(
        """---
title: 三段落测试文档
culture_domain: 妈祖
---

段落甲开头内容关于妈祖信仰的起源与历史传承。

段落乙中间内容包含独特关键词朱子文化理学思想。

段落丙结尾内容描述祖庙祭典与信众朝拜活动。
""",
        encoding="utf-8",
    )
    return path


def _run_query_pipeline(
    question: str,
    settings: Settings,
    *,
    retrieval_mode: str | None = None,
    rerank_enabled: bool | None = None,
) -> AskResult:
    return ask_pipeline_single(
        AskPipelineInput(
            question=question,
            settings=settings,
            retrieval_mode=retrieval_mode,
            rerank_enabled=rerank_enabled,
        ),
        phase_batch=settings.resources.query_phase_batch,
    )


def test_ingest_writes_sequential_chunk_index(
    test_settings: Settings, tmp_path: Path
) -> None:
    test_settings.chunking.size_chars = 40
    test_settings.chunking.overlap_ratio = 0.0
    source = _write_three_paragraph_markdown(tmp_path / "three.md")

    result = ingest_source(source, test_settings)
    assert result.chunk_count >= 3

    chunks = create_knowledge(test_settings).get_by_document_id(result.document_id)
    chunks.sort(key=lambda c: c.metadata["chunk_index"])
    indices = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indices == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.metadata["document_id"] == result.document_id


class _PromptCapturingLLM:
    def __init__(self) -> None:
        self.last_prompt: str = ""

    @property
    def provider_name(self) -> str:
        return "capturing"

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "根据材料，朱子文化理学思想有相关记载[1]。"


def test_query_expands_neighbors_into_generation_prompt(
    test_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_settings.chunking.size_chars = 40
    test_settings.chunking.overlap_ratio = 0.0
    test_settings.retrieval.adjacent_n = 1
    test_settings.retrieval.rerank_enabled = False
    source = _write_three_paragraph_markdown(tmp_path / "three.md")
    ingest_result = ingest_source(source, test_settings)
    knowledge = create_knowledge(test_settings)
    doc_chunks = knowledge.get_by_document_id(ingest_result.document_id)
    doc_chunks.sort(key=lambda c: c.metadata["chunk_index"])
    middle = doc_chunks[1]
    assert "朱子文化理学思想" in middle.text

    capturer = _PromptCapturingLLM()
    monkeypatch.setattr(
        "lab_knowledge.factories.multimodal.create",
        lambda settings: capturer,
    )

    result = _run_query_pipeline(
        "朱子文化理学思想的独特关键词是什么？",
        test_settings,
        retrieval_mode="sparse_only",
        rerank_enabled=False,
    )

    assert "段落甲" in capturer.last_prompt
    assert "朱子文化理学思想" in capturer.last_prompt
    assert "段落丙" in capturer.last_prompt
    assert result.citations[0].chunk_id == middle.chunk_id

    record = get_trace_record(test_settings, result.trace_id)
    assert record is not None
    gen_stage = next(s for s in record["stages"] if s["name"] == "generation")
    assert gen_stage["expanded_from"] == [middle.chunk_id]
    expanded_ids = gen_stage["expanded_chunk_ids"]
    assert middle.chunk_id in expanded_ids
    assert doc_chunks[0].chunk_id in expanded_ids
    assert doc_chunks[2].chunk_id in expanded_ids

    fusion_stage = next(
        (s for s in record["stages"] if s["name"] in ("fusion", "sparse")),
        None,
    )
    assert fusion_stage is not None
    fusion_ids = [c["chunk_id"] for c in fusion_stage["candidates"]]
    assert middle.chunk_id in fusion_ids
