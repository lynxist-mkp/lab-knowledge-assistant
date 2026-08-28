"""生成：回答、拒答与引用解析。"""

from __future__ import annotations

import pytest

from wenmai.config import Settings
from wenmai.generation import generate
from wenmai.models import Chunk, ScoredChunk


def _scored_chunk(
    index: int,
    *,
    text: str | None = None,
    title: str | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"chunk-{index}",
            document_id="doc-1",
            text=text or f"片段{index}关于妈祖信仰的内容。",
            metadata={"title": title or f"标题{index}"},
        ),
        score=1.0 - index * 0.01,
    )


class _StubLLM:
    def __init__(self, answer: str, *, provider_name: str = "stub") -> None:
        self._answer = answer
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def generate(self, prompt: str) -> str:
        return self._answer


def test_generate_normal_answer_with_citations(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_scored_chunk(1, text="湄洲岛是妈祖信仰的发源地。")]
    monkeypatch.setattr(
        "wenmai.factories.multimodal.create",
        lambda settings: _StubLLM("湄洲岛是妈祖信仰的发源地[1]。"),
    )

    result = generate("妈祖信仰的发源地在哪里？", chunks, test_settings)

    assert result.refused is False
    assert result.provider_name == "stub"
    assert len(result.citations) == 1
    assert result.citations[0].index == 1
    assert result.citations[0].chunk_id == "chunk-1"
    assert "妈祖" in result.citations[0].excerpt


def test_generate_refusal_returns_all_retrieved_citations(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [
        _scored_chunk(1, text="湄洲岛是妈祖信仰的发源地。"),
        _scored_chunk(2, text="祖庙是信俗活动的中心场所。"),
    ]
    monkeypatch.setattr(
        "wenmai.factories.multimodal.create",
        lambda settings: _StubLLM("拒答：检索片段不足以回答该问题。"),
    )

    result = generate("船政学堂是什么时候创办的？", chunks, test_settings)

    assert result.refused is True
    assert result.output_summary == "refusal"
    assert len(result.citations) == 2
    assert [citation.index for citation in result.citations] == [1, 2]


def test_generate_empty_chunks_yields_no_citations(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "wenmai.factories.multimodal.create",
        lambda settings: _StubLLM("暂无相关内容[1]。"),
    )

    result = generate("任意问题", [], test_settings)

    assert result.refused is False
    assert result.citations == []
    assert result.candidate_count == 0


def test_generate_ignores_out_of_range_citation_indices(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_scored_chunk(1)]
    monkeypatch.setattr(
        "wenmai.factories.multimodal.create",
        lambda settings: _StubLLM("答案引用[1]与无效[99]。"),
    )

    result = generate("问题", chunks, test_settings)

    assert result.refused is False
    assert len(result.citations) == 1
    assert result.citations[0].index == 1
