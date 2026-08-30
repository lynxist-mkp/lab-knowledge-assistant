"""查询缝：Multi-Query 在术语归一之后增加 LLM 改写 RRF 路径。"""

from __future__ import annotations

from pathlib import Path

import yaml

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import Chunk
from wenmai.pipelines.query import ask_question
from wenmai.retrieval import retrieve
from wenmai.tracing.store import get_trace_record


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


def _settings(
    test_settings: Settings,
    tmp_path: Path,
    *,
    rewriter: str = "none",
    multi_query: bool = False,
    multi_query_n: int = 3,
    lexicon: dict[str, str] | None = None,
    multimodal_behavior: str = "ok",
) -> Settings:
    if lexicon is not None:
        path = tmp_path / "synonyms.yaml"
        path.write_text(yaml.safe_dump(lexicon, allow_unicode=True), encoding="utf-8")
        lexicon_file = str(path)
    else:
        lexicon_file = str(
            Path(__file__).resolve().parents[1] / "data/lexicon/synonyms.yaml"
        )
    test_settings.query_processing.rewriter = rewriter
    test_settings.query_processing.lexicon = lexicon_file
    test_settings.query_processing.multi_query = multi_query
    test_settings.query_processing.multi_query_n = multi_query_n
    test_settings.fakes["multimodal"] = multimodal_behavior
    return test_settings


def test_multi_query_extra_path_hits_fusion(test_settings: Settings, tmp_path: Path) -> None:
    _settings(test_settings, tmp_path, multi_query=True)
    knowledge = create_knowledge(test_settings)
    _commit(
        knowledge,
        "doc-ship",
        "马尾船政学堂创办于1866年，是中国最早的近代海军学堂。",
    )

    disabled = retrieve(
        "这所院校始于哪一年？",
        test_settings,
        rerank_enabled=False,
        retrieval_mode="sparse_only",
        knowledge=knowledge,
        extra_queries=[],
    )
    assert disabled.chunks == []

    enabled = retrieve(
        "这所院校始于哪一年？",
        test_settings,
        rerank_enabled=False,
        retrieval_mode="sparse_only",
        knowledge=knowledge,
        extra_queries=["马尾船政学堂是哪年创立的？"],
    )
    assert enabled.chunks
    assert enabled.chunks[0].chunk.chunk_id == "doc-ship:0000"


def test_multi_query_disabled_misses_doc(test_settings: Settings, tmp_path: Path) -> None:
    _settings(test_settings, tmp_path, multi_query=False)
    knowledge = create_knowledge(test_settings)
    _commit(
        knowledge,
        "doc-ship",
        "马尾船政学堂创办于1866年，是中国最早的近代海军学堂。",
    )

    missed = ask_question(
        "这所院校始于哪一年？",
        test_settings,
        rerank_enabled=False,
        retrieval_mode="sparse_only",
        knowledge=knowledge,
    )
    assert missed.ranked_chunks == []


def test_multi_query_enabled_hits_doc(test_settings: Settings, tmp_path: Path) -> None:
    _settings(test_settings, tmp_path, multi_query=True)
    knowledge = create_knowledge(test_settings)
    _commit(
        knowledge,
        "doc-ship",
        "马尾船政学堂创办于1866年，是中国最早的近代海军学堂。",
    )

    hit = ask_question(
        "这所院校始于哪一年？",
        test_settings,
        rerank_enabled=False,
        retrieval_mode="sparse_only",
        knowledge=knowledge,
    )
    assert hit.ranked_chunks
    assert hit.ranked_chunks[0].chunk.chunk_id == "doc-ship:0000"


def test_multi_query_timeout_fallback(test_settings: Settings, tmp_path: Path) -> None:
    _settings(test_settings, tmp_path, multi_query=True, multimodal_behavior="timeout")
    knowledge = create_knowledge(test_settings)
    _commit(
        knowledge,
        "doc-ship",
        "马尾船政学堂创办于1866年，是中国最早的近代海军学堂。",
    )

    disabled = _settings(test_settings, tmp_path, multi_query=False)
    baseline = ask_question(
        "这所院校始于哪一年？",
        disabled,
        rerank_enabled=False,
        retrieval_mode="sparse_only",
        knowledge=knowledge,
    )

    timed_out = ask_question(
        "这所院校始于哪一年？",
        test_settings,
        rerank_enabled=False,
        retrieval_mode="sparse_only",
        knowledge=knowledge,
    )
    assert timed_out.ranked_chunks == baseline.ranked_chunks

    record = get_trace_record(test_settings, timed_out.trace_id)
    assert record is not None
    processing = next(s for s in record["stages"] if s["name"] == "query_processing")
    assert processing["method"] == "normalize"
    assert processing["candidate_count"] == 1


def test_lexicon_survives_multi_query_timeout(
    test_settings: Settings, tmp_path: Path
) -> None:
    _settings(
        test_settings,
        tmp_path,
        rewriter="lexicon",
        multi_query=True,
        multimodal_behavior="timeout",
        lexicon={"马尾学堂": "船政学堂"},
    )
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-ship", "船政学堂创办于1866年，是中国最早的近代海军学堂。")

    result = ask_question(
        "马尾学堂哪年办的？",
        test_settings,
        rerank_enabled=False,
        retrieval_mode="sparse_only",
        knowledge=knowledge,
    )
    assert result.ranked_chunks
    assert result.ranked_chunks[0].chunk.chunk_id == "doc-ship:0000"

    record = get_trace_record(test_settings, result.trace_id)
    assert record is not None
    processing = next(s for s in record["stages"] if s["name"] == "query_processing")
    assert processing["method"] == "term-normalize"
    assert "船政学堂" in processing["output_summary"]
    assert processing["candidate_count"] == 2


def test_multi_query_cap(test_settings: Settings, tmp_path: Path) -> None:
    _settings(test_settings, tmp_path, multi_query=True, multi_query_n=2)
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "占位文本。")

    result = ask_question(
        "这所院校始于哪一年？",
        test_settings,
        rerank_enabled=False,
        knowledge=knowledge,
    )
    record = get_trace_record(test_settings, result.trace_id)
    assert record is not None
    processing = next(s for s in record["stages"] if s["name"] == "query_processing")
    assert processing["method"] == "multi-query"
    assert processing["candidate_count"] == 3  # original + 2 LLM extras


def test_trace_records_rewrite_and_path_counts(
    test_settings: Settings, tmp_path: Path
) -> None:
    _settings(test_settings, tmp_path, multi_query=True)
    knowledge = create_knowledge(test_settings)
    _commit(
        knowledge,
        "doc-ship",
        "马尾船政学堂创办于1866年，是中国最早的近代海军学堂。",
    )

    result = ask_question(
        "这所院校始于哪一年？",
        test_settings,
        rerank_enabled=False,
        retrieval_mode="rrf",
        knowledge=knowledge,
    )
    record = get_trace_record(test_settings, result.trace_id)
    assert record is not None
    processing = next(s for s in record["stages"] if s["name"] == "query_processing")
    assert processing["method"] == "multi-query"
    assert processing["candidate_count"] >= 2
    assert "马尾船政学堂" in processing["output_summary"]

    fusion = next(s for s in record["stages"] if s["name"] == "fusion")
    assert "paths=[" in fusion["input_summary"]
    assert "q0:" in fusion["input_summary"]
    assert "q1:" in fusion["input_summary"]

    dense = next(s for s in record["stages"] if s["name"] == "dense")
    sparse = next(s for s in record["stages"] if s["name"] == "sparse")
    assert dense["candidate_count"] is not None
    assert sparse["candidate_count"] is not None
