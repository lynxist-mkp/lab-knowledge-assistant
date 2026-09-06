"""查询缝：术语归一在检索前增加额外 RRF 路径。"""

from __future__ import annotations

from pathlib import Path

import yaml

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import AskResult, Chunk
from wenmai.pipelines.query_orchestration import AskPipelineInput, ask_pipeline_single
from wenmai.query_processing.extras import prepare_query_extras
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


def _settings_with_lexicon(
    test_settings: Settings,
    tmp_path: Path,
    *,
    rewriter: str,
    lexicon: dict[str, str] | None = None,
    lexicon_path: Path | None = None,
) -> Settings:
    if lexicon is not None:
        path = tmp_path / "synonyms.yaml"
        path.write_text(yaml.safe_dump(lexicon, allow_unicode=True), encoding="utf-8")
        lexicon_file = str(path)
    elif lexicon_path is not None:
        lexicon_file = str(lexicon_path)
    else:
        lexicon_file = str(
            Path(__file__).resolve().parents[1] / "data/lexicon/synonyms.yaml"
        )
    test_settings.query_processing.rewriter = rewriter
    test_settings.query_processing.lexicon = lexicon_file
    return test_settings


def _run_query_pipeline(
    question: str,
    settings: Settings,
    *,
    rerank_enabled: bool | None = None,
    knowledge=None,
    record_trace: bool = True,
) -> AskResult:
    return ask_pipeline_single(
        AskPipelineInput(
            question=question,
            settings=settings,
            rerank_enabled=rerank_enabled,
            knowledge=knowledge,
            record_trace=record_trace,
        ),
        phase_batch=settings.resources.query_phase_batch,
    )


def test_miss_lexicon_identical_retrieval(test_settings: Settings, tmp_path: Path) -> None:
    _settings_with_lexicon(test_settings, tmp_path, rewriter="lexicon", lexicon={})
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "船政学堂创办于1866年，是近代海军摇篮。")

    control = retrieve(
        "船政学堂哪年办的？",
        test_settings,
        knowledge=knowledge,
    )
    _settings_with_lexicon(test_settings, tmp_path, rewriter="none")
    disabled = retrieve(
        "船政学堂哪年办的？",
        test_settings,
        knowledge=knowledge,
    )

    assert [item.chunk.chunk_id for item in control.chunks] == [
        item.chunk.chunk_id for item in disabled.chunks
    ]

    result = _run_query_pipeline(
        "船政学堂哪年办的？",
        test_settings,
        rerank_enabled=False,
        knowledge=knowledge,
        record_trace=True,
    )
    record = get_trace_record(test_settings, result.trace_id)
    assert record is not None
    processing = next(s for s in record["stages"] if s["name"] == "query_processing")
    assert processing["method"] == "normalize"


def test_synonym_extra_path_hits_fusion(test_settings: Settings, tmp_path: Path) -> None:
    _settings_with_lexicon(
        test_settings,
        tmp_path,
        rewriter="lexicon",
        lexicon={"马尾学堂": "船政学堂"},
    )
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-ship", "船政学堂创办于1866年，是中国最早的近代海军学堂。")

    without = _settings_with_lexicon(test_settings, tmp_path, rewriter="none")
    missed = retrieve(
        "马尾学堂哪年办的？",
        without,
        retrieval_mode="sparse_only",
        knowledge=knowledge,
    )
    assert missed.chunks == []

    with_lexicon = _settings_with_lexicon(
        test_settings,
        tmp_path,
        rewriter="lexicon",
        lexicon={"马尾学堂": "船政学堂"},
    )
    hit = retrieve(
        "马尾学堂哪年办的？",
        with_lexicon,
        retrieval_mode="sparse_only",
        knowledge=knowledge,
        extra_queries=prepare_query_extras("马尾学堂哪年办的？", with_lexicon).combined,
    )
    assert hit.chunks
    assert hit.chunks[0].chunk.chunk_id == "doc-ship:0000"


def test_trace_records_rewrite_strings(test_settings: Settings, tmp_path: Path) -> None:
    _settings_with_lexicon(
        test_settings,
        tmp_path,
        rewriter="lexicon",
        lexicon={"马尾学堂": "船政学堂"},
    )
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-ship", "船政学堂创办于1866年。")

    result = _run_query_pipeline(
        "马尾学堂哪年办的？",
        test_settings,
        rerank_enabled=False,
        knowledge=knowledge,
    )
    record = get_trace_record(test_settings, result.trace_id)
    assert record is not None
    processing = next(s for s in record["stages"] if s["name"] == "query_processing")
    assert processing["method"] == "term-normalize"
    assert processing["input_summary"] == "马尾学堂哪年办的？"
    assert "马尾学堂" in processing["output_summary"]
    assert "船政学堂" in processing["output_summary"]
    assert processing["candidate_count"] == 2


def test_failure_open_missing_lexicon_file(
    test_settings: Settings, tmp_path: Path
) -> None:
    _settings_with_lexicon(
        test_settings,
        tmp_path,
        rewriter="lexicon",
        lexicon_path=tmp_path / "missing-synonyms.yaml",
    )
    knowledge = create_knowledge(test_settings)
    _commit(knowledge, "doc-a", "船政学堂创办于1866年。")

    result = _run_query_pipeline(
        "船政学堂哪年办的？",
        test_settings,
        rerank_enabled=False,
        knowledge=knowledge,
    )
    assert result.ranked_chunks
    assert result.ranked_chunks[0].chunk.chunk_id == "doc-a:0000"
