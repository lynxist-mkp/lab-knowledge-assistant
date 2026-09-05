"""TraceRecorder round-trip: typed write path and read-side view compatibility."""

from __future__ import annotations

import json

from wenmai.config import Settings
from wenmai.ops.observation import get_query_detail, list_query_summaries
from wenmai.tracing.recorder import QUERY_TRACE_SCHEMA_VERSION, TraceRecorder
from wenmai.tracing.stage_result import STAGE_SCHEMA_VERSION, stage_as_dict, stage_from_dict, stage_to_dict
from wenmai.tracing.store import get_trace_record
from wenmai.storage.paths import store_path
from wenmai.tracing.stages.query import QueryStage


def test_stage_result_round_trip(test_settings: Settings) -> None:
    original = QueryStage.query_processing(
        question="妈祖信仰的发源地在哪里？",
        normalized="妈祖 信仰 发源地",
        elapsed_ms=12.5,
        culture_domain="妈祖",
    )
    restored = stage_from_dict(stage_to_dict(original))
    assert restored.name == "query_processing"
    assert restored.culture_domain == "妈祖"
    assert restored.elapsed_ms == 12.5


def test_stage_as_dict_supports_old_and_new_shapes() -> None:
    old = {"name": "dense", "method": "vector_query", "elapsed_ms": 1.0}
    new = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "name": "dense",
        "method": "vector_query",
        "elapsed_ms": 1.0,
    }
    wrapped = {"stage": new}

    assert stage_as_dict(old)["name"] == "dense"
    assert "schema_version" not in stage_as_dict(new)
    assert stage_as_dict(wrapped)["name"] == "dense"


def test_trace_recorder_writes_typed_schema(test_settings: Settings) -> None:
    recorder = TraceRecorder(question="妈祖信仰的发源地在哪里？")
    recorder.record_query_processing(
        question="妈祖信仰的发源地在哪里？",
        normalized="妈祖 信仰 发源地",
        elapsed_ms=2.0,
        culture_domain="妈祖",
        term_extras=[],
        multi_query_extras=[],
        rewriter="local",
        multi_query_provider=None,
    )
    recorder.append_retrieval_stages(
        [
            QueryStage.dense(
                knowledge=_FakeKnowledge(),
                settings=test_settings,
                scored_chunks=[],
                elapsed_ms=80.0,
            ),
            QueryStage.sparse(
                settings=test_settings,
                scored_chunks=[],
                elapsed_ms=12.0,
            ),
            QueryStage.fusion(
                settings=test_settings,
                dense_chunks=[],
                sparse_chunks=[],
                fused_chunks=[],
                elapsed_ms=1.0,
            ),
        ]
    )
    recorder.append_rerank_stages(
        [
            QueryStage.rerank_success(
                provider="fake",
                elapsed_ms=45.0,
                pre_rerank=[],
                reranked=[],
                rerank_top=5,
                rank_changes=[{"chunk_id": "doc:0002", "from": 2, "to": 1}],
            )
        ]
    )
    recorder.record_generation(
        provider="fake",
        elapsed_ms=180.0,
        input_summary="3 chunks",
        output_summary="answer text",
        candidate_count=1,
    )
    recorder.set_outcome(refused=False, refusal_reason=None, citation_count=1)
    recorder.save(test_settings)

    raw = get_trace_record(test_settings, recorder.trace_id)
    assert raw is not None
    assert raw.get("schema_version") == QUERY_TRACE_SCHEMA_VERSION
    assert all(
        stage.get("schema_version") == STAGE_SCHEMA_VERSION
        for stage in raw.get("stages") or []
        if isinstance(stage, dict)
    )

    summaries = list_query_summaries(test_settings)
    assert summaries[0].trace_id == recorder.trace_id
    assert summaries[0].question == "妈祖信仰的发源地在哪里？"
    assert summaries[0].culture_domain == "妈祖"
    assert summaries[0].status == "ok"

    detail = get_query_detail(test_settings, recorder.trace_id)
    assert detail is not None
    labels = [item.label for item in detail.stage_latencies]
    assert labels == ["查询处理", "嵌入检索", "稀疏检索", "融合", "精排", "生成"]
    assert detail.rank_changes[0].chunk_id == "doc:0002"


def test_trace_recorder_reads_old_fixture_json(test_settings: Settings) -> None:
    trace_path = store_path(test_settings, "traces")
    old_trace = {
        "trace_id": "old-fixture-id",
        "trace_type": "query",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "total_elapsed_ms": 100.0,
        "stages": [
            {
                "name": "query_processing",
                "method": "normalize",
                "provider": "local",
                "elapsed_ms": 2.0,
                "input_summary": "测试问题",
                "culture_domain": "妈祖",
            },
            {
                "name": "generation",
                "method": "llm",
                "provider": "fake",
                "elapsed_ms": 50.0,
                "output_summary": "done",
            },
        ],
        "error": None,
        "metadata": {
            "question": "测试问题",
            "outcome": {"refused": False, "refusal_reason": None, "citation_count": 0},
        },
    }
    trace_path.write_text(json.dumps(old_trace, ensure_ascii=False) + "\n", encoding="utf-8")

    summaries = list_query_summaries(test_settings)
    assert summaries[0].trace_id == "old-fixture-id"
    assert summaries[0].question == "测试问题"
    assert summaries[0].culture_domain == "妈祖"

    detail = get_query_detail(test_settings, "old-fixture-id")
    assert detail is not None
    assert detail.stage_latencies[0].elapsed_ms == 2.0


class _FakeKnowledge:
    dense_provider = "fake"
