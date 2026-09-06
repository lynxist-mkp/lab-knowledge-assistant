"""运维观测 seam：概览与 Trace 读。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.knowledge.read import ReadPath
from wenmai.models import Chunk
from wenmai.ops.observation import (
    get_query_detail,
    get_task_investigation,
    has_running_long_tasks,
    list_query_summaries,
    list_task_progress_summaries,
    load_health_snapshot,
    load_overview_stats,
)
from wenmai.storage.catalog import DocumentCatalog
from wenmai.task_progress import ChildEvidence, TaskCounters, persist_task_progress


def test_load_overview_stats_empty_catalog(test_settings: Settings) -> None:
    stats = load_overview_stats(test_settings)
    assert stats.document_count == 0
    assert stats.chunk_count == 0


def test_load_overview_stats_routes_to_alternate_collection(test_settings: Settings) -> None:
    other_id = "other-collection"
    other_settings = replace(
        test_settings,
        product=replace(test_settings.product, collection=other_id),
    )
    other_knowledge = create_knowledge(other_settings)
    other_knowledge.commit_document(
        source_path="/tmp/overview-other.md",
        sha256="overview-other",
        document_id="overview-other",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="overview-other:0000",
                document_id="overview-other",
                text="其他集合概览",
                metadata={
                    "document_id": "overview-other",
                    "title": "overview-other",
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            )
        ],
    )

    default_stats = load_overview_stats(test_settings)
    other_stats = load_overview_stats(test_settings, collection_id=other_id)

    assert default_stats.document_count == 0
    assert other_stats.document_count == 1
    assert other_stats.chunk_count == 1


def test_load_overview_stats_counts_via_knowledge_read_seam(
    test_settings: Settings,
) -> None:
    class _FakeRead:
        document_count = 1
        chunk_count = 1

    class _FakeCatalog:
        @classmethod
        def from_settings(cls, _settings: Settings) -> _FakeCatalog:
            return cls()

    original_catalog = DocumentCatalog.from_settings
    original_catalog_only = ReadPath.catalog_only
    try:
        DocumentCatalog.from_settings = _FakeCatalog.from_settings  # type: ignore[method-assign]
        ReadPath.catalog_only = classmethod(lambda cls, _catalog: _FakeRead())  # type: ignore[method-assign]
        stats = load_overview_stats(test_settings)
    finally:
        DocumentCatalog.from_settings = original_catalog  # type: ignore[method-assign]
        ReadPath.catalog_only = original_catalog_only  # type: ignore[method-assign]

    assert stats.document_count == 1
    assert stats.chunk_count == 1


def test_list_query_summaries_via_observation_seam(
    test_settings: Settings, tmp_path: Path
) -> None:
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
    trace = {
        "trace_id": "obs-read-1",
        "trace_type": "query",
        "started_at": started.isoformat(),
        "finished_at": "2026-06-01T12:00:01+00:00",
        "total_elapsed_ms": 42.0,
        "stages": [],
        "error": None,
        "metadata": {"question": "闽派文化是什么？"},
    }
    trace_path.write_text(json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8")

    summaries = list_query_summaries(test_settings)
    assert len(summaries) == 1
    assert summaries[0].trace_id == "obs-read-1"

    detail = get_query_detail(test_settings, "obs-read-1")
    assert detail is not None
    assert detail.summary.question == "闽派文化是什么？"


def test_task_investigation_links_trace_and_document(test_settings: Settings) -> None:
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace = {
        "trace_id": "ing-trace-1",
        "trace_type": "ingestion",
        "started_at": "2026-06-01T12:00:00+00:00",
        "finished_at": "2026-06-01T12:00:02+00:00",
        "total_elapsed_ms": 120.0,
        "stages": [],
        "error": None,
        "metadata": {
            "status": "ingested",
            "document_id": "doc-1",
            "source_path": "/tmp/doc-1.md",
            "chunk_count": 1,
            "chunks_with_images": 0,
        },
    }
    trace_path.write_text(json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8")
    persist_task_progress(
        test_settings,
        task_id="ingestion:ing-trace-1",
        task_type="ingestion",
        status="succeeded",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at="2026-06-01T12:00:02+00:00",
        last_progress_at="2026-06-01T12:00:02+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        config_snapshot={"pdf_load_mode": "auto"},
        links={"trace_id": "ing-trace-1", "document_id": "doc-1"},
        counters=TaskCounters(total=1, completed=1),
    )

    investigation = get_task_investigation(test_settings, "ingestion:ing-trace-1")
    assert investigation is not None
    assert investigation.trace_summaries
    assert investigation.trace_summaries[0].as_dict()["trace_id"] == "ing-trace-1"
    link_types = {(item.link_type, item.target_id) for item in investigation.links}
    assert ("trace", "ing-trace-1") in link_types
    assert ("document", "doc-1") in link_types


def test_task_investigation_links_query_trace_and_config_peers(
    test_settings: Settings,
) -> None:
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace = {
        "trace_id": "query-trace-1",
        "trace_type": "query",
        "started_at": "2026-06-01T12:10:00+00:00",
        "finished_at": "2026-06-01T12:10:01+00:00",
        "total_elapsed_ms": 80.0,
        "stages": [],
        "error": None,
        "metadata": {"question": "妈祖信仰的发源地在哪里？"},
    }
    trace_path.write_text(json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8")
    persist_task_progress(
        test_settings,
        task_id="evaluation:query-hop",
        task_type="evaluation",
        status="failed",
        started_at="2026-06-01T12:10:00+00:00",
        finished_at="2026-06-01T12:10:01+00:00",
        last_progress_at="2026-06-01T12:10:01+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf"]},
        links={"trace_id": "query-trace-1"},
        counters=TaskCounters(total=1, failed=1),
        error="dependency failure",
        failure_kind="dependency",
    )
    persist_task_progress(
        test_settings,
        task_id="evaluation:query-hop-peer",
        task_type="evaluation",
        status="succeeded",
        started_at="2026-06-01T12:11:00+00:00",
        finished_at="2026-06-01T12:11:01+00:00",
        last_progress_at="2026-06-01T12:11:01+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf"]},
        counters=TaskCounters(total=1, completed=1),
        failure_kind="none",
    )

    investigation = get_task_investigation(test_settings, "evaluation:query-hop")
    assert investigation is not None
    assert investigation.trace_summaries
    assert investigation.trace_summaries[0].as_dict()["trace_type"] == "query"
    assert investigation.config_related_tasks[0].task_id == "evaluation:query-hop-peer"
    link_types = {(item.link_type, item.target_id) for item in investigation.links}
    assert ("trace", "query-trace-1") in link_types


def test_task_investigation_links_eval_artifact(
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True)
    test_settings.evaluation.runs = str(runs_dir)
    timestamp = "2026-06-01T13:00:00+00:00"
    artifact = {
        "timestamp": timestamp,
        "item_count": 1,
        "golden_set": "golden.jsonl",
        "failed_count": 0,
        "failures": [],
        "groups": {
            "rrf": {
                "metrics": {
                    "hit_at_5": 1.0,
                    "mrr": 1.0,
                    "refusal_accuracy": 1.0,
                    "citation_coverage": 1.0,
                    "answerable_count": 1,
                    "unanswerable_count": 0,
                }
            }
        },
    }
    (runs_dir / f"{timestamp}.json").write_text(
        json.dumps(artifact, ensure_ascii=False),
        encoding="utf-8",
    )
    persist_task_progress(
        test_settings,
        task_id=f"evaluation:{timestamp}",
        task_type="evaluation",
        status="succeeded",
        started_at=timestamp,
        finished_at=timestamp,
        last_progress_at=timestamp,
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf"]},
        links={"eval_run": timestamp},
        counters=TaskCounters(total=1, completed=1),
        children=[
            ChildEvidence(
                child_id="rrf:g001",
                child_type="eval_item",
                label="rrf/g001",
                status="succeeded",
                failure_kind="none",
                summary="sample question",
                links={"eval_run": timestamp, "group": "rrf", "item_id": "g001"},
                detail={"evidence_doc_ids": ["doc-a"], "answerable": True},
            )
        ],
    )

    investigation = get_task_investigation(test_settings, f"evaluation:{timestamp}")
    assert investigation is not None
    assert investigation.eval_run is not None
    assert investigation.eval_run.timestamp == timestamp
    link_types = {(item.link_type, item.target_id) for item in investigation.links}
    assert ("eval_run", timestamp) in link_types
    assert ("eval_item", "g001") in link_types
    assert ("document", "doc-a") in link_types


def test_health_snapshot_and_task_filters(test_settings: Settings) -> None:
    persist_task_progress(
        test_settings,
        task_id="evaluation:failed",
        task_type="evaluation",
        status="failed",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at="2026-06-01T12:00:10+00:00",
        last_progress_at="2026-06-01T12:00:10+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf"]},
        links={"eval_run": "run-1"},
        counters=TaskCounters(total=2, failed=2),
        error="model timeout",
        failure_kind="timeout",
    )
    persist_task_progress(
        test_settings,
        task_id="ingestion:blocked",
        task_type="ingestion",
        status="blocked",
        started_at="2026-06-01T12:00:11+00:00",
        finished_at="2026-06-01T12:00:12+00:00",
        last_progress_at="2026-06-01T12:00:12+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        config_snapshot={"pdf_load_mode": "ocr"},
        links={"trace_id": "blocked-trace"},
        counters=TaskCounters(total=1, blocked=1),
        error="bad pdf",
        failure_kind="input",
    )
    persist_task_progress(
        test_settings,
        task_id="ingestion:partial",
        task_type="ingestion",
        status="partial_success",
        started_at="2026-06-01T12:00:13+00:00",
        finished_at="2026-06-01T12:00:14+00:00",
        last_progress_at="2026-06-01T12:00:14+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        config_snapshot={"pdf_load_mode": "auto"},
        links={"trace_id": "partial-trace"},
        counters=TaskCounters(total=1, completed=1, partial=1),
        failure_kind="none",
        stages=[],
        children=[],
    )
    latest = persist_task_progress(
        test_settings,
        task_id="ingestion:partial",
        task_type="ingestion",
        status="partial_success",
        started_at="2026-06-01T12:00:13+00:00",
        finished_at="2026-06-01T12:00:15+00:00",
        last_progress_at="2026-06-01T12:00:15+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        config_snapshot={"pdf_load_mode": "auto"},
        links={"trace_id": "partial-trace"},
        counters=TaskCounters(total=1, completed=1, partial=1),
        failure_kind="none",
        error=None,
        stages=[],
        children=[],
    )
    assert latest is not None

    health = load_health_snapshot(test_settings)
    signals = {item.name: item.count for item in health.signals}
    assert signals["failed"] == 1
    assert signals["blocked"] == 1
    assert signals["partial_success"] == 1
    assert signals["attention_needed"] == 3
    assert [item.task_id for item in health.anomalies] == [
        "evaluation:failed",
        "ingestion:blocked",
        "ingestion:partial",
    ]

    filtered = list_task_progress_summaries(
        test_settings,
        needs_attention=True,
        has_trace=True,
        task_type="ingestion",
    )
    assert [item.task_id for item in filtered] == [
        "ingestion:partial",
        "ingestion:blocked",
    ]


def test_has_running_long_tasks_reads_via_observation_seam(test_settings: Settings) -> None:
    persist_task_progress(
        test_settings,
        task_id="ingestion:running",
        task_type="ingestion",
        status="running",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at=None,
        last_progress_at="2026-06-01T12:00:00+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        config_snapshot={"pdf_load_mode": "auto"},
        counters=TaskCounters(total=1),
    )
    persist_task_progress(
        test_settings,
        task_id="sync:running",
        task_type="sync",
        status="running",
        started_at="2026-06-01T12:00:01+00:00",
        finished_at=None,
        last_progress_at="2026-06-01T12:00:01+00:00",
        trigger_source="sync_runner",
        owner_surface="ops",
        config_snapshot={"mode": "full"},
        counters=TaskCounters(total=1),
    )

    assert has_running_long_tasks(test_settings) is True
