"""运维观测 seam：概览与 Trace 读。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from tests.conftest import register_collection

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.knowledge.read import ReadPath
from wenmai.models import Chunk
from wenmai.ops.observation import (
    get_query_detail,
    get_task_investigation,
    get_trace_summary,
    has_running_long_tasks,
    list_query_summaries,
    list_task_progress_summaries,
    load_health_snapshot,
    load_overview_stats,
)
from wenmai.storage.catalog import DocumentCatalog
from wenmai.task_progress import ChildEvidence, TaskCounters, persist_task_progress
from wenmai.tracing.context import TraceContext
from wenmai.tracing.store import get_trace_record, save_trace


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    registered = register_collection(test_settings, other_id)
    return replace(
        registered,
        product=replace(test_settings.product, collection=other_id),
    )


def test_load_overview_stats_empty_catalog(test_settings: Settings) -> None:
    stats = load_overview_stats(test_settings)
    assert stats.document_count == 0
    assert stats.chunk_count == 0


def test_load_overview_stats_routes_to_alternate_collection(test_settings: Settings) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    other_knowledge = create_knowledge(_other_collection_settings(test_settings, other_id))
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

    default_stats = load_overview_stats(settings)
    other_stats = load_overview_stats(settings, collection_id=other_id)

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
    traces = [
        {
            "trace_id": "query-trace-1",
            "trace_type": "query",
            "started_at": "2026-06-01T12:10:00+00:00",
            "finished_at": "2026-06-01T12:10:01+00:00",
            "total_elapsed_ms": 80.0,
            "stages": [],
            "error": None,
            "metadata": {"question": "妈祖信仰的发源地在哪里？"},
        },
        {
            "trace_id": "query-trace-peer",
            "trace_type": "query",
            "started_at": "2026-06-01T12:11:00+00:00",
            "finished_at": "2026-06-01T12:11:01+00:00",
            "total_elapsed_ms": 60.0,
            "stages": [],
            "error": None,
            "metadata": {"question": "妈祖信仰传播到哪里？"},
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in traces) + "\n",
        encoding="utf-8",
    )
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
        links={"trace_id": "query-trace-peer"},
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


def test_has_running_long_tasks_scoped_by_collection(test_settings: Settings) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    other_settings = _other_collection_settings(settings, other_id)

    persist_task_progress(
        other_settings,
        task_id="ingestion:other-running",
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

    assert has_running_long_tasks(settings) is True
    assert has_running_long_tasks(settings, collection_id=other_id) is True
    assert has_running_long_tasks(settings, collection_id=settings.product.collection) is False


def test_load_health_snapshot_global_vs_scoped(test_settings: Settings) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    other_settings = _other_collection_settings(settings, other_id)

    from wenmai.ops.ask_evidence import ask_evidence_path, write_ask_evidence

    write_ask_evidence(
        settings,
        {
            "event": "saturation",
            "code": "busy",
            "entrypoint": "default",
            "in_flight": 1,
            "max_in_flight": 1,
            "wait_ms": 0.0,
        },
    )
    write_ask_evidence(
        other_settings,
        {
            "event": "saturation",
            "code": "timeout",
            "entrypoint": "other",
            "in_flight": 1,
            "max_in_flight": 1,
            "wait_ms": 100.0,
        },
    )

    evidence_path = ask_evidence_path(settings)
    legacy_line = json.dumps(
        {
            "event": "saturation",
            "code": "long_task_active",
            "entrypoint": "legacy",
            "in_flight": 0,
            "max_in_flight": 0,
            "wait_ms": 0.0,
        },
        ensure_ascii=False,
    )
    evidence_path.write_text(evidence_path.read_text(encoding="utf-8") + legacy_line + "\n")

    persist_task_progress(
        settings,
        task_id="evaluation:default-failed",
        task_type="evaluation",
        status="failed",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at="2026-06-01T12:00:01+00:00",
        last_progress_at="2026-06-01T12:00:01+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf"]},
        counters=TaskCounters(total=1, failed=1),
        failure_kind="runtime",
    )
    persist_task_progress(
        other_settings,
        task_id="evaluation:other-failed",
        task_type="evaluation",
        status="failed",
        started_at="2026-06-01T12:00:02+00:00",
        finished_at="2026-06-01T12:00:03+00:00",
        last_progress_at="2026-06-01T12:00:03+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf"]},
        counters=TaskCounters(total=1, failed=1),
        failure_kind="runtime",
    )

    global_health = load_health_snapshot(settings)
    global_signals = {item.name: item.count for item in global_health.signals}
    assert global_signals["failed"] == 2
    assert global_signals["ask_busy"] == 1
    assert global_signals["ask_timeout"] == 1
    assert global_signals["ask_long_task_guard"] == 1

    scoped_health = load_health_snapshot(settings, collection_id=other_id)
    scoped_signals = {item.name: item.count for item in scoped_health.signals}
    assert scoped_signals["failed"] == 1
    assert scoped_signals["ask_busy"] == 0
    assert scoped_signals["ask_timeout"] == 1
    assert scoped_signals["ask_long_task_guard"] == 0
    assert [item.task_id for item in scoped_health.anomalies] == ["evaluation:other-failed"]


def test_trace_collection_filter_legacy_default_only_smoke(
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    create_knowledge(_other_collection_settings(test_settings, other_id)).commit_document(
        source_path="/tmp/trace-scope-other.md",
        sha256="trace-scope-other",
        document_id="trace-scope-other",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="trace-scope-other:0000",
                document_id="trace-scope-other",
                text="trace scope",
                metadata={
                    "document_id": "trace-scope-other",
                    "title": "trace-scope-other",
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            )
        ],
    )
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_trace = {
        "trace_id": "legacy-query",
        "trace_type": "query",
        "started_at": "2026-06-01T12:00:00+00:00",
        "finished_at": "2026-06-01T12:00:01+00:00",
        "total_elapsed_ms": 10.0,
        "stages": [],
        "error": None,
        "metadata": {"question": "legacy"},
    }
    scoped_trace = {
        **legacy_trace,
        "trace_id": "scoped-query",
        "collection_id": other_id,
        "metadata": {"question": "scoped"},
    }
    trace_path.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in (legacy_trace, scoped_trace)
        )
        + "\n",
        encoding="utf-8",
    )

    default_summaries = list_query_summaries(settings)
    assert [item.trace_id for item in default_summaries] == ["legacy-query"]

    other_summaries = list_query_summaries(settings, collection_id=other_id)
    assert [item.trace_id for item in other_summaries] == ["scoped-query"]

    assert get_trace_summary(settings, "legacy-query") is not None
    assert get_trace_summary(settings, "legacy-query", collection_id=other_id) is None
    assert get_trace_summary(settings, "scoped-query", collection_id=other_id) is not None
    assert get_trace_summary(settings, "scoped-query") is None


def test_save_trace_stamps_collection_id(test_settings: Settings) -> None:
    trace = TraceContext(trace_type="query", metadata={"question": "stamp?"})
    save_trace(test_settings, trace)
    record = get_trace_record(test_settings, trace.trace_id)
    assert record is not None
    assert record["collection_id"] == test_settings.product.collection
    detail = get_query_detail(test_settings, trace.trace_id)
    assert detail is not None
    assert detail.summary.trace_id == trace.trace_id


def test_task_investigation_trace_summaries_respect_collection_scope(
    test_settings: Settings,
    tmp_path: Path,
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    create_knowledge(_other_collection_settings(test_settings, other_id)).commit_document(
        source_path="/tmp/investigation-other.md",
        sha256="investigation-other",
        document_id="investigation-other",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="investigation-other:0000",
                document_id="investigation-other",
                text="investigation scope",
                metadata={
                    "document_id": "investigation-other",
                    "title": "investigation-other",
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            )
        ],
    )
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_trace = {
        "trace_id": "legacy-ing",
        "trace_type": "ingestion",
        "started_at": "2026-06-01T12:00:00+00:00",
        "finished_at": "2026-06-01T12:00:01+00:00",
        "total_elapsed_ms": 10.0,
        "stages": [],
        "error": None,
        "metadata": {"document_id": "legacy-doc"},
    }
    scoped_trace = {
        **legacy_trace,
        "trace_id": "scoped-ing",
        "collection_id": other_id,
        "metadata": {"document_id": "scoped-doc"},
    }
    trace_path.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in (legacy_trace, scoped_trace)
        )
        + "\n",
        encoding="utf-8",
    )
    persist_task_progress(
        test_settings,
        task_id="ingestion:legacy-ing",
        task_type="ingestion",
        status="succeeded",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at="2026-06-01T12:00:01+00:00",
        last_progress_at="2026-06-01T12:00:01+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        config_snapshot={"mode": "full"},
        links={"trace_id": "legacy-ing"},
        counters=TaskCounters(total=1, completed=1),
    )
    persist_task_progress(
        _other_collection_settings(test_settings, other_id),
        task_id="ingestion:scoped-ing",
        task_type="ingestion",
        status="succeeded",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at="2026-06-01T12:00:01+00:00",
        last_progress_at="2026-06-01T12:00:01+00:00",
        trigger_source="ingest_api",
        owner_surface="ops",
        config_snapshot={"mode": "full"},
        links={"trace_id": "scoped-ing"},
        counters=TaskCounters(total=1, completed=1),
    )

    default_investigation = get_task_investigation(
        settings,
        "ingestion:legacy-ing",
    )
    other_investigation = get_task_investigation(
        settings,
        "ingestion:scoped-ing",
        collection_id=other_id,
    )
    assert default_investigation is not None
    assert [item.trace_id for item in default_investigation.trace_summaries] == ["legacy-ing"]
    assert other_investigation is not None
    assert [item.trace_id for item in other_investigation.trace_summaries] == ["scoped-ing"]

    missing_on_other = get_task_investigation(
        settings,
        "ingestion:legacy-ing",
        collection_id=other_id,
    )
    assert missing_on_other is None


def test_task_investigation_config_related_tasks_respect_collection_scope(
    test_settings: Settings,
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    create_knowledge(_other_collection_settings(test_settings, other_id)).commit_document(
        source_path="/tmp/investigation-peer-other.md",
        sha256="investigation-peer-other",
        document_id="investigation-peer-other",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="investigation-peer-other:0000",
                document_id="investigation-peer-other",
                text="peer scope",
                metadata={
                    "document_id": "investigation-peer-other",
                    "title": "investigation-peer-other",
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            )
        ],
    )
    trace_path = Path(test_settings.paths.traces)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    traces = [
        {
            "trace_id": "default-trace",
            "trace_type": "query",
            "started_at": "2026-06-01T12:20:00+00:00",
            "finished_at": "2026-06-01T12:20:01+00:00",
            "total_elapsed_ms": 10.0,
            "stages": [],
            "error": None,
            "metadata": {"question": "default"},
        },
        {
            "trace_id": "other-trace",
            "trace_type": "query",
            "collection_id": other_id,
            "started_at": "2026-06-01T12:21:00+00:00",
            "finished_at": "2026-06-01T12:21:01+00:00",
            "total_elapsed_ms": 20.0,
            "stages": [],
            "error": None,
            "metadata": {"question": "other"},
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in traces) + "\n",
        encoding="utf-8",
    )
    for task_id, trace_id, scoped_settings in (
        ("evaluation:default-task", "default-trace", test_settings),
        (
            "evaluation:other-task",
            "other-trace",
            _other_collection_settings(test_settings, other_id),
        ),
    ):
        persist_task_progress(
            scoped_settings,
            task_id=task_id,
            task_type="evaluation",
            status="succeeded",
            started_at="2026-06-01T12:20:00+00:00",
            finished_at="2026-06-01T12:20:01+00:00",
            last_progress_at="2026-06-01T12:20:01+00:00",
            trigger_source="eval_runner",
            owner_surface="ops",
            config_snapshot={"groups": ["rrf"]},
            links={"trace_id": trace_id},
            counters=TaskCounters(total=1, completed=1),
            failure_kind="none",
        )

    default_investigation = get_task_investigation(
        settings,
        "evaluation:default-task",
    )
    other_investigation = get_task_investigation(
        settings,
        "evaluation:other-task",
        collection_id=other_id,
    )

    assert default_investigation is not None
    assert [item.task_id for item in default_investigation.config_related_tasks] == []
    assert other_investigation is not None
    assert [item.task_id for item in other_investigation.config_related_tasks] == []


def test_task_progress_list_and_detail_respect_collection_scope(
    test_settings: Settings,
) -> None:
    from fastapi.testclient import TestClient

    from wenmai.app import create_app
    from wenmai.ops.observation import get_task_progress_detail

    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    legacy_path = Path(test_settings.observability.task_progress_file)
    if not legacy_path.is_absolute():
        legacy_path = test_settings.root / legacy_path
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "task_id": "evaluation:legacy",
                "task_type": "evaluation",
                "status": "succeeded",
                "started_at": "2026-06-01T12:00:00+00:00",
                "finished_at": "2026-06-01T12:00:01+00:00",
                "last_progress_at": "2026-06-01T12:00:01+00:00",
                "trigger_source": "eval_runner",
                "owner_surface": "ops",
                "config_fingerprint": "legacy",
                "config_snapshot": {},
                "counters": TaskCounters(total=1, completed=1).as_dict(),
                "failure_kind": "none",
                "degraded": False,
                "links": {},
                "stages": [],
                "children": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    persist_task_progress(
        _other_collection_settings(test_settings, other_id),
        task_id="evaluation:scoped",
        task_type="evaluation",
        status="succeeded",
        started_at="2026-06-01T12:00:00+00:00",
        finished_at="2026-06-01T12:00:01+00:00",
        last_progress_at="2026-06-01T12:00:01+00:00",
        trigger_source="eval_runner",
        owner_surface="ops",
        config_snapshot={"groups": ["rrf"]},
        counters=TaskCounters(total=1, completed=1),
    )

    default_summaries = list_task_progress_summaries(settings)
    assert [item.task_id for item in default_summaries] == ["evaluation:legacy"]

    other_summaries = list_task_progress_summaries(settings, collection_id=other_id)
    assert [item.task_id for item in other_summaries] == ["evaluation:scoped"]

    assert get_task_progress_detail(settings, "evaluation:legacy") is not None
    assert (
        get_task_progress_detail(settings, "evaluation:legacy", collection_id=other_id)
        is None
    )
    assert (
        get_task_progress_detail(settings, "evaluation:scoped", collection_id=other_id)
        is not None
    )
    assert get_task_progress_detail(settings, "evaluation:scoped") is None

    client = TestClient(create_app(settings))
    unknown = client.get(
        "/api/tasks/progress/evaluation:scoped",
        params={"collection_id": "missing-collection"},
    )
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "collection not found"

    missing = client.get(
        "/api/tasks/progress/evaluation:legacy",
        params={"collection_id": other_id},
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "task progress not found"
