"""Regression tests for Trace collection filtering and legacy compatibility."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lab_knowledge.config import Settings
from lab_knowledge.knowledge import create_knowledge
from lab_knowledge.models import Chunk
from lab_knowledge.tracing.context import TraceContext
from lab_knowledge.tracing.store import (
    average_query_latency_ms,
    filter_trace_records,
    get_trace_record,
    read_trace_records,
    save_trace,
    trace_belongs_to_collection,
)


def _write_traces(path: Path, *records: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )


def _query_trace(
    trace_id: str,
    *,
    elapsed_ms: float,
    collection_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "trace_id": trace_id,
        "trace_type": "query",
        "started_at": "2026-06-01T12:00:00+00:00",
        "finished_at": "2026-06-01T12:00:01+00:00",
        "total_elapsed_ms": elapsed_ms,
        "stages": [],
        "error": None,
        "metadata": {"question": trace_id},
    }
    if collection_id is not None:
        payload["collection_id"] = collection_id
    return payload


@pytest.fixture
def other_collection_id(test_settings: Settings) -> str:
    return "other-collection"


@pytest.fixture
def other_settings(test_settings: Settings, other_collection_id: str) -> Settings:
    return replace(
        test_settings,
        product=replace(test_settings.product, collection=other_collection_id),
    )


@pytest.fixture
def seeded_other_collection(other_settings: Settings) -> None:
    create_knowledge(other_settings).commit_document(
        source_path="/tmp/trace-scope-seed.md",
        sha256="trace-scope-seed",
        document_id="trace-scope-seed",
        status="ingested",
        chunks=[
            Chunk(
                chunk_id="trace-scope-seed:0000",
                document_id="trace-scope-seed",
                text="seed",
                metadata={
                    "document_id": "trace-scope-seed",
                    "title": "trace-scope-seed",
                    "culture_domain": "妈祖",
                    "审阅状态": "已通过",
                },
            )
        ],
    )


def test_trace_belongs_to_collection_legacy_default_only(
    test_settings: Settings,
    other_collection_id: str,
) -> None:
    legacy = {"trace_id": "legacy"}
    default_id = test_settings.default_collection_id

    assert trace_belongs_to_collection(
        legacy,
        default_id,
        default_collection_id=default_id,
    )
    assert not trace_belongs_to_collection(
        legacy,
        other_collection_id,
        default_collection_id=default_id,
    )


def test_filter_trace_records_splits_legacy_and_scoped(
    test_settings: Settings,
    other_collection_id: str,
) -> None:
    default_id = test_settings.default_collection_id
    records = [
        _query_trace("legacy", elapsed_ms=10.0),
        _query_trace("scoped-other", elapsed_ms=20.0, collection_id=other_collection_id),
        _query_trace("scoped-default", elapsed_ms=30.0, collection_id=default_id),
    ]

    default_records = filter_trace_records(
        records,
        collection_id=default_id,
        default_collection_id=default_id,
    )
    other_records = filter_trace_records(
        records,
        collection_id=other_collection_id,
        default_collection_id=default_id,
    )

    assert [item["trace_id"] for item in default_records] == ["legacy", "scoped-default"]
    assert [item["trace_id"] for item in other_records] == ["scoped-other"]


def test_read_trace_records_collection_scope_from_file(
    test_settings: Settings,
    other_collection_id: str,
) -> None:
    trace_path = Path(test_settings.paths.traces)
    _write_traces(
        trace_path,
        _query_trace("legacy", elapsed_ms=10.0),
        _query_trace("other-only", elapsed_ms=20.0, collection_id=other_collection_id),
    )

    default_id = test_settings.default_collection_id
    default_ids = [
        item["trace_id"]
        for item in read_trace_records(test_settings, collection_id=default_id)
    ]
    scoped_ids = [
        item["trace_id"]
        for item in read_trace_records(test_settings, collection_id=other_collection_id)
    ]

    assert default_ids == ["legacy"]
    assert scoped_ids == ["other-only"]


def test_get_trace_record_respects_collection_boundary(
    test_settings: Settings,
    other_collection_id: str,
) -> None:
    trace_path = Path(test_settings.paths.traces)
    _write_traces(
        trace_path,
        _query_trace("legacy", elapsed_ms=10.0),
        _query_trace("other-only", elapsed_ms=20.0, collection_id=other_collection_id),
    )

    default_id = test_settings.default_collection_id
    assert get_trace_record(test_settings, "legacy", collection_id=default_id) is not None
    assert get_trace_record(test_settings, "legacy", collection_id=other_collection_id) is None
    assert (
        get_trace_record(test_settings, "other-only", collection_id=other_collection_id)
        is not None
    )
    assert get_trace_record(test_settings, "other-only", collection_id=default_id) is None


def test_average_query_latency_ms_isolated_by_collection(
    test_settings: Settings,
    other_collection_id: str,
) -> None:
    trace_path = Path(test_settings.paths.traces)
    _write_traces(
        trace_path,
        _query_trace("default-fast", elapsed_ms=100.0),
        _query_trace("default-slow", elapsed_ms=300.0),
        _query_trace("other-only", elapsed_ms=999.0, collection_id=other_collection_id),
    )

    default_avg = average_query_latency_ms(
        test_settings,
        collection_id=test_settings.default_collection_id,
    )
    other_avg = average_query_latency_ms(test_settings, collection_id=other_collection_id)

    assert default_avg == pytest.approx(200.0)
    assert other_avg == pytest.approx(999.0)


def test_save_trace_stamps_configured_collection_id(
    test_settings: Settings,
    other_settings: Settings,
    other_collection_id: str,
    seeded_other_collection: None,
) -> None:
    default_trace = TraceContext(trace_type="query", metadata={"question": "default"})
    other_trace = TraceContext(trace_type="query", metadata={"question": "other"})

    save_trace(test_settings, default_trace)
    save_trace(other_settings, other_trace)

    default_record = get_trace_record(test_settings, default_trace.trace_id)
    other_record = get_trace_record(other_settings, other_trace.trace_id)

    assert default_record is not None
    assert other_record is not None
    assert default_record["collection_id"] == test_settings.product.collection
    assert other_record["collection_id"] == other_collection_id

    default_id = test_settings.default_collection_id
    assert get_trace_record(test_settings, other_trace.trace_id, collection_id=default_id) is None
    assert (
        get_trace_record(other_settings, default_trace.trace_id, collection_id=other_collection_id)
        is None
    )
