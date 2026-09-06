"""Ask-path saturation evidence for 运维观测."""

from __future__ import annotations

import json
from dataclasses import replace

from tests.conftest import register_collection

from wenmai.config import Settings
from wenmai.ops.ask_evidence import (
    ask_evidence_path,
    read_ask_evidence_records,
    summarize_ask_evidence,
    write_ask_evidence,
)
from wenmai.ops.observation import load_health_snapshot


def _other_collection_settings(
    test_settings: Settings, other_id: str = "other-collection"
) -> Settings:
    registered = register_collection(test_settings, other_id)
    return replace(
        registered,
        product=replace(test_settings.product, collection=other_id),
    )


def test_summarize_ask_evidence_counts_saturation_codes(test_settings) -> None:
    write_ask_evidence(
        test_settings,
        {
            "event": "saturation",
            "code": "busy",
            "entrypoint": "mcp",
            "in_flight": 2,
            "max_in_flight": 2,
            "wait_ms": 0.0,
        },
    )
    write_ask_evidence(
        test_settings,
        {
            "event": "saturation",
            "code": "timeout",
            "entrypoint": "http",
            "in_flight": 2,
            "max_in_flight": 2,
            "wait_ms": 30000.0,
        },
    )

    summary = summarize_ask_evidence(test_settings)
    assert summary.busy_total == 1
    assert summary.timeout_total == 1
    assert summary.saturation_total == 2


def test_health_snapshot_includes_ask_saturation_signals(test_settings) -> None:
    write_ask_evidence(
        test_settings,
        {
            "event": "saturation",
            "code": "busy",
            "entrypoint": "mcp",
            "in_flight": 1,
            "max_in_flight": 1,
            "wait_ms": 0.0,
        },
    )
    snapshot = load_health_snapshot(test_settings)
    signal_names = {item.name for item in snapshot.signals}
    assert "ask_busy" in signal_names
    assert "ask_timeout" in signal_names
    assert "ask_long_task_guard" in signal_names
    busy = next(item for item in snapshot.signals if item.name == "ask_busy")
    assert busy.count == 1


def test_write_ask_evidence_stamps_collection_id(test_settings: Settings) -> None:
    write_ask_evidence(
        test_settings,
        {
            "event": "saturation",
            "code": "busy",
            "entrypoint": "http",
            "in_flight": 1,
            "max_in_flight": 1,
            "wait_ms": 0.0,
        },
    )
    records = read_ask_evidence_records(test_settings)
    assert records[-1]["collection_id"] == test_settings.product.collection


def test_ask_evidence_collection_filter_and_legacy_visibility(
    test_settings: Settings,
) -> None:
    other_id = "other-collection"
    settings = register_collection(test_settings, other_id)
    other_settings = _other_collection_settings(settings, other_id)

    evidence_path = ask_evidence_path(settings)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_record = {
        "event": "saturation",
        "code": "busy",
        "entrypoint": "legacy",
        "in_flight": 1,
        "max_in_flight": 1,
        "wait_ms": 0.0,
    }
    scoped_record = {
        **legacy_record,
        "entrypoint": "scoped",
        "collection_id": other_id,
    }
    evidence_path.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in (legacy_record, scoped_record)
        )
        + "\n",
        encoding="utf-8",
    )

    default_summary = summarize_ask_evidence(
        settings,
        collection_id=settings.default_collection_id,
    )
    other_summary = summarize_ask_evidence(settings, collection_id=other_id)

    assert default_summary.busy_total == 1
    assert other_summary.busy_total == 1

    write_ask_evidence(
        other_settings,
        {
            "event": "saturation",
            "code": "timeout",
            "entrypoint": "other-write",
            "in_flight": 1,
            "max_in_flight": 1,
            "wait_ms": 100.0,
        },
    )
    stamped = read_ask_evidence_records(settings, collection_id=other_id)
    assert stamped[-1]["collection_id"] == other_id
    assert stamped[-1]["code"] == "timeout"
