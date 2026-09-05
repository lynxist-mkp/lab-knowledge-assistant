"""Ask-path saturation evidence for 运维观测."""

from __future__ import annotations

from wenmai.ops.ask_evidence import summarize_ask_evidence, write_ask_evidence
from wenmai.ops.observation import load_health_snapshot


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
