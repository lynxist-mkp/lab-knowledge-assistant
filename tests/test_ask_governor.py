"""Tests for the ask concurrency governor."""

from __future__ import annotations

import threading

import pytest

from wenmai.config import AskConcurrencyConfig
import wenmai.http.ask_governor as ask_governor
from wenmai.http.ask_governor import (
    AskConcurrencyGovernor,
    AskSaturationError,
    reset_ask_governor,
)
from wenmai.ops.ask_evidence import summarize_ask_evidence
from wenmai.task_progress import TaskCounters, persist_task_progress


@pytest.fixture(autouse=True)
def _reset_governor() -> None:
    reset_ask_governor()


def test_governor_allows_up_to_max_in_flight(test_settings) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=2,
        saturation_policy="busy",
    )
    governor = AskConcurrencyGovernor(test_settings.resources.ask)
    with governor.acquire(test_settings, entrypoint="test"):
        with governor.acquire(test_settings, entrypoint="test"):
            snapshot = governor.snapshot(test_settings)
            assert snapshot.in_flight == 2


def test_governor_busy_policy_rejects_when_saturated(test_settings) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=1,
        saturation_policy="busy",
    )
    governor = AskConcurrencyGovernor(test_settings.resources.ask)
    gate = threading.Event()
    holder_started = threading.Event()

    def hold_slot() -> None:
        with governor.acquire(test_settings, entrypoint="holder"):
            holder_started.set()
            gate.wait(timeout=2)

    thread = threading.Thread(target=hold_slot)
    thread.start()
    assert holder_started.wait(timeout=2)

    with pytest.raises(AskSaturationError) as exc_info:
        with governor.acquire(test_settings, entrypoint="test"):
            pass

    assert exc_info.value.code == "busy"
    gate.set()
    thread.join(timeout=2)

    evidence = summarize_ask_evidence(test_settings)
    assert evidence.busy_total >= 1


def test_governor_wait_policy_times_out(test_settings) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=1,
        max_wait_seconds=0.15,
        saturation_policy="wait",
    )
    governor = AskConcurrencyGovernor(test_settings.resources.ask)
    gate = threading.Event()
    holder_started = threading.Event()

    def hold_slot() -> None:
        with governor.acquire(test_settings, entrypoint="holder"):
            holder_started.set()
            gate.wait(timeout=2)

    thread = threading.Thread(target=hold_slot)
    thread.start()
    assert holder_started.wait(timeout=2)

    with pytest.raises(AskSaturationError) as exc_info:
        with governor.acquire(test_settings, entrypoint="test"):
            pass

    assert exc_info.value.code == "timeout"
    gate.set()
    thread.join(timeout=2)

    evidence = summarize_ask_evidence(test_settings)
    assert evidence.timeout_total >= 1


def test_governor_long_task_guard_reduces_capacity(test_settings) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=3,
        long_task_guard=True,
        long_task_max_in_flight=1,
        saturation_policy="busy",
    )
    persist_task_progress(
        test_settings,
        task_id="ingest-running",
        task_type="ingestion",
        status="running",
        started_at="2026-01-01T00:00:00Z",
        finished_at=None,
        last_progress_at="2026-01-01T00:00:00Z",
        trigger_source="ops",
        owner_surface="ops",
        config_snapshot={"pdf_load_mode": "auto"},
        counters=TaskCounters(total=1),
    )
    governor = AskConcurrencyGovernor(test_settings.resources.ask)
    snapshot = governor.snapshot(test_settings)
    assert snapshot.long_task_active is True
    assert snapshot.max_in_flight == 1

    with governor.acquire(test_settings, entrypoint="first"):
        with pytest.raises(AskSaturationError) as exc_info:
            with governor.acquire(test_settings, entrypoint="second"):
                pass
    assert exc_info.value.code == "busy"


def test_governor_long_task_active_rejects_when_capacity_is_zero(test_settings) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=3,
        long_task_guard=True,
        long_task_max_in_flight=0,
        saturation_policy="busy",
    )
    persist_task_progress(
        test_settings,
        task_id="ingest-running",
        task_type="ingestion",
        status="running",
        started_at="2026-01-01T00:00:00Z",
        finished_at=None,
        last_progress_at="2026-01-01T00:00:00Z",
        trigger_source="ops",
        owner_surface="ops",
        config_snapshot={"pdf_load_mode": "auto"},
        counters=TaskCounters(total=1),
    )
    governor = AskConcurrencyGovernor(test_settings.resources.ask)

    with pytest.raises(AskSaturationError) as exc_info:
        with governor.acquire(test_settings, entrypoint="test"):
            pass

    assert exc_info.value.code == "long_task_active"


def test_governor_long_task_guard_uses_observation_seam(test_settings, monkeypatch) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=3,
        long_task_guard=True,
        long_task_max_in_flight=1,
        saturation_policy="busy",
    )
    monkeypatch.setattr(ask_governor, "has_running_long_tasks", lambda _settings: True)

    governor = AskConcurrencyGovernor(test_settings.resources.ask)
    snapshot = governor.snapshot(test_settings)

    assert snapshot.long_task_active is True
    assert snapshot.max_in_flight == 1


def test_governor_disabled_is_noop(test_settings) -> None:
    test_settings.resources.ask = AskConcurrencyConfig(enabled=False)
    governor = AskConcurrencyGovernor(test_settings.resources.ask)
    with governor.acquire(test_settings, entrypoint="test"):
        with governor.acquire(test_settings, entrypoint="test"):
            snapshot = governor.snapshot(test_settings)
            assert snapshot.in_flight == 0
