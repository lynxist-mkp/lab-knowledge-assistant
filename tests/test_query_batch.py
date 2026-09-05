"""Query phase batching and time-window coordinator."""

from __future__ import annotations

import threading
import time

import pytest

from wenmai.components.model_guard import ModelResource, active_resource, end_batch
from wenmai.models import AskResult
from wenmai.pipelines.query_batch import QueryBatchCoordinator, reset_query_coordinator
from wenmai.pipelines.query_orchestration import run_ask_pipeline


@pytest.fixture(autouse=True)
def _reset_coordinators() -> None:
    reset_query_coordinator()
    end_batch()
    yield
    reset_query_coordinator()
    end_batch()


def test_run_ask_pipeline_phase_batch_sets_active_resource(
    test_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_settings.resources.query_phase_batch = True
    seen: list[ModelResource | None] = []

    class Job:
        question = "妈祖信仰的发源地在哪里？"
        settings = test_settings
        culture_domain = None
        retrieval_mode = None
        rerank_enabled = None
        knowledge = None
        record_trace = False
        result = None
        error = None
        batch_id = None
        batch_size = 1
        batch_wait_ms = 0.0

    from wenmai.retrieval import retrieve as retrieve_mod

    def spy_retrieve(*args, **kwargs):
        seen.append(active_resource())
        return retrieve_mod(*args, **kwargs)

    monkeypatch.setattr("wenmai.pipelines.query_orchestration.retrieve", spy_retrieve)
    run_ask_pipeline([Job()], phase_batch=True, batch_meta=None)
    assert ModelResource.BGE_M3 in seen


def test_query_batch_coordinator_waits_for_window(test_settings) -> None:
    test_settings.resources.batch_window_seconds = 0.05
    test_settings.resources.batch_window_max_size = 4
    coordinator = QueryBatchCoordinator(test_settings)
    results: list[AskResult] = []
    errors: list[BaseException] = []

    def worker(question: str) -> None:
        try:
            results.append(
                coordinator.submit(
                    question,
                    test_settings,
                    record_trace=False,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(f"问题{i}？",))
        for i in range(2)
    ]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    elapsed = time.monotonic() - started

    assert not errors
    assert len(results) == 2
    assert elapsed >= 0.04


def test_query_batch_coordinator_flushes_at_max_size(test_settings) -> None:
    test_settings.resources.batch_window_seconds = 10.0
    test_settings.resources.batch_window_max_size = 2
    coordinator = QueryBatchCoordinator(test_settings)
    results: list[AskResult] = []
    errors: list[BaseException] = []

    def worker(question: str) -> None:
        try:
            results.append(
                coordinator.submit(
                    question,
                    test_settings,
                    record_trace=False,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(f"问题{i}？",))
        for i in range(2)
    ]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    elapsed = time.monotonic() - started

    assert not errors
    assert len(results) == 2
    assert elapsed < 2.0


def test_query_batch_coordinator_passes_batch_metadata(
    test_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_settings.resources.batch_window_seconds = 0
    captured: list[dict[str, object]] = []

    def fake_run(jobs, *, phase_batch, batch_meta=None):
        if batch_meta:
            captured.append(batch_meta)
        for job in jobs:
            job.result = AskResult(
                answer="测试",
                citations=[],
                trace_id="trace-1",
                refused=False,
                refusal_reason=None,
                ranked_chunks=[],
            )

    monkeypatch.setattr(
        "wenmai.pipelines.query_batch.run_ask_pipeline",
        fake_run,
    )
    coordinator = QueryBatchCoordinator(test_settings)
    result = coordinator.submit("测试问题？", test_settings, record_trace=False)
    assert result.answer == "测试"
    assert captured
    assert captured[0]["batch_size"] == 1
