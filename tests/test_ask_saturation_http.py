"""HTTP ask saturation responses."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from wenmai.app import create_app
from wenmai.config import AskConcurrencyConfig
from wenmai.http.ask_governor import get_ask_governor, reset_ask_governor
from wenmai.task_progress import TaskCounters, persist_task_progress


def test_http_ask_returns_503_when_governor_is_busy(test_settings) -> None:
    reset_ask_governor()
    test_settings.resources.ask = AskConcurrencyConfig(
        enabled=True,
        max_in_flight=1,
        saturation_policy="busy",
    )
    app = create_app(test_settings)
    client = TestClient(app)
    gate = threading.Event()
    holder_started = threading.Event()
    governor = get_ask_governor(test_settings)

    def hold_governor() -> None:
        with governor.acquire(test_settings, entrypoint="holder"):
            holder_started.set()
            gate.wait(timeout=2)

    thread = threading.Thread(target=hold_governor)
    thread.start()
    assert holder_started.wait(timeout=2)

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})
    gate.set()
    thread.join(timeout=2)

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["code"] == "busy"


def test_http_ask_returns_503_when_long_task_guard_blocks(test_settings) -> None:
    reset_ask_governor()
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
    app = create_app(test_settings)
    client = TestClient(app)

    response = client.post("/ask", json={"question": "妈祖信仰的发源地在哪里？"})

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["code"] == "long_task_active"
    assert body["detail"]["max_in_flight"] == 0
