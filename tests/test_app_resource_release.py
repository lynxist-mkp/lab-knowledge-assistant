"""Integration: lifespan shutdown and idle unload release model resources (#37)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import wenmai.app as app_module
from wenmai.app import create_app
from wenmai.components.model_guard import (
    ModelResource,
    active_resource,
    begin_batch,
    configure,
    in_batch,
    register_unload,
    release_all_resources,
)
from wenmai.config import Settings


class FakeTimer:
    instances: list[FakeTimer] = []

    def __init__(
        self,
        interval: float,
        function,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> None:
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs or {}
        self.cancelled = False
        FakeTimer.instances.append(self)

    def start(self) -> None:
        return None

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        self.function(*self.args, **self.kwargs)


@pytest.fixture(autouse=True)
def reset_app_resource_state() -> None:
    configure(exclusive=True)
    release_all_resources()
    with app_module._idle_lock:
        app_module._cancel_idle_timer()
        app_module._in_flight = 0
    FakeTimer.instances.clear()
    yield
    configure(exclusive=True)
    release_all_resources()
    with app_module._idle_lock:
        app_module._cancel_idle_timer()
        app_module._in_flight = 0
    FakeTimer.instances.clear()


@pytest.fixture
def idle_settings(test_settings: Settings) -> Settings:
    test_settings.resources.process_idle_unload = True
    test_settings.resources.process_idle_timeout_seconds = 0.05
    return test_settings


def test_shutdown_cancels_pending_idle_timer_and_releases(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("wenmai.app.threading.Timer", FakeTimer)
    unload = MagicMock()
    register_unload(ModelResource.MLX_VLM, unload)
    begin_batch(ModelResource.MLX_VLM)

    with TestClient(create_app(idle_settings)) as client:
        client.get("/")
        assert len(FakeTimer.instances) == 1
        timer = FakeTimer.instances[0]
        assert not timer.cancelled

    assert timer.cancelled
    assert not in_batch()
    assert active_resource() is None
    unload.assert_called()


def test_idle_release_then_shutdown_is_idempotent(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("wenmai.app.threading.Timer", FakeTimer)
    unload = MagicMock()
    register_unload(ModelResource.CROSS_ENCODER, unload)
    import wenmai.components.model_guard as model_guard

    with model_guard._lock:
        model_guard._active = ModelResource.CROSS_ENCODER

    with TestClient(create_app(idle_settings)) as client:
        client.get("/")
        FakeTimer.instances[0].fire()
        assert active_resource() is None
        unload.assert_called_once()

    assert active_resource() is None
    assert unload.call_count == 2
