"""Process idle timeout: unload weights after HTTP quiet period (#36)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import lab_knowledge.app as app_module
from lab_knowledge.app import create_app
from lab_knowledge.components.model_guard import (
    ModelResource,
    active_resource,
    begin_batch,
    configure,
    end_batch,
    register_unload,
    release_all_resources,
)
from lab_knowledge.config import Settings


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
def reset_idle_state() -> None:
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


def test_release_on_idle_after_request(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("lab_knowledge.app.threading.Timer", FakeTimer)
    with patch("lab_knowledge.app.release_all_resources") as mock_release:
        with TestClient(create_app(idle_settings)) as client:
            client.get("/")
            assert len(FakeTimer.instances) == 1
            assert FakeTimer.instances[0].interval == 0.05
            FakeTimer.instances[0].fire()
            mock_release.assert_called_once()


def test_new_request_cancels_pending_idle_timer(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("lab_knowledge.app.threading.Timer", FakeTimer)
    with patch("lab_knowledge.app.release_all_resources"):
        with TestClient(create_app(idle_settings)) as client:
            client.get("/")
            first_timer = FakeTimer.instances[0]
            client.get("/")
            second_timer = FakeTimer.instances[1]
            assert first_timer.cancelled
            assert second_timer is not first_timer


def test_idle_timer_skips_unload_during_batch(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("lab_knowledge.app.threading.Timer", FakeTimer)
    begin_batch(ModelResource.BGE_M3)
    try:
        with patch("lab_knowledge.app.release_all_resources") as mock_release:
            with TestClient(create_app(idle_settings)) as client:
                client.get("/")
                FakeTimer.instances[0].fire()
                mock_release.assert_not_called()
    finally:
        end_batch()


def test_idle_unload_disabled(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_settings.resources.process_idle_unload = False
    monkeypatch.setattr("lab_knowledge.app.threading.Timer", FakeTimer)
    with TestClient(create_app(test_settings)) as client:
        client.get("/")
        assert FakeTimer.instances == []


def test_idle_release_clears_active_and_unloads(
    idle_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("lab_knowledge.app.threading.Timer", FakeTimer)
    unload = MagicMock()
    register_unload(ModelResource.BGE_M3, unload)
    import lab_knowledge.components.model_guard as model_guard

    with model_guard._lock:
        model_guard._active = ModelResource.BGE_M3

    with TestClient(create_app(idle_settings)) as client:
        client.get("/")
        FakeTimer.instances[0].fire()
        assert active_resource() is None
        unload.assert_called_once()
