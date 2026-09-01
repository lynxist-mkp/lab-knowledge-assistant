"""FastAPI lifespan: release model resources on shutdown."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

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


@pytest.fixture(autouse=True)
def reset_model_guard() -> None:
    configure(exclusive=True)
    release_all_resources()
    yield
    release_all_resources()


def test_release_all_resources_on_shutdown(test_settings: Settings) -> None:
    with patch("wenmai.app.release_all_resources") as mock_release:
        with TestClient(create_app(test_settings)) as client:
            client.get("/")
        mock_release.assert_called_once()


def test_shutdown_clears_active_batch(test_settings: Settings) -> None:
    begin_batch(ModelResource.MLX_VLM)
    assert in_batch()
    assert active_resource() is ModelResource.MLX_VLM

    with TestClient(create_app(test_settings)) as client:
        client.get("/")

    assert not in_batch()
    assert active_resource() is None


def test_shutdown_invokes_registered_unload_callbacks(
    test_settings: Settings,
) -> None:
    unload = MagicMock()
    register_unload(ModelResource.BGE_M3, unload)
    begin_batch(ModelResource.BGE_M3)

    with TestClient(create_app(test_settings)) as client:
        client.get("/")

    assert active_resource() is None
    unload.assert_called()
