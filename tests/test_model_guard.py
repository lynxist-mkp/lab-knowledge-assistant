"""Exclusive model residency guard."""

from __future__ import annotations

from unittest.mock import MagicMock

from wenmai.components.model_guard import (
    ModelResource,
    active_resource,
    batch_resource,
    begin_batch,
    configure,
    end_batch,
    hold,
    in_batch,
    register_unload,
    release,
    release_all_resources,
)


def setup_function() -> None:
    configure(exclusive=True)
    end_batch()
    for resource in ModelResource:
        release(resource)


def teardown_function() -> None:
    configure(exclusive=True)
    end_batch()
    for resource in ModelResource:
        release(resource)


def test_hold_tracks_active_resource() -> None:
    assert active_resource() is None
    with hold(ModelResource.BGE_M3):
        assert active_resource() is ModelResource.BGE_M3
    assert active_resource() is None


def test_bge_unloads_when_acquiring_mlx_vlm() -> None:
    from wenmai.components.embedding.bge_m3 import BgeM3Embedding

    embedder = BgeM3Embedding()
    embedder._model = object()
    with hold(ModelResource.MLX_VLM):
        assert embedder._model is None


def test_release_unloads_held_resource() -> None:
    unload = MagicMock()
    from wenmai.components.model_guard import register_unload

    register_unload(ModelResource.BGE_M3, unload)
    with hold(ModelResource.BGE_M3):
        unload.reset_mock()
    unload.assert_called_once()


def test_exclusive_disabled_is_noop() -> None:
    configure(exclusive=False)
    with hold(ModelResource.MLX_VLM):
        with hold(ModelResource.BGE_M3):
            assert active_resource() is None


def test_begin_batch_holds_until_end_batch() -> None:
    unload = MagicMock()
    from wenmai.components.model_guard import register_unload

    register_unload(ModelResource.MLX_VLM, unload)
    begin_batch(ModelResource.MLX_VLM)
    assert in_batch()
    assert batch_resource() is ModelResource.MLX_VLM
    with hold(ModelResource.MLX_VLM):
        assert active_resource() is ModelResource.MLX_VLM
    unload.reset_mock()
    end_batch()
    assert not in_batch()
    assert active_resource() is None
    unload.assert_called_once()


def test_hold_during_batch_does_not_release_on_exit() -> None:
    unload = MagicMock()
    register_unload(ModelResource.BGE_M3, unload)
    begin_batch(ModelResource.BGE_M3)
    with hold(ModelResource.BGE_M3):
        unload.reset_mock()
    unload.assert_not_called()
    end_batch()
    unload.assert_called_once()


def test_release_all_clears_active_and_batch() -> None:
    begin_batch(ModelResource.MLX_VLM)
    assert in_batch()
    assert active_resource() is ModelResource.MLX_VLM
    release_all_resources()
    assert not in_batch()
    assert active_resource() is None


def test_release_all_unloads_every_resource() -> None:
    unloads = {resource: MagicMock() for resource in ModelResource}
    for resource, unload in unloads.items():
        register_unload(resource, unload)
    release_all_resources()
    for unload in unloads.values():
        unload.assert_called_once()


def test_release_all_idempotent() -> None:
    unload = MagicMock()
    register_unload(ModelResource.MLX_VLM, unload)
    release_all_resources()
    release_all_resources()
    assert active_resource() is None
    assert not in_batch()


def test_release_all_during_batch() -> None:
    unload = MagicMock()
    register_unload(ModelResource.CROSS_ENCODER, unload)
    begin_batch(ModelResource.CROSS_ENCODER)
    with hold(ModelResource.CROSS_ENCODER):
        unload.reset_mock()
    unload.assert_not_called()
    release_all_resources()
    assert not in_batch()
    assert active_resource() is None
    unload.assert_called_once()
