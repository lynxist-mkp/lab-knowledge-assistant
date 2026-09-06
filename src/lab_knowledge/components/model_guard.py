"""Exclusive model residency: at most one heavy weight in memory at a time."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum


class ModelResource(StrEnum):
    BGE_M3 = "bge_m3"
    MLX_VLM = "mlx_vlm"
    CROSS_ENCODER = "cross_encoder"


_lock = threading.RLock()
_active: ModelResource | None = None
_batch: ModelResource | None = None
_exclusive = True
_unload_handlers: dict[ModelResource, list[object]] = {
    ModelResource.BGE_M3: [],
    ModelResource.MLX_VLM: [],
    ModelResource.CROSS_ENCODER: [],
}


def configure(*, exclusive: bool) -> None:
    global _exclusive
    _exclusive = exclusive


def is_exclusive() -> bool:
    return _exclusive


def active_resource() -> ModelResource | None:
    with _lock:
        return _active


def in_batch() -> bool:
    with _lock:
        return _batch is not None


def batch_resource() -> ModelResource | None:
    with _lock:
        return _batch


def register_unload(resource: ModelResource, callback: object) -> None:
    handlers = _unload_handlers.setdefault(resource, [])
    if callback not in handlers:
        handlers.append(callback)


def _unload_resource(resource: ModelResource) -> None:
    for callback in _unload_handlers.get(resource, []):
        callback()  # type: ignore[operator]


def _evict_others(resource: ModelResource) -> None:
    if resource != ModelResource.MLX_VLM:
        _unload_resource(ModelResource.MLX_VLM)
    if resource != ModelResource.BGE_M3:
        _unload_resource(ModelResource.BGE_M3)
    if resource != ModelResource.CROSS_ENCODER:
        _unload_resource(ModelResource.CROSS_ENCODER)


def begin_batch(resource: ModelResource) -> None:
    """Hold one resource for a multi-call phase; released by end_batch()."""
    if not _exclusive:
        return
    global _active, _batch
    with _lock:
        if _batch is not None:
            raise RuntimeError(f"batch already active for {_batch}")
        _evict_others(resource)
        _active = resource
        _batch = resource


def end_batch() -> None:
    """Release the active batch resource and unload weights."""
    if not _exclusive:
        return
    global _active, _batch
    with _lock:
        if _batch is None:
            return
        resource = _batch
        _batch = None
        _active = None
    _unload_resource(resource)


def release(resource: ModelResource) -> None:
    """Drop the held resource immediately (unload weights / kill subprocess)."""
    if not _exclusive:
        return
    global _active
    with _lock:
        if _batch is not None:
            return
        if _active != resource:
            return
        _active = None
    _unload_resource(resource)


def release_all_resources() -> None:
    """End any active batch and unload every registered model resource."""
    if not _exclusive:
        return
    global _active, _batch
    with _lock:
        _batch = None
        _active = None
    for resource in ModelResource:
        _unload_resource(resource)


@contextmanager
def phase_batch(resource: ModelResource, enabled: bool) -> Iterator[None]:
    """Begin/end a model batch when *enabled*; no-op otherwise."""
    if enabled:
        begin_batch(resource)
    try:
        yield
    finally:
        if enabled:
            end_batch()


@contextmanager
def hold(resource: ModelResource) -> Iterator[None]:
    global _active
    if not _exclusive:
        yield
        return
    acquired = False
    with _lock:
        if _batch is not None:
            if _batch != resource:
                raise RuntimeError(
                    f"batch holds {_batch}, cannot acquire {resource}"
                )
        else:
            _evict_others(resource)
            _active = resource
            acquired = True
    try:
        yield
    finally:
        if acquired:
            release(resource)
