"""入库后处理：切分后的清洗 / 补元数据 / 图转文。"""

from __future__ import annotations

from typing import Protocol

from wenmai.config import Settings
from wenmai.factories.transform import run_registered
from wenmai.models import Chunk
from wenmai.tracing.context import TraceContext


class _TraceRecorderLike(Protocol):
    @property
    def trace_context(self) -> TraceContext: ...


def prepare_chunks(
    chunks: list[Chunk],
    settings: Settings,
    recorder: TraceContext | _TraceRecorderLike,
) -> list[Chunk]:
    """Run configured 入库后处理 stages via the transform registry."""
    trace = (
        recorder
        if isinstance(recorder, TraceContext)
        else recorder.trace_context
    )
    return run_registered(chunks, settings, trace)
