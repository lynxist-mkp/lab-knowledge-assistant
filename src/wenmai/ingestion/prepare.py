"""入库后处理：切分后的清洗 / 补元数据 / 图转文。"""

from __future__ import annotations

from wenmai.config import Settings
from wenmai.factories.transform import run_registered
from wenmai.models import Chunk
from wenmai.tracing.context import TraceContext


def prepare_chunks(
    chunks: list[Chunk],
    settings: Settings,
    trace: TraceContext,
) -> list[Chunk]:
    """Run configured 入库后处理 stages via the transform registry."""
    return run_registered(chunks, settings, trace)
