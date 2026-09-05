"""入库后处理：切分后的清洗 / 补元数据 / 图转文。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from wenmai.config import Settings
from wenmai.models import Chunk


class TransformTraceRecorder(Protocol):
    """Narrow observability seam for transform registry stages."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    @contextmanager
    def stage(
        self,
        name: str,
        method: str,
        provider: str,
        input_summary: str = "",
    ) -> Iterator[dict[str, Any]]: ...


def prepare_chunks(
    chunks: list[Chunk],
    settings: Settings,
    recorder: TransformTraceRecorder,
) -> list[Chunk]:
    """Run configured 入库后处理 stages via the transform registry."""
    from wenmai.factories.transform import run_registered

    return run_registered(chunks, settings, recorder)
