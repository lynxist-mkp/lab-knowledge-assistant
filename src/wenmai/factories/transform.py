from __future__ import annotations

from wenmai.components.transform.base import BaseTransform
from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.factories.registry import ProviderRegistry
from wenmai.models import Chunk
from wenmai.tracing.context import TraceContext

registry: ProviderRegistry[BaseTransform] = ProviderRegistry()


def run_registered(chunks: list[Chunk], settings: Settings, trace: TraceContext) -> list[Chunk]:
    """Run transform stages listed in config. Unregistered names are skipped."""
    ensure_providers()
    for stage_name in settings.transform.stages:
        provider = getattr(settings.transform, stage_name, None)
        if not provider:
            continue
        impl_cls = registry.get(f"{stage_name}.{provider}")
        if impl_cls is None:
            continue
        impl = impl_cls(settings=settings)
        chunks = impl.apply(chunks, trace)
    return chunks
