from __future__ import annotations

from lab_knowledge.components.transform.base import BaseTransform
from lab_knowledge.config import Settings
from lab_knowledge.factories.loader import ensure_providers
from lab_knowledge.factories.registry import ProviderRegistry
from lab_knowledge.ingestion.prepare import TransformTraceRecorder
from lab_knowledge.models import Chunk

registry: ProviderRegistry[BaseTransform] = ProviderRegistry()


def run_registered(
    chunks: list[Chunk],
    settings: Settings,
    recorder: TransformTraceRecorder,
) -> list[Chunk]:
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
        chunks = impl.apply(chunks, recorder)
    return chunks
