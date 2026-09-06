from __future__ import annotations

from lab_knowledge.components.splitter.base import BaseSplitter
from lab_knowledge.config import Settings
from lab_knowledge.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseSplitter] = ProviderRegistry()


def create(settings: Settings) -> BaseSplitter:
    return registry.create(
        settings.providers.splitter,
        size_chars=settings.chunking.size_chars,
        overlap_chars=settings.chunking.overlap_chars,
        behavior=settings.fake_behavior("splitter"),
    )
