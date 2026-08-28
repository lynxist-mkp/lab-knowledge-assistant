from __future__ import annotations

from wenmai.components.splitter.base import BaseSplitter
from wenmai.config import Settings
from wenmai.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseSplitter] = ProviderRegistry()


def create(settings: Settings) -> BaseSplitter:
    return registry.create(
        settings.providers.splitter,
        size_chars=settings.chunking.size_chars,
        overlap_chars=settings.chunking.overlap_chars,
        behavior=settings.fake_behavior("splitter"),
    )
