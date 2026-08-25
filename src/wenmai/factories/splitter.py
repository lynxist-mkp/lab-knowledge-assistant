from __future__ import annotations

from wenmai.components.splitter.base import BaseSplitter
from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseSplitter] = ProviderRegistry()


def create(settings: Settings) -> BaseSplitter:
    ensure_providers()
    return registry.create(
        settings.providers.splitter,
        size_chars=settings.chunking.size_chars,
        overlap_chars=settings.chunking.overlap_chars,
        behavior=settings.fake_behavior("splitter"),
    )
