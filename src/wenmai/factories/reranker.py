from __future__ import annotations

from wenmai.components.reranker.base import BaseReranker
from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseReranker] = ProviderRegistry()


def create(settings: Settings) -> BaseReranker:
    ensure_providers()
    return registry.create(
        settings.providers.reranker,
        model_name=settings.providers.reranker_model,
        behavior=settings.fake_behavior("reranker"),
    )
