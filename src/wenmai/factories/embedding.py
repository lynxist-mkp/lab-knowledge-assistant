from __future__ import annotations

from wenmai.components.embedding.base import BaseEmbedding
from wenmai.config import Settings
from wenmai.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseEmbedding] = ProviderRegistry()


def create(settings: Settings) -> BaseEmbedding:
    return registry.create(
        settings.providers.embedding,
        model_name=settings.providers.embedding_model,
        behavior=settings.fake_behavior("embedding"),
    )
