from __future__ import annotations

from wenmai.components.embedding.base import BaseEmbedding
from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseEmbedding] = ProviderRegistry()


def create(settings: Settings) -> BaseEmbedding:
    ensure_providers()
    return registry.create(
        settings.providers.embedding,
        model_name=settings.providers.embedding_model,
        behavior=settings.fake_behavior("embedding"),
    )
