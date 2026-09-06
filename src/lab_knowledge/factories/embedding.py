from __future__ import annotations

from lab_knowledge.components.embedding.base import BaseEmbedding
from lab_knowledge.config import Settings
from lab_knowledge.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseEmbedding] = ProviderRegistry()


def create(settings: Settings) -> BaseEmbedding:
    return registry.create(
        settings.providers.embedding,
        model_name=settings.providers.embedding_model,
        behavior=settings.fake_behavior("embedding"),
    )
