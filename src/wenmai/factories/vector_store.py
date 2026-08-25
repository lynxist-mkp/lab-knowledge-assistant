from __future__ import annotations

from wenmai.components.vector_store.base import BaseVectorStore
from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.factories.registry import ProviderRegistry
from wenmai.storage.paths import store_path

registry: ProviderRegistry[BaseVectorStore] = ProviderRegistry()


def create(settings: Settings) -> BaseVectorStore:
    ensure_providers()
    return registry.create(
        settings.providers.vector_store,
        persist_path=str(store_path(settings, "chroma")),
        collection=settings.product.collection,
        behavior=settings.fake_behavior("vector_store"),
    )
