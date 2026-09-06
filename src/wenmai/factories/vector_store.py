from __future__ import annotations

from wenmai.components.vector_store.base import BaseVectorStore
from wenmai.config import Settings
from wenmai.factories.registry import ProviderRegistry
from wenmai.storage.paths import collection_storage_bindings

registry: ProviderRegistry[BaseVectorStore] = ProviderRegistry()


def create(settings: Settings) -> BaseVectorStore:
    bindings = collection_storage_bindings(settings)
    return registry.create(
        settings.providers.vector_store,
        persist_path=str(bindings.chroma_path),
        collection=bindings.collection_id,
        behavior=settings.fake_behavior("vector_store"),
    )
