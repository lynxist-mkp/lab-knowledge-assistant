from __future__ import annotations

from lab_knowledge.components.vector_store.base import BaseVectorStore
from lab_knowledge.config import Settings
from lab_knowledge.factories.registry import ProviderRegistry
from lab_knowledge.storage.paths import collection_storage_bindings

registry: ProviderRegistry[BaseVectorStore] = ProviderRegistry()


def create(settings: Settings) -> BaseVectorStore:
    bindings = collection_storage_bindings(settings)
    return registry.create(
        settings.providers.vector_store,
        persist_path=str(bindings.chroma_persist_path()),
        collection=bindings.collection_id,
        behavior=settings.fake_behavior("vector_store"),
    )
