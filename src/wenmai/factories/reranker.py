from __future__ import annotations

from wenmai.components.reranker.base import BaseReranker
from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseReranker] = ProviderRegistry()

_cache: dict[tuple[str, str, str], BaseReranker] = {}


def create(settings: Settings) -> BaseReranker:
    ensure_providers()
    key = (
        settings.providers.reranker,
        settings.providers.reranker_model,
        settings.fake_behavior("reranker"),
    )
    cached = _cache.get(key)
    if cached is not None:
        return cached
    reranker = registry.create(
        settings.providers.reranker,
        model_name=settings.providers.reranker_model,
        behavior=settings.fake_behavior("reranker"),
    )
    _cache[key] = reranker
    return reranker


def clear_cache() -> None:
    _cache.clear()
