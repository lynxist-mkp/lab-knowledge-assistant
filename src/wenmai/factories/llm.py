from __future__ import annotations

from wenmai.components.llm.base import BaseLLM
from wenmai.components.vision.base import BaseVisionLLM
from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseLLM] = ProviderRegistry()
vision_registry: ProviderRegistry[BaseVisionLLM] = ProviderRegistry()


def create(settings: Settings) -> BaseLLM:
    ensure_providers()
    return registry.create(
        settings.providers.llm,
        model=settings.providers.llm_model,
        behavior=settings.fake_behavior("llm"),
        settings=settings,
    )


def create_vision(settings: Settings) -> BaseVisionLLM:
    ensure_providers()
    return vision_registry.create(
        settings.providers.vision,
        model=settings.providers.vision_model,
        behavior=settings.fake_behavior("vision"),
        settings=settings,
    )
