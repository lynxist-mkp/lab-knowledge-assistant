from __future__ import annotations

from wenmai.components.multimodal.base import BaseMultimodal
from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseMultimodal] = ProviderRegistry()


def create(settings: Settings) -> BaseMultimodal:
    """Create the multimodal provider used for 生成 and 图转文."""
    ensure_providers()
    return registry.create(
        settings.providers.multimodal,
        model=settings.providers.multimodal_model,
        vision_model=settings.providers.caption_model or settings.providers.multimodal_model,
        behavior=settings.fake_behavior("multimodal"),
        vision_behavior=settings.fake_behavior("caption"),
        settings=settings,
    )
