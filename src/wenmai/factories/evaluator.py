from __future__ import annotations

from wenmai.components.evaluator.base import BaseEvaluator
from wenmai.config import Settings
from wenmai.factories.loader import ensure_providers
from wenmai.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseEvaluator] = ProviderRegistry()


def create(settings: Settings) -> BaseEvaluator:
    ensure_providers()
    return registry.create(
        settings.providers.evaluator,
        behavior=settings.fake_behavior("evaluator"),
    )
