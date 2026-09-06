from __future__ import annotations

from lab_knowledge.components.evaluator.base import BaseEvaluator
from lab_knowledge.config import Settings
from lab_knowledge.factories.loader import ensure_providers
from lab_knowledge.factories.registry import ProviderRegistry

registry: ProviderRegistry[BaseEvaluator] = ProviderRegistry()


def create(settings: Settings) -> BaseEvaluator:
    ensure_providers()
    return registry.create(
        settings.providers.evaluator,
        behavior=settings.fake_behavior("evaluator"),
        settings=settings,
    )
