from __future__ import annotations

from typing import Any

from wenmai.components.evaluator.base import BaseEvaluator
from wenmai.components.fakes import apply_behavior
from wenmai.factories.evaluator import registry


@registry.register("fake")
class FakeEvaluator(BaseEvaluator):
    def __init__(self, behavior: str = "ok", **kwargs: object) -> None:
        self.behavior = behavior

    @property
    def provider_name(self) -> str:
        return "fake"

    def score(self, **kwargs: Any) -> dict[str, Any]:
        apply_behavior(self.behavior, "evaluator")
        if self.behavior == "garbage":
            return {}
        return {"value": 0.0}
