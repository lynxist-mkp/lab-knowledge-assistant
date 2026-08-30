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
        if "user_input" in kwargs or "retrieved_contexts" in kwargs:
            return {
                "faithfulness": {"value": 0.9, "status": "ok"},
                "context_precision": {"value": 0.8, "status": "ok"},
            }
        return {"value": 0.0}
