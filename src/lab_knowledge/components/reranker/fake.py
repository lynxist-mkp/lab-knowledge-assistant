from __future__ import annotations

from lab_knowledge.components.fakes import apply_behavior
from lab_knowledge.components.reranker.base import BaseReranker
from lab_knowledge.factories.reranker import registry


@registry.register("fake")
class FakeReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "fake",
        behavior: str = "ok",
        **kwargs: object,
    ) -> None:
        self.model_name = model_name
        self.behavior = behavior

    @property
    def provider_name(self) -> str:
        return "fake"

    def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        apply_behavior(self.behavior, "reranker")
        if self.behavior == "garbage":
            return []
        return [(index, 1.0 - index * 0.01) for index in range(len(texts))]
