from __future__ import annotations

import hashlib

from lab_knowledge.components.embedding.base import BaseEmbedding
from lab_knowledge.components.fakes import apply_behavior
from lab_knowledge.factories.embedding import registry


def _vector(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for index in range(dim):
        byte = digest[index % len(digest)]
        values.append((byte / 255.0) * 2.0 - 1.0)
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


@registry.register("fake")
class FakeEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str = "fake",
        behavior: str = "ok",
        dimension: int = 8,
        **kwargs: object,
    ) -> None:
        self._model_name = model_name
        self.behavior = behavior
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _guard(self) -> None:
        apply_behavior(self.behavior, "embedding")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._guard()
        if self.behavior == "garbage":
            return [[0.0] * self._dimension for _ in texts]
        return [_vector(text, self._dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
