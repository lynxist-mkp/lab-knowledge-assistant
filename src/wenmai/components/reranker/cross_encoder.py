from __future__ import annotations

from wenmai.components.reranker.base import BaseReranker
from wenmai.factories.reranker import registry


@registry.register("cross_encoder")
class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2", **kwargs: object) -> None:
        self.model_name = model_name
        self._model: object | None = None

    @property
    def provider_name(self) -> str:
        return "cross_encoder"

    def _get_model(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        if not texts:
            return []
        model = self._get_model()
        pairs = [[query, text] for text in texts]
        scores = model.predict(pairs)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(index, float(score)) for index, score in ranked]
