from __future__ import annotations

import weakref

from lab_knowledge.components.model_guard import ModelResource, hold, register_unload
from lab_knowledge.components.reranker.base import BaseReranker
from lab_knowledge.factories.reranker import registry


@registry.register("cross_encoder")
class CrossEncoderReranker(BaseReranker):
    _instances: weakref.WeakSet[CrossEncoderReranker] = weakref.WeakSet()

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        **kwargs: object,
    ) -> None:
        self.model_name = model_name
        self._model: object | None = None
        CrossEncoderReranker._instances.add(self)

    @property
    def provider_name(self) -> str:
        return "cross_encoder"

    def _load_model(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def unload(self) -> None:
        self._model = None

    @classmethod
    def unload_all(cls) -> None:
        for instance in list(cls._instances):
            instance.unload()

    def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        if not texts:
            return []
        with hold(ModelResource.CROSS_ENCODER):
            model = self._load_model()
            pairs = [[query, text] for text in texts]
            scores = model.predict(pairs)
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            return [(index, float(score)) for index, score in ranked]


register_unload(ModelResource.CROSS_ENCODER, CrossEncoderReranker.unload_all)
