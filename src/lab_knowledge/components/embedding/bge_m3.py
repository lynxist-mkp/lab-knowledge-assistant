from __future__ import annotations

import weakref

from lab_knowledge.components.embedding.base import BaseEmbedding
from lab_knowledge.components.model_guard import ModelResource, hold, register_unload
from lab_knowledge.factories.embedding import registry


@registry.register("bge_m3")
class BgeM3Embedding(BaseEmbedding):
    _instances: weakref.WeakSet[BgeM3Embedding] = weakref.WeakSet()

    def __init__(self, model_name: str = "BAAI/bge-m3", **kwargs: object) -> None:
        self.model_name = model_name
        self._model = None
        self._device: str | None = None
        BgeM3Embedding._instances.add(self)

    @property
    def provider_name(self) -> str:
        return "bge_m3"

    @property
    def dimension(self) -> int:
        return 1024

    def _load_model(self) -> object:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._device = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    @classmethod
    def unload_all(cls) -> None:
        for instance in list(cls._instances):
            instance.unload()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with hold(ModelResource.BGE_M3):
            model = self._load_model()
            vectors = model.encode(texts, normalize_embeddings=True)
            return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


register_unload(ModelResource.BGE_M3, BgeM3Embedding.unload_all)
