from __future__ import annotations

from wenmai.components.embedding.base import BaseEmbedding
from wenmai.factories.embedding import registry


@registry.register("bge_m3")
class BgeM3Embedding(BaseEmbedding):
    def __init__(self, model_name: str = "BAAI/bge-m3", **kwargs: object) -> None:
        self.model_name = model_name
        self._model = None
        self._device: str | None = None

    @property
    def provider_name(self) -> str:
        return "bge_m3"

    @property
    def dimension(self) -> int:
        return 1024

    def _ensure(self) -> object:
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
