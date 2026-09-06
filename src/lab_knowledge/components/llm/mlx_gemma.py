from __future__ import annotations

from lab_knowledge.components.fakes import apply_behavior
from lab_knowledge.components.gemma.client import GemmaMlxClient
from lab_knowledge.components.llm.base import BaseLLM
from lab_knowledge.config import Settings
from lab_knowledge.factories.llm import registry


@registry.register("mlx_gemma")
class MlxGemmaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "",
        behavior: str = "ok",
        settings: Settings | None = None,
        **kwargs: object,
    ) -> None:
        if settings is None:
            raise ValueError("settings is required for mlx_gemma LLM")
        self._model = model or settings.gemma.model
        self.behavior = behavior
        self._client = GemmaMlxClient(settings)

    @property
    def provider_name(self) -> str:
        return self._client.provider_name

    def generate(self, prompt: str) -> str:
        apply_behavior(self.behavior, "llm")
        if self.behavior == "garbage":
            return "<<<not-json>>>"
        return self._client.generate_text(prompt)
