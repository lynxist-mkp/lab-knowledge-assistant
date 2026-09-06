from __future__ import annotations

from pathlib import Path

from lab_knowledge.components.fakes import apply_behavior
from lab_knowledge.components.gemma.client import GemmaMlxClient
from lab_knowledge.components.vision.base import BaseVisionLLM
from lab_knowledge.config import Settings
from lab_knowledge.factories.llm import vision_registry


@vision_registry.register("mlx_gemma")
class MlxGemmaVisionLLM(BaseVisionLLM):
    def __init__(
        self,
        model: str = "",
        behavior: str = "ok",
        settings: Settings | None = None,
        **kwargs: object,
    ) -> None:
        if settings is None:
            raise ValueError("settings is required for mlx_gemma vision")
        self._model = model or settings.gemma.model
        self.behavior = behavior
        self._client = GemmaMlxClient(settings)

    @property
    def provider_name(self) -> str:
        return self._client.provider_name

    def caption(self, image_path: Path, prompt: str) -> str:
        apply_behavior(self.behavior, "vision")
        if self.behavior == "garbage":
            return ""
        return self._client.caption_image(image_path, prompt)
