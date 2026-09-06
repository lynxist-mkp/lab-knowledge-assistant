from __future__ import annotations

from pathlib import Path

from lab_knowledge.components.fakes import apply_behavior
from lab_knowledge.components.gemma.client import GemmaMlxClient
from lab_knowledge.components.multimodal.base import BaseMultimodal
from lab_knowledge.config import Settings
from lab_knowledge.factories.multimodal import registry


@registry.register("mlx_gemma")
class MlxGemmaMultimodal(BaseMultimodal):
    def __init__(
        self,
        model: str = "",
        vision_model: str = "",
        behavior: str = "ok",
        vision_behavior: str | None = None,
        settings: Settings | None = None,
        **kwargs: object,
    ) -> None:
        if settings is None:
            raise ValueError("settings is required for mlx_gemma multimodal")
        text_model = model or settings.gemma.model
        image_model = vision_model or model or settings.gemma.model
        self.behavior = behavior
        self.vision_behavior = behavior if vision_behavior is None else vision_behavior
        self._text_client = GemmaMlxClient(settings, model=text_model)
        self._vision_client = (
            self._text_client
            if image_model == text_model
            else GemmaMlxClient(settings, model=image_model)
        )

    @property
    def provider_name(self) -> str:
        return self._text_client.provider_name

    def generate(self, prompt: str) -> str:
        apply_behavior(self.behavior, "llm")
        if self.behavior == "garbage":
            return "<<<not-json>>>"
        return self._text_client.generate_text(prompt)

    def caption(self, image_path: Path, prompt: str) -> str:
        apply_behavior(self.vision_behavior, "vision")
        if self.vision_behavior == "garbage":
            return ""
        return self._vision_client.caption_image(image_path, prompt)
