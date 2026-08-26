from __future__ import annotations

from pathlib import Path

from wenmai.components.fakes import apply_behavior
from wenmai.components.llm.base import BaseLLM
from wenmai.components.vision.base import BaseVisionLLM
from wenmai.factories.llm import registry, vision_registry

_ENRICHER_MARKER = "入库助手"
_QA_MARKER = "文脉助手"


@registry.register("fake")
class FakeLLM(BaseLLM):
    def __init__(self, model: str = "fake", behavior: str = "ok", **kwargs: object) -> None:
        self.model = model
        self.behavior = behavior

    @property
    def provider_name(self) -> str:
        return "fake"

    def generate(self, prompt: str) -> str:
        apply_behavior(self.behavior, "llm")
        if self.behavior == "garbage":
            return "<<<not-json>>>"
        if _QA_MARKER in prompt:
            return "湄洲岛是妈祖信仰的发源地，祖庙是信俗活动的中心场所[1]。"
        if _ENRICHER_MARKER in prompt:
            return (
                '{"title": "妈祖祖庙", "summary": "湄洲岛妈祖信仰中心", '
                '"tags": ["妈祖"], "culture_domain": "妈祖"}'
            )
        return "占位回答[1]。"


@vision_registry.register("fake")
class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, model: str = "fake", behavior: str = "ok", **kwargs: object) -> None:
        self.model = model
        self.behavior = behavior

    @property
    def provider_name(self) -> str:
        return "fake"

    def caption(self, image_path: Path, prompt: str) -> str:
        apply_behavior(self.behavior, "vision")
        if self.behavior == "garbage":
            return ""
        return f"图片占位说明：{image_path.name}"
