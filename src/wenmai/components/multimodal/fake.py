from __future__ import annotations

from pathlib import Path

from wenmai.components.fakes import apply_behavior
from wenmai.components.multimodal.base import BaseMultimodal
from wenmai.factories.multimodal import registry

_ENRICHER_MARKER = "入库助手"
_QA_MARKER = "文脉助手"
_REFUSAL_PREFIX = "拒答："


@registry.register("fake")
class FakeMultimodal(BaseMultimodal):
    def __init__(
        self,
        model: str = "fake",
        behavior: str = "ok",
        vision_behavior: str | None = None,
        **kwargs: object,
    ) -> None:
        self.model = model
        self.behavior = behavior
        self.vision_behavior = behavior if vision_behavior is None else vision_behavior

    @property
    def provider_name(self) -> str:
        return "fake"

    def generate(self, prompt: str) -> str:
        apply_behavior(self.behavior, "llm")
        if self.behavior == "garbage":
            return "<<<not-json>>>"
        if self.behavior == "refuse":
            return f"{_REFUSAL_PREFIX}检索片段不足以回答该问题。"
        if _QA_MARKER in prompt:
            return "湄洲岛是妈祖信仰的发源地，祖庙是信俗活动的中心场所[1]。"
        if _ENRICHER_MARKER in prompt:
            return (
                '{"title": "妈祖祖庙", "summary": "湄洲岛妈祖信仰中心", '
                '"tags": ["妈祖"], "culture_domain": "妈祖"}'
            )
        return "占位回答[1]。"

    def caption(self, image_path: Path, prompt: str) -> str:
        apply_behavior(self.vision_behavior, "vision")
        if self.vision_behavior == "garbage":
            return ""
        return f"图片占位说明：{image_path.name}"
