from __future__ import annotations

from pathlib import Path

from lab_knowledge.components.fakes import apply_behavior
from lab_knowledge.components.multimodal.base import BaseMultimodal
from lab_knowledge.factories.multimodal import registry

_ENRICHER_MARKER = "入库助手"
_QA_MARKER = "课题组知识助手"
_MULTI_QUERY_MARKER = "多路改写"
_GRAY_REVIEW_MARKER = "灰区复判"
_GRAY_REJECT_HINT = "不值得入库"
_REFUSAL_PREFIX = "拒答："


def _behavior_applies_to_prompt(behavior: str, prompt: str) -> bool:
    if behavior == "error":
        return True
    if behavior != "timeout":
        return False
    if _QA_MARKER in prompt:
        return False
    return any(
        marker in prompt
        for marker in (
            _MULTI_QUERY_MARKER,
            _ENRICHER_MARKER,
            _GRAY_REVIEW_MARKER,
        )
    )


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
        if self.behavior == "error" and _behavior_applies_to_prompt(
            self.behavior, prompt
        ):
            apply_behavior(self.behavior, "llm")
        if self.behavior == "timeout" and _behavior_applies_to_prompt(
            self.behavior, prompt
        ):
            apply_behavior(self.behavior, "llm")
        if self.behavior == "garbage":
            return "<<<not-json>>>"
        if self.behavior == "refuse":
            return f"{_REFUSAL_PREFIX}检索片段不足以回答该问题。"
        if _GRAY_REVIEW_MARKER in prompt:
            return "不通过" if _GRAY_REJECT_HINT in prompt else "通过"
        if _MULTI_QUERY_MARKER in prompt:
            return (
                "马尾船政学堂是哪年创立的？\n"
                "船政学院创办年份是多少？\n"
                "1866年成立的船政教育机构何时建立？"
            )
        if _QA_MARKER in prompt:
            return "湄洲岛是妈祖信仰的发源地，祖庙是信俗活动的中心场所[1]。"
        if _ENRICHER_MARKER in prompt:
            return (
                '{"title": "妈祖祖庙", "summary": "湄洲岛妈祖信仰中心", '
                '"tags": ["妈祖"], "culture_domain": "检索增强"}'
            )
        return "占位回答[1]。"

    def caption(self, image_path: Path, prompt: str) -> str:
        apply_behavior(self.vision_behavior, "vision")
        if self.vision_behavior == "garbage":
            return ""
        if _GRAY_REVIEW_MARKER in prompt:
            return "不通过" if _GRAY_REJECT_HINT in prompt else "通过"
        return f"图片占位说明：{image_path.name}"
