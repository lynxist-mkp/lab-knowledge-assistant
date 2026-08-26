from __future__ import annotations

import re
from pathlib import Path

from wenmai.components.fakes import apply_behavior
from wenmai.components.llm.base import BaseLLM
from wenmai.components.vision.base import BaseVisionLLM
from wenmai.factories.llm import registry, vision_registry

_ENRICHER_MARKER = "入库助手"
_QA_MARKER = "文脉助手"
_REFUSAL_PREFIX = "拒答："
_QUESTION_STOP_TERMS = frozenset(
    {"什么", "时候", "哪里", "如何", "为什么", "创办", "是什么", "多少", "哪些"}
)


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
            if _qa_should_refuse(prompt):
                return _qa_refusal_response(prompt)
            return "湄洲岛是妈祖信仰的发源地，祖庙是信俗活动的中心场所[1]。"
        if _ENRICHER_MARKER in prompt:
            return (
                '{"title": "妈祖祖庙", "summary": "湄洲岛妈祖信仰中心", '
                '"tags": ["妈祖"], "culture_domain": "妈祖"}'
            )
        return "占位回答[1]。"


def _qa_prompt_sections(prompt: str) -> tuple[str, str]:
    question_match = re.search(r"问题：\s*\n(.+?)\n\n检索片段", prompt, re.DOTALL)
    context_match = re.search(r"检索片段：\s*\n(.+)$", prompt, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""
    context = context_match.group(1) if context_match else ""
    return question, context


def _qa_question_terms(question: str) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fff]{2,}", question)
    return [term for term in terms if term not in _QUESTION_STOP_TERMS]


def _qa_should_refuse(prompt: str) -> bool:
    question, context = _qa_prompt_sections(prompt)
    if not question or not context:
        return False
    terms = _qa_question_terms(question)
    if not terms:
        return False
    return not any(term in context for term in terms)


def _qa_refusal_response(prompt: str) -> str:
    _, context = _qa_prompt_sections(prompt)
    indices = sorted({int(match) for match in re.findall(r"\[(\d+)\]", context)})
    if not indices:
        indices = [1]
    source_refs = "、".join(f"[{index}]" for index in indices)
    return f"{_REFUSAL_PREFIX}检索片段不足以回答该问题。实际检索到的来源：{source_refs}"


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
