from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from lab_knowledge.components.fakes import apply_behavior
from lab_knowledge.components.splitter.base import BaseSplitter
from lab_knowledge.factories.splitter import registry

# 段落 → 句号问号叹号 → 分号逗号 → 字符
ZH_SEPARATORS = ["\n\n", "。", "？", "！", "；", "，", ""]


@registry.register("recursive_zh")
class RecursiveZhSplitter(BaseSplitter):
    def __init__(
        self, size_chars: int = 600, overlap_chars: int = 90, **kwargs: object
    ) -> None:
        self.size_chars = size_chars
        self.overlap_chars = overlap_chars
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=size_chars,
            chunk_overlap=overlap_chars,
            separators=ZH_SEPARATORS,
            keep_separator=True,
        )

    @property
    def provider_name(self) -> str:
        return "recursive_zh"

    def split(self, text: str) -> list[str]:
        pieces = [piece.strip() for piece in self._splitter.split_text(text)]
        return [piece for piece in pieces if piece]


@registry.register("fake")
class FakeSplitter(BaseSplitter):
    def __init__(
        self,
        size_chars: int = 600,
        overlap_chars: int = 90,
        behavior: str = "ok",
        **kwargs: object,
    ) -> None:
        self.size_chars = size_chars
        self.overlap_chars = overlap_chars
        self.behavior = behavior

    @property
    def provider_name(self) -> str:
        return "fake"

    def split(self, text: str) -> list[str]:
        apply_behavior(self.behavior, "splitter")
        if self.behavior == "garbage":
            return ["<<<garbage>>>"]
        stripped = text.strip()
        return [stripped] if stripped else []
