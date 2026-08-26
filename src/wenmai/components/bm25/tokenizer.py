from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jieba

from wenmai.config import Settings


@dataclass(frozen=True)
class ChineseTokenizer:
    domain_dict_path: Path
    stopwords_path: Path
    root: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> ChineseTokenizer:
        return cls(
            domain_dict_path=_resolve_path(settings, settings.bm25.domain_dict),
            stopwords_path=_resolve_path(settings, settings.bm25.stopwords),
            root=settings.root,
        )

    def __post_init__(self) -> None:
        for word in _load_lines(self.domain_dict_path):
            jieba.add_word(word, freq=10_000)
        object.__setattr__(self, "_stopwords", _load_lines(self.stopwords_path))

    def tokenize(self, text: str) -> list[str]:
        tokens = jieba.lcut(text)
        return [token for token in tokens if token.strip() and token not in self._stopwords]


def _resolve_path(settings: Settings, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = settings.root / path
    return path


def _load_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}
