from __future__ import annotations

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]: ...
