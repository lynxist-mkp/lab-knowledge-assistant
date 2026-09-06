from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSplitter(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def split(self, text: str) -> list[str]: ...
