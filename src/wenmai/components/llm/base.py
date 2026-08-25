from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def generate(self, prompt: str) -> str: ...
