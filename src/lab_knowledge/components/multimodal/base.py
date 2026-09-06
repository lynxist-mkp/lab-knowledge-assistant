from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseMultimodal(ABC):
    """One provider for 生成 text and 图转文 caption."""

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def generate(self, prompt: str) -> str: ...

    @abstractmethod
    def caption(self, image_path: Path, prompt: str) -> str: ...
