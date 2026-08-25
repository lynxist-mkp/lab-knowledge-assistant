from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseVisionLLM(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def caption(self, image_path: Path, prompt: str) -> str: ...
