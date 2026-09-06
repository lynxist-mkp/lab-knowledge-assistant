from __future__ import annotations

import os

import httpx

from lab_knowledge.components.llm.base import BaseLLM
from lab_knowledge.factories.llm import registry

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEFAULT_TIMEOUT = 60.0


@registry.register("deepseek")
class DeepSeekLLM(BaseLLM):
    def __init__(self, model: str = "deepseek-v4-flash", **kwargs: object) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        self._model = model
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def generate(self, prompt: str) -> str:
        response = httpx.post(
            f"{_DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=_DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek response missing message content")
        return content.strip()
