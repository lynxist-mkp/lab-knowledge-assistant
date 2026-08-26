from __future__ import annotations

import base64
import os
from pathlib import Path

import httpx

from wenmai.components.vision.base import BaseVisionLLM
from wenmai.factories.llm import vision_registry

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEFAULT_TIMEOUT = 90.0
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _guess_mime(path: Path) -> str:
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")


@vision_registry.register("deepseek")
class DeepSeekVisionLLM(BaseVisionLLM):
    def __init__(self, model: str = "deepseek-v4-flash-vision-exp", **kwargs: object) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        self._model = model
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def caption(self, image_path: Path, prompt: str) -> str:
        image_bytes = image_path.read_bytes()
        mime = _guess_mime(image_path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{encoded}"
        response = httpx.post(
            f"{_DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            },
            timeout=_DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek vision response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek vision response missing message content")
        return content.strip()
