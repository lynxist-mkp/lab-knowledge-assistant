from __future__ import annotations

import os
from pathlib import Path

from wenmai.components.chat_completions.client import (
    build_text_messages,
    build_vision_messages,
    extract_message_content,
    post_chat_completion,
)
from wenmai.components.multimodal.base import BaseMultimodal
from wenmai.factories.multimodal import registry

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_TEXT_TIMEOUT = 60.0
_VISION_TIMEOUT = 90.0


@registry.register("deepseek")
class DeepSeekMultimodal(BaseMultimodal):
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        vision_model: str = "deepseek-v4-flash-vision-exp",
        **kwargs: object,
    ) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        self._model = model
        self._vision_model = vision_model or model
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def generate(self, prompt: str) -> str:
        payload = post_chat_completion(
            url=f"{_DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            model=self._model,
            messages=build_text_messages(prompt),
            timeout=_TEXT_TIMEOUT,
        )
        return extract_message_content(payload, error_prefix="DeepSeek")

    def caption(self, image_path: Path, prompt: str) -> str:
        payload = post_chat_completion(
            url=f"{_DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            model=self._vision_model,
            messages=build_vision_messages(prompt, image_path),
            timeout=_VISION_TIMEOUT,
        )
        return extract_message_content(payload, error_prefix="DeepSeek vision")
