from __future__ import annotations

import base64
from pathlib import Path

import httpx

from wenmai.components.gemma.mlx_server import get_gemma_server_manager
from wenmai.config import Settings

_DEFAULT_TIMEOUT = 120.0
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _guess_mime(path: Path) -> str:
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")


def _extract_message_content(payload: dict[str, object]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("mlx_vlm response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("mlx_vlm response choice is not an object")
    message = first.get("message") or {}
    if not isinstance(message, dict):
        raise RuntimeError("mlx_vlm response missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("mlx_vlm response missing message content")
    return content.strip()


class GemmaMlxClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._config = settings.gemma
        self._manager = get_gemma_server_manager(self._config)
        self._chat_url = self._config.server_url.rstrip("/") + "/v1/chat/completions"

    @property
    def provider_name(self) -> str:
        return "mlx_gemma"

    def generate_text(self, prompt: str, *, max_tokens: int = 512) -> str:
        self._manager.ensure_running()
        try:
            response = httpx.post(
                self._chat_url,
                json={
                    "model": self._config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout=_DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return _extract_message_content(response.json())
        finally:
            self._manager.touch()

    def caption_image(self, image_path: Path, prompt: str, *, max_tokens: int = 256) -> str:
        image_bytes = image_path.read_bytes()
        mime = _guess_mime(image_path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{encoded}"
        self._manager.ensure_running()
        try:
            response = httpx.post(
                self._chat_url,
                json={
                    "model": self._config.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout=_DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return _extract_message_content(response.json())
        finally:
            self._manager.touch()
