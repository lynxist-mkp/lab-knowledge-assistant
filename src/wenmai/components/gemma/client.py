from __future__ import annotations

from pathlib import Path

from wenmai.components.chat_completions.client import (
    build_text_messages,
    build_vision_messages,
    extract_message_content,
    post_chat_completion,
)
from wenmai.components.mlx.server import MlxVlmProcessConfig, get_mlx_vlm_manager
from wenmai.config import Settings

_DEFAULT_TIMEOUT = 120.0


class GemmaMlxClient:
    def __init__(self, settings: Settings, *, model: str | None = None) -> None:
        self._settings = settings
        self._config = settings.gemma
        self._model = model or self._config.model
        self._manager = get_mlx_vlm_manager(
            MlxVlmProcessConfig(
                mlx_python=self._config.mlx_python,
                server_port=self._config.server_port,
                server_url=self._config.server_url,
                model=self._config.model,
                resolve_script=self._config.resolve_script,
                idle_timeout_seconds=self._config.idle_timeout_seconds,
                reuse_healthy=True,
                ready_timeout_seconds=180.0,
            )
        )
        self._chat_url = self._config.server_url.rstrip("/") + "/v1/chat/completions"

    @property
    def provider_name(self) -> str:
        return "mlx_gemma"

    def generate_text(self, prompt: str, *, max_tokens: int = 512) -> str:
        self._manager.ensure_running()
        try:
            payload = post_chat_completion(
                url=self._chat_url,
                model=self._model,
                messages=build_text_messages(prompt),
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=_DEFAULT_TIMEOUT,
            )
            return extract_message_content(payload, error_prefix="mlx_vlm")
        finally:
            self._manager.touch()

    def caption_image(self, image_path: Path, prompt: str, *, max_tokens: int = 256) -> str:
        self._manager.ensure_running()
        try:
            payload = post_chat_completion(
                url=self._chat_url,
                model=self._model,
                messages=build_vision_messages(prompt, image_path),
                max_tokens=max_tokens,
                temperature=0.0,
                timeout=_DEFAULT_TIMEOUT,
            )
            return extract_message_content(payload, error_prefix="mlx_vlm")
        finally:
            self._manager.touch()
