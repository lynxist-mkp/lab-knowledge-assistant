from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def guess_image_mime(path: Path) -> str:
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")


def image_to_data_url(image_path: Path) -> str:
    mime = guess_image_mime(image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_multimodal_user_content(text: str, image_path: Path) -> list[dict[str, Any]]:
    data_url = image_to_data_url(image_path)
    return [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": text},
    ]


def build_text_messages(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": prompt}]


def build_vision_messages(prompt: str, image_path: Path) -> list[dict[str, Any]]:
    return [{"role": "user", "content": build_multimodal_user_content(prompt, image_path)}]


def extract_message_content(
    payload: dict[str, Any],
    *,
    error_prefix: str = "chat completion",
) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"{error_prefix} response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError(f"{error_prefix} response choice is not an object")
    message = first.get("message") or {}
    if not isinstance(message, dict):
        raise RuntimeError(f"{error_prefix} response missing message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"{error_prefix} response missing message content")
    return content.strip()


def post_chat_completion(
    *,
    url: str,
    messages: list[dict[str, Any]],
    model: str,
    timeout: float,
    headers: dict[str, str] | None = None,
    **extra_json: Any,
) -> dict[str, Any]:
    response = httpx.post(
        url,
        headers=headers,
        json={"model": model, "messages": messages, **extra_json},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("chat completion response is not an object")
    return payload
