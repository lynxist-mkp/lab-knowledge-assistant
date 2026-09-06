"""Unit tests for shared OpenAI-shaped chat completion helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from lab_knowledge.components.chat_completions.client import (
    build_text_messages,
    build_vision_messages,
    extract_message_content,
    guess_image_mime,
    image_to_data_url,
    post_chat_completion,
)


def test_guess_image_mime_known_and_default(tmp_path: Path) -> None:
    assert guess_image_mime(tmp_path / "photo.jpg") == "image/jpeg"
    assert guess_image_mime(tmp_path / "photo.JPEG") == "image/jpeg"
    assert guess_image_mime(tmp_path / "photo.png") == "image/png"
    assert guess_image_mime(tmp_path / "photo.unknown") == "image/png"


def test_image_to_data_url_encodes_bytes(tmp_path: Path) -> None:
    image_path = tmp_path / "probe.png"
    image_path.write_bytes(b"\x89PNG\r\n")
    data_url = image_to_data_url(image_path)
    assert data_url.startswith("data:image/png;base64,")
    assert "iVBORw0KGgo" not in data_url  # tiny payload; just ensure prefix


def test_build_text_messages() -> None:
    messages = build_text_messages("hello")
    assert messages == [{"role": "user", "content": "hello"}]


def test_build_vision_messages_orders_image_before_text(tmp_path: Path) -> None:
    image_path = tmp_path / "a.webp"
    image_path.write_bytes(b"RIFF")
    messages = build_vision_messages("describe", image_path)
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/webp;base64,")
    assert content[1] == {"type": "text", "text": "describe"}


def test_extract_message_content_success() -> None:
    payload = {"choices": [{"message": {"content": "  answer  "}}]}
    assert extract_message_content(payload) == "answer"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({}, "missing choices"),
        ({"choices": []}, "missing choices"),
        ({"choices": ["bad"]}, "choice is not an object"),
        ({"choices": [{"message": "bad"}]}, "missing message"),
        ({"choices": [{"message": {"content": ""}}]}, "missing message content"),
        ({"choices": [{"message": {"content": "   "}}]}, "missing message content"),
        ({"choices": [{"message": {"content": 42}}]}, "missing message content"),
    ],
)
def test_extract_message_content_failures(payload: dict[str, object], match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        extract_message_content(payload, error_prefix="test")


def test_post_chat_completion_posts_json_and_returns_payload() -> None:
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    with patch(
        "lab_knowledge.components.chat_completions.client.httpx.post",
        return_value=mock_response,
    ) as post:
        payload = post_chat_completion(
            url="https://example.com/v1/chat/completions",
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            timeout=30.0,
            headers={"Authorization": "Bearer tok"},
            max_tokens=64,
        )

    assert payload["choices"][0]["message"]["content"] == "ok"
    post.assert_called_once_with(
        "https://example.com/v1/chat/completions",
        headers={"Authorization": "Bearer tok"},
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
        },
        timeout=30.0,
    )
