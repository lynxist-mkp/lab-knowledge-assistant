"""Narrow seam: GemmaMlxClient talks to mlx_vlm.server without loading weights."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from wenmai.components.gemma.client import GemmaMlxClient
from wenmai.config import Settings


def _settings() -> Settings:
    repo = Path(__file__).resolve().parents[1]
    return Settings.load(repo / "settings.yaml")


def test_generate_text_posts_chat_completion_and_touches_server() -> None:
    settings = _settings()
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "湄洲岛是妈祖信仰的发源地[1]。"}}]
    }
    mock_manager = MagicMock()

    with (
        patch("wenmai.components.gemma.client.get_mlx_vlm_manager", return_value=mock_manager),
        patch("httpx.post", return_value=mock_response) as post,
    ):
        text = GemmaMlxClient(settings).generate_text("妈祖信仰的发源地在哪里？")

    assert "妈祖" in text
    post.assert_called_once()
    url = post.call_args.args[0]
    body = post.call_args.kwargs["json"]
    assert url.endswith("/v1/chat/completions")
    assert body["messages"][0]["content"] == "妈祖信仰的发源地在哪里？"
    mock_manager.ensure_running.assert_called_once()
    mock_manager.touch.assert_called_once()


def test_caption_image_sends_image_url_and_prompt() -> None:
    settings = _settings()
    image_path = Path(__file__).resolve().parent / "_fixtures" / "probe.png"
    image_path.parent.mkdir(exist_ok=True)
    if not image_path.exists():
        image_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x04\x00\x01\x8d\x0b\x0b\x04\x00\x00\x00\x00IEND\xaeB`\x82"
        )

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "图片主要是红色和黄色。"}}]
    }
    mock_manager = MagicMock()

    with (
        patch("wenmai.components.gemma.client.get_mlx_vlm_manager", return_value=mock_manager),
        patch("httpx.post", return_value=mock_response) as post,
    ):
        text = GemmaMlxClient(settings).caption_image(image_path, "描述颜色")

    assert "红色" in text
    body = post.call_args.kwargs["json"]
    content = body["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["text"] == "描述颜色"


def test_mlx_gemma_multimodal_provider_uses_client() -> None:
    from wenmai.factories import multimodal as multimodal_factory

    settings = _settings()
    settings.fakes["multimodal"] = "ok"
    mock_client = MagicMock()
    mock_client.provider_name = "mlx_gemma"
    mock_client.generate_text.return_value = '{"title":"t"}'
    mock_client.caption_image.return_value = "图说明"

    with patch(
        "wenmai.components.multimodal.mlx_gemma.GemmaMlxClient", return_value=mock_client
    ):
        provider = multimodal_factory.create(settings)
        assert provider.provider_name == "mlx_gemma"
        assert provider.generate("test") == '{"title":"t"}'
        mock_client.generate_text.assert_called_once()
        image = Path(__file__).resolve().parent / "_fixtures" / "probe.png"
        assert provider.caption(image, "prompt") == "图说明"
        mock_client.caption_image.assert_called_once()
