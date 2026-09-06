#!/usr/bin/env python3
"""Run PaddleOCR-VL document parsing in an isolated environment.

Outputs a JSON payload consumed by lab_knowledge.components.paddleocr.adapter.
Must be invoked via subprocess from the main RAG venv — do not import paddleocr
from the main project.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

DISCARDED_LAYOUT_LABELS = frozenset({"header_image", "footer_image"})
IMAGE_LAYOUT_LABEL = "image"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a PDF with PaddleOCR-VL.")
    parser.add_argument("input", type=Path, help="PDF file path")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument(
        "--vl-rec-backend",
        default="mlx-vlm-server",
        help="VLM backend for PaddleOCR-VL",
    )
    parser.add_argument(
        "--vl-rec-server-url",
        default="http://localhost:8111/",
        help="mlx-vlm-server base URL",
    )
    parser.add_argument(
        "--vl-rec-api-model-name",
        default="mlx-community/PaddleOCR-VL-1.6-5bit",
        help="Model name exposed by mlx-vlm-server",
    )
    return parser.parse_args()


def _image_to_b64(image: Any) -> tuple[str, str]:
    from PIL import Image

    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii"), "image/png"


def _block_from_layout_item(item: Any) -> dict[str, Any] | None:
    label = str(getattr(item, "label", None) or item.get("label") or "").lower()
    if not label:
        return None
    if label in DISCARDED_LAYOUT_LABELS:
        return {"label": label}

    content = getattr(item, "content", None) or item.get("content")
    image = getattr(item, "image", None) or item.get("image")

    block: dict[str, Any] = {"label": label}
    if label == IMAGE_LAYOUT_LABEL and image is not None:
        image_b64, mime_type = _image_to_b64(image)
        block["image_b64"] = image_b64
        block["mime_type"] = mime_type
        return block

    if content:
        block["content"] = str(content).strip()
    return block


def _blocks_from_markdown_dict(markdown: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    texts = str(markdown.get("markdown_texts") or "").strip()
    if texts and "<img" not in texts:
        blocks.append({"label": "text", "content": texts})
    images = markdown.get("markdown_images") or {}
    for image in images.values():
        if image is not None:
            image_b64, mime_type = _image_to_b64(image)
            blocks.append(
                {"label": "image", "image_b64": image_b64, "mime_type": mime_type}
            )
    return blocks


def _extract_pages(results: list[Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for page_index, result in enumerate(results, start=1):
        blocks: list[dict[str, Any]] = []

        layout_items = getattr(result, "layout_parsing_result", None)
        if layout_items is not None:
            raw_blocks = getattr(layout_items, "blocks", None) or layout_items.get("blocks")
            if raw_blocks:
                for item in raw_blocks:
                    block = _block_from_layout_item(item)
                    if block is not None:
                        blocks.append(block)

        parsing_res = getattr(result, "parsing_res_list", None)
        if parsing_res:
            for item in parsing_res:
                block = _block_from_layout_item(item)
                if block is not None:
                    blocks.append(block)

        markdown = getattr(result, "markdown", None)
        if markdown is None and hasattr(result, "get"):
            markdown = result.get("markdown")
        if markdown and not blocks:
            if isinstance(markdown, dict):
                blocks.extend(_blocks_from_markdown_dict(markdown))
            else:
                blocks.append({"label": "text", "content": str(markdown).strip()})

        pages.append({"page_number": page_index, "blocks": blocks})
    return pages


def main() -> int:
    args = _parse_args()
    if not args.input.is_file():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 2

    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(
        vl_rec_backend=args.vl_rec_backend,
        vl_rec_server_url=args.vl_rec_server_url,
        vl_rec_api_model_name=args.vl_rec_api_model_name,
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_layout_detection=True,
        use_chart_recognition=True,
        use_seal_recognition=True,
    )

    results = pipeline.predict(str(args.input))
    payload = {"pages": _extract_pages(results)}
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
