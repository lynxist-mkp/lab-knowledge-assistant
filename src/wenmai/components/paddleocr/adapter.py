from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from wenmai.components.paddleocr.mlx_server import get_mlx_server_manager
from wenmai.config import PaddleOCR, PdfLoad
from wenmai.storage.images import ImageStore

SubprocessRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]

PdfRoute = str  # "markitdown" | "paddleocr-vl"
DISCARDED_LAYOUT_LABELS = frozenset({"header_image", "footer_image"})
TEXT_LAYOUT_LABELS = frozenset({"chart", "seal"})
IMAGE_LAYOUT_LABEL = "image"

_STDERR_EXCERPT_CHARS = 500


def measure_pdf_chars_per_page(path: Path) -> float:
    pdf = pdfium.PdfDocument(str(path))
    try:
        page_count = len(pdf)
        if page_count == 0:
            return 0.0
        total_chars = 0
        for page_index in range(page_count):
            page = pdf[page_index]
            textpage = page.get_textpage()
            total_chars += len(textpage.get_text_range().strip())
        return total_chars / page_count
    finally:
        pdf.close()


def choose_pdf_route(
    path: Path,
    pdf_load: PdfLoad,
    override_mode: str | None = None,
) -> PdfRoute:
    mode = (override_mode or pdf_load.mode or "auto").lower()
    if mode == "markitdown":
        return "markitdown"
    if mode == "ocr":
        return "paddleocr-vl"
    if mode != "auto":
        raise ValueError(f"unsupported pdf load mode: {mode}")

    chars_per_page = measure_pdf_chars_per_page(path)
    if chars_per_page >= pdf_load.chars_per_page_threshold:
        return "markitdown"
    return "paddleocr-vl"


def build_text_from_ocr_payload(
    payload: dict[str, Any],
    image_store: ImageStore,
    *,
    document_id: str,
    source_path: str,
) -> str:
    sections: list[str] = []
    for page in payload.get("pages") or []:
        page_number = int(page.get("page_number") or 1)
        page_parts: list[str] = []
        for block in page.get("blocks") or []:
            label = str(block.get("label") or "").lower()
            if label in DISCARDED_LAYOUT_LABELS:
                continue
            if label == IMAGE_LAYOUT_LABEL:
                image_b64 = block.get("image_b64")
                if not image_b64:
                    continue
                image_bytes = base64.b64decode(image_b64)
                mime_type = str(block.get("mime_type") or "image/png")
                image_id = image_store.save(
                    document_id=document_id,
                    source_path=source_path,
                    page=page_number,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                )
                page_parts.append(f"[IMAGE: {image_id}]")
                continue

            content = str(block.get("content") or "").strip()
            if content:
                page_parts.append(content)

        if page_parts:
            sections.append("\n\n".join(page_parts))
    return "\n\n".join(sections).strip()


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _stderr_excerpt(stderr: str) -> str:
    text = stderr.strip()
    if len(text) <= _STDERR_EXCERPT_CHARS:
        return text
    return text[:_STDERR_EXCERPT_CHARS] + "…"


def _build_command(
    pdf_path: Path,
    output_path: Path,
    config: PaddleOCR,
) -> list[str]:
    return [
        "env",
        "-u",
        "PYTHONHOME",
        "-u",
        "PYTHONPATH",
        config.python,
        config.script,
        str(pdf_path),
        "-o",
        str(output_path),
        "--vl-rec-backend",
        "mlx-vlm-server",
        "--vl-rec-server-url",
        config.server_url,
        "--vl-rec-api-model-name",
        config.vl_rec_api_model_name,
    ]


def parse_scanned_pdf(
    pdf_path: str | Path,
    *,
    config: PaddleOCR,
    runner: SubprocessRunner | None = None,
    server_manager: Any | None = None,
) -> dict[str, Any]:
    """Run PaddleOCR-VL in an isolated env and return structured layout JSON."""
    pdf = Path(pdf_path)
    run = runner or _default_runner
    manager = server_manager or get_mlx_server_manager(config)
    manager.ensure_running()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        result = run(_build_command(pdf, output_path, config))
        if result.returncode != 0:
            excerpt = _stderr_excerpt(result.stderr or "")
            detail = f" (stderr: {excerpt})" if excerpt else ""
            raise RuntimeError(
                f"PaddleOCR-VL parsing failed with exit code {result.returncode}{detail}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        manager.touch()
        return payload
    finally:
        output_path.unlink(missing_ok=True)
