from __future__ import annotations

import base64
import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from wenmai.components.mlx.server import MlxVlmProcessConfig, get_mlx_vlm_manager
from wenmai.config import PaddleOCR
from wenmai.storage.document_images import DocumentImages

SubprocessRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]

DISCARDED_LAYOUT_LABELS = frozenset({"header_image", "footer_image"})
TEXT_LAYOUT_LABELS = frozenset({"chart", "seal"})
IMAGE_LAYOUT_LABEL = "image"

_STDERR_EXCERPT_CHARS = 500


def build_text_from_ocr_payload(
    payload: dict[str, Any],
    images: DocumentImages,
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
                page_parts.append(
                    images.attach(
                        document_id=document_id,
                        source_path=source_path,
                        page=page_number,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                    )
                )
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
    *,
    vl_rec_api_model_name: str | None = None,
) -> list[str]:
    model_name = vl_rec_api_model_name or config.vl_rec_api_model_name
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
        model_name,
    ]


def _get_paddle_mlx_manager(config: PaddleOCR):
    resolve_script = Path(config.script).resolve().parent / "resolve_modelscope_model.py"
    fallback = config.mlx_fallback_model or None
    if fallback == config.mlx_model:
        fallback = None
    return get_mlx_vlm_manager(
        MlxVlmProcessConfig(
            mlx_python=config.mlx_python,
            server_port=config.server_port,
            server_url=config.server_url,
            model=config.mlx_model,
            resolve_script=str(resolve_script),
            idle_timeout_seconds=config.idle_timeout_seconds,
            fallback_model=fallback,
            reuse_healthy=True,
            ready_timeout_seconds=120.0,
        )
    )


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
    manager = server_manager or _get_paddle_mlx_manager(config)
    manager.ensure_running()
    active_model = getattr(manager, "active_mlx_model", None)
    vl_model = active_model or config.vl_rec_api_model_name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        result = run(
            _build_command(pdf, output_path, config, vl_rec_api_model_name=vl_model)
        )
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
