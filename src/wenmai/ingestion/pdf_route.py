"""PDF load routing for 文档解析 — digital vs scanned, owned by load."""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

from wenmai.config import PdfLoad

PdfRoute = str  # "markitdown" | "paddleocr-vl"
PDF_LOAD_MODES = frozenset({"auto", "markitdown", "ocr"})


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


def validate_pdf_load_mode(mode: str | None) -> None:
    if mode is None:
        return
    if mode.lower() not in PDF_LOAD_MODES:
        raise ValueError(f"unsupported pdf load mode: {mode!r}")


def choose_pdf_route(
    path: Path,
    pdf_load: PdfLoad,
    override_mode: str | None = None,
) -> PdfRoute:
    validate_pdf_load_mode(override_mode)
    mode = (override_mode or pdf_load.mode or "auto").lower()
    if mode == "markitdown":
        return "markitdown"
    if mode == "ocr":
        return "paddleocr-vl"
    if mode != "auto":
        raise ValueError(f"unsupported pdf load mode: {mode!r}")

    chars_per_page = measure_pdf_chars_per_page(path)
    if chars_per_page >= pdf_load.chars_per_page_threshold:
        return "markitdown"
    return "paddleocr-vl"
