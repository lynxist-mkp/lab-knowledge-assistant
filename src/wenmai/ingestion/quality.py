"""入库质量门 — document-level effective-char ratio before load."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pypdfium2 as pdfium

from wenmai.config import QualityGate, Settings
from wenmai.ingestion.pdf_route import measure_pdf_chars_per_page

_EFFECTIVE_CHAR = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

QualityBand = Literal["reject", "approve", "gray"]


def effective_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    effective = len(_EFFECTIVE_CHAR.findall(text))
    return effective / len(text)


@dataclass(frozen=True)
class SourcePeek:
    text: str
    defer_reject: bool = False


@dataclass(frozen=True)
class QualityGateResult:
    ratio: float
    band: QualityBand


def evaluate_quality_gate(
    text: str,
    gate: QualityGate,
    *,
    defer_reject: bool = False,
) -> QualityGateResult:
    ratio = effective_char_ratio(text)
    if ratio < gate.reject_below:
        # Scanned PDF peek text is unreliable; route to 灰区 instead of hard reject or 已通过.
        band = "gray" if defer_reject else "reject"
    elif ratio > gate.approve_above:
        band = "approve"
    else:
        band = "gray"
    return QualityGateResult(ratio=ratio, band=band)


def peek_source(path: Path, settings: Settings) -> SourcePeek:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return SourcePeek(text=_peek_markdown_text(path))
    if suffix == ".pdf":
        text = _peek_pdf_text(path)
        scanned = measure_pdf_chars_per_page(path) < settings.pdf_load.chars_per_page_threshold
        return SourcePeek(text=text, defer_reject=scanned)
    raise ValueError(f"unsupported source type: {path.suffix}")


def peek_source_text(path: Path) -> str:
    return _peek_markdown_text(path) if path.suffix.lower() == ".md" else _peek_pdf_text(path)


def _peek_markdown_text(path: Path) -> str:
    raw = path.read_bytes().decode("utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return raw


def _peek_pdf_text(path: Path, *, max_pages: int | None = None) -> str:
    pdf = pdfium.PdfDocument(str(path))
    try:
        parts: list[str] = []
        limit = len(pdf) if max_pages is None else min(max_pages, len(pdf))
        for page_index in range(limit):
            page = pdf[page_index]
            textpage = page.get_textpage()
            parts.append(textpage.get_text_range())
        return "".join(parts)
    finally:
        pdf.close()


def peek_pdf_text(path: Path, *, max_pages: int | None = None) -> str:
    """Extract PDF text for 入库质量门 / 灰区复判 previews."""
    return _peek_pdf_text(path, max_pages=max_pages)
