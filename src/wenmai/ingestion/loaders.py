from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import yaml
from markitdown import MarkItDown

from wenmai.config import Settings
from wenmai.storage.images import ImageStore

_IMAGE_PLACEHOLDER = "[IMAGE: {image_id}]"


@dataclass
class LoadedDocument:
    document_id: str
    text: str
    title: str
    url: str
    page: int
    source_path: str
    extra: dict[str, Any]


def load_source(path: Path, settings: Settings) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return load_markdown(path)
    if suffix == ".pdf":
        return load_pdf(path, settings)
    raise ValueError(f"unsupported source type: {path.suffix}")


def load_markdown(path: Path) -> LoadedDocument:
    raw_bytes = path.read_bytes()
    raw = raw_bytes.decode("utf-8")
    front_matter: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            parsed = yaml.safe_load(parts[1]) or {}
            if isinstance(parsed, dict):
                front_matter = parsed
            body = parts[2].lstrip("\n")
    title = str(front_matter.get("title") or _first_heading(body) or path.stem)
    url = str(front_matter.get("source_url") or front_matter.get("url") or "")
    page = int(front_matter.get("page") or 1)
    document_id = hashlib.sha256(raw_bytes).hexdigest()
    extra = {
        key: value
        for key, value in front_matter.items()
        if key not in {"title", "source_url", "url", "page"}
    }
    return LoadedDocument(
        document_id=document_id,
        text=body,
        title=title,
        url=url,
        page=page,
        source_path=str(path),
        extra=extra,
    )


def load_pdf(path: Path, settings: Settings) -> LoadedDocument:
    raw_bytes = path.read_bytes()
    document_id = hashlib.sha256(raw_bytes).hexdigest()
    source_path = str(path)

    markdown = MarkItDown().convert(str(path)).text_content.strip()
    image_store = ImageStore(settings)
    page_placeholders = _extract_pdf_images(path, image_store, document_id, source_path)
    text = _inject_image_placeholders(markdown, page_placeholders)

    return LoadedDocument(
        document_id=document_id,
        text=text,
        title=path.stem,
        url="",
        page=1,
        source_path=source_path,
        extra={"doc_type": "pdf"},
    )


def _first_heading(text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_pdf_images(
    path: Path,
    image_store: ImageStore,
    document_id: str,
    source_path: str,
) -> dict[int, list[str]]:
    placeholders_by_page: dict[int, list[str]] = {}
    pdf = pdfium.PdfDocument(str(path))
    try:
        for page_index in range(len(pdf)):
            page_number = page_index + 1
            page = pdf[page_index]
            for image_obj in page.get_objects(filter=(pdfium.raw.FPDF_PAGEOBJ_IMAGE,)):
                pil_image = image_obj.get_bitmap().to_pil()
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                image_id = image_store.save(
                    document_id=document_id,
                    source_path=source_path,
                    page=page_number,
                    image_bytes=buffer.getvalue(),
                    mime_type="image/png",
                )
                placeholders_by_page.setdefault(page_number, []).append(
                    _IMAGE_PLACEHOLDER.format(image_id=image_id)
                )
    finally:
        pdf.close()
    return placeholders_by_page


def _inject_image_placeholders(markdown: str, placeholders_by_page: dict[int, list[str]]) -> str:
    if not placeholders_by_page:
        return markdown

    sections: list[str] = []
    for page_number in sorted(placeholders_by_page):
        placeholders = placeholders_by_page[page_number]
        if page_number == 1:
            sections.append(markdown)
        sections.extend(placeholders)
    return "\n\n".join(section for section in sections if section)
