"""Source semantics and basic literature metadata for single-file ingest."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pypdfium2 as pdfium

SourceKind = Literal["group_doc", "personal_literature"]

SOURCE_KIND_GROUP: SourceKind = "group_doc"
SOURCE_KIND_PERSONAL: SourceKind = "personal_literature"

SOURCE_KIND_LABELS: dict[SourceKind, str] = {
    SOURCE_KIND_GROUP: "组内资料",
    SOURCE_KIND_PERSONAL: "个人文献库",
}

_YEAR_IN_FILENAME_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
_AUTHOR_TITLE_YEAR_RE = re.compile(
    r"^(?P<authors>.+?)\s*[-–—]\s*(?P<title>.+?)(?:\s*[\[(（](?P<year>(?:19|20)\d{2})[\])）])?$"
)
_YEAR_PREFIX_RE = re.compile(r"^(?:(?:19|20)\d{2})\s*[-–—]\s*(?P<rest>.+)$")


@dataclass(frozen=True)
class LiteratureMetadataOverrides:
    title: str | None = None
    authors: str | None = None
    year: int | None = None


@dataclass(frozen=True)
class ResolvedSourceMetadata:
    source_kind: SourceKind
    title: str
    authors: str
    publication_year: int | None

    def chunk_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "source_kind": self.source_kind,
            "source_label": SOURCE_KIND_LABELS[self.source_kind],
        }
        if self.authors:
            fields["authors"] = self.authors
        if self.publication_year is not None:
            fields["publication_year"] = self.publication_year
        return fields

    def trace_fields(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "source_label": SOURCE_KIND_LABELS[self.source_kind],
            "authors": self.authors,
            "publication_year": self.publication_year,
        }


def normalize_source_kind(value: object, *, default: SourceKind = SOURCE_KIND_GROUP) -> SourceKind:
    if value in (SOURCE_KIND_GROUP, SOURCE_KIND_PERSONAL):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in SOURCE_KIND_LABELS.values():
            for kind, label in SOURCE_KIND_LABELS.items():
                if label == normalized:
                    return kind
    return default


def extract_frontmatter_literature(front_matter: dict[str, Any]) -> dict[str, object]:
    extracted: dict[str, object] = {}
    source_kind = front_matter.get("source_kind") or front_matter.get("资料来源")
    if source_kind is not None:
        extracted["source_kind"] = source_kind

    authors = front_matter.get("authors") or front_matter.get("author") or front_matter.get("作者")
    if isinstance(authors, str) and authors.strip():
        extracted["authors"] = authors.strip()
    elif isinstance(authors, list):
        cleaned = [str(item).strip() for item in authors if str(item).strip()]
        if cleaned:
            extracted["authors"] = ", ".join(cleaned)

    year = (
        front_matter.get("year")
        or front_matter.get("publication_year")
        or front_matter.get("年份")
    )
    parsed_year = _coerce_year(year)
    if parsed_year is not None:
        extracted["publication_year"] = parsed_year

    title = front_matter.get("literature_title") or front_matter.get("paper_title")
    if isinstance(title, str) and title.strip():
        extracted["literature_title"] = title.strip()

    return extracted


def extract_pdf_embedded_metadata(path: Path) -> dict[str, object]:
    extracted: dict[str, object] = {}
    pdf = pdfium.PdfDocument(str(path))
    try:
        metadata = pdf.get_metadata_dict()
    finally:
        pdf.close()

    title = _clean_text(metadata.get("Title"))
    if title:
        extracted["literature_title"] = title

    authors = _clean_text(metadata.get("Author"))
    if authors:
        extracted["authors"] = authors

    year = _year_from_pdf_date(metadata.get("CreationDate"))
    if year is not None:
        extracted["publication_year"] = year

    return extracted


def extract_filename_literature(path: Path) -> dict[str, object]:
    stem = path.stem.strip()
    if not stem:
        return {}

    extracted: dict[str, object] = {}
    working = stem
    year_match = _YEAR_IN_FILENAME_RE.search(working)
    if year_match:
        extracted["publication_year"] = int(year_match.group(0))

    prefix_match = _YEAR_PREFIX_RE.match(working)
    if prefix_match:
        working = prefix_match.group("rest").strip()

    author_title_match = _AUTHOR_TITLE_YEAR_RE.match(working)
    if author_title_match:
        authors = author_title_match.group("authors").strip()
        title = author_title_match.group("title").strip()
        if authors:
            extracted["authors"] = authors
        if title:
            extracted["literature_title"] = title
        year = _coerce_year(author_title_match.group("year"))
        if year is not None:
            extracted["publication_year"] = year
        return extracted

    if year_match and len(working) > 4:
        remainder = working.replace(year_match.group(0), "", 1).strip(" -_–—")
        if remainder:
            extracted["literature_title"] = remainder

    return extracted


def resolve_source_metadata(
    *,
    path: Path,
    loaded_title: str,
    front_matter: dict[str, Any] | None = None,
    embedded: dict[str, object] | None = None,
    source_kind: SourceKind = SOURCE_KIND_GROUP,
    overrides: LiteratureMetadataOverrides | None = None,
) -> ResolvedSourceMetadata:
    merged: dict[str, object] = {}
    if front_matter:
        merged.update(extract_frontmatter_literature(front_matter))
    if embedded:
        merged.update(embedded)
    if source_kind == SOURCE_KIND_PERSONAL:
        merged.update(extract_filename_literature(path))

    resolved_kind = normalize_source_kind(merged.get("source_kind"), default=source_kind)
    if overrides is None:
        overrides = LiteratureMetadataOverrides()

    title = _first_non_empty(
        overrides.title,
        merged.get("literature_title"),
        loaded_title,
        path.stem,
    )
    authors = _first_non_empty(overrides.authors, merged.get("authors"), default="")
    publication_year = overrides.year
    if publication_year is None:
        publication_year = _coerce_year(merged.get("publication_year"))

    return ResolvedSourceMetadata(
        source_kind=resolved_kind,
        title=str(title),
        authors=str(authors),
        publication_year=publication_year,
    )


def _first_non_empty(*values: object, default: str = "") -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _coerce_year(value: object) -> int | None:
    if isinstance(value, int):
        return value if 1900 <= value <= 2100 else None
    if isinstance(value, str):
        match = _YEAR_IN_FILENAME_RE.search(value.strip())
        if match:
            return int(match.group(0))
    return None


def _year_from_pdf_date(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"D:(?P<year>(?:19|20)\d{2})", value)
    if match:
        return int(match.group("year"))
    return _coerce_year(value)


__all__ = [
    "LiteratureMetadataOverrides",
    "ResolvedSourceMetadata",
    "SOURCE_KIND_GROUP",
    "SOURCE_KIND_LABELS",
    "SOURCE_KIND_PERSONAL",
    "SourceKind",
    "extract_filename_literature",
    "extract_frontmatter_literature",
    "extract_pdf_embedded_metadata",
    "normalize_source_kind",
    "resolve_source_metadata",
]
