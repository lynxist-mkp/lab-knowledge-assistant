"""Source semantics and literature metadata extraction."""

from __future__ import annotations

from pathlib import Path

from wenmai.ingestion.source_metadata import (
    SOURCE_KIND_GROUP,
    SOURCE_KIND_PERSONAL,
    LiteratureMetadataOverrides,
    extract_filename_literature,
    extract_frontmatter_literature,
    resolve_source_metadata,
)


def test_extract_frontmatter_literature_reads_personal_fields() -> None:
    extracted = extract_frontmatter_literature(
        {
            "source_kind": "personal_literature",
            "author": "Vaswani et al.",
            "year": 2017,
            "literature_title": "Attention Is All You Need",
        }
    )

    assert extracted["source_kind"] == "personal_literature"
    assert extracted["authors"] == "Vaswani et al."
    assert extracted["publication_year"] == 2017
    assert extracted["literature_title"] == "Attention Is All You Need"


def test_extract_filename_literature_parses_author_title_year() -> None:
    extracted = extract_filename_literature(
        Path("Vaswani et al - Attention Is All You Need (2017).pdf")
    )

    assert extracted["authors"] == "Vaswani et al"
    assert extracted["literature_title"] == "Attention Is All You Need"
    assert extracted["publication_year"] == 2017


def test_resolve_source_metadata_prefers_request_overrides() -> None:
    resolved = resolve_source_metadata(
        path=Path("ignored.pdf"),
        loaded_title="stem-title",
        front_matter={"author": "Front Author", "year": 2018},
        source_kind=SOURCE_KIND_PERSONAL,
        overrides=LiteratureMetadataOverrides(
            title="Override Title",
            authors="Override Author",
            year=2020,
        ),
    )

    assert resolved.source_kind == SOURCE_KIND_PERSONAL
    assert resolved.title == "Override Title"
    assert resolved.authors == "Override Author"
    assert resolved.publication_year == 2020
    assert resolved.chunk_fields()["source_label"] == "个人文献库"


def test_resolve_source_metadata_defaults_to_group_doc() -> None:
    resolved = resolve_source_metadata(
        path=Path("notes.md"),
        loaded_title="组会纪要",
    )

    assert resolved.source_kind == SOURCE_KIND_GROUP
    assert resolved.title == "组会纪要"
    assert resolved.chunk_fields()["source_kind"] == SOURCE_KIND_GROUP
