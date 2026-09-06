from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LiteratureMetadataOverrides(BaseModel):
    title: str | None = None
    authors: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)


class IngestRequest(BaseModel):
    source_path: str
    pdf_load_mode: str | None = None
    source_kind: Literal["group_doc", "personal_literature"] = "group_doc"
    literature: LiteratureMetadataOverrides | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    culture_domain: str | None = None
    retrieval_mode: Literal["rrf", "dense_only", "sparse_only"] | None = None
    rerank_enabled: bool | None = None
