from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_path: str
    pdf_load_mode: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    culture_domain: str | None = None
    retrieval_mode: Literal["rrf", "dense_only", "sparse_only"] | None = None
    rerank_enabled: bool | None = None
