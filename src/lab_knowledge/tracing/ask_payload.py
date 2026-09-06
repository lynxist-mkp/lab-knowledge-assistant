"""AskOutcome — stable seam from 提问编排 to QueryTrace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lab_knowledge.config import Settings
from lab_knowledge.models import ScoredChunk

if TYPE_CHECKING:
    from lab_knowledge.generation.generate import GenerationError, GenerationResult
    from lab_knowledge.retrieval.fusion import RetrievalResult


@dataclass(frozen=True)
class AskTracePayload:
    """Trace assembly inputs — no orchestration work-bag fields."""

    settings: Settings
    normalized: str
    retrieval_result: RetrievalResult
    extras_elapsed_ms: float = 0.0
    term_extras: list[str] = field(default_factory=list)
    multi_query_extras: list[str] = field(default_factory=list)
    rewriter_provider_name: str = "none"
    rerank_stages: list[Any] = field(default_factory=list)
    chunks: list[ScoredChunk] = field(default_factory=list)
    expanded_chunks: list[ScoredChunk] = field(default_factory=list)
    expanded_from: list[str] = field(default_factory=list)
    expanded_chunk_ids: list[str] = field(default_factory=list)
    generation: GenerationResult | None = None

    @property
    def multi_query_provider(self) -> str | None:
        if self.settings.query_processing.multi_query:
            return self.settings.providers.multimodal
        return None


@dataclass(frozen=True)
class AskOutcome:
    """One stable outcome shape for success and generation failure."""

    question: str
    culture_domain: str | None
    payload: AskTracePayload
    generation_error: GenerationError | None = None


__all__ = ["AskOutcome", "AskTracePayload"]
