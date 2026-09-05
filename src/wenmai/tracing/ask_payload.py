"""AskTracePayload — typed input for QueryTrace write seam."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wenmai.config import Settings
from wenmai.models import ScoredChunk

if TYPE_CHECKING:
    from wenmai.generation.generate import GenerationResult
    from wenmai.retrieval.fusion import RetrievalResult


@dataclass(frozen=True)
class AskTracePayload:
    """Everything QueryTrace finalize needs — no OrchestrationWork duck-typing."""

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


def ask_trace_payload_from_work(work: Any) -> AskTracePayload:
    """Build payload from orchestration work (internal adapter)."""
    assert work.retrieval_result is not None
    return AskTracePayload(
        settings=work.settings,
        normalized=work.normalized,
        retrieval_result=work.retrieval_result,
        extras_elapsed_ms=work.extras_elapsed_ms,
        term_extras=list(work.term_extras),
        multi_query_extras=list(work.multi_query_extras),
        rewriter_provider_name=work.rewriter_provider_name,
        rerank_stages=list(work.rerank_stages),
        chunks=list(work.chunks or []),
        expanded_chunks=list(work.expanded_chunks or []),
        expanded_from=list(work.expanded_from),
        expanded_chunk_ids=list(work.expanded_chunk_ids),
        generation=work.generation,
    )


__all__ = ["AskTracePayload", "ask_trace_payload_from_work"]
