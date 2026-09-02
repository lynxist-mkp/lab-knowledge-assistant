from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from wenmai.config import Settings
from wenmai.tracing.context import TraceContext
from wenmai.tracing.stage_result import StageResult, stage_to_dict
from wenmai.tracing.stages.query import QueryStage
from wenmai.storage.paths import store_path
from wenmai.tracing.writer import JsonlTraceWriter

QUERY_TRACE_SCHEMA_VERSION = 2


class TraceRecorder:
    """Central owner for query trace assembly and typed serialization."""

    def __init__(
        self,
        *,
        question: str,
        batch_id: str | None = None,
        batch_size: int = 1,
        batch_wait_ms: float = 0.0,
    ) -> None:
        metadata: dict[str, Any] = {"question": question}
        if batch_id:
            metadata["batch_id"] = batch_id
            metadata["batch_size"] = batch_size
            metadata["batch_wait_ms"] = round(batch_wait_ms, 1)
        self._context = TraceContext(trace_type="query", metadata=metadata)

    @property
    def trace_id(self) -> str:
        return self._context.trace_id

    @property
    def error(self) -> str | None:
        return self._context.error

    @error.setter
    def error(self, value: str | None) -> None:
        self._context.error = value

    def append_stage(self, stage: StageResult) -> None:
        self._context.append_stage(stage)

    def record_query_processing(
        self,
        *,
        question: str,
        normalized: str,
        elapsed_ms: float,
        culture_domain: str | None = None,
        term_extras: list[str] | None = None,
        multi_query_extras: list[str] | None = None,
        rewriter: str = "local",
        multi_query_provider: str | None = None,
    ) -> None:
        self.append_stage(
            QueryStage.query_processing(
                question=question,
                normalized=normalized,
                elapsed_ms=elapsed_ms,
                culture_domain=culture_domain,
                term_extras=term_extras,
                multi_query_extras=multi_query_extras,
                rewriter=rewriter,
                multi_query_provider=multi_query_provider,
            )
        )

    def append_retrieval_stages(self, stages: Iterable[StageResult]) -> None:
        for stage in stages:
            self.append_stage(stage)

    def append_rerank_stages(self, stages: Iterable[StageResult]) -> None:
        for stage in stages:
            self.append_stage(stage)

    def record_generation(
        self,
        *,
        provider: str,
        elapsed_ms: float,
        input_summary: str,
        output_summary: str,
        candidate_count: int,
        error: str | None = None,
        expanded_from: list[str] | None = None,
        expanded_chunk_ids: list[str] | None = None,
    ) -> None:
        self.append_stage(
            QueryStage.generation(
                provider=provider,
                elapsed_ms=elapsed_ms,
                input_summary=input_summary,
                output_summary=output_summary,
                candidate_count=candidate_count,
                error=error,
                expanded_from=expanded_from,
                expanded_chunk_ids=expanded_chunk_ids,
            )
        )

    def set_outcome(
        self,
        *,
        refused: bool,
        refusal_reason: str | None,
        citation_count: int,
    ) -> None:
        self._context.metadata["outcome"] = {
            "refused": refused,
            "refusal_reason": refusal_reason,
            "citation_count": citation_count,
        }

    def to_dict(self) -> dict[str, Any]:
        if self._context.finished_at is None:
            self._context.close()
        payload = self._context.to_dict()
        payload["schema_version"] = QUERY_TRACE_SCHEMA_VERSION
        payload["stages"] = [stage_to_dict(stage) for stage in self._context.stages]
        return payload

    def save(self, settings: Settings) -> None:
        JsonlTraceWriter(store_path(settings, "traces")).write_payload(self.to_dict())

    def finalize_ask_generation_error(
        self,
        *,
        work: object,
        question: str,
        culture_domain: str | None,
        error: object,
    ) -> None:
        from wenmai.factories import query_rewrite as query_rewrite_factory

        assert work.retrieval_result is not None
        rewriter = query_rewrite_factory.create(work.settings)
        mq_provider = (
            work.settings.providers.multimodal
            if work.settings.query_processing.multi_query
            else None
        )
        self.record_query_processing(
            question=question,
            normalized=work.normalized,
            elapsed_ms=work.extras_elapsed_ms,
            culture_domain=culture_domain,
            term_extras=work.term_extras,
            multi_query_extras=work.multi_query_extras,
            rewriter=rewriter.provider_name,
            multi_query_provider=mq_provider,
        )
        self.append_retrieval_stages(work.retrieval_result.stages)
        self.append_rerank_stages(work.rerank_stages)

        scored_chunks = work.chunks or []
        generation_input = f"{len(work.expanded_chunks or scored_chunks)} chunks"
        generation_error = f"{type(error).__name__}: {error}"
        self.record_generation(
            provider=error.provider_name,
            elapsed_ms=0.0,
            input_summary=generation_input,
            output_summary="generation failed",
            candidate_count=0,
            error=generation_error,
            expanded_from=work.expanded_from,
            expanded_chunk_ids=work.expanded_chunk_ids,
        )
        self.error = generation_error

    def finalize_ask_work(
        self,
        *,
        work: object,
        question: str,
        culture_domain: str | None,
    ) -> object:
        """Assemble query trace from orchestration work and return AskResult."""
        from wenmai.factories import query_rewrite as query_rewrite_factory
        from wenmai.generation import GenerationError, QueryGenerationError
        from wenmai.models import AskResult

        assert work.retrieval_result is not None
        rewriter = query_rewrite_factory.create(work.settings)
        mq_provider = (
            work.settings.providers.multimodal
            if work.settings.query_processing.multi_query
            else None
        )
        self.record_query_processing(
            question=question,
            normalized=work.normalized,
            elapsed_ms=work.extras_elapsed_ms,
            culture_domain=culture_domain,
            term_extras=work.term_extras,
            multi_query_extras=work.multi_query_extras,
            rewriter=rewriter.provider_name,
            multi_query_provider=mq_provider,
        )
        self.append_retrieval_stages(work.retrieval_result.stages)
        self.append_rerank_stages(work.rerank_stages)

        scored_chunks = work.chunks or []
        generation_input = f"{len(work.expanded_chunks or [])} chunks"
        assert work.generation is not None
        gen_result = work.generation

        self.record_generation(
            provider=gen_result.provider_name,
            elapsed_ms=0.0,
            input_summary=generation_input,
            output_summary=gen_result.output_summary,
            candidate_count=gen_result.candidate_count,
            expanded_from=work.expanded_from,
            expanded_chunk_ids=work.expanded_chunk_ids,
        )
        self.set_outcome(
            refused=gen_result.refused,
            refusal_reason=gen_result.refusal_reason,
            citation_count=len(gen_result.citations),
        )
        return AskResult(
            answer=gen_result.answer,
            citations=gen_result.citations,
            trace_id=self.trace_id,
            refused=gen_result.refused,
            refusal_reason=gen_result.refusal_reason,
            ranked_chunks=scored_chunks,
        )
