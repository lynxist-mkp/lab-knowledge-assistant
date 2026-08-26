from __future__ import annotations

import re

from wenmai.config import Settings
from wenmai.factories import embedding as embedding_factory
from wenmai.factories import llm as llm_factory
from wenmai.factories import vector_store as vector_store_factory
from wenmai.models import AskResult, Citation, ScoredChunk
from wenmai.storage.paths import store_path
from wenmai.tracing.context import TraceContext
from wenmai.tracing.writer import JsonlTraceWriter

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class QueryGenerationError(Exception):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


def _load_qa_prompt(settings: Settings) -> str:
    prompt_path = settings.root / settings.paths.prompts
    return prompt_path.read_text(encoding="utf-8")


def _normalize_question(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question.strip())
    return collapsed


def _build_context(scored_chunks: list[ScoredChunk]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(scored_chunks, start=1):
        chunk = item.chunk
        title = str(chunk.metadata.get("title") or chunk.metadata.get("chunk_title") or "")
        header = f"[{index}]"
        if title:
            header += f" {title}"
        blocks.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def _build_prompt(template: str, question: str, scored_chunks: list[ScoredChunk]) -> str:
    return template.format(question=question, context=_build_context(scored_chunks))


def _excerpt(text: str, limit: int = 160) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _extract_citations(answer: str, scored_chunks: list[ScoredChunk]) -> list[Citation]:
    indices = sorted({int(match) for match in _CITATION_PATTERN.findall(answer)})
    citations: list[Citation] = []
    for index in indices:
        if index < 1 or index > len(scored_chunks):
            continue
        chunk = scored_chunks[index - 1].chunk
        citations.append(
            Citation(
                index=index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                title=str(chunk.metadata.get("title") or chunk.metadata.get("chunk_title") or ""),
                excerpt=_excerpt(chunk.text),
                url=str(chunk.metadata.get("url") or ""),
            )
        )
    return citations


def _candidate_records(scored_chunks: list[ScoredChunk]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": item.chunk.chunk_id,
            "score": round(item.score, 6),
        }
        for item in scored_chunks
    ]


def ask_question(question: str, settings: Settings) -> AskResult:
    trace = TraceContext(trace_type="query", metadata={"question": question})
    writer = JsonlTraceWriter(store_path(settings, "traces"))
    normalized = _normalize_question(question)
    top_k = settings.retrieval.dense_k

    try:
        with trace.stage(
            "query_processing",
            method="normalize",
            provider="local",
            input_summary=question,
        ) as stage_info:
            stage_info["output_summary"] = normalized
            stage_info["candidate_count"] = 1

        embedder = embedding_factory.create(settings)
        query_vector = embedder.embed_query(normalized)

        store = vector_store_factory.create(settings)
        with trace.stage(
            "dense",
            method="vector_query",
            provider=store.provider_name,
            input_summary=f"k={top_k}",
        ) as dense_info:
            scored_chunks = store.query(query_vector, top_k=top_k)
            dense_info["candidate_count"] = len(scored_chunks)
            dense_info["output_summary"] = f"retrieved {len(scored_chunks)} chunks"
            dense_info["candidates"] = _candidate_records(scored_chunks)

        llm = llm_factory.create(settings)
        template = _load_qa_prompt(settings)
        prompt = _build_prompt(template, normalized, scored_chunks)

        with trace.stage(
            "generation",
            method="llm",
            provider=llm.provider_name,
            input_summary=f"{len(scored_chunks)} chunks",
        ) as generation_info:
            try:
                answer = llm.generate(prompt)
            except Exception as exc:
                generation_info["output_summary"] = "generation failed"
                generation_info["error"] = f"{type(exc).__name__}: {exc}"
                trace.error = generation_info["error"]
                raise QueryGenerationError(str(exc), trace.trace_id) from exc
            generation_info["output_summary"] = f"{len(answer)} chars"
            generation_info["candidate_count"] = len(scored_chunks)

        citations = _extract_citations(answer, scored_chunks)
        return AskResult(answer=answer, citations=citations, trace_id=trace.trace_id)
    finally:
        trace.close()
        writer.write(trace)
