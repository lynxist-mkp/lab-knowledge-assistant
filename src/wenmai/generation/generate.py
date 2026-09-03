from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from wenmai.config import Settings
from wenmai.factories import multimodal as multimodal_factory
from wenmai.generation.expand import expand_for_generation
from wenmai.knowledge import Knowledge
from wenmai.models import Citation, ScoredChunk

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_REFUSAL_PREFIX = "拒答："
_INSUFFICIENT_EVIDENCE_ANSWER = "拒答：检索未返回可用片段，无法依据材料回答该问题。"

RefusalReason = Literal["insufficient_evidence", "model_refused"]
GenerateFn = Callable[[str, list[ScoredChunk], Settings], "GenerationResult"]


class GenerationError(Exception):
    """LLM generation failed."""

    def __init__(self, message: str, *, provider_name: str = "unknown") -> None:
        super().__init__(message)
        self.provider_name = provider_name


class QueryGenerationError(GenerationError):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    refused: bool
    citations: list[Citation]
    provider_name: str
    output_summary: str
    candidate_count: int
    refusal_reason: RefusalReason | None = None
    error: str | None = None


@dataclass(frozen=True)
class ContextGeneration:
    """Deep 生成 outcome: expanded context + LLM/拒答 result."""

    result: GenerationResult
    expanded_chunks: list[ScoredChunk]
    expanded_from: list[str]
    expanded_chunk_ids: list[str]


def _load_qa_prompt(settings: Settings) -> str:
    prompt_path = settings.root / settings.paths.prompts
    return prompt_path.read_text(encoding="utf-8")


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


def _citation_for_index(index: int, scored_chunks: list[ScoredChunk]) -> Citation | None:
    if index < 1 or index > len(scored_chunks):
        return None
    chunk = scored_chunks[index - 1].chunk
    return Citation(
        index=index,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=str(chunk.metadata.get("title") or chunk.metadata.get("chunk_title") or ""),
        excerpt=_excerpt(chunk.text),
        url=str(chunk.metadata.get("url") or ""),
    )


def _extract_citations(answer: str, scored_chunks: list[ScoredChunk]) -> list[Citation]:
    indices = sorted({int(match) for match in _CITATION_PATTERN.findall(answer)})
    citations: list[Citation] = []
    for index in indices:
        citation = _citation_for_index(index, scored_chunks)
        if citation is not None:
            citations.append(citation)
    return citations


def _citations_from_retrieved(scored_chunks: list[ScoredChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for index in range(1, len(scored_chunks) + 1):
        citation = _citation_for_index(index, scored_chunks)
        if citation is not None:
            citations.append(citation)
    return citations


def _is_refusal(answer: str) -> bool:
    return answer.strip().startswith(_REFUSAL_PREFIX)


def _resolve_response(answer: str, scored_chunks: list[ScoredChunk]) -> tuple[bool, list[Citation]]:
    refused = _is_refusal(answer)
    if refused:
        return True, _citations_from_retrieved(scored_chunks)
    return False, _extract_citations(answer, scored_chunks)


def generate(
    question: str, scored_chunks: list[ScoredChunk], settings: Settings
) -> GenerationResult:
    """LLM + 拒答/出处 — implementation detail of generate_with_context."""
    if not scored_chunks:
        return GenerationResult(
            answer=_INSUFFICIENT_EVIDENCE_ANSWER,
            refused=True,
            citations=[],
            provider_name="local",
            output_summary="insufficient_evidence",
            candidate_count=0,
            refusal_reason="insufficient_evidence",
        )

    llm = multimodal_factory.create(settings)
    template = _load_qa_prompt(settings)
    prompt = _build_prompt(template, question, scored_chunks)

    try:
        answer = llm.generate(prompt)
    except Exception as exc:
        raise GenerationError(str(exc), provider_name=llm.provider_name) from exc

    refused, citations = _resolve_response(answer, scored_chunks)
    return GenerationResult(
        answer=answer,
        refused=refused,
        citations=citations,
        provider_name=llm.provider_name,
        output_summary="refusal" if refused else f"{len(answer)} chars",
        candidate_count=len(scored_chunks),
        refusal_reason="model_refused" if refused else None,
    )


def generate_with_context(
    question: str,
    scored_chunks: list[ScoredChunk],
    knowledge: Knowledge,
    settings: Settings,
    *,
    generate_fn: GenerateFn | None = None,
) -> ContextGeneration:
    """生成 interface：邻块扩展 + LLM + 拒答/出处."""
    expanded, expanded_from, expanded_chunk_ids = expand_for_generation(
        scored_chunks,
        knowledge,
        settings.retrieval.adjacent_n,
    )
    gen = generate_fn or generate
    result = gen(question, expanded, settings)
    return ContextGeneration(
        result=result,
        expanded_chunks=expanded,
        expanded_from=expanded_from,
        expanded_chunk_ids=expanded_chunk_ids,
    )
