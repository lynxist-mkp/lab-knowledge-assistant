from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from lab_knowledge.config import Settings
from lab_knowledge.factories import multimodal as multimodal_factory
from lab_knowledge.knowledge.domain import authors, publication_year, source_kind, source_label
from lab_knowledge.models import Citation, ScoredChunk

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_REFUSAL_PREFIX = "拒答："
_INSUFFICIENT_EVIDENCE_ANSWER = "拒答：检索未返回可用片段，无法依据材料回答该问题。"
_LOW_OVERLAP_ANSWER = "拒答：检索片段与问题缺少足够重合的依据，无法依据材料回答该问题。"

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")
_STOPWORDS = frozenset(
    "的了吗呢是在有和与及或对从到为被把等什么怎么多少哪里何时"
    "今日今天昨晚本周一个一些如何是否可以"
)

RefusalReason = Literal["insufficient_evidence", "model_refused"]


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
        source_kind=source_kind(chunk),
        source_label=source_label(chunk),
        authors=authors(chunk),
        publication_year=publication_year(chunk),
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


def _content_tokens(text: str) -> set[str]:
    parts = _TOKEN_RE.findall(text.lower())
    tokens: set[str] = set()
    cjk_chars: list[str] = []
    for part in parts:
        if part in _STOPWORDS:
            continue
        if len(part) == 1 and "\u4e00" <= part <= "\u9fff":
            cjk_chars.append(part)
            continue
        tokens.add(part)
    for left, right in zip(cjk_chars, cjk_chars[1:]):
        bigram = left + right
        if bigram not in _STOPWORDS:
            tokens.add(bigram)
    return tokens


def question_evidence_overlap(question: str, scored_chunks: list[ScoredChunk]) -> float:
    """Fraction of question content tokens that appear in any retrieved chunk text."""
    question_tokens = _content_tokens(question)
    if not question_tokens:
        return 1.0
    context_tokens: set[str] = set()
    for item in scored_chunks:
        context_tokens |= _content_tokens(item.chunk.text)
        title = str(
            item.chunk.metadata.get("title") or item.chunk.metadata.get("chunk_title") or ""
        )
        if title:
            context_tokens |= _content_tokens(title)
    if not context_tokens:
        return 0.0
    hit = sum(1 for token in question_tokens if token in context_tokens)
    return hit / len(question_tokens)


def _is_refusal(answer: str) -> bool:
    return answer.strip().startswith(_REFUSAL_PREFIX)


def _resolve_response(answer: str, scored_chunks: list[ScoredChunk]) -> tuple[bool, list[Citation]]:
    refused = _is_refusal(answer)
    if refused:
        return True, _citations_from_retrieved(scored_chunks)
    return False, _extract_citations(answer, scored_chunks)


def _insufficient_evidence(
    *,
    answer: str,
    scored_chunks: list[ScoredChunk],
    output_summary: str,
) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        refused=True,
        citations=_citations_from_retrieved(scored_chunks) if scored_chunks else [],
        provider_name="local",
        output_summary=output_summary,
        candidate_count=len(scored_chunks),
        refusal_reason="insufficient_evidence",
    )


def generate(
    question: str, scored_chunks: list[ScoredChunk], settings: Settings
) -> GenerationResult:
    if not scored_chunks:
        return _insufficient_evidence(
            answer=_INSUFFICIENT_EVIDENCE_ANSWER,
            scored_chunks=[],
            output_summary="insufficient_evidence",
        )

    min_overlap = settings.generation.min_question_overlap
    if min_overlap > 0:
        overlap = question_evidence_overlap(question, scored_chunks)
        if overlap < min_overlap:
            return _insufficient_evidence(
                answer=_LOW_OVERLAP_ANSWER,
                scored_chunks=scored_chunks,
                output_summary="insufficient_evidence_low_overlap",
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
