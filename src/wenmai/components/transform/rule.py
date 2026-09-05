from __future__ import annotations

import re

from wenmai.components.transform.base import BaseTransform
from wenmai.config import Settings
from wenmai.factories.transform import registry
from wenmai.ingestion.prepare import TransformTraceRecorder
from wenmai.ingestion.quality import effective_char_ratio
from wenmai.models import Chunk

_HEADER_FOOTER_LINE = re.compile(
    r"^\s*(?:"
    r"第\s*\d+\s*页"
    r"|-\s*\d+\s*-"
    r"|Page\s+\d+.*"
    r"|版权所有.*"
    r"|Copyright.*"
    r"|闽\s*ICP\s*备.*"
    r")\s*$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[。！？；.!?;]$")
_NEW_BLOCK = re.compile(r"^(?:#+\s|[-*•]\s|\d+[.)]\s)")


def _strip_header_footer_lines(text: str) -> str:
    lines = text.splitlines()
    kept = [line for line in lines if not _HEADER_FOOTER_LINE.match(line)]
    return "\n".join(kept)


def _collapse_whitespace(text: str) -> str:
    collapsed = re.sub(r"[ \t]+", " ", text)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def _should_join(previous: str, nxt: str) -> bool:
    if _SENTENCE_END.search(previous):
        return False
    if _NEW_BLOCK.match(nxt):
        return False
    if not previous or not nxt:
        return False
    return True


def _fix_line_wraps(text: str) -> str:
    lines = text.splitlines()
    merged: list[str] = []
    buffer = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged.append(buffer)
                buffer = ""
            if merged and merged[-1] != "":
                merged.append("")
            continue
        if not buffer:
            buffer = stripped
            continue
        if _should_join(buffer, stripped):
            buffer += stripped
        else:
            merged.append(buffer)
            buffer = stripped
    if buffer:
        merged.append(buffer)
    return "\n".join(merged)


def clean_chunk_text(text: str) -> str:
    cleaned = _strip_header_footer_lines(text)
    cleaned = _fix_line_wraps(cleaned)
    return _collapse_whitespace(cleaned)


@registry.register("refiner.rule")
class RuleRefiner(BaseTransform):
    name = "refiner"

    def __init__(self, settings: Settings, **kwargs: object) -> None:
        self._min_ratio = settings.transform.refiner_min_ratio

    def apply(self, chunks: list[Chunk], recorder: TransformTraceRecorder) -> list[Chunk]:
        kept: list[Chunk] = []
        discarded: list[dict[str, str]] = []

        with recorder.stage(
            "transform",
            method="rule",
            provider="refiner",
            input_summary=f"{len(chunks)} chunks",
        ) as stage_info:
            for chunk in chunks:
                cleaned = clean_chunk_text(chunk.text)
                ratio = effective_char_ratio(cleaned)
                if ratio < self._min_ratio:
                    discarded.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "reason": (
                                f"effective_char_ratio {ratio:.2f} below {self._min_ratio:.2f}"
                            ),
                        }
                    )
                    continue
                kept.append(
                    Chunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=cleaned,
                        embedding=chunk.embedding,
                        metadata=dict(chunk.metadata),
                    )
                )

            stage_info["candidate_count"] = len(kept)
            stage_info["output_summary"] = (
                f"{len(kept)} kept, {len(discarded)} discarded"
                if discarded
                else f"{len(kept)} kept"
            )
            if discarded:
                recorder.metadata.setdefault("transform_discarded", []).extend(discarded)

        return kept
