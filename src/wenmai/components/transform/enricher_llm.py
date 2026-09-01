from __future__ import annotations

import json
import time

from wenmai.components.transform.base import BaseTransform
from wenmai.config import Settings
from wenmai.factories import multimodal as multimodal_factory
from wenmai.factories.transform import registry
from wenmai.models import Chunk
from wenmai.tracing.context import TraceContext

_DIGEST_CHAR_LIMIT = 8000
_CHUNK_EXCERPT_CHARS = 500


def _load_prompt(settings: Settings) -> str:
    prompt_path = settings.root / settings.transform.enricher_prompt
    return prompt_path.read_text(encoding="utf-8")


def _build_prompt(template: str, document_digest: str, domains: list[str]) -> str:
    return template.format(chunk_text=document_digest, domains="、".join(domains))


def _build_document_digest(chunks: list[Chunk]) -> str:
    lines: list[str] = []
    first = chunks[0].metadata if chunks else {}
    for key in ("title", "url", "culture_domain", "space", "source_path"):
        value = first.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value.strip()}")
    if lines:
        lines.append("")
    for index, chunk in enumerate(chunks, start=1):
        excerpt = chunk.text[:_CHUNK_EXCERPT_CHARS].strip()
        if excerpt:
            lines.append(f"[chunk {index}] {excerpt}")
    digest = "\n".join(lines).strip()
    if len(digest) > _DIGEST_CHAR_LIMIT:
        return digest[:_DIGEST_CHAR_LIMIT] + "…"
    return digest


def _parse_response(raw: str, domains: list[str]) -> dict[str, object]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")

    result: dict[str, object] = {}
    title = data.get("title")
    if isinstance(title, str) and title.strip():
        result["chunk_title"] = title.strip()

    summary = data.get("summary")
    if isinstance(summary, str) and summary.strip():
        result["summary"] = summary.strip()

    tags = data.get("tags")
    if isinstance(tags, list):
        cleaned = [str(tag).strip() for tag in tags if str(tag).strip()]
        if cleaned:
            result["tags"] = cleaned

    domain = data.get("culture_domain")
    if isinstance(domain, str) and domain.strip():
        normalized = domain.strip()
        if normalized in domains:
            result["culture_domain"] = normalized
        elif "其他" in domains:
            result["culture_domain"] = "其他"

    return result


@registry.register("enricher.llm")
class LlmEnricher(BaseTransform):
    name = "enricher"

    def __init__(self, settings: Settings, **kwargs: object) -> None:
        self._settings = settings
        self._llm = multimodal_factory.create(settings)
        self._template = _load_prompt(settings)
        self._domains = list(settings.transform.domains)

    def apply(self, chunks: list[Chunk], trace: TraceContext) -> list[Chunk]:
        if not chunks:
            return chunks

        started = time.perf_counter()
        stage_error: str | None = None
        enriched_count = 0
        output_summary = "enriched 0 chunks"

        try:
            digest = _build_document_digest(chunks)
            prompt = _build_prompt(self._template, digest, self._domains)
            raw = self._llm.generate(prompt)
            parsed = _parse_response(raw, self._domains)
            if parsed:
                for chunk in chunks:
                    chunk.metadata.update(parsed)
                enriched_count = len(chunks)
            output_summary = f"enriched document ({enriched_count}/{len(chunks)} chunks)"
        except Exception as exc:
            stage_error = f"{type(exc).__name__}: {exc}"
            output_summary = f"document enrich failed: {stage_error}"

        trace.record_stage(
            name="enricher",
            method="llm",
            provider=self._llm.provider_name,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_summary=f"{len(chunks)} chunks (document-level)",
            output_summary=output_summary,
            candidate_count=len(chunks),
            error=stage_error,
        )
        return chunks
