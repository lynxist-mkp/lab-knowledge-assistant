from __future__ import annotations

import json
import time

from wenmai.config import Settings
from wenmai.factories import multimodal as multimodal_factory
from wenmai.models import Chunk
from wenmai.tracing.context import TraceContext


def _load_prompt(settings: Settings) -> str:
    prompt_path = settings.root / settings.transform.enricher_prompt
    return prompt_path.read_text(encoding="utf-8")


def _build_prompt(template: str, chunk_text: str, domains: list[str]) -> str:
    return template.format(chunk_text=chunk_text, domains="、".join(domains))


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


class LlmEnricher:
    def __init__(self, settings: Settings, **kwargs: object) -> None:
        self._settings = settings
        self._llm = multimodal_factory.create(settings)
        self._template = _load_prompt(settings)
        self._domains = list(settings.transform.domains)

    def apply(self, chunks: list[Chunk], trace: TraceContext) -> list[Chunk]:
        started = time.perf_counter()
        failures: list[str] = []
        enriched_count = 0

        for chunk in chunks:
            try:
                prompt = _build_prompt(self._template, chunk.text, self._domains)
                raw = self._llm.generate(prompt)
                parsed = _parse_response(raw, self._domains)
                if parsed:
                    chunk.metadata.update(parsed)
                    enriched_count += 1
            except Exception as exc:
                failures.append(f"{chunk.chunk_id}: {type(exc).__name__}: {exc}")

        output_summary = f"enriched {enriched_count}/{len(chunks)} chunks"
        stage_error: str | None = None
        if failures:
            output_summary += f"; {len(failures)} failed"
            stage_error = failures[0]
            if len(failures) > 1:
                stage_error += f" (+{len(failures) - 1} more)"

        trace.record_stage(
            name="enricher",
            method="llm",
            provider=self._llm.provider_name,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_summary=f"{len(chunks)} chunks",
            output_summary=output_summary,
            candidate_count=len(chunks),
            error=stage_error,
        )
        return chunks
