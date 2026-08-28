from __future__ import annotations

import re
import time
from pathlib import Path

from wenmai.components.transform.base import BaseTransform
from wenmai.config import Settings
from wenmai.factories import multimodal as multimodal_factory
from wenmai.factories.transform import registry
from wenmai.models import Chunk
from wenmai.storage.images import ImageStore
from wenmai.tracing.context import TraceContext

_IMAGE_PLACEHOLDER = re.compile(r"\[IMAGE:\s*([a-f0-9]+)\s*\]")


def _load_prompt(settings: Settings) -> str:
    prompt_path = settings.root / settings.transform.captioner_prompt
    return prompt_path.read_text(encoding="utf-8")


def _replace_placeholders(text: str, image_id: str, caption: str) -> str:
    pattern = re.compile(rf"\[IMAGE:\s*{re.escape(image_id)}\s*\]")
    return pattern.sub(caption, text)


@registry.register("captioner.vision")
class VisionCaptioner(BaseTransform):
    name = "captioner"

    def __init__(self, settings: Settings, **kwargs: object) -> None:
        self._settings = settings
        self._vision = multimodal_factory.create(settings)
        self._template = _load_prompt(settings)
        self._image_store = ImageStore(settings)

    def apply(self, chunks: list[Chunk], trace: TraceContext) -> list[Chunk]:
        started = time.perf_counter()
        captioned = 0
        failures: list[str] = []

        for chunk in chunks:
            placeholders = _IMAGE_PLACEHOLDER.findall(chunk.text)
            if not placeholders:
                continue
            updated_text = chunk.text
            for image_id in placeholders:
                record = self._image_store.get(image_id)
                if record is None:
                    failures.append(f"{chunk.chunk_id}: missing image {image_id}")
                    continue
                try:
                    caption = self._vision.caption(Path(record.file_path), self._template)
                    if not caption.strip():
                        failures.append(f"{chunk.chunk_id}: empty caption for {image_id}")
                        continue
                    updated_text = _replace_placeholders(updated_text, image_id, caption.strip())
                    captioned += 1
                except Exception as exc:
                    failures.append(
                        f"{chunk.chunk_id}/{image_id}: {type(exc).__name__}: {exc}"
                    )
            chunk.text = updated_text

        output_summary = f"captioned {captioned} images"
        stage_error: str | None = None
        if failures:
            output_summary += f"; {len(failures)} kept placeholder"
            stage_error = failures[0]
            if len(failures) > 1:
                stage_error += f" (+{len(failures) - 1} more)"

        trace.record_stage(
            name="captioner",
            method="vision",
            provider=self._vision.provider_name,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_summary=f"{len(chunks)} chunks",
            output_summary=output_summary,
            candidate_count=len(chunks),
            error=stage_error,
        )
        return chunks
