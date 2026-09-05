from __future__ import annotations

import re

from wenmai.components.transform.base import BaseTransform
from wenmai.config import Settings
from wenmai.factories import multimodal as multimodal_factory
from wenmai.factories.transform import registry
from wenmai.ingestion.prepare import TransformTraceRecorder
from wenmai.models import Chunk
from wenmai.storage.document_images import IMAGE_PLACEHOLDER_RE, DocumentImages


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
        self._images = DocumentImages(settings)

    def apply(self, chunks: list[Chunk], recorder: TransformTraceRecorder) -> list[Chunk]:
        with recorder.stage(
            name="captioner",
            method="vision",
            provider=self._vision.provider_name,
            input_summary=f"{len(chunks)} chunks",
        ) as stage_info:
            captioned = 0
            failures: list[str] = []

            for chunk in chunks:
                placeholders = IMAGE_PLACEHOLDER_RE.findall(chunk.text)
                if not placeholders:
                    continue
                existing_ids = chunk.metadata.get("image_ids")
                image_ids = list(existing_ids) if isinstance(existing_ids, list) else []
                for image_id in placeholders:
                    if image_id not in image_ids:
                        image_ids.append(image_id)
                chunk.metadata["image_ids"] = image_ids
                updated_text = chunk.text
                for image_id in placeholders:
                    image_path = self._images.get(image_id)
                    if image_path is None:
                        failures.append(f"{chunk.chunk_id}: missing image {image_id}")
                        continue
                    try:
                        caption = self._vision.caption(image_path, self._template)
                        if not caption.strip():
                            failures.append(f"{chunk.chunk_id}: empty caption for {image_id}")
                            continue
                        updated_text = _replace_placeholders(
                            updated_text, image_id, caption.strip()
                        )
                        captioned += 1
                    except Exception as exc:
                        failures.append(
                            f"{chunk.chunk_id}/{image_id}: {type(exc).__name__}: {exc}"
                        )
                chunk.text = updated_text

            stage_info["candidate_count"] = len(chunks)
            stage_info["output_summary"] = f"captioned {captioned} images"
            if failures:
                stage_info["output_summary"] += f"; {len(failures)} kept placeholder"
                stage_info["error"] = failures[0]
                if len(failures) > 1:
                    stage_info["error"] += f" (+{len(failures) - 1} more)"
        return chunks
