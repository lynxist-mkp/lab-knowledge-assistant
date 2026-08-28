"""配图 lifecycle: attach → placeholder → caption → delete with document."""

from __future__ import annotations

import time
from pathlib import Path

from wenmai.components.multimodal.base import BaseMultimodal
from wenmai.config import Settings
from wenmai.models import Chunk
from wenmai.storage.images import (
    IMAGE_PLACEHOLDER_RE,
    ImageStore,
    find_image_ids,
    format_image_placeholder,
    replace_image_placeholder,
)
from wenmai.tracing.context import TraceContext

__all__ = [
    "IMAGE_PLACEHOLDER_RE",
    "DocumentImages",
    "find_image_ids",
    "format_image_placeholder",
]


class DocumentImages:
    """Owns 配图 invariant: extract/save, caption placeholders, delete with doc."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = ImageStore(settings)

    def attach(
        self,
        *,
        document_id: str,
        source_path: str,
        page: int,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> str:
        """Save image bytes and return the `[IMAGE: id]` placeholder for text."""
        image_id = self._store.save(
            document_id=document_id,
            source_path=source_path,
            page=page,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )
        return format_image_placeholder(image_id)

    def exists(self, image_id: str) -> bool:
        return self._store.get(image_id) is not None

    def caption_chunks(
        self,
        chunks: list[Chunk],
        *,
        multimodal: BaseMultimodal,
        prompt_template: str,
        trace: TraceContext,
    ) -> list[Chunk]:
        started = time.perf_counter()
        captioned = 0
        failures: list[str] = []

        for chunk in chunks:
            placeholders = find_image_ids(chunk.text)
            if not placeholders:
                continue
            updated_text = chunk.text
            for image_id in placeholders:
                record = self._store.get(image_id)
                if record is None:
                    failures.append(f"{chunk.chunk_id}: missing image {image_id}")
                    continue
                try:
                    caption = multimodal.caption(Path(record.file_path), prompt_template)
                    if not caption.strip():
                        failures.append(f"{chunk.chunk_id}: empty caption for {image_id}")
                        continue
                    updated_text = replace_image_placeholder(
                        updated_text, image_id, caption.strip()
                    )
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
            provider=multimodal.provider_name,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_summary=f"{len(chunks)} chunks",
            output_summary=output_summary,
            candidate_count=len(chunks),
            error=stage_error,
        )
        return chunks

    def delete_for_document(self, document_id: str) -> None:
        self._store.delete_by_document_id(document_id)
