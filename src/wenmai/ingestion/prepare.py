"""入库后处理：切分后的清洗 / 补元数据 / 图转文。"""

from __future__ import annotations

from wenmai.components.transform.enricher_llm import LlmEnricher
from wenmai.components.transform.rule import RuleRefiner
from wenmai.config import Settings
from wenmai.factories import multimodal as multimodal_factory
from wenmai.models import Chunk
from wenmai.storage.document_images import DocumentImages
from wenmai.tracing.context import TraceContext


def prepare_chunks(
    chunks: list[Chunk],
    settings: Settings,
    trace: TraceContext,
    *,
    images: DocumentImages | None = None,
) -> list[Chunk]:
    """Run configured 入库后处理 stages in fixed order."""
    stages = list(settings.transform.stages)
    if "refiner" in stages:
        chunks = RuleRefiner(settings).apply(chunks, trace)
    if "enricher" in stages:
        chunks = LlmEnricher(settings).apply(chunks, trace)
    if "captioner" in stages:
        doc_images = images or DocumentImages(settings)
        prompt_path = settings.root / settings.transform.captioner_prompt
        template = prompt_path.read_text(encoding="utf-8")
        multimodal = multimodal_factory.create(settings)
        chunks = doc_images.caption_chunks(
            chunks,
            multimodal=multimodal,
            prompt_template=template,
            trace=trace,
        )
    return chunks
