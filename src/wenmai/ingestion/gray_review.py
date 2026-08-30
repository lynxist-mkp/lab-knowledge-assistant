"""灰区复判 — digital text vs scanned page images, before load completes."""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

from wenmai.config import Settings
from wenmai.factories import multimodal as multimodal_factory
from wenmai.ingestion.quality import SourcePeek

_PROMPT_TEXT = (
    "你是知识库入库助手，正在做灰区复判。判断下列抽文是否值得入库。"
    "只回答「通过」或「不通过」。\n\n"
)
_PROMPT_VISION = (
    "你是知识库入库助手，正在做灰区复判。根据页图判断材料是否值得入库。"
    "只回答「通过」或「不通过」。"
)


@dataclass(frozen=True)
class GrayReviewOutcome:
    pending: bool
    hard_reject: bool
    passed: bool
    provider: str
    method: str
    output_summary: str
    error: str | None = None


def parse_verdict(raw: str) -> bool:
    text = raw.strip()
    if not text:
        return False
    head = text[:40]
    if "不通过" in head:
        return False
    if "通过" in head:
        return True
    return False


def run_gray_review(path: Path, peek: SourcePeek, settings: Settings) -> GrayReviewOutcome:
    provider = multimodal_factory.create(settings)
    timeout = settings.quality_gate.timeout_seconds
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_judge, path, peek, settings, provider)
            raw = future.result(timeout=timeout)
    except FuturesTimeoutError:
        elapsed = (time.perf_counter() - started) * 1000
        return GrayReviewOutcome(
            pending=False,
            hard_reject=True,
            passed=False,
            provider=provider.provider_name,
            method="vlm" if peek.defer_reject else "mllm",
            output_summary=f"timeout after {elapsed:.0f}ms",
            error="gray_review timeout",
        )
    except Exception as exc:
        return GrayReviewOutcome(
            pending=False,
            hard_reject=True,
            passed=False,
            provider=provider.provider_name,
            method="vlm" if peek.defer_reject else "mllm",
            output_summary="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    passed = parse_verdict(raw)
    method = "vlm" if peek.defer_reject else "mllm"
    return GrayReviewOutcome(
        pending=not passed,
        hard_reject=False,
        passed=passed,
        provider=provider.provider_name,
        method=method,
        output_summary=f"verdict={'通过' if passed else '不通过'}",
    )


def _judge(path: Path, peek: SourcePeek, settings: Settings, provider: object) -> str:
    if peek.defer_reject and path.suffix.lower() == ".pdf":
        return _judge_pages(path, settings, provider)
    excerpt = peek.text[: settings.quality_gate.preview_chars]
    prompt = f"{_PROMPT_TEXT}{excerpt}"
    return provider.generate(prompt)  # type: ignore[union-attr]


def _judge_pages(path: Path, settings: Settings, provider: object) -> str:
    pages = min(settings.quality_gate.preview_pages, 3)
    with tempfile.TemporaryDirectory(prefix="wenmai-gray-") as tmp:
        images = _render_pdf_pages(path, Path(tmp), pages)
        if not images:
            raise RuntimeError("no page images for 灰区复判")
        prompt = _PROMPT_VISION
        replies = [provider.caption(image, prompt) for image in images]  # type: ignore[union-attr]
    return "\n".join(replies)


def _render_pdf_pages(path: Path, dest: Path, page_count: int) -> list[Path]:
    pdf = pdfium.PdfDocument(str(path))
    written: list[Path] = []
    try:
        limit = min(page_count, len(pdf))
        for index in range(limit):
            page = pdf[index]
            bitmap = page.render(scale=1.0)
            image = dest / f"page-{index:02d}.png"
            bitmap.to_pil().save(image)
            written.append(image)
    finally:
        pdf.close()
    return written
