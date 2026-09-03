from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from wenmai.config import Settings
from wenmai.knowledge import create_knowledge
from wenmai.models import IngestResult
from wenmai.pipelines.ingestion import ingest_source, run_prepare_commit_batch

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST = Path("data/corpus/manifest.yaml")
DEFAULT_ITEMS_DIR = Path("data/corpus/items")


@dataclass
class IngestManifestResult:
    total: int
    attempted: int
    ingested: int
    skipped: int
    missing: int
    failed: int
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        if self.attempted == 0:
            return True
        return self.failed < self.attempted


@dataclass
class _CorpusPrepareCommitJob:
    source_path: Path
    settings: Settings
    pdf_load_mode: str | None
    on_stage: None
    knowledge: object | None
    item_id: str
    index: int
    result: IngestResult | None = None
    error: BaseException | None = None


def load_corpus_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid manifest: expected mapping at {path}")
    items = payload.get("items") or []
    if not isinstance(items, list):
        raise ValueError(f"invalid manifest: items must be a list at {path}")
    return items


_STUB_MARKERS = (
    "markitdown 未能提取正文",
    "源文件为以地图图件为主的 PDF",
)

_MIN_PDF_BYTES = 1024


def _is_stub_markdown(md_path: Path) -> bool:
    raw = md_path.read_text(encoding="utf-8")
    return any(marker in raw for marker in _STUB_MARKERS)


def item_source_path(items_dir: Path, item_id: str, *, prefer_pdf: bool = False) -> Path:
    """Resolve ingest source: stub .md or prefer_pdf uses .tmp/{id}.pdf when present."""
    md_path = items_dir / f"{item_id}.md"
    pdf_path = items_dir / ".tmp" / f"{item_id}.pdf"
    if pdf_path.is_file() and pdf_path.stat().st_size >= _MIN_PDF_BYTES:
        if prefer_pdf or (md_path.is_file() and _is_stub_markdown(md_path)):
            return pdf_path
    return md_path


def item_pdf_load_mode(item: dict[str, Any], source_path: Path) -> str | None:
    """Per-item PDF route override from manifest."""
    mode = item.get("pdf_load_mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    return None


def _record_terminal_result(
    result: IngestResult,
    *,
    index: int,
    total: int,
    item_id: str,
    ingested: int,
    skipped: int,
    failed: int,
    errors: list[str],
) -> tuple[int, int, int]:
    if result.status == "skipped":
        skipped += 1
        logger.info(
            "[%s/%s] SKIP %s: 未变更 (document_id=%s)",
            index,
            total,
            item_id,
            result.document_id[:12],
        )
    elif result.status == "rejected":
        failed += 1
        message = f"{item_id}: quality gate rejected"
        errors.append(message)
        logger.error("[%s/%s] REJECT %s", index, total, item_id)
    else:
        ingested += 1
        logger.info(
            "[%s/%s] OK %s: status=%s chunks=%s trace=%s",
            index,
            total,
            item_id,
            result.status,
            result.chunk_count,
            result.trace_id,
        )
    return ingested, skipped, failed


def ingest_corpus_manifest(
    settings: Settings,
    manifest_path: Path,
    items_dir: Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    item_ids: list[str] | None = None,
    batched: bool = True,
) -> IngestManifestResult:
    """Ingest corpus items listed in manifest.yaml from items_dir/{id}.md."""
    items = load_corpus_manifest(manifest_path)
    if item_ids is not None:
        wanted = set(item_ids)
        selected = [item for item in items if str(item.get("id") or "") in wanted]
    elif limit is None:
        selected = items
    else:
        selected = items[:limit]

    if batched and not dry_run:
        return _ingest_corpus_manifest_batched(settings, selected, items_dir)

    return _ingest_corpus_manifest_sequential(
        settings, selected, items_dir, dry_run=dry_run
    )


def _ingest_corpus_manifest_batched(
    settings: Settings,
    selected: list[dict[str, Any]],
    items_dir: Path,
) -> IngestManifestResult:
    knowledge = create_knowledge(settings)
    attempted = 0
    ingested = 0
    skipped = 0
    missing = 0
    failed = 0
    errors: list[str] = []
    jobs: list[_CorpusPrepareCommitJob] = []

    for index, item in enumerate(selected, start=1):
        item_id = str(item.get("id") or "")
        if not item_id:
            logger.warning("[%s/%s] SKIP: manifest item missing id", index, len(selected))
            continue

        source_path = item_source_path(
            items_dir,
            item_id,
            prefer_pdf=bool(item.get("prefer_pdf")),
        )
        if not source_path.is_file():
            logger.info(
                "[%s/%s] SKIP %s: 文件不存在 %s",
                index,
                len(selected),
                item_id,
                source_path,
            )
            missing += 1
            continue

        attempted += 1
        jobs.append(
            _CorpusPrepareCommitJob(
                source_path=source_path,
                settings=settings,
                pdf_load_mode=item_pdf_load_mode(item, source_path),
                on_stage=None,
                knowledge=knowledge,
                item_id=item_id,
                index=index,
            )
        )

    if jobs:
        run_prepare_commit_batch(jobs, batch_id="corpus-manifest")

    for job in jobs:
        if job.error is not None:
            failed += 1
            message = f"{job.item_id}: {job.error}"
            errors.append(message)
            logger.error("[%s/%s] FAIL %s", job.index, len(selected), message)
            continue
        if job.result is None:
            failed += 1
            message = f"{job.item_id}: no ingest result"
            errors.append(message)
            logger.error("[%s/%s] FAIL %s", job.index, len(selected), message)
            continue
        ingested, skipped, failed = _record_terminal_result(
            job.result,
            index=job.index,
            total=len(selected),
            item_id=job.item_id,
            ingested=ingested,
            skipped=skipped,
            failed=failed,
            errors=errors,
        )

    return IngestManifestResult(
        total=len(selected),
        attempted=attempted,
        ingested=ingested,
        skipped=skipped,
        missing=missing,
        failed=failed,
        errors=errors,
    )


def _ingest_corpus_manifest_sequential(
    settings: Settings,
    selected: list[dict[str, Any]],
    items_dir: Path,
    *,
    dry_run: bool,
) -> IngestManifestResult:
    attempted = 0
    ingested = 0
    skipped = 0
    missing = 0
    failed = 0
    errors: list[str] = []

    for index, item in enumerate(selected, start=1):
        item_id = str(item.get("id") or "")
        if not item_id:
            logger.warning("[%s/%s] SKIP: manifest item missing id", index, len(selected))
            continue

        source_path = item_source_path(
            items_dir,
            item_id,
            prefer_pdf=bool(item.get("prefer_pdf")),
        )
        if not source_path.is_file():
            logger.info(
                "[%s/%s] SKIP %s: 文件不存在 %s",
                index,
                len(selected),
                item_id,
                source_path,
            )
            missing += 1
            continue

        attempted += 1
        pdf_mode = item_pdf_load_mode(item, source_path)
        if dry_run:
            logger.info(
                "[%s/%s] DRY-RUN %s <- %s (pdf_load_mode=%s)",
                index,
                len(selected),
                item_id,
                source_path,
                pdf_mode,
            )
            continue

        try:
            result = ingest_source(source_path, settings, pdf_load_mode=pdf_mode)
        except Exception as exc:  # noqa: BLE001 - batch ingest logs and continues
            failed += 1
            message = f"{item_id}: {exc}"
            errors.append(message)
            logger.error("[%s/%s] FAIL %s", index, len(selected), message)
            continue

        ingested, skipped, failed = _record_terminal_result(
            result,
            index=index,
            total=len(selected),
            item_id=item_id,
            ingested=ingested,
            skipped=skipped,
            failed=failed,
            errors=errors,
        )

    return IngestManifestResult(
        total=len(selected),
        attempted=attempted,
        ingested=ingested,
        skipped=skipped,
        missing=missing,
        failed=failed,
        errors=errors,
    )
