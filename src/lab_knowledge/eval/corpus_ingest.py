from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lab_knowledge.config import Settings
from lab_knowledge.knowledge import Knowledge, create_knowledge
from lab_knowledge.models import IngestResult
from lab_knowledge.pipelines.ingestion import run_prepare_commit_batch
from lab_knowledge.tracing.context import StageRecord

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
class _CorpusIngestJob:
    index: int
    item_id: str
    source_path: Path
    settings: Settings
    knowledge: Knowledge | None = None
    pdf_load_mode: str | None = None
    on_stage: Callable[[StageRecord], None] | None = None
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


def item_source_path(items_dir: Path, item_id: str) -> Path:
    return items_dir / f"{item_id}.md"


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
    batched: bool = True,
) -> IngestManifestResult:
    """Ingest corpus items listed in manifest.yaml from items_dir/{id}.md."""
    items = load_corpus_manifest(manifest_path)
    selected = items if limit is None else items[:limit]

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
    jobs: list[_CorpusIngestJob] = []
    total = len(selected)

    for index, item in enumerate(selected, start=1):
        item_id = str(item.get("id") or "")
        if not item_id:
            logger.warning("[%s/%s] SKIP: manifest item missing id", index, total)
            continue

        source_path = item_source_path(items_dir, item_id)
        if not source_path.is_file():
            logger.info(
                "[%s/%s] SKIP %s: 文件不存在 %s",
                index,
                total,
                item_id,
                source_path,
            )
            missing += 1
            continue

        attempted += 1
        jobs.append(
            _CorpusIngestJob(
                index=index,
                item_id=item_id,
                source_path=source_path,
                settings=settings,
                knowledge=knowledge,
            )
        )

    run_prepare_commit_batch(jobs, batch_id="corpus-manifest")

    for job in jobs:
        if job.error is not None:
            failed += 1
            message = f"{job.item_id}: {job.error}"
            errors.append(message)
            logger.error("[%s/%s] FAIL %s", job.index, total, message)
            continue

        if job.result is None:
            failed += 1
            message = f"{job.item_id}: no ingest result"
            errors.append(message)
            logger.error("[%s/%s] FAIL %s", job.index, total, message)
            continue

        ingested, skipped, failed = _record_terminal_result(
            job.result,
            index=job.index,
            total=total,
            item_id=job.item_id,
            ingested=ingested,
            skipped=skipped,
            failed=failed,
            errors=errors,
        )

    return IngestManifestResult(
        total=total,
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
    from lab_knowledge.pipelines.ingestion import ingest_source

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

        source_path = item_source_path(items_dir, item_id)
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
        if dry_run:
            logger.info(
                "[%s/%s] DRY-RUN %s <- %s",
                index,
                len(selected),
                item_id,
                source_path,
            )
            continue

        try:
            result = ingest_source(source_path, settings)
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
