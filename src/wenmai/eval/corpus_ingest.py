from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from wenmai.config import Settings
from wenmai.pipelines.ingestion import ingest_source

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


def ingest_corpus_manifest(
    settings: Settings,
    manifest_path: Path,
    items_dir: Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> IngestManifestResult:
    """Ingest corpus items listed in manifest.yaml from items_dir/{id}.md."""
    items = load_corpus_manifest(manifest_path)
    selected = items if limit is None else items[:limit]

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

        if result.status == "skipped":
            skipped += 1
            logger.info(
                "[%s/%s] SKIP %s: 未变更 (document_id=%s)",
                index,
                len(selected),
                item_id,
                result.document_id[:12],
            )
        elif result.status == "rejected":
            failed += 1
            message = f"{item_id}: quality gate rejected"
            errors.append(message)
            logger.error("[%s/%s] REJECT %s", index, len(selected), item_id)
        else:
            ingested += 1
            logger.info(
                "[%s/%s] OK %s: status=%s chunks=%s trace=%s",
                index,
                len(selected),
                item_id,
                result.status,
                result.chunk_count,
                result.trace_id,
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
