#!/usr/bin/env python3
"""按 manifest.yaml 批量入库语料正文（data/corpus/items/{id}.md）。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from wenmai.components.model_guard import configure
from wenmai.config import Settings
from wenmai.eval.corpus_ingest import (
    DEFAULT_ITEMS_DIR,
    DEFAULT_MANIFEST,
    ingest_corpus_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按语料清单批量入库 Markdown 正文（两阶段同模型批次，支持幂等跳过）"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="语料清单路径（默认 data/corpus/manifest.yaml）",
    )
    parser.add_argument(
        "--items-dir",
        type=Path,
        default=DEFAULT_ITEMS_DIR,
        help="语料正文目录（默认 data/corpus/items）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要入库的文件，不实际写入知识库",
    )
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    items_dir = args.items_dir if args.items_dir.is_absolute() else repo_root / args.items_dir

    if not manifest_path.exists():
        print(f"找不到清单: {manifest_path}", file=sys.stderr)
        raise SystemExit(2)

    settings = Settings.load(repo_root / "settings.yaml")
    configure(exclusive=settings.resources.single_model_exclusive)
    result = ingest_corpus_manifest(
        settings,
        manifest_path,
        items_dir,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    print("\n=== 入库摘要 ===")
    print(f"清单条目: {result.total}")
    print(f"尝试入库: {result.attempted}")
    print(f"新入库: {result.ingested}")
    print(f"幂等跳过: {result.skipped}")
    print(f"文件缺失: {result.missing}")
    print(f"失败: {result.failed}")

    raise SystemExit(0 if result.success else 1)


if __name__ == "__main__":
    main()
