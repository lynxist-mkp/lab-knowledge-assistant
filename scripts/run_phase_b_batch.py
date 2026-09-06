#!/usr/bin/env python3
"""Phase B 一键跑批：语料入库 → 四组消融 → 改写对比 → Ragas 汇总。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lab_knowledge.config import Settings
from lab_knowledge.eval.corpus_ingest import DEFAULT_ITEMS_DIR, DEFAULT_MANIFEST
from lab_knowledge.eval.phase_b import (
    DEFAULT_BAD_CASES_PATH,
    format_metrics_summary,
    run_phase_b_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase B 评测跑批：入库、四组消融、改写对比与 Ragas 汇总"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="跳过语料入库，直接跑评测",
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
        "--bad-cases",
        type=Path,
        default=DEFAULT_BAD_CASES_PATH,
        help="Bad Case 笔记路径（默认 .scratch/lab-knowledge/phase-b-bad-cases.md）",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="禁用 Ragas 评分（默认在 judge 密钥可用时自动开启）",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="强制开启 Ragas（无密钥时会写入 unavailable）",
    )
    parser.add_argument("--ingest-limit", type=int, default=None, help="入库只处理前 N 条")
    parser.add_argument(
        "--ingest-dry-run",
        action="store_true",
        help="入库阶段仅 dry-run（评测仍会执行）",
    )
    args = parser.parse_args()

    if args.ragas and args.no_ragas:
        print("不能同时指定 --ragas 与 --no-ragas", file=sys.stderr)
        raise SystemExit(2)

    ragas: bool | None
    if args.no_ragas:
        ragas = False
    elif args.ragas:
        ragas = True
    else:
        ragas = None

    repo_root = Path(__file__).resolve().parents[1]
    settings = Settings.load(repo_root / "settings.yaml")

    try:
        result = run_phase_b_batch(
            settings,
            skip_ingest=args.skip_ingest,
            manifest_path=args.manifest,
            items_dir=args.items_dir,
            bad_cases_path=args.bad_cases,
            ragas=ragas,
            ingest_limit=args.ingest_limit,
            ingest_dry_run=args.ingest_dry_run,
        )
    except RuntimeError as exc:
        print(f"Phase B 失败: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    summary_payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    print(format_metrics_summary(summary_payload))
    print(f"phase_b_summary={result.summary_path}")
    print(
        f"ablation={settings.root / settings.evaluation.runs}"
        f"/{result.ablation_run.timestamp}.json"
    )
    print(
        f"rewrite_compare={settings.root / settings.evaluation.runs}"
        f"/{result.rewrite_compare_run.timestamp}.json"
    )
    if result.bad_cases_path is not None:
        print(f"bad_cases={result.bad_cases_path}")


if __name__ == "__main__":
    main()
