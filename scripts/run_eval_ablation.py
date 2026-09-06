#!/usr/bin/env python3
"""Run one golden-set eval: Hit@5, MRR, refusal accuracy for each configured group."""

from __future__ import annotations

import argparse
import logging

from lab_knowledge.config import Settings
from lab_knowledge.eval import run_eval, run_rewrite_compare

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run golden-set evaluation.")
    parser.add_argument(
        "--rewrite-compare",
        action="store_true",
        help="Compare rewrite off vs on (RRF+Rerank backbone only).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N golden items (for smoke / CE batch checks).",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Ablation groups to run (default: all from settings).",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip Ragas judge scoring.",
    )
    args = parser.parse_args()
    settings = Settings.load()
    if args.rewrite_compare:
        run = run_rewrite_compare(settings)
        print(
            f"rewrite compare {run.timestamp} items={run.item_count} "
            f"failed={run.failed_count}"
        )
        return
    ragas = False if args.no_ragas else None
    run = run_eval(
        settings,
        ragas=ragas,
        item_limit=args.limit,
        groups=args.groups,
    )
    print(
        f"eval run {run.timestamp} items={run.item_count} failed={run.failed_count}"
    )


if __name__ == "__main__":
    main()
