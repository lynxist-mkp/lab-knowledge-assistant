#!/usr/bin/env python3
"""Run one golden-set eval: Hit@5, MRR, refusal accuracy for each configured group."""

from __future__ import annotations

import argparse

from wenmai.config import Settings
from wenmai.eval import run_eval, run_rewrite_compare


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden-set evaluation.")
    parser.add_argument(
        "--rewrite-compare",
        action="store_true",
        help="Compare rewrite off vs on (RRF+Rerank backbone only).",
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
    run = run_eval(settings)
    print(
        f"eval run {run.timestamp} items={run.item_count} failed={run.failed_count}"
    )


if __name__ == "__main__":
    main()
