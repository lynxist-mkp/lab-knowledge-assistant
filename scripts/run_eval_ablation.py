#!/usr/bin/env python3
"""Run one golden-set eval: Hit@5, MRR, refusal accuracy for each configured group."""

from __future__ import annotations

from wenmai.config import Settings
from wenmai.eval import run_eval


def main() -> None:
    settings = Settings.load()
    run = run_eval(settings)
    print(
        f"eval run {run.timestamp} items={run.item_count} failed={run.failed_count}"
    )


if __name__ == "__main__":
    main()
