#!/usr/bin/env python3
"""Run golden-set ablation batch: Hit@5, MRR, refusal accuracy for each configured group."""

from __future__ import annotations

from wenmai.config import Settings
from wenmai.eval.runner import run_ablation_batch


def main() -> None:
    settings = Settings.load()
    output_path = run_ablation_batch(settings)
    print(f"ablation run written to {output_path}")


if __name__ == "__main__":
    main()
