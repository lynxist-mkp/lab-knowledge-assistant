#!/usr/bin/env python3
"""Download (or locate cached) a ModelScope model and print its local directory path."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve ModelScope model to a local path.")
    parser.add_argument(
        "model_id",
        help="ModelScope repo id, e.g. mlx-community/PaddleOCR-VL-1.6-5bit",
    )
    args = parser.parse_args()

    from modelscope import snapshot_download

    path = snapshot_download(args.model_id, revision="master")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
