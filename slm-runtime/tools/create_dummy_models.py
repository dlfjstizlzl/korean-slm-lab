#!/usr/bin/env python3
"""Create sparse dummy model files for the retained dummy runtime backend."""

import argparse
from pathlib import Path


MODEL_SIZES_MB = {
    "router.bin": 100,
    "search.bin": 300,
    "sql.bin": 500,
    "summary.bin": 400,
    "code.bin": 1024,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
        help="directory that receives the dummy model files",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply every size by this value (useful for quick tests)",
    )
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error("--scale must be greater than zero")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, size_mb in MODEL_SIZES_MB.items():
        byte_count = max(1, int(size_mb * args.scale * 1024 * 1024))
        path = args.output / name
        with path.open("wb") as model_file:
            model_file.truncate(byte_count)
        print(f"created {path} ({byte_count / (1024 * 1024):.1f} MB, sparse)")


if __name__ == "__main__":
    main()
