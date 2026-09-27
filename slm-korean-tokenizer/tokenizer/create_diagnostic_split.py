#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import CorpusRecord, read_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic stratified diagnostic split without replacing source files.")
    parser.add_argument("--input", default="data/korean_train.txt")
    parser.add_argument("--train-output", default="data/diagnostic_train.txt")
    parser.add_argument("--eval-output", default="data/diagnostic_eval.txt")
    parser.add_argument("--metadata-output", default="data/diagnostic_split.json")
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def serialize(records: list[CorpusRecord]) -> str:
    lines = [f"{record.category}\t{record.text}" if record.category else record.text for record in records]
    return "\n".join(lines) + "\n"


def require_writable_destinations(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to replace existing diagnostic files: {existing}; pass --overwrite")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.eval_ratio < 1.0:
        raise ValueError("--eval-ratio must be between 0 and 1")
    records = read_corpus(args.input, deduplicate=True)
    grouped: dict[str, list[CorpusRecord]] = defaultdict(list)
    for record in records:
        grouped[record.category or "uncategorized"].append(record)

    train: list[CorpusRecord] = []
    evaluation: list[CorpusRecord] = []
    counts: dict[str, dict[str, int]] = {}
    for index, (category, category_records) in enumerate(sorted(grouped.items())):
        shuffled = list(category_records)
        random.Random(args.seed + index).shuffle(shuffled)
        eval_count = max(1, round(len(shuffled) * args.eval_ratio))
        evaluation.extend(shuffled[:eval_count])
        train.extend(shuffled[eval_count:])
        counts[category] = {"train": len(shuffled) - eval_count, "eval": eval_count}

    random.Random(args.seed + 1000).shuffle(train)
    random.Random(args.seed + 2000).shuffle(evaluation)
    destinations = [Path(args.train_output), Path(args.eval_output), Path(args.metadata_output)]
    require_writable_destinations(destinations, args.overwrite)
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    destinations[0].write_text(serialize(train), encoding="utf-8")
    destinations[1].write_text(serialize(evaluation), encoding="utf-8")
    metadata = {
        "source": args.input,
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "train_lines": len(train),
        "eval_lines": len(evaluation),
        "categories": counts,
        "warning": "This split measures only within-corpus generalization of the synthetic/curated seed data, not real Korean generalization.",
    }
    destinations[2].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
