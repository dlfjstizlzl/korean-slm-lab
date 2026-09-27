#!/usr/bin/env python3
"""Build the large Korean-focused validation set for Stage-2 planning.

Korean prose and technical text hold the entire residual gap after Stage-1
recovery, and the 361k Korean characters in the balanced set are a thin basis
for that claim. This builds a much larger Korean-only set from the same
held-out splits.

NIKL is forbidden here exactly as it is in the recovery pipeline.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from build_recovery_data import (  # noqa: E402
    assert_record_allowed,
    assert_source_allowed,
    chunk_text,
    digest,
    line_key,
    read_lines,
)

PROCESSED = Path("data/processed")

PLAN = {
    "korean_general": (0.60, [(PROCESSED / "fineweb_general_holdout.txt", "fineweb_holdout")]),
    "korean_technical": (0.30, [(PROCESSED / "aihub_documents_valid_clean.txt", "aihub_document_valid")]),
    "dialogue_summary": (0.10, [
        (PROCESSED / "aihub_dialogue_valid_clean.txt", "aihub_dialogue_valid"),
        (PROCESSED / "aihub_summaries_valid_clean.txt", "aihub_summary_valid"),
    ]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/eval/recovery_korean_large_eval.jsonl")
    parser.add_argument("--report", default="outputs/korean_large_eval_report.json")
    parser.add_argument("--train-file", default="data/processed/recovery_train.jsonl")
    parser.add_argument("--balanced-eval", default="data/eval/recovery_balanced_eval.jsonl")
    parser.add_argument("--target-chars", type=int, default=2_400_000)
    parser.add_argument("--chunk", type=int, default=900)
    parser.add_argument("--min-chunk", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_keys = {
        line_key(json.loads(line)["text"])
        for line in Path(args.train_file).open(encoding="utf-8") if line.strip()
    }

    records: list[dict] = []
    dropped = Counter()
    for category, (share, specs) in PLAN.items():
        budget = int(args.target_chars * share)
        per_spec = max(1, budget // len(specs))
        pool: list[tuple[str, str]] = []
        for path, source in specs:
            pool.extend(read_lines(path, source, per_spec * 4))
        rng = random.Random(f"{args.seed}:{category}")
        rng.shuffle(pool)
        total = 0
        for text, source in pool:
            for piece in chunk_text(text, args.chunk, args.min_chunk):
                # Hard leakage guard against the recovery training corpus.
                if line_key(piece) in train_keys:
                    dropped[category] += 1
                    continue
                records.append({
                    "id": digest(category, source, piece)[:16],
                    "category": category, "source": source,
                    "text": piece, "characters": len(piece),
                })
                total += len(piece)
            if total >= budget:
                break

    keys = {line_key(record["text"]) for record in records}
    overlap = keys & train_keys
    if overlap:
        raise RuntimeError(f"{len(overlap)} chunks leak from recovery train into the Korean eval")
    for record in records:
        assert_source_allowed(record["source"])
        assert_record_allowed(record["text"], record["source"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_category = Counter()
    chars = Counter()
    for record in records:
        by_category[record["category"]] += 1
        chars[record["category"]] += record["characters"]
    total_chars = sum(chars.values())

    balanced_overlap = 0
    if Path(args.balanced_eval).exists():
        balanced_keys = {
            line_key(json.loads(line)["text"])
            for line in Path(args.balanced_eval).open(encoding="utf-8") if line.strip()
        }
        balanced_overlap = len(keys & balanced_keys)

    report = {
        "path": args.output,
        "seed": args.seed,
        "records": len(records),
        "characters": total_chars,
        "by_category": {
            category: {
                "records": by_category[category], "characters": chars[category],
                "share": chars[category] / total_chars,
            }
            for category in sorted(by_category)
        },
        "leakage": {
            "chunks_dropped_for_train_overlap": dict(dropped),
            "residual_train_overlap": 0,
            "balanced_eval_overlap": balanced_overlap,
            "note": "Balanced-eval overlap is expected and harmless: both draw from the same "
                    "held-out pool and neither is used for training.",
        },
        "nikl": {"used": False, "guard": "assert_source_allowed / assert_record_allowed"},
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "by_category"}, ensure_ascii=False, indent=2))
    print(json.dumps(report["by_category"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
