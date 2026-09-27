#!/usr/bin/env python3
"""Canonical repetition metrics for the skill_manager category.

Two different numbers were reported for the same corpus (0.2356 and 0.2364)
because the build-time statistic samples the first 5,000 documents written and
also counts the holdout split, while an ad-hoc check used the full training
population. Neither was wrong; they measured different things.

This module is the single source of truth. Everything that judges condition J
calls `skill_duplicate_stats`, so v1, v2 and v3 are always compared by the same
function, over the same population, with the same denominator.

Canonical definition
--------------------
population : every skill_manager record in the *training* file
             (`stage2_train*.jsonl`), no sampling, no cap
unit       : one document
statistic  : mean over documents of the per-document duplicate n-gram ratio
n-gram     : whitespace-split word tokens, contiguous, size 4 and size 8
ratio      : 1 - (distinct n-grams / total n-grams); 0.0 when a document has
             fewer than n words
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from stage2_dedup import ngram_duplicate_ratio  # noqa: E402

CATEGORY = "skill_manager"


def iter_skill_records(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("category") == CATEGORY:
                yield record


def duplicate_line_ratio(records: list[dict]) -> float:
    lines = unique = 0
    for record in records:
        pieces = [line.strip() for line in record["text"].split("\n") if line.strip()]
        if pieces:
            lines += len(pieces)
            unique += len(set(pieces))
    return 1.0 - unique / lines if lines else 0.0


def skill_duplicate_stats(path: str | Path) -> dict:
    """The canonical condition-J measurement. Full population, no sampling."""
    records = list(iter_skill_records(path))
    if not records:
        return {"documents": 0}
    four = [ngram_duplicate_ratio(record["text"], 4) for record in records]
    eight = [ngram_duplicate_ratio(record["text"], 8) for record in records]
    hashes = Counter(record.get("sha256") or record["text"] for record in records)
    return {
        "corpus": str(path),
        "documents": len(records),
        "characters": sum(record["chars"] for record in records),
        "duplicate_4gram": sum(four) / len(four),
        "duplicate_8gram": sum(eight) / len(eight),
        "duplicate_line_ratio": duplicate_line_ratio(records),
        "duplicate_documents": sum(count - 1 for count in hashes.values() if count > 1),
    }


def skeleton_stats(records: list[dict]) -> dict:
    from stage2_skill_synth import skeleton

    counts = Counter(skeleton(record["text"]) for record in records)
    shapes = Counter(record.get("shape", "?") for record in records)
    return {
        "distinct_skeletons": len(counts),
        "top_skeleton_share": counts.most_common(1)[0][1] / len(records) if records else 0.0,
        "shapes": dict(shapes),
    }


if __name__ == "__main__":
    for target in sys.argv[1:]:
        print(json.dumps(skill_duplicate_stats(target), ensure_ascii=False, indent=2))
