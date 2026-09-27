#!/usr/bin/env python3
"""Attribute skill_manager repetition to its actual causes.

Decomposes duplicate n-grams into: query material, plan skeleton, serialization
scaffolding, and control-token scaffolding — so the generator fix targets the
real source instead of the metric.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from skill_metrics import iter_skill_records, skeleton_stats  # noqa: E402
from stage2_dedup import ngram_duplicate_ratio  # noqa: E402
from stage2_skill_synth import CONTROL, skeleton  # noqa: E402

CONTROL_RE = re.compile("|".join(re.escape(token) for token in CONTROL))


def strip_controls(text: str) -> str:
    return CONTROL_RE.sub(" ", text)


def strip_json_scaffold(text: str) -> str:
    """Remove punctuation and structural keys, leaving mostly query material."""
    text = re.sub(r'"(?:goal|plan|step|skill|input|input_from|query|status|error|'
                  r'attempts|attempt|fallback|final|context)"', " ", text)
    return re.sub(r"[{}\[\],:\"]", " ", text)


def extract_queries(records: list[dict]) -> list[str]:
    """Recover the query material each document was built from."""
    out = []
    for record in records:
        text = record["text"]
        found = re.findall(r'"(?:input|query|goal)":\s*"([^"]+)"', text)
        if not found:
            stripped = strip_controls(text)
            lines = [line.strip() for line in stripped.split("\n") if line.strip()]
            found = [re.sub(r"^\d+\.\s*", "", line) for line in lines[:1]]
        out.extend(found)
    return out


def analyze(path: str | Path) -> dict:
    records = list(iter_skill_records(path))
    if not records:
        return {"documents": 0}

    full4 = sum(ngram_duplicate_ratio(r["text"], 4) for r in records) / len(records)
    no_ctrl4 = sum(ngram_duplicate_ratio(strip_controls(r["text"]), 4)
                   for r in records) / len(records)
    no_scaffold4 = sum(ngram_duplicate_ratio(strip_json_scaffold(strip_controls(r["text"])), 4)
                       for r in records) / len(records)

    queries = extract_queries(records)
    query_counts = Counter(queries)
    query_text = " \n ".join(queries)
    query_dup4 = ngram_duplicate_ratio(query_text, 4)

    prefixes = Counter(q[:40] for q in queries if len(q) >= 40)

    by_shape: dict[str, float] = {}
    shape_groups: dict[str, list[dict]] = {}
    for record in records:
        shape_groups.setdefault(record.get("shape", "?"), []).append(record)
    for shape, group in shape_groups.items():
        by_shape[shape] = sum(ngram_duplicate_ratio(r["text"], 4) for r in group) / len(group)

    by_skeleton = Counter()
    skeleton_dup: dict[str, list[float]] = {}
    for record in records:
        key = skeleton(record["text"])
        by_skeleton[key] += 1
        skeleton_dup.setdefault(key, []).append(ngram_duplicate_ratio(record["text"], 4))
    contribution = sorted(
        ((key, len(values), sum(values) / len(values),
          len(values) / len(records) * (sum(values) / len(values)))
         for key, values in skeleton_dup.items()),
        key=lambda item: -item[3])[:10]

    # Most repeated 4-grams overall
    grams = Counter()
    for record in records:
        words = record["text"].split()
        for index in range(len(words) - 3):
            grams[tuple(words[index:index + 4])] += 1

    serialization = Counter(
        "json_compact" if r["text"].lstrip().startswith("{") and "\n" not in r["text"]
        else "json_indented" if r["text"].lstrip().startswith("{")
        else "context_prefixed" if r["text"].lstrip().startswith("<이전")
        else "dsl_inline" if "\n" not in r["text"]
        else "line_plan"
        for r in records)

    return {
        "corpus": str(path),
        "documents": len(records),
        "duplicate_4gram_full": full4,
        "duplicate_4gram_without_control_tokens": no_ctrl4,
        "duplicate_4gram_without_control_or_json_scaffold": no_scaffold4,
        "attribution": {
            "control_token_scaffold": full4 - no_ctrl4,
            "json_serialization_scaffold": no_ctrl4 - no_scaffold4,
            "query_and_plan_residue": no_scaffold4,
        },
        "queries": {
            "total": len(queries),
            "unique": len(query_counts),
            "unique_ratio": len(query_counts) / len(queries) if queries else 0.0,
            "mean_length": sum(len(q) for q in queries) / len(queries) if queries else 0.0,
            "duplicate_4gram_over_query_stream": query_dup4,
            "top_repeated": query_counts.most_common(5),
            "distinct_40char_prefixes": len(prefixes),
            "top_prefix_share": (prefixes.most_common(1)[0][1] / sum(prefixes.values())
                                 if prefixes else 0.0),
        },
        "by_shape_duplicate_4gram": by_shape,
        "serialization_mix": dict(serialization),
        "top_skeleton_contribution": [
            {"skeleton": key, "documents": count, "mean_dup4": mean, "contribution": share}
            for key, count, mean, share in contribution
        ],
        "top_repeated_4grams": [{"gram": " ".join(g), "count": c} for g, c in grams.most_common(20)],
        **skeleton_stats(records),
    }


if __name__ == "__main__":
    for target in sys.argv[1:]:
        print(json.dumps(analyze(target), ensure_ascii=False, indent=2))
