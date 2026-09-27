#!/usr/bin/env python3
"""Deterministic ablation over skill-generator variants.

All variants share the same seed, the same character budget and the same source
documents. The goal is the *smallest* change that brings canonical duplicate
4-gram to or below the v1 baseline without losing structural diversity — not
the most elaborate configuration.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

import stage2_skill_synth as v1gen  # noqa: E402
import stage2_skill_synth_v2 as v2gen  # noqa: E402
from stage2_dedup import content_hash, ngram_duplicate_ratio  # noqa: E402

SOURCES = [
    ("aihub", "data/processed/stage2_aihub_technical.jsonl"),
    ("wikipedia", "data/raw/korean_technical_wiki.jsonl"),
]


def load_groups(limit_per_source: int, balanced: bool, multi_position: bool) -> list[dict]:
    """Build candidate groups. Flags switch the ablated behaviours on and off."""
    per_source: dict[str, list[dict]] = {}
    for name, path in SOURCES:
        groups: list[dict] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record["text"]
                if multi_position:
                    candidates = v2gen.extract_candidates(text)
                else:
                    head = text.split(". ")[0][:120].strip()
                    candidates = [head] if 12 <= len(head) <= 120 else []
                if candidates:
                    groups.append({"candidates": candidates, "source": name,
                                   "doc_id": record.get("source_id", "")})
                if len(groups) >= limit_per_source:
                    break
        per_source[name] = groups

    if not balanced:
        # Original behaviour: read sources in order until the cap.
        merged: list[dict] = []
        for name, _ in SOURCES:
            merged.extend(per_source[name])
        return merged[:60000]

    # Deterministic stratified interleave so no source dominates.
    merged = []
    streams = [per_source[name] for name, _ in SOURCES]
    position = 0
    while any(position < len(stream) for stream in streams):
        for stream in streams:
            if position < len(stream):
                merged.append(stream[position])
        position += 1
    return merged[:60000]


def measure(records: list[dict], label: str) -> dict:
    four = [ngram_duplicate_ratio(r["text"], 4) for r in records]
    eight = [ngram_duplicate_ratio(r["text"], 8) for r in records]
    skeletons = Counter(v2gen.skeleton(r["text"]) for r in records)
    hashes = Counter(content_hash(r["text"]) for r in records)
    queries: list[str] = []
    import re
    for record in records:
        queries.extend(re.findall(r'"(?:input|query|goal)":\s*"([^"]+)"', record["text"]))
    lines = unique_lines = 0
    for record in records:
        pieces = [line.strip() for line in record["text"].split("\n") if line.strip()]
        lines += len(pieces)
        unique_lines += len(set(pieces))
    return {
        "variant": label,
        "documents": len(records),
        "characters": sum(r["chars"] for r in records),
        "duplicate_4gram": sum(four) / len(four),
        "duplicate_8gram": sum(eight) / len(eight),
        "duplicate_line_ratio": 1.0 - unique_lines / lines if lines else 0.0,
        "duplicate_documents": sum(c - 1 for c in hashes.values() if c > 1),
        "distinct_skeletons": len(skeletons),
        "top_skeleton_share": skeletons.most_common(1)[0][1] / len(records),
        "unique_query_ratio": (len(set(queries)) / len(queries)) if queries else None,
        "mean_query_length": (sum(map(len, queries)) / len(queries)) if queries else None,
        "source_mix": dict(Counter(r.get("query_source", "n/a") for r in records)),
        "shapes": dict(Counter(r["shape"] for r in records)),
        "serializations": dict(Counter(r.get("serialization", "n/a") for r in records)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-chars", type=int, default=4_442_590,
                        help="Match the v2 skill budget so variants are comparable.")
    parser.add_argument("--seed", type=int, default=42, help="Unchanged; no seed search.")
    parser.add_argument("--limit-per-source", type=int, default=40000)
    parser.add_argument("--report", default="outputs/stage2_skill_ablation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []

    # A: current generator, current query harvest (first sentence, source order)
    groups_a = load_groups(args.limit_per_source, balanced=False, multi_position=False)
    flat_a = [group["candidates"][0] for group in groups_a]
    records_a = v1gen.generate(flat_a, args.target_chars, seed=args.seed)
    for record in records_a:
        record.setdefault("serialization", "n/a")
    results.append(measure(records_a, "A_current"))

    # B: diversified query extraction only (v2 generator, unbalanced sources)
    groups_b = load_groups(args.limit_per_source, balanced=False, multi_position=True)
    results.append(measure(v2gen.generate(groups_b, args.target_chars, seed=args.seed),
                           "B_diversified_query"))

    # C: B plus deterministic source balancing
    groups_c = load_groups(args.limit_per_source, balanced=True, multi_position=True)
    results.append(measure(v2gen.generate(groups_c, args.target_chars, seed=args.seed),
                           "C_query_plus_source_balance"))

    # D: C is already the full generator (skeleton + serialization changes are
    #    intrinsic to v2), so D re-runs it with a wider candidate window to test
    #    whether more material helps further.
    groups_d = load_groups(args.limit_per_source * 2, balanced=True, multi_position=True)
    results.append(measure(v2gen.generate(groups_d, args.target_chars, seed=args.seed),
                           "D_full_wider_pool"))

    payload = {"seed": args.seed, "target_chars": args.target_chars,
               "seed_search_performed": False, "variants": results}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    header = (f"{'variant':30s} {'docs':>7s} {'dup4':>7s} {'dup8':>7s} {'dupline':>8s} "
              f"{'skel':>5s} {'top':>6s} {'uniqQ':>6s}")
    print(header)
    for row in results:
        print(f"{row['variant']:30s} {row['documents']:7,d} {row['duplicate_4gram']:7.4f} "
              f"{row['duplicate_8gram']:7.4f} {row['duplicate_line_ratio']:8.4f} "
              f"{row['distinct_skeletons']:5d} {row['top_skeleton_share']:6.3f} "
              f"{(row['unique_query_ratio'] or 0):6.3f}")


if __name__ == "__main__":
    main()
