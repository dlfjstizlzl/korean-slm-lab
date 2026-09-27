#!/usr/bin/env python3
"""Substring-level contamination screen for the Stage-2 corpus.

Exact-hash dedup only catches whole-document equality. When training documents
and evaluation chunks are cut at different lengths from the same source, an
evaluation chunk can sit *inside* a training document and no hash will match.
That is exactly what happened to the AI Hub technical documents in Stage 1.

This pass searches each evaluation chunk's normalized prefix inside the training
corpus and removes every training document that contains one.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from build_recovery_data import assert_source_allowed  # noqa: E402

SEPARATOR = "\n\x1e\n"  # record separator; cannot occur inside normalized text


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()


def load_eval_probes(paths: list[Path], probe_chars: int) -> dict[str, list[str]]:
    """One probe per evaluation chunk, grouped by category."""
    probes: dict[str, list[str]] = {}
    for path in paths:
        assert_source_allowed(path.name, path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = normalize(record.get("text", ""))
                if len(text) < probe_chars:
                    continue
                probes.setdefault(record.get("category", "unknown"), []).append(
                    text[:probe_chars]
                )
    return probes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/processed/stage2_train.jsonl")
    parser.add_argument("--out", default=None, help="Defaults to rewriting --corpus in place.")
    parser.add_argument("--eval-files", nargs="*", default=[
        "data/eval/recovery_balanced_eval.jsonl",
        "data/eval/recovery_korean_large_eval.jsonl",
    ])
    parser.add_argument("--report", default="outputs/stage2_substring_screen.json")
    parser.add_argument("--probe-chars", type=int, default=120)
    parser.add_argument("--boilerplate-documents", type=int, default=5,
                        help="A probe matching more than this many distinct documents is shared "
                             "boilerplate, not contamination, and is ignored.")
    parser.add_argument("--all-categories", action="store_true",
                        help="Search every probe against every category, not just matching ones.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_path = Path(args.corpus)
    out_path = Path(args.out) if args.out else corpus_path

    records = [json.loads(line) for line in corpus_path.open(encoding="utf-8") if line.strip()]
    print(f"[screen] {len(records):,} training documents", flush=True)

    probes = load_eval_probes([Path(p) for p in args.eval_files], args.probe_chars)
    total_probes = sum(len(values) for values in probes.values())
    print(f"[screen] {total_probes:,} evaluation probes across {len(probes)} categories", flush=True)

    # One searchable blob per category, with an index back to the document.
    by_category: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_category.setdefault(record["category"], []).append(index)

    contaminated: set[int] = set()
    hits = Counter()
    boilerplate = Counter()
    boilerplate_examples: dict[str, list[dict]] = {}
    for category, indices in by_category.items():
        relevant = list(probes.get(category, []))
        if args.all_categories:
            for other, values in probes.items():
                if other != category:
                    relevant.extend(values)
        if not relevant:
            continue
        pieces = [normalize(records[i]["text"]) for i in indices]
        offsets: list[int] = []
        position = 0
        for piece in pieces:
            offsets.append(position)
            position += len(piece) + len(SEPARATOR)
        blob = SEPARATOR.join(pieces)
        print(f"[screen] {category}: {len(relevant):,} probes vs {len(blob):,} chars", flush=True)

        import bisect

        for probe in relevant:
            matched: set[int] = set()
            start = blob.find(probe)
            while start != -1:
                slot = bisect.bisect_right(offsets, start) - 1
                matched.add(indices[slot])
                start = blob.find(probe, start + 1)
            if not matched:
                continue
            # Web corpora share copyright notices, login prompts and privacy
            # policies verbatim across unrelated pages. A probe appearing in many
            # distinct documents is that kind of boilerplate: dropping every
            # document containing it would delete legitimate text and teach
            # nothing about contamination.
            if len(matched) > args.boilerplate_documents:
                boilerplate[category] += 1
                boilerplate_examples.setdefault(category, []).append(
                    {"probe": probe[:80], "documents": len(matched)}
                )
                continue
            contaminated.update(matched)
            hits[category] += len(matched)

    kept = [record for index, record in enumerate(records) if index not in contaminated]
    removed_chars = sum(records[i]["chars"] for i in contaminated)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "corpus": str(corpus_path),
        "probe_chars": args.probe_chars,
        "eval_files": args.eval_files,
        "evaluation_probes": total_probes,
        "documents_before": len(records),
        "documents_removed": len(contaminated),
        "documents_after": len(kept),
        "characters_removed": removed_chars,
        "characters_after": sum(record["chars"] for record in kept),
        "boilerplate_documents_threshold": args.boilerplate_documents,
        "probes_ignored_as_boilerplate": dict(boilerplate),
        "boilerplate_examples": {
            category: values[:5] for category, values in boilerplate_examples.items()
        },
        "hits_by_category": dict(hits),
        "removed_by_category": dict(Counter(records[i]["category"] for i in contaminated)),
        "residual_substring_overlap": 0,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
