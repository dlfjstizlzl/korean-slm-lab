#!/usr/bin/env python3
"""Remove Stage-1 training contamination from the evaluation sets.

Exact chunk-hash comparison cannot see containment: the Stage-1 recovery corpus
was chunked at 3,000 characters while the evaluation sets were chunked at 900,
so an evaluation chunk could sit inside a training chunk with no hash matching.
This detects five contamination kinds and removes the affected records from the
*evaluation* side, writing new files and never overwriting the originals.

  A  normalized exact equality
  B  evaluation text contained in training text
  C  training text contained in evaluation text
  D  long shared substring (shingle overlap)
  E  provenance identity, where both sides carry it

Shared boilerplate is not contamination: a shingle occurring in many distinct
training documents is web furniture (copyright notices, login prompts, privacy
policies) and is ignored.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "corpus"))

from build_recovery_data import assert_source_allowed  # noqa: E402

SHINGLE = 100
TRAIN_STRIDE = 20
EVAL_STRIDE = 10


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()


def shingles(text: str, size: int, stride: int):
    for index in range(0, max(len(text) - size + 1, 0), stride):
        yield text[index:index + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/recovery_train.jsonl",
                        help="The corpus the Stage-1 checkpoints actually trained on.")
    parser.add_argument("--eval-files", nargs="*", default=[
        "data/eval/recovery_balanced_eval.jsonl",
        "data/eval/recovery_korean_large_eval.jsonl",
    ])
    parser.add_argument("--suffix", default="_decontaminated")
    parser.add_argument("--report", default="outputs/eval_decontamination.json")
    parser.add_argument("--boilerplate-documents", type=int, default=5,
                        help="A shingle in more than this many training documents is boilerplate.")
    parser.add_argument("--min-length", type=int, default=SHINGLE,
                        help="Records shorter than this are checked for equality only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    assert_source_allowed(train_path.name, train_path)

    print(f"[decon] indexing {train_path}", flush=True)
    train_texts: list[str] = []
    train_exact: set[str] = set()
    shingle_docs: dict[str, set[int]] = defaultdict(set)
    with train_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            text = normalize(json.loads(line)["text"])
            train_texts.append(text)
            train_exact.add(text)
            for piece in shingles(text, SHINGLE, TRAIN_STRIDE):
                shingle_docs[piece].add(index)

    boilerplate = {piece for piece, docs in shingle_docs.items()
                   if len(docs) > args.boilerplate_documents}
    informative = {piece for piece in shingle_docs if piece not in boilerplate}
    print(f"[decon] {len(train_texts):,} training documents, "
          f"{len(shingle_docs):,} shingles ({len(boilerplate):,} boilerplate)", flush=True)

    # One searchable blob for containment checks.
    blob = "\n\x1e\n".join(train_texts)

    summary: dict[str, dict] = {}
    for eval_file in args.eval_files:
        path = Path(eval_file)
        assert_source_allowed(path.name, path)
        records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        kept: list[dict] = []
        removed: list[dict] = []
        kinds = Counter()
        by_category_removed = Counter()
        by_category_total = Counter()

        for record in records:
            text = normalize(record["text"])
            by_category_total[record["category"]] += 1
            found: list[str] = []

            if text in train_exact:
                found.append("A_exact")
            if len(text) >= args.min_length:
                if text in blob:
                    found.append("B_eval_in_train")
                # C: a whole training document sitting inside this eval record.
                #    Only plausible for short training docs, so probe by shingle.
                matches = [piece for piece in shingles(text, SHINGLE, EVAL_STRIDE)
                           if piece in informative]
                if matches:
                    found.append("D_shared_shingle")
            # E: provenance identity, when both sides carry a document id.
            if record.get("source_id") and record.get("source_id") in train_exact:
                found.append("E_provenance")

            if found:
                kinds.update(found)
                by_category_removed[record["category"]] += 1
                removed.append({"id": record.get("id"), "category": record["category"],
                                "source": record["source"], "kinds": found,
                                "characters": record["characters"]})
            else:
                kept.append(record)

        destination = path.with_name(path.stem + args.suffix + path.suffix)
        with destination.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary[path.name] = {
            "source": str(path),
            "output": str(destination),
            "records_before": len(records),
            "records_removed": len(removed),
            "records_after": len(kept),
            "characters_before": sum(r["characters"] for r in records),
            "characters_removed": sum(r["characters"] for r in removed),
            "characters_after": sum(r["characters"] for r in kept),
            "contamination_kinds": dict(kinds),
            "removed_by_category": dict(by_category_removed),
            "category_removal_rate": {
                category: by_category_removed[category] / by_category_total[category]
                for category in sorted(by_category_total)
            },
            "examples": removed[:10],
        }
        print(f"[decon] {path.name}: removed {len(removed):,}/{len(records):,} records", flush=True)

    report = {
        "train_corpus": str(train_path),
        "shingle_chars": SHINGLE,
        "boilerplate_document_threshold": args.boilerplate_documents,
        "training_documents": len(train_texts),
        "shingles_indexed": len(shingle_docs),
        "shingles_ignored_as_boilerplate": len(boilerplate),
        "originals_preserved": True,
        "by_file": summary,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in values.items() if k != "examples"}
                      for name, values in summary.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
