#!/usr/bin/env python3
"""Repository-level 90/10 split of the extra StarCoder JSON corpus.

Splitting by repository (rather than by document) keeps near-duplicate files
from the same project out of both sides at once. Each kept document contributes
two surface forms, exactly as the first structured pass did: the compact form
the SkillLM runtime actually transports, and the document's own lines as they
appear in a checked-in file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = "data/external/json/starcoder_json_extra.jsonl"
OUTPUT_DIR = "data/processed/structured"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split the extra JSON corpus by repository.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--eval-file", default="data/eval_json_new.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--max-line-chars", type=int, default=4_000)
    parser.add_argument("--report", default="outputs/json_extra_split_report.json")
    parser.add_argument(
        "--protect-eval",
        nargs="*",
        default=[
            "data/eval_json.txt",
            "data/eval_sql.txt",
            "data/eval_skilllm_structured.txt",
            "data/eval_skill_dsl.txt",
            "data/eval_code_url.txt",
            "data/korean_eval.txt",
            "data/processed/mixed/mixed_validation.txt",
        ],
        help="Benchmark files whose lines must never enter the training split",
    )
    parser.add_argument(
        "--fairness-corpora",
        nargs="*",
        default=[
            "data/processed/tokenizer_train_mixed10_structured.txt",
            "data/processed/tokenizer_train_mixed10.txt",
            "data/processed/tokenizer_train_real.txt",
        ],
        help="Existing training corpora; lines they contain are dropped from the new holdout "
             "so no baseline tokenizer has already seen the new benchmark",
    )
    return parser.parse_args()


def normalize_line(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\t", "    ")).strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score(seed: int, namespace: str, value: str) -> float:
    raw = hashlib.blake2b(f"{seed}\0{namespace}\0{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(raw, "big") / float(1 << 64)


def records_for(document: str, parsed, document_id: str, max_line: int) -> list[str]:
    rows: list[str] = []
    compact = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if len(compact) <= max_line:
        rows.append(compact)
    for line in document.split("\n"):
        line = normalize_line(line)
        if line and len(line) <= max_line:
            rows.append(line)
    return rows


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    if not source.exists():
        raise SystemExit(f"missing {source}; run corpus/harvest_json_extra.py first")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    valid_rows: list[dict] = []
    train_seen: set[str] = set()
    valid_seen: set[str] = set()
    repositories: dict[str, bool] = {}
    counts: Counter = Counter()

    with source.open("r", encoding="utf-8") as handle:
        for raw in handle:
            record = json.loads(raw)
            text = str(record["text"])
            repository = str(record.get("repository") or "")
            # Documents with no repository label fall back to their own id so
            # they still land deterministically on one side of the split.
            bucket_key = repository or f"__doc__:{record.get('sha256')}"
            if bucket_key not in repositories:
                repositories[bucket_key] = score(args.seed, "json-extra-repo", bucket_key) < args.valid_ratio
            is_valid = repositories[bucket_key]
            try:
                parsed = json.loads(text)
            except Exception:
                counts["unparseable_skipped"] += 1
                continue
            document_id = str(record.get("sha256"))
            rows = records_for(text, parsed, document_id, args.max_line_chars)
            target, seen = (valid_rows, valid_seen) if is_valid else (train_rows, train_seen)
            for line in rows:
                key = sha256(line)
                if key in seen:
                    counts["duplicate_line_dropped"] += 1
                    continue
                seen.add(key)
                target.append({"text": line, "source": "starcoder_json_extra", "repository": repository,
                               "document": document_id})
            counts["valid_documents" if is_valid else "train_documents"] += 1

    def hashes_of(paths) -> set[str]:
        collected: set[str] = set()
        for relative in paths:
            path = Path(relative)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.rstrip("\n")
                    if stripped:
                        collected.add(sha256(stripped))
        return collected

    # The same upstream documents can be re-harvested from the stream, and the
    # two passes normalize whitespace differently, so a document-level hash does
    # not catch them. Clean line by line, in both directions.
    protected = hashes_of(args.protect_eval)
    kept_train = []
    for row in train_rows:
        if sha256(row["text"]) in protected:
            counts["train_lines_removed_for_benchmark_leakage"] += 1
            continue
        kept_train.append(row)
    train_rows = kept_train
    train_seen = {sha256(row["text"]) for row in train_rows}

    # A line present on both sides is removed from validation only.
    cross = 0
    seen_by_baselines = hashes_of(args.fairness_corpora)
    unfair = 0
    deduped_valid = []
    for row in valid_rows:
        if sha256(row["text"]) in train_seen:
            cross += 1
            continue
        if sha256(row["text"]) in seen_by_baselines:
            unfair += 1
            continue
        deduped_valid.append(row)

    for name, rows in (("json_extra_train", train_rows), ("json_extra_valid", deduped_valid)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out_dir / f"{name}.txt").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row["text"] + "\n")

    Path(args.eval_file).write_text(
        "".join(row["text"] + "\n" for row in deduped_valid), encoding="utf-8"
    )

    train_repos = sum(1 for key, is_valid in repositories.items() if not is_valid)
    report = {
        "input": str(source),
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "split_unit": "repository",
        "repositories_total": len(repositories),
        "train_repositories": train_repos,
        "valid_repositories": len(repositories) - train_repos,
        "train_documents": counts["train_documents"],
        "valid_documents": counts["valid_documents"],
        "train_lines": len(train_rows),
        "train_characters": sum(len(r["text"]) for r in train_rows),
        "valid_lines": len(deduped_valid),
        "valid_characters": sum(len(r["text"]) for r in deduped_valid),
        "validation_lines_removed_for_train_overlap": cross,
        "validation_lines_removed_seen_by_baselines": unfair,
        "train_lines_removed_for_benchmark_leakage": counts["train_lines_removed_for_benchmark_leakage"],
        "duplicate_line_dropped": counts["duplicate_line_dropped"],
        "unparseable_skipped": counts["unparseable_skipped"],
        "eval_file": args.eval_file,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
