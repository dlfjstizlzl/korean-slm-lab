#!/usr/bin/env python3
"""Build the Stage-1 embedding-recovery training corpus and balanced eval set.

Two deterministic artifacts:

  data/eval/recovery_balanced_eval.jsonl   category-balanced held-out evaluation
  data/processed/recovery_train.jsonl      recovery training documents

The evaluation set is drawn exclusively from held-out splits that the tokenizer
experiments already reserved, and every record carries its source and category so
per-category NLL can be reported. Train/eval leakage is removed from the *train*
side; the eval set is never modified to fit the training data.

NIKL is forbidden. It stays sealed as an external evaluation corpus and is
blocked here at the path level, per source, and per record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tokenizer"))

from common import CONTROL_TOKENS, normalize_text  # noqa: E402

# --------------------------------------------------------------------------- #
# NIKL safeguard
# --------------------------------------------------------------------------- #

FORBIDDEN_SOURCE_MARKERS = ("nikl", "newspaper", "nirw", "nwrw")


class ForbiddenSourceError(RuntimeError):
    """Raised when a NIKL-derived path or record reaches the recovery pipeline."""


def assert_source_allowed(source: str, path: str | Path | None = None) -> None:
    """Hard block on NIKL. Recovery training must never see this corpus."""
    haystacks = [str(source).lower()]
    if path is not None:
        haystacks.append(str(path).lower())
    for haystack in haystacks:
        for marker in FORBIDDEN_SOURCE_MARKERS:
            if marker in haystack:
                raise ForbiddenSourceError(
                    f"NIKL-derived data is forbidden in recovery training/eval: "
                    f"source={source!r} path={path!r} (matched {marker!r})"
                )


NIKL_ARTICLE_ID = re.compile(r"\b(?:NIRW|NWRW)\d{7,}")


def assert_record_allowed(text: str, source: str) -> None:
    if NIKL_ARTICLE_ID.search(text):
        raise ForbiddenSourceError(f"record from {source!r} carries a NIKL article id")


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #

def digest(*parts: str) -> str:
    return hashlib.blake2b("\0".join(parts).encode("utf-8"), digest_size=16).hexdigest()


def line_key(text: str) -> str:
    """Normalized key for exact line-overlap detection."""
    collapsed = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()
    return hashlib.blake2b(collapsed.encode("utf-8"), digest_size=16).hexdigest()


def chunk_text(text: str, target: int, minimum: int) -> list[str]:
    """Split a document into ~`target`-character chunks on whitespace boundaries."""
    text = text.strip()
    if len(text) <= target:
        return [text] if len(text) >= minimum else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target, len(text))
        if end < len(text):
            window = text.rfind(" ", start + minimum, end)
            if window > start:
                end = window
        chunk = text[start:end].strip()
        if len(chunk) >= minimum:
            chunks.append(chunk)
        start = end
    return chunks


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #

def read_lines(path: Path, source: str, limit_chars: int) -> list[tuple[str, str]]:
    assert_source_allowed(source, path)
    out: list[tuple[str, str]] = []
    total = 0
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = normalize_text(raw)
            if not text or text.startswith("#"):
                continue
            assert_record_allowed(text, source)
            out.append((text, source))
            total += len(text)
            if total >= limit_chars:
                break
    return out


def read_jsonl(path: Path, sources: set[str] | None, limit_chars: int) -> list[tuple[str, str]]:
    assert_source_allowed(path.name, path)
    out: list[tuple[str, str]] = []
    total = 0
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            source = str(record.get("source", path.stem))
            if sources is not None and source not in sources:
                continue
            text = normalize_text(record.get("text", ""))
            if not text:
                continue
            assert_source_allowed(source, path)
            assert_record_allowed(text, source)
            out.append((text, source))
            total += len(text)
            if total >= limit_chars:
                break
    return out


# --------------------------------------------------------------------------- #
# Skill DSL synthesis
# --------------------------------------------------------------------------- #

SKILL_TEMPLATES = [
    "<검색>{q}",
    "<요약>{q}",
    "<SQL>{q}",
    "<코드>{q}",
    "<검색>{q}<종료>",
    "<요약><이전0>",
    "<요약><이전1>{q}",
    "<검색><이전2>{q}<종료>",
    '<SQL>{q}\n{{"skill":"sql","input":"{q}"}}',
    '<코드>{q}\n{{"skill":"code","input":"{q}"}}',
]


def build_skill_dsl(queries: list[tuple[str, str]], limit_chars: int, seed: int) -> list[tuple[str, str]]:
    """Synthesize Skill-DSL text from held-out Korean queries.

    The Skill DSL is a project-defined format with no natural corpus, so it is
    generated. Train and eval draw from disjoint query pools (the AI Hub train
    and validation splits respectively), so the synthesis cannot leak.
    """
    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    total = 0
    for index, (query, _) in enumerate(queries):
        template = SKILL_TEMPLATES[index % len(SKILL_TEMPLATES)]
        text = template.format(q=query)
        if rng.random() < 0.25:
            text = f"{text}\n{rng.choice(CONTROL_TOKENS)}"
        out.append((text, "skill_dsl_synthetic"))
        total += len(text)
        if total >= limit_chars:
            break
    return out


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #

PROCESSED = Path("data/processed")

EVAL_PLAN = {
    "korean_general": (0.40, [("lines", PROCESSED / "fineweb_general_holdout.txt", "fineweb_holdout")]),
    "korean_technical": (0.20, [("lines", PROCESSED / "aihub_documents_valid_clean.txt", "aihub_document_valid")]),
    "dialogue": (0.10, [
        ("lines", PROCESSED / "aihub_dialogue_valid_clean.txt", "aihub_dialogue_valid"),
        ("lines", PROCESSED / "aihub_summaries_valid_clean.txt", "aihub_summary_valid"),
    ]),
    "json_structured": (0.10, [
        ("jsonl", PROCESSED / "structured/json_extra_valid.jsonl", None),
        ("jsonl", PROCESSED / "structured/structured_valid.jsonl",
         {"starcoder_json_pretty", "starcoder_json_compact"}),
        ("jsonl", PROCESSED / "mixed/mixed_validation.jsonl", {"structured_data"}),
    ]),
    "code_url": (0.10, [
        ("jsonl", PROCESSED / "mixed/mixed_validation.jsonl",
         {"source_code", "identifiers_technical", "urls_api"}),
    ]),
    "sql": (0.05, [("jsonl", PROCESSED / "structured/structured_valid.jsonl", {"stack_sql", "spider_sql"})]),
    "skill_dsl": (0.05, [("skill", PROCESSED / "aihub_queries_valid_clean.txt", "aihub_query_valid")]),
}

TRAIN_PLAN = {
    "korean_general": (0.40, [("lines", PROCESSED / "fineweb_general_train.txt", "fineweb_train")]),
    "korean_technical": (0.30, [("lines", PROCESSED / "aihub_documents_train.txt", "aihub_document_train")]),
    "dialogue": (0.10, [
        ("lines", PROCESSED / "aihub_dialogue_train.txt", "aihub_dialogue_train"),
        ("lines", PROCESSED / "aihub_summaries_train.txt", "aihub_summary_train"),
    ]),
    "code_url": (0.07, [
        ("jsonl", PROCESSED / "mixed/mixed_pool.jsonl",
         {"source_code", "identifiers_technical", "urls_api", "cli_paths"}),
    ]),
    "json_structured": (0.07, [
        ("jsonl", PROCESSED / "structured/json_extra_train.jsonl", None),
        ("jsonl", PROCESSED / "structured/structured_train.jsonl",
         {"starcoder_json_pretty", "starcoder_json_compact"}),
    ]),
    "sql": (0.03, [("jsonl", PROCESSED / "structured/structured_train.jsonl", {"stack_sql", "spider_sql"})]),
    "skill_dsl": (0.03, [("skill", PROCESSED / "aihub_queries_train.txt", "aihub_query_train")]),
}


def gather(plan: dict, budget: int, seed: int) -> dict[str, list[tuple[str, str]]]:
    collected: dict[str, list[tuple[str, str]]] = {}
    for category, (share, specs) in plan.items():
        target = int(budget * share)
        per_spec = max(1, target // max(len(specs), 1))
        records: list[tuple[str, str]] = []
        for kind, path, selector in specs:
            # Over-read so short records still reach the character target.
            headroom = per_spec * 3
            if kind == "lines":
                records.extend(read_lines(path, str(selector), headroom))
            elif kind == "jsonl":
                records.extend(read_jsonl(path, selector, headroom))
            elif kind == "skill":
                queries = read_lines(path, str(selector), headroom)
                records.extend(build_skill_dsl(queries, headroom, seed))
        collected[category] = records
    return collected


def take_chars(records: list[tuple[str, str]], budget: int, chunk: int, minimum: int,
               seed: int, category: str) -> list[dict]:
    """Deterministically shuffle, chunk, and cut a category to its character budget."""
    rng = random.Random(f"{seed}:{category}")
    ordered = list(records)
    rng.shuffle(ordered)
    out: list[dict] = []
    total = 0
    for text, source in ordered:
        for piece in chunk_text(text, chunk, minimum):
            out.append({
                "id": digest(category, source, piece)[:16],
                "category": category, "source": source, "text": piece,
                "characters": len(piece),
            })
            total += len(piece)
            if total >= budget:
                return out
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-out", default="data/eval/recovery_balanced_eval.jsonl")
    parser.add_argument("--train-out", default="data/processed/recovery_train.jsonl")
    parser.add_argument("--report", default="outputs/recovery_data_report.json")
    parser.add_argument("--eval-chars", type=int, default=600_000)
    parser.add_argument("--train-chars", type=int, default=9_000_000)
    parser.add_argument("--eval-chunk", type=int, default=900)
    parser.add_argument("--train-chunk", type=int, default=3000)
    parser.add_argument("--min-chunk", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng_seed = args.seed

    print("[eval] gathering held-out sources", flush=True)
    eval_pool = gather(EVAL_PLAN, args.eval_chars, rng_seed)
    eval_records: list[dict] = []
    for category, (share, _) in EVAL_PLAN.items():
        eval_records.extend(take_chars(
            eval_pool[category], int(args.eval_chars * share),
            args.eval_chunk, args.min_chunk, rng_seed, category,
        ))

    eval_keys = {line_key(record["text"]) for record in eval_records}

    print("[train] gathering training sources", flush=True)
    train_pool = gather(TRAIN_PLAN, args.train_chars, rng_seed)
    train_records: list[dict] = []
    dropped = Counter()
    for category, (share, _) in TRAIN_PLAN.items():
        candidates = take_chars(
            train_pool[category], int(args.train_chars * share) * 2,
            args.train_chunk, args.min_chunk, rng_seed, category,
        )
        budget = int(args.train_chars * share)
        total = 0
        for record in candidates:
            # Leakage guard: drop any train chunk that matches an eval chunk.
            if line_key(record["text"]) in eval_keys:
                dropped[category] += 1
                continue
            train_records.append(record)
            total += record["characters"]
            if total >= budget:
                break

    # Final assertion: the two sets must not intersect.
    train_keys = {line_key(record["text"]) for record in train_records}
    overlap = train_keys & eval_keys
    if overlap:
        raise RuntimeError(f"{len(overlap)} chunks leak between recovery train and eval")

    for path, records in ((Path(args.eval_out), eval_records), (Path(args.train_out), train_records)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def summarize(records: list[dict]) -> dict:
        by_category = Counter()
        chars_by_category = Counter()
        by_source = Counter()
        for record in records:
            by_category[record["category"]] += 1
            chars_by_category[record["category"]] += record["characters"]
            by_source[record["source"]] += 1
        total = sum(chars_by_category.values())
        return {
            "records": len(records),
            "characters": total,
            "by_category": {
                category: {
                    "records": by_category[category],
                    "characters": chars_by_category[category],
                    "share": chars_by_category[category] / total if total else 0.0,
                }
                for category in sorted(by_category)
            },
            "by_source": dict(by_source.most_common()),
        }

    report = {
        "seed": rng_seed,
        "eval": {"path": args.eval_out, "chunk_chars": args.eval_chunk, **summarize(eval_records)},
        "train": {"path": args.train_out, "chunk_chars": args.train_chunk, **summarize(train_records)},
        "leakage": {
            "eval_chunks": len(eval_keys),
            "train_chunks_dropped_for_overlap": dict(dropped),
            "residual_overlap": 0,
        },
        "nikl": {
            "used": False,
            "guard": "assert_source_allowed blocks nikl/newspaper/nirw in every path and source; "
                     "assert_record_allowed rejects any record carrying a NIRW article id.",
        },
        "skill_dsl": {
            "synthetic": True,
            "train_pool": "data/processed/aihub_queries_train.txt",
            "eval_pool": "data/processed/aihub_queries_valid_clean.txt",
            "note": "Disjoint query pools; the Skill DSL has no natural corpus.",
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"eval": report["eval"]["by_category"], "train": report["train"]["by_category"]},
                     ensure_ascii=False, indent=2))
    print(f"eval  -> {args.eval_out}  ({report['eval']['characters']:,} chars)")
    print(f"train -> {args.train_out} ({report['train']['characters']:,} chars)")


if __name__ == "__main__":
    main()
