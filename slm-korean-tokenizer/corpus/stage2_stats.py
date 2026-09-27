#!/usr/bin/env python3
"""Tokenize the Stage-2 corpus with KO-32K and report measured statistics.

Nothing here is estimated: the corpus is actually tokenized, and token counts,
per-category rates, vocabulary coverage and the resulting epoch fraction are all
measured.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tokenizer"))

from common import CONTROL_TOKENS  # noqa: E402

DEFAULT_TOKENIZER = "outputs/tokenizer-ko-32000-mixed10-structured-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/processed/stage2_train.jsonl")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--report", default="outputs/stage2_token_stats.json")
    parser.add_argument("--csv", default="outputs/stage2_token_stats.csv")
    parser.add_argument("--token-cache", default="data/processed/stage2_token_lengths.jsonl")
    parser.add_argument("--target-tokens", type=int, default=60_000_000)
    parser.add_argument("--batch", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)

    chars = Counter()
    tokens = Counter()
    documents = Counter()
    usage = Counter()
    cache = Path(args.token_cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    batch_texts: list[str] = []
    batch_meta: list[tuple[str, str]] = []
    processed = 0

    def flush(handle) -> None:
        nonlocal batch_texts, batch_meta
        if not batch_texts:
            return
        encoded = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        for (category, source), ids in zip(batch_meta, encoded):
            tokens[category] += len(ids)
            usage.update(ids)
            handle.write(json.dumps({"category": category, "source": source,
                                     "tokens": len(ids)}, ensure_ascii=False) + "\n")
        batch_texts = []
        batch_meta = []

    with Path(args.corpus).open("r", encoding="utf-8") as corpus_handle, \
            cache.open("w", encoding="utf-8") as cache_handle:
        for line in corpus_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            category = record["category"]
            documents[category] += 1
            chars[category] += record["chars"]
            batch_texts.append(record["text"])
            batch_meta.append((category, record["source"]))
            if len(batch_texts) >= args.batch:
                flush(cache_handle)
                processed += args.batch
                if processed % 20480 == 0:
                    print(f"  {processed:,} documents, {sum(tokens.values()):,} tokens", flush=True)
        flush(cache_handle)

    total_chars = sum(chars.values())
    total_tokens = sum(tokens.values())

    # Vocabulary coverage
    used = len(usage)
    unused = vocab_size - used
    control_ids = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in CONTROL_TOKENS}
    top = [
        {"id": token_id, "token": tokenizer.convert_ids_to_tokens(token_id),
         "count": count, "share": count / total_tokens}
        for token_id, count in usage.most_common(40)
    ]
    counts_sorted = sorted(usage.values(), reverse=True)
    cumulative = 0
    head_for_90 = 0
    for index, value in enumerate(counts_sorted, 1):
        cumulative += value
        if cumulative >= 0.9 * total_tokens:
            head_for_90 = index
            break

    def probe(name: str, strings: list[str]) -> dict:
        out = {}
        for text in strings:
            ids = tokenizer.encode(text, add_special_tokens=False)
            out[text] = {"ids": ids, "count": sum(usage.get(i, 0) for i in ids)}
        return out

    report = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "corpus": args.corpus,
        "totals": {
            "documents": sum(documents.values()),
            "characters": total_chars,
            "tokens": total_tokens,
            "tokens_per_character": total_tokens / total_chars if total_chars else 0.0,
            "characters_per_token": total_chars / total_tokens if total_tokens else 0.0,
        },
        "by_category": {
            category: {
                "documents": documents[category],
                "characters": chars[category],
                "tokens": tokens[category],
                "tokens_per_character": tokens[category] / chars[category] if chars[category] else 0.0,
                "characters_per_token": chars[category] / tokens[category] if tokens[category] else 0.0,
                "char_share": chars[category] / total_chars if total_chars else 0.0,
                "token_share": tokens[category] / total_tokens if total_tokens else 0.0,
                "share_drift_pp": (
                    (tokens[category] / total_tokens - chars[category] / total_chars) * 100
                    if total_chars and total_tokens else 0.0
                ),
            }
            for category in sorted(chars)
        },
        "vocabulary_coverage": {
            "used_tokens": used,
            "unused_tokens": unused,
            "used_share": used / vocab_size,
            "tokens_covering_90_percent": head_for_90,
            "top_tokens": top,
            "control_tokens": {
                token: {"id": token_id, "count": usage.get(token_id, 0)}
                for token, token_id in control_ids.items()
            },
            "json_structural": probe("json", ['{"', '":', '":"', '"},', '[{', '}]']),
            "sql_keywords": probe("sql", ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"]),
            "korean_samples": probe("korean", ["메모리", "프로세스", "데이터", "설명", "시스템"]),
        },
        "training_budget": {
            "target_tokens": args.target_tokens,
            "corpus_tokens": total_tokens,
            "epochs_required": args.target_tokens / total_tokens if total_tokens else None,
            "repeated_token_fraction": (
                max(0.0, (args.target_tokens - total_tokens) / args.target_tokens)
                if total_tokens else None
            ),
            "one_epoch_budget": total_tokens,
            "within_one_epoch": total_tokens >= args.target_tokens,
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with Path(args.csv).open("w", encoding="utf-8") as handle:
        handle.write("category,documents,characters,tokens,tokens_per_char,char_share,token_share,share_drift_pp\n")
        for category, values in report["by_category"].items():
            handle.write(f"{category},{values['documents']},{values['characters']},{values['tokens']},"
                         f"{values['tokens_per_character']:.6f},{values['char_share']:.6f},"
                         f"{values['token_share']:.6f},{values['share_drift_pp']:.4f}\n")

    print(json.dumps({"totals": report["totals"], "budget": report["training_budget"],
                      "vocab": {k: v for k, v in report["vocabulary_coverage"].items()
                                if k in ("used_tokens", "unused_tokens", "tokens_covering_90_percent")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
