#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from benchmark_tokenizers import embedding_estimate, get_hidden_size
from common import BYTE_FALLBACK_RE, CATEGORIES, CONTROL_TOKENS, assert_control_tokens_atomic, load_tokenizer

EXAMPLES = [
    "오늘 학교에서 운영체제 수업을 들었다.",
    "가상메모리는 프로세스에게 물리 메모리와 독립된 주소 공간을 제공한다.",
    "Transformer의 self-attention 연산은 sequence length에 따라 계산량이 증가한다.",
    "Apple Silicon은 CPU와 GPU가 unified memory를 공유한다.",
    "<검색>Apple Silicon 통합 메모리 구조",
    "<요약><이전0>",
    "<SQL>지난달 가입한 사용자 수를 조회한다.",
    "https://example.com/test?q=memory",
]

DEFAULT_DATASETS = {
    "training_corpus": "data/processed/tokenizer_train_real.txt",
    "fineweb_holdout": "data/processed/fineweb_general_holdout.txt",
    "aihub_dialogue_valid": "data/processed/aihub_dialogue_valid_clean.txt",
    "aihub_document_valid": "data/processed/aihub_documents_valid_clean.txt",
    "aihub_query_valid": "data/processed/aihub_queries_valid_clean.txt",
    "aihub_summary_valid": "data/processed/aihub_summaries_valid_clean.txt",
    "independent_eval_40": "data/korean_eval.txt",
    "skill_dsl": "data/eval_skill_dsl.txt",
    "code_url": "data/eval_code_url.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real-data Korean tokenizer candidates by source.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=[
            "outputs/real/mf2/tokenizer-ko-16000",
            "outputs/real/mf2/tokenizer-ko-32000",
            "outputs/real/mf5/tokenizer-ko-16000",
            "outputs/real/mf5/tokenizer-ko-32000",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-json", default="outputs/tokenizer_real_benchmark.json")
    parser.add_argument("--output-markdown", default="outputs/TOKENIZER_REAL_BENCHMARK.md")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def candidate_name(path: str) -> str:
    value = Path(path)
    frequency = value.parent.name
    return f"KO-{value.name.rsplit('-', 1)[-1]}-{frequency}"


def nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def iter_batches(path: str | Path, batch_size: int):
    batch = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.rstrip("\r\n")
            if "\t" in text:
                candidate, remainder = text.split("\t", 1)
                if candidate in CATEGORIES:
                    text = remainder.strip()
            if not text:
                continue
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def token_properties(tokenizer, token_id: int, cache: dict[int, tuple[bool, bool]]) -> tuple[bool, bool]:
    if token_id in cache:
        return cache[token_id]
    token = str(tokenizer.convert_ids_to_tokens(token_id))
    explicit_byte = bool(BYTE_FALLBACK_RE.fullmatch(token))
    try:
        surface = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        surface = tokenizer.decode([token_id], skip_special_tokens=False)
    invalid_surface = "\ufffd" in surface
    cache[token_id] = (explicit_byte, invalid_surface)
    return cache[token_id]


def benchmark_file(tokenizer, path: str | Path, batch_size: int) -> dict:
    counts: list[int] = []
    total_chars = total_tokens = unknown = explicit_byte = invalid_surface = fallback_like = 0
    cache: dict[int, tuple[bool, bool]] = {}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for texts in iter_batches(path, batch_size):
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for text, ids in zip(texts, encoded):
            counts.append(len(ids))
            total_chars += len(text)
            total_tokens += len(ids)
            for value in ids:
                token_id = int(value)
                is_unknown = unk_id is not None and token_id == int(unk_id)
                is_byte, is_invalid = token_properties(tokenizer, token_id, cache)
                unknown += is_unknown
                explicit_byte += is_byte
                invalid_surface += is_invalid
                fallback_like += is_unknown or is_byte or is_invalid
    return {
        "path": str(path),
        "documents": len(counts),
        "total_characters": total_chars,
        "total_tokens": total_tokens,
        "characters_per_token": total_chars / (total_tokens or 1),
        "tokens_per_character": total_tokens / (total_chars or 1),
        "average_tokens_per_document": total_tokens / (len(counts) or 1),
        "median_tokens_per_document": statistics.median(counts) if counts else 0,
        "p95_tokens_per_document": nearest_rank(counts, 0.95),
        "unknown_tokens": unknown,
        "explicit_byte_tokens": explicit_byte,
        "explicit_byte_percent": 100.0 * explicit_byte / (total_tokens or 1),
        "invalid_standalone_utf8_tokens": invalid_surface,
        "invalid_standalone_utf8_percent": 100.0 * invalid_surface / (total_tokens or 1),
        "fallback_like_tokens": fallback_like,
        "fallback_like_percent": 100.0 * fallback_like / (total_tokens or 1),
        "observed_token_ids": len(cache),
    }


def inspect_examples(tokenizer) -> list[dict]:
    result = []
    for text in EXAMPLES:
        ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        result.append({
            "text": text,
            "token_strings": [str(value) for value in tokenizer.convert_ids_to_tokens(ids)],
            "token_ids": ids,
            "token_count": len(ids),
        })
    return result


def markdown(payload: dict) -> str:
    names = list(payload["tokenizers"])
    lines = [
        "# Real-data tokenizer benchmark",
        "",
        "> FineWeb holdout and AI Hub VL are in-domain validation. The 40-sentence independent set is too small for a final Korean-generalization claim.",
        "",
    ]
    for dataset, path in payload["datasets"].items():
        lines += [f"## {dataset}", "", f"`{path}`", ""]
        lines += ["| Tokenizer | Vocab rows | Tokens | Token reduction vs Qwen | Chars/token | Median | p95 | Invalid UTF-8 fragments |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in names:
            item = payload["tokenizers"][name]
            metrics = item["datasets"][dataset]
            lines.append(
                f"| {name} | {item['embedding_vocab_rows']:,} | {metrics['total_tokens']:,} | "
                f"{metrics['token_reduction_percent']:+.2f}% | {metrics['characters_per_token']:.3f} | "
                f"{metrics['median_tokens_per_document']} | {metrics['p95_tokens_per_document']} | "
                f"{metrics['invalid_standalone_utf8_percent']:.3f}% |"
            )
        lines.append("")
    lines += [
        "## Training-to-validation comparison",
        "",
        "| Tokenizer | Train reduction | FineWeb reduction | AI Hub VL reduction | Independent reduction | Largest train advantage | Warning |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, values in payload["training_validation_gap"].items():
        lines.append(
            f"| {name} | {values['train_reduction_percent']:.2f}% | "
            f"{values['fineweb_reduction_percent']:.2f}% | {values['aihub_reduction_percent']:.2f}% | "
            f"{values['independent_reduction_percent']:.2f}% | {values['largest_train_advantage_points']:.2f} pp | "
            f"{values['warning'] or 'none'} |"
        )
    lines.append("")
    lines += ["## Tokenization examples", ""]
    for text in EXAMPLES:
        lines += [f"### {text}", ""]
        for name in names:
            entry = next(item for item in payload["tokenizers"][name]["examples"] if item["text"] == text)
            lines.append(f"- `{name}` ({entry['token_count']}): `{entry['token_strings']}` / `{entry['token_ids']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    for path in DEFAULT_DATASETS.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    hidden_size = get_hidden_size(config)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    base = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    augmented = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    augmented.add_special_tokens({"additional_special_tokens": CONTROL_TOKENS}, replace_additional_special_tokens=False)
    loaded = [("Qwen original", base, int(config.vocab_size)), ("Qwen + Skill", augmented, max(int(config.vocab_size), len(augmented)))]
    for path in args.tokenizers:
        tokenizer = load_tokenizer(path, args.trust_remote_code)
        assert_control_tokens_atomic(tokenizer)
        loaded.append((candidate_name(path), tokenizer, len(tokenizer)))

    payload = {
        "base_model": args.base_model,
        "datasets": DEFAULT_DATASETS,
        "definitions": {
            "invalid_standalone_utf8": "Token IDs whose individual decode contains U+FFFD; full sequences may still decode correctly.",
            "fallback_like": "Union of explicit <0xNN>, unknown, and invalid-standalone-surface token occurrences.",
        },
        "model": {"hidden_size": hidden_size, "original_vocab_rows": int(config.vocab_size), "tie_word_embeddings": tied},
        "tokenizers": {},
    }
    for name, tokenizer, vocab_rows in loaded:
        print(f"[Benchmark] {name}", flush=True)
        datasets = {}
        for dataset, path in DEFAULT_DATASETS.items():
            print(f"  - {dataset}", flush=True)
            datasets[dataset] = benchmark_file(tokenizer, path, args.batch_size)
        payload["tokenizers"][name] = {
            "tokenizer_size": len(tokenizer),
            "embedding_vocab_rows": vocab_rows,
            "embedding_estimate": embedding_estimate(vocab_rows, hidden_size, tied),
            "control_token_ids": {
                token: tokenizer.encode(token, add_special_tokens=False) for token in CONTROL_TOKENS
            },
            "datasets": datasets,
            "examples": inspect_examples(tokenizer),
        }

    for dataset in DEFAULT_DATASETS:
        baseline = payload["tokenizers"]["Qwen original"]["datasets"][dataset]["total_tokens"]
        for item in payload["tokenizers"].values():
            metrics = item["datasets"][dataset]
            ratio = metrics["total_tokens"] / (baseline or 1)
            metrics["compression_ratio"] = ratio
            metrics["token_reduction_percent"] = 100.0 * (1.0 - ratio)

    aihub_sets = [
        "aihub_dialogue_valid",
        "aihub_document_valid",
        "aihub_query_valid",
        "aihub_summary_valid",
    ]
    base_aihub_tokens = sum(
        payload["tokenizers"]["Qwen original"]["datasets"][dataset]["total_tokens"]
        for dataset in aihub_sets
    )
    training_validation_gap = {}
    for name, item in payload["tokenizers"].items():
        train_reduction = item["datasets"]["training_corpus"]["token_reduction_percent"]
        fineweb_reduction = item["datasets"]["fineweb_holdout"]["token_reduction_percent"]
        independent_reduction = item["datasets"]["independent_eval_40"]["token_reduction_percent"]
        aihub_tokens = sum(item["datasets"][dataset]["total_tokens"] for dataset in aihub_sets)
        aihub_reduction = 100.0 * (1.0 - aihub_tokens / (base_aihub_tokens or 1))
        validation_reductions = [fineweb_reduction, aihub_reduction, independent_reduction]
        largest_advantage = max(train_reduction - value for value in validation_reductions)
        warning = None
        if train_reduction > 0.1 and any(value <= 0 for value in validation_reductions):
            warning = "training improves while at least one Korean validation set does not"
        training_validation_gap[name] = {
            "train_reduction_percent": train_reduction,
            "fineweb_reduction_percent": fineweb_reduction,
            "aihub_total_tokens": aihub_tokens,
            "aihub_reduction_percent": aihub_reduction,
            "independent_reduction_percent": independent_reduction,
            "largest_train_advantage_points": largest_advantage,
            "warning": warning,
        }
    payload["training_validation_gap"] = training_validation_gap

    candidates = [name for name in payload["tokenizers"] if name.startswith("KO-")]
    validation_sets = [
        "fineweb_holdout", "aihub_dialogue_valid", "aihub_document_valid",
        "aihub_query_valid", "aihub_summary_valid", "independent_eval_40", "code_url",
    ]
    ranking = []
    for name in candidates:
        reductions = [payload["tokenizers"][name]["datasets"][dataset]["token_reduction_percent"] for dataset in validation_sets]
        weighted_tokens = sum(payload["tokenizers"][name]["datasets"][dataset]["total_tokens"] for dataset in validation_sets)
        ranking.append({
            "tokenizer": name,
            "embedding_vocab_rows": payload["tokenizers"][name]["embedding_vocab_rows"],
            "validation_tokens": weighted_tokens,
            "minimum_validation_reduction_percent": min(reductions),
            "all_validation_sets_better_than_qwen": all(value > 0 for value in reductions),
        })
    ranking.sort(key=lambda item: (item["validation_tokens"], item["embedding_vocab_rows"]))
    payload["candidate_ranking"] = ranking
    payload["recommendation_note"] = "Select only after considering every validation source, fragmentation, code/URL, and vocabulary-dependent parameters. NIKL external evaluation is still unavailable."
    payload["versions"] = {"transformers": __import__("transformers").__version__, "torch": torch.__version__}

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_markdown).write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"candidate_ranking": ranking}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
