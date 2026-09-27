#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from common import (
    CONTROL_TOKENS,
    assert_control_tokens_atomic,
    load_tokenizer,
    read_corpus,
    token_metrics,
    tokenizer_name,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline and Korean-focused tokenizers on one corpus.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--eval-file", default="data/korean_eval.txt")
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["outputs/tokenizer-ko-16000", "outputs/tokenizer-ko-24000", "outputs/tokenizer-ko-32000"],
    )
    parser.add_argument("--output-json", default="outputs/benchmark.json")
    parser.add_argument("--output-markdown", default="outputs/BENCHMARK.md")
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def get_hidden_size(config) -> int:
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not find hidden size in model config")


def embedding_estimate(vocab_size: int, hidden_size: int, tied: bool) -> dict:
    input_parameters = vocab_size * hidden_size
    vocabulary_dependent_parameters = input_parameters * (1 if tied else 2)
    return {
        "embedding_vocab_size": vocab_size,
        "input_embedding_parameters": input_parameters,
        "vocabulary_dependent_parameters": vocabulary_dependent_parameters,
        "estimated_input_embedding_memory_fp16_bytes": input_parameters * 2,
        "estimated_input_embedding_memory_int8_bytes": input_parameters,
        "estimated_input_embedding_memory_int4_bytes": (input_parameters + 1) // 2,
        "estimated_vocabulary_dependent_memory_fp16_bytes": vocabulary_dependent_parameters * 2,
        "estimated_vocabulary_dependent_memory_int8_bytes": vocabulary_dependent_parameters,
        "estimated_vocabulary_dependent_memory_int4_bytes": (vocabulary_dependent_parameters + 1) // 2,
    }


def choose_balanced(results: list[dict]) -> tuple[str | None, str]:
    candidates = [item for item in results if item["kind"] == "candidate" and item.get("control_tokens_atomic")]
    if not candidates:
        return None, "No candidate preserved every control token as one token."
    best_tokens = min(item["metrics"]["total_tokens"] for item in candidates)
    best_fallback = min(item["metrics"]["byte_fallback_percent"] for item in candidates)
    eligible = [
        item
        for item in candidates
        if item["metrics"]["total_tokens"] <= best_tokens * 1.02
        and item["metrics"]["byte_fallback_percent"] <= best_fallback + 0.1
    ]
    if not eligible:
        token_eligible = [item for item in candidates if item["metrics"]["total_tokens"] <= best_tokens * 1.02]
        chosen = min(token_eligible, key=lambda item: item["metrics"]["vocab_size"])
        reason = (
            "No candidate simultaneously met the 2% token-count and 0.1-point byte-fallback constraints. "
            "Selected the smallest vocabulary within 2% of the best token count; inspect byte fallback before use."
        )
        return chosen["name"], reason
    chosen = min(eligible, key=lambda item: item["metrics"]["vocab_size"])
    reason = (
        "Smallest vocabulary within 2% of the best candidate token count, with all control tokens atomic "
        "and explicit byte-fallback rate no more than 0.1 percentage point above the best candidate."
    )
    return chosen["name"], reason


def human_bytes(value: int) -> str:
    return f"{value / (1024 * 1024):,.2f} MiB"


def build_markdown(payload: dict) -> str:
    lines = [
        "# Measured tokenizer benchmark",
        "",
        "> Generated from the local evaluation corpus. Memory figures are parameter-count estimates only.",
        "",
        f"- Base model: `{payload['base_model']}`",
        f"- Evaluation sentences: {payload['corpus']['eval_lines']:,}",
        f"- Evaluation characters: {payload['corpus']['eval_characters']:,}",
        f"- Hidden size: {payload['model']['hidden_size']:,}",
        f"- `tie_word_embeddings`: `{str(payload['model']['tie_word_embeddings']).lower()}`",
        "",
        "| Tokenizer | Vocab | Tokens | Reduction | Avg/sentence | p95 | Chars/token | Byte fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["tokenizers"]:
        metrics = item["metrics"]
        reduction = item.get("token_reduction_percent", 0.0)
        lines.append(
            f"| {item['name']} | {metrics['vocab_size']:,} | {metrics['total_tokens']:,} | "
            f"{reduction:.2f}% | {metrics['average_tokens_per_sentence']:.2f} | "
            f"{metrics['p95_tokens_per_sentence']} | {metrics['characters_per_token']:.3f} | "
            f"{metrics['byte_fallback_percent']:.3f}% |"
        )
    lines += ["", "## Skill DSL atomic-token check", ""]
    for item in payload["tokenizers"]:
        if item["kind"] == "baseline":
            continue
        status = "PASS" if item.get("control_tokens_atomic") else "FAIL"
        lines.append(f"- `{item['name']}`: {status}")
    lines += [
        "",
        "## Vocabulary-dependent parameter estimate",
        "",
        "For untied models this includes both the input embedding and output LM head. Quantized sizes ignore metadata, scales, padding, and runtime overhead.",
        "",
        "| Tokenizer | Input embedding params | Vocabulary-dependent params | FP16 estimate | INT8 estimate | INT4 estimate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in payload["tokenizers"]:
        estimate = item["embedding_estimate"]
        lines.append(
            f"| {item['name']} | {estimate['input_embedding_parameters']:,} | "
            f"{estimate['vocabulary_dependent_parameters']:,} | "
            f"{human_bytes(estimate['estimated_vocabulary_dependent_memory_fp16_bytes'])} | "
            f"{human_bytes(estimate['estimated_vocabulary_dependent_memory_int8_bytes'])} | "
            f"{human_bytes(estimate['estimated_vocabulary_dependent_memory_int4_bytes'])} |"
        )
    lines += [
        "",
        "## Balanced recommendation",
        "",
        f"**{payload['recommendation']['tokenizer'] or 'None'}** — {payload['recommendation']['reason']}",
        "",
        "The recommendation deliberately does not select solely by minimum token count. Inspect rare Korean terms, ASCII/code, numbers, URLs, and downstream loss before continued pretraining.",
        "",
    ]
    if payload.get("categories"):
        lines += ["## Category breakdown", ""]
        for category, entries in payload["categories"].items():
            lines.append(f"### {category}")
            lines.append("")
            for name, metrics in entries.items():
                lines.append(f"- `{name}`: {metrics['total_tokens']:,} tokens, {metrics['characters_per_token']:.3f} chars/token")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = read_corpus(args.eval_file, deduplicate=args.deduplicate)
    if not records:
        raise ValueError(f"No usable evaluation records in {args.eval_file}")
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    hidden_size = get_hidden_size(config)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    baseline = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    loaded = [("Qwen original", args.base_model, baseline, "baseline")]
    for path in args.tokenizers:
        loaded.append((tokenizer_name(path), path, load_tokenizer(path, args.trust_remote_code), "candidate"))

    results: list[dict] = []
    baseline_total = None
    for name, source, tokenizer, kind in loaded:
        metrics = token_metrics(tokenizer, records)
        if kind == "baseline":
            baseline_total = metrics["total_tokens"]
            controls = None
            atomic = False
        else:
            try:
                controls = assert_control_tokens_atomic(tokenizer)
                atomic = True
            except ValueError as error:
                controls = {"error": str(error)}
                atomic = False
        item = {
            "name": name,
            "source": source,
            "kind": kind,
            "metrics": metrics,
            "control_tokens_atomic": atomic,
            "control_token_ids": controls,
            "embedding_estimate": embedding_estimate(
                int(getattr(config, "vocab_size")) if kind == "baseline" else metrics["vocab_size"],
                hidden_size,
                tied,
            ),
        }
        results.append(item)
    assert baseline_total is not None
    for item in results:
        ratio = item["metrics"]["total_tokens"] / baseline_total if baseline_total else 0.0
        item["compression_ratio"] = ratio
        item["token_reduction_percent"] = 100.0 * (1.0 - ratio)

    categorized: dict[str, list] = defaultdict(list)
    for record in records:
        if record.category:
            categorized[record.category].append(record)
    category_payload: dict[str, dict] = {}
    for category, category_records in categorized.items():
        category_payload[category] = {name: token_metrics(tokenizer, category_records) for name, _, tokenizer, _ in loaded}

    recommendation, reason = choose_balanced(results)
    payload = {
        "base_model": args.base_model,
        "corpus": {
            "eval_file": args.eval_file,
            "eval_lines": len(records),
            "eval_characters": sum(len(record.text) for record in records),
            "nfc_normalized": True,
        },
        "model": {
            "original_vocab": int(getattr(config, "vocab_size")),
            "hidden_size": hidden_size,
            "tie_word_embeddings": tied,
        },
        "tokenizers": results,
        "categories": category_payload,
        "recommendation": {"tokenizer": recommendation, "reason": reason},
        "byte_fallback_note": "Counts explicit <0xNN> pieces. HF byte-level BPE can encode bytes without exposing pieces in this form.",
    }
    write_json(args.output_json, payload)
    Path(args.output_markdown).write_text(build_markdown(payload), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
