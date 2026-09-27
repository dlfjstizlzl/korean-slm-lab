#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from benchmark_real import benchmark_file
from benchmark_tokenizers import embedding_estimate, get_hidden_size
from common import BYTE_FALLBACK_RE, CONTROL_TOKENS, assert_control_tokens_atomic, load_tokenizer


COMMON_DATASETS = {
    "fineweb_holdout": "data/processed/fineweb_general_holdout.txt",
    "aihub_dialogue_valid": "data/processed/aihub_dialogue_valid_clean.txt",
    "aihub_document_valid": "data/processed/aihub_documents_valid_clean.txt",
    "aihub_query_valid": "data/processed/aihub_queries_valid_clean.txt",
    "aihub_summary_valid": "data/processed/aihub_summaries_valid_clean.txt",
    "independent_eval_40": "data/korean_eval.txt",
    "skill_dsl": "data/eval_skill_dsl.txt",
    "code_url": "data/eval_code_url.txt",
    "mixed_validation": "data/processed/mixed/mixed_validation.txt",
}

TRAINING_CORPORA = {
    "KO-32K-current": "data/processed/tokenizer_train_real.txt",
    "KO-32K-mixed10": "data/processed/tokenizer_train_mixed10.txt",
    "KO-32K-mixed15": "data/processed/tokenizer_train_mixed15.txt",
}

EXAMPLES = [
    "가상메모리는 프로세스에게 물리 메모리와 독립된 주소 공간을 제공한다.",
    "Qwen3 모델을 llama.cpp에서 Q4_K_M으로 실행한다.",
    '{"skill":"search","input":"Apple Silicon unified memory"}',
    "SELECT id, name FROM users WHERE created_at >= '2026-09-01';",
    "python tokenizer/train_tokenizers.py --vocab-size 32000",
    "models/Qwen3-0.6B-Q4_0.gguf",
    "https://github.com/ggerganov/llama.cpp",
    "std::shared_ptr<Model> model;",
]

AIHUB_DATASETS = (
    "aihub_dialogue_valid",
    "aihub_document_valid",
    "aihub_query_valid",
    "aihub_summary_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark current and mixed KO-32K tokenizers.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--current-tokenizer", default="outputs/real/mf2/tokenizer-ko-32000")
    parser.add_argument("--mixed10-tokenizer", default="outputs/tokenizer-ko-32000-mixed10")
    parser.add_argument("--mixed15-tokenizer", default="outputs/tokenizer-ko-32000-mixed15")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-json", default="outputs/tokenizer_mixed_benchmark.json")
    parser.add_argument("--output-markdown", default="outputs/TOKENIZER_MIXED_BENCHMARK.md")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


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


def token_surface(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([token_id], skip_special_tokens=False)


def classify_surface(surface: str, internal: str, is_special: bool) -> str:
    if is_special:
        return "special/control"
    if BYTE_FALLBACK_RE.fullmatch(internal) or "\ufffd" in surface or not surface:
        return "byte-like"
    stripped = surface.strip()
    hangul = sum("가" <= char <= "힣" for char in stripped)
    english = sum(char.isascii() and char.isalpha() for char in stripped)
    digits = sum(char.isdigit() for char in stripped)
    punctuation = sum(unicodedata.category(char).startswith(("P", "S")) for char in stripped)
    if hangul and english:
        return "mixed Korean-English"
    if hangul:
        return "Korean/Hangul dominant"
    if stripped and digits and digits >= max(1, len(stripped) - punctuation):
        return "numeric"
    code_markers = ("_", "::", "->", "//", "/*", "*/", "<", ">", "{", "}", "[", "]", "/", "\\")
    if any(marker in stripped for marker in code_markers) or (stripped and punctuation / len(stripped) >= 0.4):
        return "code-symbol dominant"
    return "ASCII/English"


def vocab_composition(tokenizer) -> dict:
    special_ids = {int(value) for value in tokenizer.added_tokens_decoder}
    counts = {
        "Korean/Hangul dominant": 0,
        "ASCII/English": 0,
        "code-symbol dominant": 0,
        "mixed Korean-English": 0,
        "numeric": 0,
        "byte-like": 0,
        "special/control": 0,
    }
    examples: dict[str, list[str]] = {name: [] for name in counts}
    for token_id in range(len(tokenizer)):
        internal = str(tokenizer.convert_ids_to_tokens(token_id))
        surface = token_surface(tokenizer, token_id)
        category = classify_surface(surface, internal, token_id in special_ids)
        counts[category] += 1
        if len(examples[category]) < 12:
            examples[category].append(surface if surface and "\ufffd" not in surface else internal)
    return {
        "method": "Rule-based heuristic over each token's standalone decoded surface; U+FFFD surfaces count as byte-like.",
        "counts": counts,
        "percent": {name: 100.0 * value / (len(tokenizer) or 1) for name, value in counts.items()},
        "examples": examples,
    }


def add_reduction(metrics: dict, baseline_tokens: int) -> None:
    ratio = metrics["total_tokens"] / (baseline_tokens or 1)
    metrics["compression_ratio"] = ratio
    metrics["token_reduction_percent"] = 100.0 * (1.0 - ratio)


def markdown(payload: dict) -> str:
    names = ["Qwen original", "KO-32K-current", "KO-32K-mixed10", "KO-32K-mixed15"]
    lines = [
        "# KO-32K mixed-workload tokenizer benchmark",
        "",
        "> Positive reduction means fewer tokens than Qwen. Negative reduction means a regression.",
        "",
        "## Core comparison",
        "",
        "| Tokenizer | FineWeb reduction | AI Hub total reduction | Independent reduction | Skill DSL reduction | Code/URL reduction | Vocab rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        summary = payload["summary"][name]
        lines.append(
            f"| {name} | {summary['fineweb_reduction_percent']:.2f}% | "
            f"{summary['aihub_reduction_percent']:.2f}% | {summary['independent_reduction_percent']:.2f}% | "
            f"{summary['skill_reduction_percent']:.2f}% | {summary['code_url_reduction_percent']:.2f}% | "
            f"{summary['embedding_vocab_rows']:,} |"
        )

    lines += ["", "## Korean penalty and code recovery", ""]
    lines += [
        "| Tokenizer | FineWeb penalty | AI Hub penalty | Independent penalty | Code regression | Recovery | Recovery ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("KO-32K-current", "KO-32K-mixed10", "KO-32K-mixed15"):
        item = payload["tradeoff"][name]
        lines.append(
            f"| {name} | {item['fineweb_penalty_points']:.2f} pp | {item['aihub_penalty_points']:.2f} pp | "
            f"{item['independent_penalty_points']:.2f} pp | {item['code_url_regression_percent']:.2f}% | "
            f"{item['code_recovery_points']:.2f} pp | {item['code_recovery_ratio_percent']:.2f}% |"
        )

    lines += ["", "## Vocabulary composition", ""]
    lines += ["| Category | Current | Mixed10 | Mixed15 |", "|---|---:|---:|---:|"]
    for category in payload["tokenizers"]["KO-32K-current"]["vocab_composition"]["counts"]:
        values = [payload["tokenizers"][name]["vocab_composition"]["counts"][category] for name in names[1:]]
        lines.append(f"| {category} | {values[0]:,} | {values[1]:,} | {values[2]:,} |")

    lines += ["", "## Dataset details", ""]
    for dataset, path in payload["datasets"].items():
        lines += [f"### {dataset}", "", f"`{path}`", ""]
        lines += ["| Tokenizer | Tokens | Reduction vs Qwen | Chars/token | Median | p95 | Invalid UTF-8 fragments |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name in names:
            item = payload["tokenizers"][name]["datasets"][dataset]
            lines.append(
                f"| {name} | {item['total_tokens']:,} | {item['token_reduction_percent']:.2f}% | "
                f"{item['characters_per_token']:.3f} | {item['median_tokens_per_document']} | "
                f"{item['p95_tokens_per_document']} | {item['invalid_standalone_utf8_percent']:.3f}% |"
            )
        lines.append("")

    lines += ["## Tokenization examples", ""]
    for text in EXAMPLES:
        lines += [f"### {text}", ""]
        for name in names:
            example = next(value for value in payload["tokenizers"][name]["examples"] if value["text"] == text)
            lines.append(f"- `{name}` ({example['token_count']}): `{example['token_strings']}` / `{example['token_ids']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = {
        "KO-32K-current": args.current_tokenizer,
        "KO-32K-mixed10": args.mixed10_tokenizer,
        "KO-32K-mixed15": args.mixed15_tokenizer,
    }
    for path in [*COMMON_DATASETS.values(), *TRAINING_CORPORA.values(), *paths.values()]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    base = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    augmented = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    augmented.add_special_tokens({"additional_special_tokens": CONTROL_TOKENS}, replace_additional_special_tokens=False)
    loaded = [("Qwen original", base, int(config.vocab_size)), ("Qwen + Skill", augmented, int(config.vocab_size))]
    for name, path in paths.items():
        tokenizer = load_tokenizer(path, args.trust_remote_code)
        assert_control_tokens_atomic(tokenizer)
        if len(tokenizer) != 32_000:
            raise ValueError(f"{name} must have exactly 32,000 tokens, got {len(tokenizer):,}")
        loaded.append((name, tokenizer, len(tokenizer)))

    hidden_size = get_hidden_size(config)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    payload = {
        "base_model": args.base_model,
        "datasets": COMMON_DATASETS,
        "training_corpora": TRAINING_CORPORA,
        "model": {"hidden_size": hidden_size, "original_vocab_rows": int(config.vocab_size), "tie_word_embeddings": tied},
        "tokenizers": {},
    }
    for name, tokenizer, vocab_rows in loaded:
        print(f"[Benchmark] {name}", flush=True)
        datasets = {}
        for dataset, path in COMMON_DATASETS.items():
            print(f"  - {dataset}", flush=True)
            datasets[dataset] = benchmark_file(tokenizer, path, args.batch_size)
        item = {
            "tokenizer_size": len(tokenizer),
            "embedding_vocab_rows": vocab_rows,
            "embedding_estimate": embedding_estimate(vocab_rows, hidden_size, tied),
            "control_token_ids": {token: tokenizer.encode(token, add_special_tokens=False) for token in CONTROL_TOKENS},
            "datasets": datasets,
            "examples": inspect_examples(tokenizer),
        }
        if name.startswith("KO-"):
            item["vocab_composition"] = vocab_composition(tokenizer)
        payload["tokenizers"][name] = item

    for dataset in COMMON_DATASETS:
        baseline = payload["tokenizers"]["Qwen original"]["datasets"][dataset]["total_tokens"]
        for item in payload["tokenizers"].values():
            add_reduction(item["datasets"][dataset], baseline)

    training = {}
    for name in ("KO-32K-current", "KO-32K-mixed10", "KO-32K-mixed15"):
        path = TRAINING_CORPORA[name]
        print(f"[Training benchmark] Qwen vs {name}", flush=True)
        qwen_metrics = benchmark_file(base, path, args.batch_size)
        candidate_tokenizer = next(tokenizer for candidate, tokenizer, _ in loaded if candidate == name)
        candidate_metrics = benchmark_file(candidate_tokenizer, path, args.batch_size)
        add_reduction(candidate_metrics, qwen_metrics["total_tokens"])
        training[name] = {"qwen": qwen_metrics, "candidate": candidate_metrics}
    payload["training"] = training

    summary = {}
    base_aihub = sum(payload["tokenizers"]["Qwen original"]["datasets"][name]["total_tokens"] for name in AIHUB_DATASETS)
    korean_datasets = ("fineweb_holdout", *AIHUB_DATASETS, "independent_eval_40")
    base_korean = sum(payload["tokenizers"]["Qwen original"]["datasets"][name]["total_tokens"] for name in korean_datasets)
    for name in ("Qwen original", "Qwen + Skill", "KO-32K-current", "KO-32K-mixed10", "KO-32K-mixed15"):
        item = payload["tokenizers"][name]
        aihub_tokens = sum(item["datasets"][dataset]["total_tokens"] for dataset in AIHUB_DATASETS)
        korean_tokens = sum(item["datasets"][dataset]["total_tokens"] for dataset in korean_datasets)
        summary[name] = {
            "embedding_vocab_rows": item["embedding_vocab_rows"],
            "fineweb_reduction_percent": item["datasets"]["fineweb_holdout"]["token_reduction_percent"],
            "aihub_tokens": aihub_tokens,
            "aihub_reduction_percent": 100.0 * (1.0 - aihub_tokens / (base_aihub or 1)),
            "independent_reduction_percent": item["datasets"]["independent_eval_40"]["token_reduction_percent"],
            "skill_reduction_percent": item["datasets"]["skill_dsl"]["token_reduction_percent"],
            "code_url_reduction_percent": item["datasets"]["code_url"]["token_reduction_percent"],
            "korean_aggregate_tokens": korean_tokens,
            "korean_aggregate_reduction_percent": 100.0 * (1.0 - korean_tokens / (base_korean or 1)),
        }
    payload["summary"] = summary

    current = summary["KO-32K-current"]
    current_regression = max(0.0, -current["code_url_reduction_percent"])
    tradeoff = {}
    for name in ("KO-32K-current", "KO-32K-mixed10", "KO-32K-mixed15"):
        item = summary[name]
        regression = max(0.0, -item["code_url_reduction_percent"])
        recovery = current_regression - regression
        tradeoff[name] = {
            "fineweb_penalty_points": current["fineweb_reduction_percent"] - item["fineweb_reduction_percent"],
            "aihub_penalty_points": current["aihub_reduction_percent"] - item["aihub_reduction_percent"],
            "independent_penalty_points": current["independent_reduction_percent"] - item["independent_reduction_percent"],
            "korean_aggregate_penalty_points": current["korean_aggregate_reduction_percent"] - item["korean_aggregate_reduction_percent"],
            "code_url_regression_percent": regression,
            "code_recovery_points": recovery,
            "code_recovery_ratio_percent": 100.0 * recovery / (current_regression or 1),
        }
    payload["tradeoff"] = tradeoff
    payload["pareto_axes"] = {
        name: {
            "x_korean_aggregate_reduction_percent": summary[name]["korean_aggregate_reduction_percent"],
            "y_code_url_regression_percent": tradeoff[name]["code_url_regression_percent"],
        }
        for name in tradeoff
    }
    payload["versions"] = {"transformers": __import__("transformers").__version__, "torch": torch.__version__}

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_markdown).write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"summary": summary, "tradeoff": tradeoff, "pareto_axes": payload["pareto_axes"]}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
