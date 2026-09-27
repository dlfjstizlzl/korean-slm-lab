#!/usr/bin/env python3
"""Benchmark KO-32K-mixed10-structured against the established baselines.

Reuses the existing measurement code unchanged so the structured variant is
scored on exactly the same footing as the earlier experiment, and adds the
JSON / SQL / SkillLM-payload datasets the structured corpus targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from benchmark_real import benchmark_file
from benchmark_tokenizers import embedding_estimate, get_hidden_size
from benchmark_mixed import (
    AIHUB_DATASETS,
    add_reduction,
    classify_surface,
    inspect_examples,
    token_surface,
    vocab_composition,
)
from common import CONTROL_TOKENS, assert_control_tokens_atomic, load_tokenizer

DATASETS = {
    "fineweb_holdout": "data/processed/fineweb_general_holdout.txt",
    "aihub_dialogue_valid": "data/processed/aihub_dialogue_valid_clean.txt",
    "aihub_document_valid": "data/processed/aihub_documents_valid_clean.txt",
    "aihub_query_valid": "data/processed/aihub_queries_valid_clean.txt",
    "aihub_summary_valid": "data/processed/aihub_summaries_valid_clean.txt",
    "independent_eval_40": "data/korean_eval.txt",
    "skill_dsl": "data/eval_skill_dsl.txt",
    "code_url": "data/eval_code_url.txt",
    "mixed_validation": "data/processed/mixed/mixed_validation.txt",
    "json_validation": "data/eval_json.txt",
    "sql_validation": "data/eval_sql.txt",
    "skilllm_structured": "data/eval_skilllm_structured.txt",
}

EXAMPLES = [
    "가상메모리는 프로세스에게 물리 메모리와 독립된 주소 공간을 제공한다.",
    '{"skill":"search","input":"Apple Silicon unified memory"}',
    '{"tasks":[{"id":0,"skill":"search","input":"메모리"},{"id":1,"skill":"summary","input_from":0}]}',
    "SELECT id, name FROM users WHERE created_at >= '2026-09-01';",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name;",
    "https://github.com/ggerganov/llama.cpp",
    "python tokenizer/train_tokenizers.py --vocab-size 32000",
]

# Surface forms that dominate JSON and SQL; reported to show where the
# structured budget actually landed inside the vocabulary.
PROBE_STRINGS = [
    '":', '{"', '"}', '":"', '","', "[{", "}]", "[]", "{}",
    "SELECT", "FROM", "WHERE", "JOIN", "GROUP", "ORDER", "INSERT", "VALUES",
    "NULL", "true", "false", "COUNT(", "INTO", "CREATE", "TABLE",
]

ORDER = [
    "Qwen original",
    "Qwen + Skill",
    "KO-32K-current",
    "KO-32K-mixed10",
    "KO-32K-mixed10-structured",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the structured KO-32K variant.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--current-tokenizer", default="outputs/real/mf2/tokenizer-ko-32000")
    parser.add_argument("--mixed10-tokenizer", default="outputs/tokenizer-ko-32000-mixed10")
    parser.add_argument("--structured-tokenizer", default="outputs/tokenizer-ko-32000-mixed10-structured")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-json", default="outputs/tokenizer_structured_benchmark.json")
    parser.add_argument("--output-markdown", default="outputs/TOKENIZER_STRUCTURED_BENCHMARK.md")
    return parser.parse_args()


def probe(tokenizer) -> dict:
    return {
        text: {
            "tokens": [str(v) for v in tokenizer.convert_ids_to_tokens(tokenizer.encode(text, add_special_tokens=False))],
            "count": len(tokenizer.encode(text, add_special_tokens=False)),
        }
        for text in PROBE_STRINGS
    }


def build_tokenizers(args: argparse.Namespace) -> dict:
    base = AutoTokenizer.from_pretrained(args.base_model)
    skill = AutoTokenizer.from_pretrained(args.base_model)
    skill.add_special_tokens(
        {"additional_special_tokens": list(dict.fromkeys(
            [str(v) for v in skill.additional_special_tokens] + CONTROL_TOKENS))},
        replace_additional_special_tokens=True,
    )
    return {
        "Qwen original": base,
        "Qwen + Skill": skill,
        "KO-32K-current": load_tokenizer(args.current_tokenizer),
        "KO-32K-mixed10": load_tokenizer(args.mixed10_tokenizer),
        "KO-32K-mixed10-structured": load_tokenizer(args.structured_tokenizer),
    }


def aggregate(datasets: dict, keys) -> int:
    return sum(datasets[key]["total_tokens"] for key in keys)


def markdown(payload: dict) -> str:
    names = ORDER
    trained = ["KO-32K-current", "KO-32K-mixed10", "KO-32K-mixed10-structured"]
    out = [
        "# KO-32K structured-workload tokenizer benchmark",
        "",
        "> Positive reduction means fewer tokens than Qwen original. Negative means a regression.",
        "",
        "## Core comparison",
        "",
        "| Tokenizer | Korean (aggregate) | Skill DSL | Code/URL | JSON | SQL | SkillLM structured | Mixed validation | Vocab rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        s = payload["summary"][name]
        out.append(
            f"| {name} | {s['korean_aggregate_reduction_percent']:.2f}% | {s['skill_reduction_percent']:.2f}% | "
            f"{s['code_url_reduction_percent']:.2f}% | {s['json_reduction_percent']:.2f}% | "
            f"{s['sql_reduction_percent']:.2f}% | {s['skilllm_structured_reduction_percent']:.2f}% | "
            f"{s['mixed_validation_reduction_percent']:.2f}% | {s['embedding_vocab_rows']:,} |"
        )

    out += ["", "## Change versus KO-32K-mixed10", ""]
    out += [
        "| Metric | mixed10 | structured | Delta (pp) |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Korean aggregate reduction", "korean_aggregate_reduction_percent"),
        ("FineWeb reduction", "fineweb_reduction_percent"),
        ("AI Hub reduction", "aihub_reduction_percent"),
        ("Independent reduction", "independent_reduction_percent"),
        ("Skill DSL reduction", "skill_reduction_percent"),
        ("Code/URL reduction", "code_url_reduction_percent"),
        ("JSON reduction", "json_reduction_percent"),
        ("SQL reduction", "sql_reduction_percent"),
        ("SkillLM structured reduction", "skilllm_structured_reduction_percent"),
        ("Mixed validation reduction", "mixed_validation_reduction_percent"),
    ):
        a = payload["summary"]["KO-32K-mixed10"][key]
        b = payload["summary"]["KO-32K-mixed10-structured"][key]
        out.append(f"| {label} | {a:.2f}% | {b:.2f}% | {b - a:+.2f} |")

    out += ["", "## Vocabulary composition", ""]
    out += ["| Category | " + " | ".join(trained) + " |", "|---|" + "---:|" * len(trained)]
    for category in payload["tokenizers"]["KO-32K-current"]["vocab_composition"]["counts"]:
        values = [payload["tokenizers"][n]["vocab_composition"]["counts"][category] for n in trained]
        out.append(f"| {category} | " + " | ".join(f"{v:,}" for v in values) + " |")

    out += ["", "## Structured surface probes (token count per string)", ""]
    out += ["| String | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for text in PROBE_STRINGS:
        values = [payload["tokenizers"][n]["probes"][text]["count"] for n in names]
        out.append(f"| `{text}` | " + " | ".join(str(v) for v in values) + " |")

    out += ["", "## Dataset details", ""]
    for dataset, path in payload["datasets"].items():
        out += [f"### {dataset}", "", f"`{path}`", ""]
        out += [
            "| Tokenizer | Tokens | Reduction vs Qwen | Chars/token | Invalid UTF-8 fragments |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in names:
            item = payload["tokenizers"][name]["datasets"][dataset]
            out.append(
                f"| {name} | {item['total_tokens']:,} | {item['token_reduction_percent']:.2f}% | "
                f"{item['characters_per_token']:.3f} | {item['invalid_standalone_utf8_percent']:.3f}% |"
            )
        out.append("")

    out += ["## Tokenization examples", ""]
    for text in EXAMPLES:
        out += [f"### {text}", ""]
        for name in names:
            example = next(v for v in payload["tokenizers"][name]["examples"] if v["text"] == text)
            out.append(
                f"- `{name}` ({example['token_count']}): `{example['token_strings']}` / `{example['token_ids']}`"
            )
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(args.base_model)
    hidden = get_hidden_size(config)
    tied = bool(getattr(config, "tie_word_embeddings", True))
    tokenizers = build_tokenizers(args)

    payload: dict = {
        "base_model": args.base_model,
        "hidden_size": hidden,
        "tie_word_embeddings": tied,
        "datasets": DATASETS,
        "tokenizers": {},
    }

    for name, tokenizer in tokenizers.items():
        print(f"[{name}] measuring", flush=True)
        rows = {key: benchmark_file(tokenizer, path, args.batch_size) for key, path in DATASETS.items()}
        entry = {
            "tokenizer_size": len(tokenizer),
            "embedding_vocab_rows": len(tokenizer) if name.startswith("KO-") else 151_936,
            "datasets": rows,
            "examples": inspect_examples_for(tokenizer),
            "probes": probe(tokenizer),
        }
        entry["embedding_estimate"] = embedding_estimate(entry["embedding_vocab_rows"], hidden, tied)
        if name.startswith("KO-"):
            entry["vocab_composition"] = vocab_composition(tokenizer)
            entry["control_token_ids"] = assert_control_tokens_atomic(tokenizer)
        payload["tokenizers"][name] = entry

    baseline = payload["tokenizers"]["Qwen original"]["datasets"]
    korean_keys = ("fineweb_holdout",) + AIHUB_DATASETS + ("independent_eval_40",)
    for name, entry in payload["tokenizers"].items():
        for key, metrics in entry["datasets"].items():
            add_reduction(metrics, baseline[key]["total_tokens"])

    payload["summary"] = {}
    for name, entry in payload["tokenizers"].items():
        rows = entry["datasets"]
        payload["summary"][name] = {
            "fineweb_reduction_percent": rows["fineweb_holdout"]["token_reduction_percent"],
            "aihub_reduction_percent": 100.0 * (1.0 - aggregate(rows, AIHUB_DATASETS) / aggregate(baseline, AIHUB_DATASETS)),
            "independent_reduction_percent": rows["independent_eval_40"]["token_reduction_percent"],
            "korean_aggregate_reduction_percent": 100.0 * (1.0 - aggregate(rows, korean_keys) / aggregate(baseline, korean_keys)),
            "skill_reduction_percent": rows["skill_dsl"]["token_reduction_percent"],
            "code_url_reduction_percent": rows["code_url"]["token_reduction_percent"],
            "json_reduction_percent": rows["json_validation"]["token_reduction_percent"],
            "sql_reduction_percent": rows["sql_validation"]["token_reduction_percent"],
            "skilllm_structured_reduction_percent": rows["skilllm_structured"]["token_reduction_percent"],
            "mixed_validation_reduction_percent": rows["mixed_validation"]["token_reduction_percent"],
            "embedding_vocab_rows": entry["embedding_vocab_rows"],
        }

    ref = payload["summary"]["KO-32K-mixed10"]
    cur = payload["summary"]["KO-32K-mixed10-structured"]
    payload["recovery_vs_mixed10"] = {
        "korean_aggregate_penalty_points": ref["korean_aggregate_reduction_percent"] - cur["korean_aggregate_reduction_percent"],
        "fineweb_penalty_points": ref["fineweb_reduction_percent"] - cur["fineweb_reduction_percent"],
        "aihub_penalty_points": ref["aihub_reduction_percent"] - cur["aihub_reduction_percent"],
        "independent_penalty_points": ref["independent_reduction_percent"] - cur["independent_reduction_percent"],
        "skill_delta_points": cur["skill_reduction_percent"] - ref["skill_reduction_percent"],
        "code_url_delta_points": cur["code_url_reduction_percent"] - ref["code_url_reduction_percent"],
        "json_regression_mixed10_percent": -ref["json_reduction_percent"],
        "json_regression_structured_percent": -cur["json_reduction_percent"],
        "json_recovery_points": cur["json_reduction_percent"] - ref["json_reduction_percent"],
        "sql_regression_mixed10_percent": -ref["sql_reduction_percent"],
        "sql_regression_structured_percent": -cur["sql_reduction_percent"],
        "sql_recovery_points": cur["sql_reduction_percent"] - ref["sql_reduction_percent"],
        "skilllm_structured_recovery_points": cur["skilllm_structured_reduction_percent"] - ref["skilllm_structured_reduction_percent"],
    }

    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_markdown).write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["recovery_vs_mixed10"], indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


def inspect_examples_for(tokenizer) -> list[dict]:
    result = []
    for text in EXAMPLES:
        ids = [int(v) for v in tokenizer.encode(text, add_special_tokens=False)]
        result.append({
            "text": text,
            "token_strings": [str(v) for v in tokenizer.convert_ids_to_tokens(ids)],
            "token_ids": ids,
            "token_count": len(ids),
        })
    return result


if __name__ == "__main__":
    main()
