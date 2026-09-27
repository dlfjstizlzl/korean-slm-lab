#!/usr/bin/env python3
"""Benchmark the JSON-reinforced structured-v2 tokenizers.

Adds the new StarCoder JSON holdout as its own dataset, kept strictly separate
from the original JSON benchmark so the two are never collapsed into one number.
Everything else reuses the existing measurement code unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from benchmark_real import benchmark_file
from benchmark_tokenizers import embedding_estimate, get_hidden_size
from benchmark_mixed import AIHUB_DATASETS, add_reduction, vocab_composition
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
    "json_old": "data/eval_json.txt",
    "json_new": "data/eval_json_new.txt",
    "sql_validation": "data/eval_sql.txt",
    "skilllm_structured": "data/eval_skilllm_structured.txt",
}

EXAMPLES = [
    "가상메모리는 프로세스에게 물리 메모리와 독립된 주소 공간을 제공한다.",
    '{"skill":"search","input":"Apple Silicon unified memory"}',
    '{"tasks":[{"id":0,"skill":"search","input":"메모리"},{"id":1,"skill":"summary","input_from":0}]}',
    '{"model":"Qwen3-0.6B-Q4_0.gguf","backend":"llama","gpu_layers":"auto"}',
    '{"a":[1,2,3],"b":{"c":[{"d":null},{"e":[true,false]}]}}',
    '{"lat":37.5665,"lon":126.9780,"alt":38.5,"count":102938,"ratio":0.00123}',
    '{"제목":"가상메모리","설명":"프로세스에게 독립된 주소 공간을 제공한다.","태그":["메모리","OS"]}',
    "SELECT id, name FROM users WHERE created_at >= '2026-09-01';",
]

PROBE_STRINGS = [
    "{", "}", "[", "]", '":', '",', '"}', '":"', '","', "[{", "}]", "{}", "[]",
    "true", "false", "null", '"id"', '"name"', '"input"', '"output"',
    '"skill"', '"tasks"', '"model"', '"backend"',
    '{"', '"id":', '"name":', '":[', '":{', '"},{', '"]}',
]

ORDER = [
    "Qwen original",
    "KO-32K-current",
    "KO-32K-mixed10",
    "KO-32K-mixed10-structured",
    "KO-32K-mixed10-structured-v2-json2",
    "KO-32K-mixed10-structured-v2",
]
TRAINED = ORDER[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark structured-v2 JSON reinforcement.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-json", default="outputs/tokenizer_structured_v2_benchmark.json")
    parser.add_argument("--output-markdown", default="outputs/TOKENIZER_STRUCTURED_V2_BENCHMARK.md")
    return parser.parse_args()


PATHS = {
    "KO-32K-current": "outputs/real/mf2/tokenizer-ko-32000",
    "KO-32K-mixed10": "outputs/tokenizer-ko-32000-mixed10",
    "KO-32K-mixed10-structured": "outputs/tokenizer-ko-32000-mixed10-structured",
    "KO-32K-mixed10-structured-v2-json2": "outputs/tokenizer-ko-32000-mixed10-structured-v2-json2",
    "KO-32K-mixed10-structured-v2": "outputs/tokenizer-ko-32000-mixed10-structured-v2",
}

# JSON share of the total corpus, for the dose-response table.
JSON_SHARE = {
    "KO-32K-current": 0.0,
    "KO-32K-mixed10": 0.058,
    "KO-32K-mixed10-structured": 1.234,
    "KO-32K-mixed10-structured-v2-json2": 2.002,
    "KO-32K-mixed10-structured-v2": 3.003,
}


def aggregate(rows: dict, keys) -> int:
    return sum(rows[key]["total_tokens"] for key in keys)


def probes(tokenizer) -> dict:
    return {t: len(tokenizer.encode(t, add_special_tokens=False)) for t in PROBE_STRINGS}


def examples_for(tokenizer) -> list[dict]:
    out = []
    for text in EXAMPLES:
        ids = [int(v) for v in tokenizer.encode(text, add_special_tokens=False)]
        out.append({
            "text": text,
            "token_strings": [str(v) for v in tokenizer.convert_ids_to_tokens(ids)],
            "token_ids": ids,
            "token_count": len(ids),
        })
    return out


def markdown(payload: dict) -> str:
    names = ORDER
    out = [
        "# KO-32K structured-v2 (JSON reinforcement) benchmark",
        "",
        "> Positive reduction means fewer tokens than Qwen original. Negative means a regression.",
        "> `json_old` is the original JSON holdout, kept unchanged for comparability.",
        "> `json_new` is the freshly harvested StarCoder holdout, disjoint from every corpus here.",
        "",
        "## Core comparison",
        "",
        "| Tokenizer | JSON share | Korean | Skill DSL | Code/URL | JSON-old | JSON-new | SQL | SkillLM payload | Vocab |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        s = payload["summary"][name]
        share = f"{JSON_SHARE[name]:.2f}%" if name in JSON_SHARE else "—"
        out.append(
            f"| {name} | {share} | {s['korean_aggregate_reduction_percent']:.2f}% | "
            f"{s['skill_reduction_percent']:.2f}% | {s['code_url_reduction_percent']:.2f}% | "
            f"{s['json_old_reduction_percent']:.2f}% | {s['json_new_reduction_percent']:.2f}% | "
            f"{s['sql_reduction_percent']:.2f}% | {s['skilllm_reduction_percent']:.2f}% | "
            f"{s['embedding_vocab_rows']:,} |"
        )

    out += ["", "## Change versus KO-32K-mixed10-structured (the v1 baseline)", ""]
    out += ["| Metric | v1 | v2-json2 (2%) | v2 (3%) |", "|---|---:|---:|---:|"]
    base = payload["summary"]["KO-32K-mixed10-structured"]
    for label, key in (
        ("Korean aggregate", "korean_aggregate_reduction_percent"),
        ("FineWeb", "fineweb_reduction_percent"),
        ("AI Hub", "aihub_reduction_percent"),
        ("Independent Korean", "independent_reduction_percent"),
        ("Skill DSL", "skill_reduction_percent"),
        ("Code/URL", "code_url_reduction_percent"),
        ("JSON-old", "json_old_reduction_percent"),
        ("JSON-new", "json_new_reduction_percent"),
        ("SQL", "sql_reduction_percent"),
        ("SkillLM payload", "skilllm_reduction_percent"),
    ):
        a = payload["summary"]["KO-32K-mixed10-structured-v2-json2"][key]
        b = payload["summary"]["KO-32K-mixed10-structured-v2"][key]
        out.append(
            f"| {label} | {base[key]:.2f}% | {a:.2f}% ({a - base[key]:+.2f}) | {b:.2f}% ({b - base[key]:+.2f}) |"
        )

    out += ["", "## JSON dose response", "",
            "| JSON share of corpus | JSON-old regression | JSON-new regression | SkillLM payload | Korean aggregate |",
            "|---:|---:|---:|---:|---:|"]
    for name in ("KO-32K-mixed10", "KO-32K-mixed10-structured",
                 "KO-32K-mixed10-structured-v2-json2", "KO-32K-mixed10-structured-v2"):
        s = payload["summary"][name]
        out.append(
            f"| {JSON_SHARE[name]:.2f}% | {-s['json_old_reduction_percent']:.2f}% | "
            f"{-s['json_new_reduction_percent']:.2f}% | {-s['skilllm_reduction_percent']:.2f}% | "
            f"{s['korean_aggregate_reduction_percent']:.2f}% |"
        )

    out += ["", "## Vocabulary composition", ""]
    out += ["| Category | " + " | ".join(TRAINED) + " |", "|---|" + "---:|" * len(TRAINED)]
    for category in payload["tokenizers"]["KO-32K-current"]["vocab_composition"]["counts"]:
        values = [payload["tokenizers"][n]["vocab_composition"]["counts"][category] for n in TRAINED]
        out.append(f"| {category} | " + " | ".join(f"{v:,}" for v in values) + " |")

    out += ["", "## JSON surface probes (tokens per string)", ""]
    out += ["| String | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for text in PROBE_STRINGS:
        values = [payload["tokenizers"][n]["probes"][text] for n in names]
        out.append(f"| `{text}` | " + " | ".join(str(v) for v in values) + " |")

    out += ["", "## Dataset details", ""]
    for dataset, path in payload["datasets"].items():
        out += [f"### {dataset}", "", f"`{path}`", ""]
        out += ["| Tokenizer | Tokens | Reduction vs Qwen | Chars/token | Invalid UTF-8 fragments |",
                "|---|---:|---:|---:|---:|"]
        for name in names:
            item = payload["tokenizers"][name]["datasets"][dataset]
            out.append(
                f"| {name} | {item['total_tokens']:,} | {item['token_reduction_percent']:.2f}% | "
                f"{item['characters_per_token']:.3f} | {item['invalid_standalone_utf8_percent']:.3f}% |"
            )
        out.append("")

    out += ["## Representative tokenization", ""]
    for text in EXAMPLES:
        out += [f"### {text}", ""]
        for name in names:
            e = next(v for v in payload["tokenizers"][name]["examples"] if v["text"] == text)
            out.append(f"- `{name}` ({e['token_count']}): `{e['token_strings']}` / `{e['token_ids']}`")
        out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(args.base_model)
    hidden = get_hidden_size(config)
    tied = bool(getattr(config, "tie_word_embeddings", True))

    tokenizers = {"Qwen original": AutoTokenizer.from_pretrained(args.base_model)}
    for name, path in PATHS.items():
        tokenizers[name] = load_tokenizer(path)

    payload: dict = {"base_model": args.base_model, "hidden_size": hidden,
                     "tie_word_embeddings": tied, "datasets": DATASETS,
                     "json_share_percent": JSON_SHARE, "tokenizers": {}}

    for name in ORDER:
        tokenizer = tokenizers[name]
        print(f"[{name}] measuring", flush=True)
        rows = {key: benchmark_file(tokenizer, path, args.batch_size) for key, path in DATASETS.items()}
        entry = {
            "tokenizer_size": len(tokenizer),
            "embedding_vocab_rows": len(tokenizer) if name.startswith("KO-") else 151_936,
            "datasets": rows,
            "examples": examples_for(tokenizer),
            "probes": probes(tokenizer),
        }
        entry["embedding_estimate"] = embedding_estimate(entry["embedding_vocab_rows"], hidden, tied)
        if name.startswith("KO-"):
            entry["vocab_composition"] = vocab_composition(tokenizer)
            entry["control_token_ids"] = assert_control_tokens_atomic(tokenizer)
        payload["tokenizers"][name] = entry

    baseline = payload["tokenizers"]["Qwen original"]["datasets"]
    korean_keys = ("fineweb_holdout",) + AIHUB_DATASETS + ("independent_eval_40",)
    for entry in payload["tokenizers"].values():
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
            "json_old_reduction_percent": rows["json_old"]["token_reduction_percent"],
            "json_new_reduction_percent": rows["json_new"]["token_reduction_percent"],
            "sql_reduction_percent": rows["sql_validation"]["token_reduction_percent"],
            "skilllm_reduction_percent": rows["skilllm_structured"]["token_reduction_percent"],
            "mixed_validation_reduction_percent": rows["mixed_validation"]["token_reduction_percent"],
            "embedding_vocab_rows": entry["embedding_vocab_rows"],
        }

    v1 = payload["summary"]["KO-32K-mixed10-structured"]
    payload["recovery_vs_structured_v1"] = {}
    for name in ("KO-32K-mixed10-structured-v2-json2", "KO-32K-mixed10-structured-v2"):
        cur = payload["summary"][name]
        payload["recovery_vs_structured_v1"][name] = {
            "korean_aggregate_penalty_points": v1["korean_aggregate_reduction_percent"] - cur["korean_aggregate_reduction_percent"],
            "json_old_recovery_points": cur["json_old_reduction_percent"] - v1["json_old_reduction_percent"],
            "json_new_reduction_percent": cur["json_new_reduction_percent"],
            "skilllm_recovery_points": cur["skilllm_reduction_percent"] - v1["skilllm_reduction_percent"],
            "sql_penalty_points": v1["sql_reduction_percent"] - cur["sql_reduction_percent"],
            "code_url_penalty_points": v1["code_url_reduction_percent"] - cur["code_url_reduction_percent"],
            "skill_dsl_delta_points": cur["skill_reduction_percent"] - v1["skill_reduction_percent"],
        }

    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_markdown).write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["recovery_vs_structured_v1"], indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
