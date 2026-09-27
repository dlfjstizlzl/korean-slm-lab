#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from common import CONTROL_TOKENS, CorpusRecord, load_tokenizer, normalize_text, read_corpus, token_metrics, tokenizer_name, write_json

INSPECTION_TEXTS = [
    "가상메모리는 프로세스에게 물리 메모리와 독립된 주소 공간을 제공한다.",
    "운영체제는 페이지 폴트가 발생하면 필요한 페이지를 물리 메모리에 적재한다.",
    "Apple Silicon은 CPU와 GPU가 통합 메모리를 공유한다.",
    "소형 언어모델은 제한된 자원에서 효율적인 추론을 수행할 수 있다.",
    "<검색>Apple Silicon 통합 메모리",
    "<요약><이전0>",
    "<SQL>지난달 주문량을 조회한다.",
    "https://example.com/test?q=memory",
    "SELECT name FROM users WHERE id = 123;",
]

WORD_RE = re.compile(r"[가-힣]+|[A-Za-z_][A-Za-z0-9_]*|\d+")
URL_RE = re.compile(r"https?://", re.IGNORECASE)
CODE_RE = re.compile(r"\b(?:SELECT|FROM|WHERE|ORDER BY|def|return|class|import)\b|[{};]|\w+\s*=", re.IGNORECASE)
TECHNICAL_TERMS = {
    "메모리", "프로세스", "운영체제", "페이지", "토크나이저", "언어모델", "임베딩", "데이터베이스",
    "인덱스", "알고리즘", "체크포인트", "로그잇", "GPU", "CPU", "분산", "바이트", "모델", "tokenizer",
}
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit tokenizer training reliability and separate code, data, and approach effects.")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-file", default="data/korean_train.txt")
    parser.add_argument("--eval-file", default="data/korean_eval.txt")
    parser.add_argument(
        "--hf-tokenizers", nargs="+",
        default=["outputs/fixed/tokenizer-ko-16000", "outputs/fixed/tokenizer-ko-24000", "outputs/fixed/tokenizer-ko-32000"],
    )
    parser.add_argument(
        "--sentencepiece-tokenizers",
        nargs="*",
        default=None,
        help="SentencePiece directories; when omitted, existing outputs/algorithms 8K/16K Unigram/BPE artifacts are discovered",
    )
    parser.add_argument("--diagnostic-benchmark", default="outputs/diagnostic/benchmark_eval.json")
    parser.add_argument("--diagnostic-train-benchmark", default="outputs/diagnostic/benchmark_train.json")
    parser.add_argument("--min-frequency-train-benchmark", default="outputs/minfreq2/benchmark_train.json")
    parser.add_argument("--min-frequency-eval-benchmark", default="outputs/minfreq2/benchmark_eval.json")
    parser.add_argument("--output-json", default="outputs/diagnostics.json")
    parser.add_argument("--output-markdown", default="outputs/DIAGNOSTICS.md")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def raw_texts(path: str | Path) -> list[str]:
    values = []
    for raw in Path(path).open(encoding="utf-8"):
        line = raw.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line and line.split("\t", 1)[0] in {"general", "technical", "skill_dsl", "mixed_ascii"}:
            line = line.split("\t", 1)[1]
        values.append(line)
    return values


def unicode_audit(path: str | Path) -> dict:
    texts = raw_texts(path)
    changed = []
    jamo = []
    combining = []
    compatibility = Counter()
    zero_width = Counter()
    nbsp = 0
    controls = Counter()
    for text in texts:
        if text != unicodedata.normalize("NFC", text):
            changed.append(text)
        if any(0x1100 <= ord(char) <= 0x11FF or 0xA960 <= ord(char) <= 0xA97F or 0xD7B0 <= ord(char) <= 0xD7FF for char in text):
            jamo.append(text)
        if any(unicodedata.combining(char) for char in text):
            combining.append(text)
        for char in text:
            if unicodedata.normalize("NFKC", char) != char:
                compatibility[char] += 1
            if char in ZERO_WIDTH:
                zero_width[char] += 1
            if char == "\u00a0":
                nbsp += 1
            if unicodedata.category(char) == "Cc":
                controls[char] += 1
    def named(counter: Counter) -> list[dict]:
        return [
            {"character": char, "codepoint": f"U+{ord(char):04X}", "name": unicodedata.name(char, "UNKNOWN"), "count": count}
            for char, count in counter.most_common(20)
        ]
    return {
        "lines": len(texts),
        "nfc_changed_lines": len(changed),
        "nfc_changed_samples": changed[:5],
        "hangul_jamo_sequence_lines": len(jamo),
        "hangul_jamo_samples": jamo[:5],
        "combining_mark_lines": len(combining),
        "compatibility_characters": named(compatibility),
        "zero_width_characters": named(zero_width),
        "non_breaking_spaces": nbsp,
        "control_characters": named(controls),
    }


def diagnostic_category(record: CorpusRecord) -> str:
    if record.category == "general":
        return "general_korean"
    if record.category == "technical":
        return "technical_korean"
    if record.category == "skill_dsl" or any(token in record.text for token in CONTROL_TOKENS):
        return "skill_dsl"
    if URL_RE.search(record.text):
        return "url"
    if CODE_RE.search(record.text):
        return "code"
    return "mixed_ascii"


def corpus_stats(records: list[CorpusRecord]) -> dict:
    texts = [record.text for record in records]
    all_text = "".join(texts)
    non_space = [char for char in all_text if not char.isspace()]
    denominator = len(non_space) or 1
    word_counts = Counter(match.group(0).lower() for text in texts for match in WORD_RE.finditer(text))
    char_counts = Counter(all_text)
    hangul_counts = Counter(char for char in all_text if "가" <= char <= "힣")
    categories = Counter(diagnostic_category(record) for record in records)
    sentence_lengths = [len(text) for text in texts]
    tech_lines = sum(any(term.lower() in text.lower() for term in TECHNICAL_TERMS) for text in texts)
    return {
        "total_lines": len(texts),
        "total_characters": len(all_text),
        "unique_lines": len(set(texts)),
        "unique_unicode_characters": len(char_counts),
        "unique_hangul_syllables": len(hangul_counts),
        "hangul_syllable_characters": sum(hangul_counts.values()),
        "hangul_ratio_percent": 100.0 * sum("가" <= char <= "힣" for char in non_space) / denominator,
        "ascii_ratio_percent": 100.0 * sum(ord(char) < 128 for char in non_space) / denominator,
        "english_alphabet_ratio_percent": 100.0 * sum(char.isascii() and char.isalpha() for char in non_space) / denominator,
        "digit_ratio_percent": 100.0 * sum(char.isdigit() for char in non_space) / denominator,
        "punctuation_ratio_percent": 100.0 * sum(unicodedata.category(char).startswith("P") for char in non_space) / denominator,
        "unique_word_like_units": len(word_counts),
        "total_word_like_units": sum(word_counts.values()),
        "hapax_word_like_units": sum(value == 1 for value in word_counts.values()),
        "hapax_type_ratio_percent": 100.0 * sum(value == 1 for value in word_counts.values()) / (len(word_counts) or 1),
        "top_100_terms": [{"term": term, "count": count} for term, count in word_counts.most_common(100)],
        "top_100_hangul_syllables": [{"syllable": term, "count": count} for term, count in hangul_counts.most_common(100)],
        "average_sentence_characters": statistics.mean(sentence_lengths) if sentence_lengths else 0.0,
        "median_sentence_characters": statistics.median(sentence_lengths) if sentence_lengths else 0.0,
        "technical_term_line_ratio_percent": 100.0 * tech_lines / (len(texts) or 1),
        "code_or_url_line_ratio_percent": 100.0 * sum(diagnostic_category(record) in {"code", "url"} for record in records) / (len(records) or 1),
        "skill_dsl_line_ratio_percent": 100.0 * sum(diagnostic_category(record) == "skill_dsl" for record in records) / (len(records) or 1),
        "categories": dict(categories),
    }


def distribution_gap(train_records: list[CorpusRecord], eval_records: list[CorpusRecord]) -> dict:
    train_text = "".join(record.text for record in train_records)
    eval_text = "".join(record.text for record in eval_records)
    train_chars = set(train_text)
    eval_chars = set(eval_text)
    train_words = {match.group(0).lower() for record in train_records for match in WORD_RE.finditer(record.text)}
    eval_words = [match.group(0).lower() for record in eval_records for match in WORD_RE.finditer(record.text)]
    unseen_chars = sorted(eval_chars - train_chars)
    unseen_words = sorted(set(eval_words) - train_words)
    normalized_train = {normalize_text(record.text) for record in train_records}
    normalized_eval = {normalize_text(record.text) for record in eval_records}
    return {
        "exact_duplicate_lines": len(normalized_train & normalized_eval),
        "eval_unique_characters": len(eval_chars),
        "eval_unseen_characters": len(unseen_chars),
        "eval_unseen_character_ratio_percent": 100.0 * len(unseen_chars) / (len(eval_chars) or 1),
        "eval_unseen_character_samples": [
            {"character": char, "codepoint": f"U+{ord(char):04X}", "name": unicodedata.name(char, "UNKNOWN")}
            for char in unseen_chars[:100]
        ],
        "eval_word_like_tokens": len(eval_words),
        "eval_unseen_word_like_types": len(unseen_words),
        "eval_unseen_word_like_type_ratio_percent": 100.0 * len(unseen_words) / (len(set(eval_words)) or 1),
        "eval_unseen_word_like_samples": unseen_words[:100],
    }


def component_dict(component) -> dict | None:
    if component is None:
        return None
    try:
        return json.loads(component.__getstate__())
    except Exception:
        return {"repr": repr(component)}


def pipeline_info(tokenizer) -> dict:
    backend = tokenizer.backend_tokenizer
    trainer = backend.model.get_trainer()
    added = []
    for token_id, token in tokenizer.added_tokens_decoder.items():
        added.append({"id": int(token_id), "token": str(token), "special": bool(token.special), "normalized": bool(token.normalized)})
    return {
        "python_class": type(tokenizer).__name__,
        "model_type": type(backend.model).__name__,
        "model_repr_prefix": repr(backend.model)[:500],
        "trainer_type": type(trainer).__name__,
        "trainer_repr": repr(trainer),
        "normalizer": component_dict(backend.normalizer),
        "pre_tokenizer": component_dict(backend.pre_tokenizer),
        "decoder": component_dict(backend.decoder),
        "post_processor": component_dict(backend.post_processor),
        "byte_level_behavior": type(backend.pre_tokenizer).__name__ == "ByteLevel" or "ByteLevel" in repr(backend.pre_tokenizer),
        "special_tokens": list(tokenizer.all_special_tokens),
        "added_tokens": added,
    }


def compare_pipeline(base: dict, candidate: dict) -> dict:
    fields = ["model_type", "normalizer", "pre_tokenizer", "decoder", "post_processor", "byte_level_behavior"]
    comparison = {field: base[field] == candidate[field] for field in fields}
    comparison["all_preprocessing_components_match"] = all(comparison[field] for field in fields[1:])
    return comparison


def inspect_tokenization(tokenizer, texts: list[str]) -> list[dict]:
    output = []
    for text in texts:
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        tokens = tokenizer.convert_ids_to_tokens(ids)
        if isinstance(tokens, str):
            tokens = [tokens]
        invalid_surfaces = 0
        single_hangul = 0
        if hasattr(tokenizer, "decode"):
            for token_id in ids:
                try:
                    surface = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                except TypeError:
                    surface = tokenizer.decode([token_id], skip_special_tokens=False)
                invalid_surfaces += "\ufffd" in surface
                single_hangul += bool(re.fullmatch(r"[가-힣]", surface))
        output.append({
            "text": text,
            "token_strings": [str(token) for token in tokens],
            "token_ids": [int(value) for value in ids],
            "token_count": len(ids),
            "individually_invalid_utf8_surfaces": invalid_surfaces,
            "single_hangul_surface_tokens": single_hangul,
        })
    return output


def surface_fragment_metrics(tokenizer, records: list[CorpusRecord]) -> dict:
    if not hasattr(tokenizer, "decode"):
        return {"available": False}
    cache: dict[int, tuple[bool, bool]] = {}
    total = invalid = single_hangul = 0
    for record in records:
        for token_id in tokenizer.encode(record.text, add_special_tokens=False):
            token_id = int(token_id)
            if token_id not in cache:
                try:
                    surface = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                except TypeError:
                    surface = tokenizer.decode([token_id], skip_special_tokens=False)
                cache[token_id] = ("\ufffd" in surface, bool(re.fullmatch(r"[가-힣]", surface)))
            is_invalid, is_single_hangul = cache[token_id]
            total += 1
            invalid += is_invalid
            single_hangul += is_single_hangul
    return {
        "available": True,
        "total_token_occurrences": total,
        "individually_invalid_utf8_surface_occurrences": invalid,
        "individually_invalid_utf8_surface_percent": 100.0 * invalid / (total or 1),
        "single_hangul_surface_occurrences": single_hangul,
        "single_hangul_surface_percent": 100.0 * single_hangul / (total or 1),
        "unique_token_ids_observed": len(cache),
    }


def category_benchmark(tokenizers: list[tuple[str, object]], records: list[CorpusRecord]) -> dict:
    grouped: dict[str, list[CorpusRecord]] = defaultdict(list)
    for record in records:
        grouped[diagnostic_category(record)].append(record)
    result = {}
    for category, subset in sorted(grouped.items()):
        entries = {}
        baseline_total = None
        for name, tokenizer in tokenizers:
            metrics = token_metrics(tokenizer, subset)
            if baseline_total is None:
                baseline_total = metrics["total_tokens"]
            metrics["change_vs_qwen_percent"] = 100.0 * (metrics["total_tokens"] / baseline_total - 1.0) if baseline_total else 0.0
            entries[name] = metrics
        result[category] = entries
    return result


def skill_ablation(base_model: str, skill_records: list[CorpusRecord], ko_tokenizer, trust_remote_code: bool) -> dict:
    original = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    augmented = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    augmented.add_special_tokens({"additional_special_tokens": CONTROL_TOKENS}, replace_additional_special_tokens=False)
    variants = [("Qwen original", original), ("Qwen + control tokens", augmented), ("KO + control tokens", ko_tokenizer)]
    result = {}
    baseline = None
    for name, tokenizer in variants:
        metrics = token_metrics(tokenizer, skill_records)
        if baseline is None:
            baseline = metrics["total_tokens"]
        result[name] = {
            **metrics,
            "change_vs_qwen_percent": 100.0 * (metrics["total_tokens"] / baseline - 1.0) if baseline else 0.0,
            "control_token_ids": {token: tokenizer.encode(token, add_special_tokens=False) for token in CONTROL_TOKENS},
        }
    return result


def sentencepiece_proto(path: str | Path) -> dict:
    from sentencepiece import sentencepiece_model_pb2

    path = Path(path)
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString((path / "tokenizer.model").read_bytes())
    trainer = proto.trainer_spec
    normalizer = proto.normalizer_spec
    metadata_path = path / "sentencepiece_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return {
        "path": str(path),
        "actual_vocab_size": len(proto.pieces),
        "model_type": trainer.ModelType.Name(trainer.model_type),
        "requested_vocab_size": metadata.get("requested_vocab_size"),
        "model_proto_vocab_size": trainer.vocab_size,
        "hard_vocab_limit": trainer.hard_vocab_limit,
        "input_sentence_size": trainer.input_sentence_size,
        "shuffle_input_sentence": trainer.shuffle_input_sentence,
        "character_coverage": trainer.character_coverage,
        "byte_fallback": trainer.byte_fallback,
        "normalization_rule_name": normalizer.name,
        "add_dummy_prefix": normalizer.add_dummy_prefix,
        "split_by_whitespace": trainer.split_by_whitespace,
        "split_by_unicode_script": trainer.split_by_unicode_script,
        "split_digits": trainer.split_digits,
        "user_defined_symbols": list(trainer.user_defined_symbols),
    }


def benchmark_table(tokenizers: list[tuple[str, object]], train_records: list[CorpusRecord], eval_records: list[CorpusRecord]) -> list[dict]:
    rows = []
    train_baseline = eval_baseline = None
    for name, tokenizer in tokenizers:
        train = token_metrics(tokenizer, train_records)
        evaluation = token_metrics(tokenizer, eval_records)
        if train_baseline is None:
            train_baseline = train["total_tokens"]
            eval_baseline = evaluation["total_tokens"]
        rows.append({
            "tokenizer": name,
            "train": train,
            "train_change_vs_qwen_percent": 100.0 * (train["total_tokens"] / train_baseline - 1.0),
            "eval": evaluation,
            "eval_change_vs_qwen_percent": 100.0 * (evaluation["total_tokens"] / eval_baseline - 1.0),
        })
    return rows


def classify_case(rows: list[dict]) -> dict:
    candidates = [row for row in rows[1:] if row["tokenizer"].startswith("tokenizer-ko")]
    if candidates and all(row["train_change_vs_qwen_percent"] < 0 for row in candidates) and all(row["eval_change_vs_qwen_percent"] > 0 for row in candidates):
        return {
            "case": "A",
            "interpretation": "The retrained tokenizers strongly compress their training corpus but regress on held-out evaluation: pipeline learning works, while corpus diversity and overfitting are the primary concern.",
        }
    if candidates and all(row["train_change_vs_qwen_percent"] > 0 for row in candidates):
        return {"case": "B", "interpretation": "The retrained tokenizers are worse even on their own training corpus; inspect implementation and configuration first."}
    return {"case": "C_OR_MIXED", "interpretation": "Results are mixed; category and distribution evidence is required."}


def build_markdown(payload: dict) -> str:
    lines = [
        "# Tokenizer reliability diagnostics",
        "",
        f"**Case {payload['case_classification']['case']}** — {payload['case_classification']['interpretation']}",
        "",
        "## Train versus external evaluation",
        "",
        "| Tokenizer | Train tokens | Train Δ vs Qwen | Eval tokens | Eval Δ vs Qwen |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["benchmarks"]:
        lines.append(
            f"| {row['tokenizer']} | {row['train']['total_tokens']:,} | {row['train_change_vs_qwen_percent']:+.2f}% | "
            f"{row['eval']['total_tokens']:,} | {row['eval_change_vs_qwen_percent']:+.2f}% |"
        )
    lines += ["", "## Category benchmark", ""]
    for category, entries in payload["category_benchmark"].items():
        lines.append(f"### {category}")
        lines.append("")
        for name, metrics in entries.items():
            lines.append(f"- `{name}`: {metrics['total_tokens']:,} ({metrics['change_vs_qwen_percent']:+.2f}%)")
        lines.append("")
    lines += ["## Skill-token ablation", ""]
    for name, metrics in payload["skill_token_ablation"].items():
        lines.append(f"- `{name}`: {metrics['total_tokens']:,} ({metrics['change_vs_qwen_percent']:+.2f}%)")
    lines += ["", "## Unicode audit", ""]
    for split in ("train", "eval"):
        audit = payload["unicode_audit"][split]
        lines.append(
            f"- {split}: NFC-changed {audit['nfc_changed_lines']}, Jamo-sequence {audit['hangul_jamo_sequence_lines']}, "
            f"compatibility chars {sum(item['count'] for item in audit['compatibility_characters'])}, "
            f"zero-width {sum(item['count'] for item in audit['zero_width_characters'])}, NBSP {audit['non_breaking_spaces']}"
        )
    lines += ["", "## Pipeline", ""]
    for name, comparison in payload["pipeline_comparison"].items():
        lines.append(f"- `{name}` preprocessing pipeline matches Qwen: `{comparison['all_preprocessing_components_match']}`")
    lines += ["", "## Representative tokenization", ""]
    for text in INSPECTION_TEXTS:
        lines.append(f"### {text}")
        lines.append("")
        for name, entries in payload["tokenization_inspection"].items():
            entry = next(item for item in entries if item["text"] == text)
            lines.append(f"- `{name}` ({entry['token_count']}): `{entry['token_strings']}`")
        lines.append("")
    lines += [
        "## Scope warning",
        "",
        "The deterministic diagnostic split measures only generalization within this synthetic/curated corpus. It is not evidence of real Korean generalization.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    train_records = read_corpus(args.train_file, deduplicate=True)
    eval_records = read_corpus(args.eval_file, deduplicate=True)
    base = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    hf = [(tokenizer_name(path), load_tokenizer(path, args.trust_remote_code)) for path in args.hf_tokenizers]
    default_sp_paths = [
        Path("outputs/algorithms/tokenizer-sp-unigram-ko-8000"),
        Path("outputs/algorithms/tokenizer-sp-unigram-ko-16000"),
        Path("outputs/algorithms/tokenizer-sp-bpe-ko-8000"),
        Path("outputs/algorithms/tokenizer-sp-bpe-ko-16000"),
    ]
    sp_paths = (
        [Path(path) for path in args.sentencepiece_tokenizers]
        if args.sentencepiece_tokenizers is not None
        else [path for path in default_sp_paths if path.is_dir()]
    )
    sp = [(tokenizer_name(str(path)), load_tokenizer(path, args.trust_remote_code)) for path in sp_paths]
    tokenizers = [("Qwen original", base), *hf, *sp]

    pipelines = {"Qwen original": pipeline_info(base)}
    comparisons = {}
    for name, tokenizer in hf:
        pipelines[name] = pipeline_info(tokenizer)
        comparisons[name] = compare_pipeline(pipelines["Qwen original"], pipelines[name])

    benchmarks = benchmark_table(tokenizers, train_records, eval_records)
    skill_records = [record for record in eval_records if diagnostic_category(record) == "skill_dsl"]
    def optional_json(path_value: str) -> dict | None:
        path = Path(path_value)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    payload = {
        "base_model": args.base_model,
        "benchmarks": benchmarks,
        "case_classification": classify_case(benchmarks),
        "unicode_audit": {"train": unicode_audit(args.train_file), "eval": unicode_audit(args.eval_file)},
        "corpus_stats": {"train": corpus_stats(train_records), "eval": corpus_stats(eval_records)},
        "distribution_gap": distribution_gap(train_records, eval_records),
        "category_benchmark": category_benchmark(tokenizers[: 1 + len(hf)], eval_records),
        "pipeline_introspection": pipelines,
        "pipeline_comparison": comparisons,
        "tokenization_inspection": {name: inspect_tokenization(tokenizer, INSPECTION_TEXTS) for name, tokenizer in tokenizers},
        "surface_fragment_metrics": {
            name: {
                "train": surface_fragment_metrics(tokenizer, train_records),
                "eval": surface_fragment_metrics(tokenizer, eval_records),
            }
            for name, tokenizer in tokenizers
        },
        "skill_token_ablation": skill_ablation(args.base_model, skill_records, hf[0][1], args.trust_remote_code),
        "sentencepiece_model_proto": [sentencepiece_proto(path) for path in sp_paths],
        "diagnostic_split_benchmark": {
            "train": optional_json(args.diagnostic_train_benchmark),
            "eval": optional_json(args.diagnostic_benchmark),
        },
        "min_frequency_2_benchmark": {
            "train": optional_json(args.min_frequency_train_benchmark),
            "eval": optional_json(args.min_frequency_eval_benchmark),
        },
    }
    write_json(args.output_json, payload)
    Path(args.output_markdown).write_text(build_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "case": payload["case_classification"],
        "benchmark_rows": [
            {
                "tokenizer": row["tokenizer"],
                "train_tokens": row["train"]["total_tokens"],
                "train_delta_percent": row["train_change_vs_qwen_percent"],
                "eval_tokens": row["eval"]["total_tokens"],
                "eval_delta_percent": row["eval_change_vs_qwen_percent"],
            }
            for row in benchmarks
        ],
        "distribution_gap": payload["distribution_gap"],
        "skill_token_ablation": payload["skill_token_ablation"],
    }, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
