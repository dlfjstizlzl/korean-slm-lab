#!/usr/bin/env python3
"""External Korean evaluation of every tokenizer candidate on NIKL Newspaper 2025.

NIKL is never used for training. The manifest produced by
``corpus/build_nikl_eval.py`` fixes the article set, so all candidates are scored
on exactly the same articles, and articles that exactly match a training line are
already excluded there.

The corpus is read once and each article is encoded by every tokenizer in turn,
rather than re-reading 3.4 GB per candidate. Only per-article token counts are
retained (int32), never the article text.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time

import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from benchmark_tokenizers import embedding_estimate, get_hidden_size  # noqa: E402
from common import BYTE_FALLBACK_RE, assert_control_tokens_atomic, load_tokenizer  # noqa: E402
from nikl import DEFAULT_ROOT, character_profile, corpus_files, iter_articles  # noqa: E402

CANDIDATES = {
    "Qwen original": None,
    "KO-32K-current": "outputs/real/mf2/tokenizer-ko-32000",
    "KO-32K-mixed10": "outputs/tokenizer-ko-32000-mixed10",
    "KO-32K-mixed10-structured": "outputs/tokenizer-ko-32000-mixed10-structured",
    "KO-32K-mixed10-structured-v2": "outputs/tokenizer-ko-32000-mixed10-structured-v2",
}
SUPPLEMENTAL = {
    "KO-32K-mixed10-structured-v2-json2": "outputs/tokenizer-ko-32000-mixed10-structured-v2-json2",
}

LENGTH_BUCKETS = ((0, 500), (500, 1_000), (1_000, 2_000), (2_000, 4_000), (4_000, None))
ASCII_BUCKETS = (
    # Bucketed on English letters + digits, not raw ASCII: Korean prose is ~30%
    # ASCII purely because spaces are ASCII, which puts every article in one bin.
    # Thresholds follow the measured per-article distribution (p25 1.9, median
    # 3.6, p75 6.6, p90 10.6).
    ("very low (<2%)", 0.0, 0.02),
    ("low (2-4%)", 0.02, 0.04),
    ("medium (4-7%)", 0.04, 0.07),
    ("high (7-12%)", 0.07, 0.12),
    ("very high (>=12%)", 0.12, 1.01),
)

# Prior in-domain results, for the generalization-gap table. Measured earlier in
# this project and reproduced from outputs/tokenizer_structured_v2_benchmark.json.
PRIOR = {
    "KO-32K-current": {"fineweb": 35.15, "aihub": 40.32, "independent": 34.85},
    "KO-32K-mixed10": {"fineweb": 34.66, "aihub": 40.03, "independent": 36.47},
    "KO-32K-mixed10-structured": {"fineweb": 34.61, "aihub": 39.98, "independent": 37.45},
    "KO-32K-mixed10-structured-v2": {"fineweb": 34.63, "aihub": 39.97, "independent": 37.77},
    "KO-32K-mixed10-structured-v2-json2": {"fineweb": 34.62, "aihub": 39.97, "independent": 37.55},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark tokenizers on NIKL Newspaper 2025.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--manifest", default="data/eval/nikl_eval_manifest.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--mode", choices=["full", "stratified"], default="full")
    parser.add_argument("--articles-per-topic", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--include-supplemental", action="store_true", default=True)
    parser.add_argument("--with-title", action="store_true",
                        help="Score title+body instead of body only (supplemental metric)")
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--output-json", default="outputs/nikl_tokenizer_benchmark.json")
    parser.add_argument("--output-markdown", default="outputs/NIKL_TOKENIZER_BENCHMARK.md")
    parser.add_argument("--progress-every", type=int, default=50_000)
    return parser.parse_args()


def length_bucket(n: int) -> str:
    for low, high in LENGTH_BUCKETS:
        if high is None or low <= n < high:
            return f"{low}-{high}" if high is not None else f"{low}+"
    return "unknown"


def ascii_bucket(ratio: float) -> str:
    for name, low, high in ASCII_BUCKETS:
        if low <= ratio < high:
            return name
    return ASCII_BUCKETS[-1][0]


class Fragmentation:
    """Byte-fallback accounting, using the same definition as the earlier benchmarks.

    The classification depends only on the token id, so the whole vocabulary is
    classified once into boolean lookup tables and batches are then counted with
    numpy. Doing it per token in Python would be ~2.3 billion iterations across
    the full corpus and six candidates.
    """

    def __init__(self, tokenizer) -> None:
        size = len(tokenizer)
        self.byte_flags = np.zeros(size, dtype=bool)
        self.invalid_flags = np.zeros(size, dtype=bool)
        internals = tokenizer.convert_ids_to_tokens(list(range(size)))
        for token_id, internal in enumerate(internals):
            if BYTE_FALLBACK_RE.fullmatch(str(internal)):
                self.byte_flags[token_id] = True
        # A standalone token whose decoded surface contains U+FFFD is a byte
        # fragment that is not valid UTF-8 on its own.
        for token_id in range(size):
            try:
                surface = tokenizer.decode([token_id], skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
            except TypeError:
                surface = tokenizer.decode([token_id], skip_special_tokens=False)
            if "�" in surface:
                self.invalid_flags[token_id] = True
        unk = getattr(tokenizer, "unk_token_id", None)
        self.unk_id = int(unk) if unk is not None else None
        self.tokens = 0
        self.byte_like = 0
        self.invalid = 0
        self.unknown = 0

    def add_batch(self, flat_ids) -> None:
        if len(flat_ids) == 0:
            return
        array = np.asarray(flat_ids, dtype=np.int64)
        self.tokens += int(array.size)
        self.byte_like += int(np.count_nonzero(self.byte_flags[array]))
        self.invalid += int(np.count_nonzero(self.invalid_flags[array]))
        if self.unk_id is not None:
            self.unknown += int(np.count_nonzero(array == self.unk_id))


def percentile(sorted_values, p: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * p))]


def markdown(payload: dict) -> str:
    names = payload["order"]
    overall = payload["overall"]
    mode = payload["mode"]
    out = [
        "# NIKL Newspaper 2025 — external Korean tokenizer evaluation",
        "",
        "> NIKL is used for evaluation only and was never added to any training corpus.",
        "> Articles whose body exactly matches a training line are excluded from this set.",
        "",
        f"**Evaluation mode: {mode.upper()}** — {payload['articles']:,} articles, "
        f"{payload['characters']:,} body characters. Scored text: "
        f"{'title + body' if payload['with_title'] else 'body only'}.",
        "",
        "## Overall",
        "",
        "| Tokenizer | Tokens | Reduction vs Qwen | Chars/token | Mean/article | Median | p90 | p95 | p99 | Vocab rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        row = overall[name]
        out.append(
            f"| {name} | {row['total_tokens']:,} | {row['token_reduction_percent']:.2f}% | "
            f"{row['characters_per_token']:.3f} | {row['mean_tokens_per_article']:,.1f} | "
            f"{row['median_tokens_per_article']:,} | {row['p90']:,} | {row['p95']:,} | "
            f"{row['p99']:,} | {row['embedding_vocab_rows']:,} |"
        )

    out += ["", "## Generalization gap against the in-domain benchmarks", "",
            "| Tokenizer | FineWeb | AI Hub | Independent 40 | **NIKL (external)** | NIKL − FineWeb |",
            "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        if name not in PRIOR:
            continue
        prior = PRIOR[name]
        nikl = overall[name]["token_reduction_percent"]
        out.append(
            f"| {name} | {prior['fineweb']:.2f}% | {prior['aihub']:.2f}% | {prior['independent']:.2f}% | "
            f"**{nikl:.2f}%** | {nikl - prior['fineweb']:+.2f} pp |"
        )

    out += ["", "## By topic", "", "| Topic | Articles | Qwen tokens | " +
            " | ".join(n for n in names if n != "Qwen original") + " |",
            "|---|---:|---:|" + "---:|" * (len(names) - 1)]
    for topic, row in sorted(payload["by_topic"].items(), key=lambda kv: -kv[1]["articles"]):
        cells = " | ".join(f"{row['reduction'][n]:.2f}%" for n in names if n != "Qwen original")
        out.append(f"| {topic} | {row['articles']:,} | {row['qwen_tokens']:,} | {cells} |")

    out += ["", "## Publisher robustness (reduction across 20 publishers)", "",
            "| Tokenizer | min | max | mean | std |", "|---|---:|---:|---:|---:|"]
    for name in names:
        if name == "Qwen original":
            continue
        row = payload["publisher_robustness"][name]
        out.append(f"| {name} | {row['min']:.2f}% | {row['max']:.2f}% | {row['mean']:.2f}% | {row['std']:.2f} |")

    out += ["", "## By article length", "", "| Bucket | Articles | " +
            " | ".join(n for n in names if n != "Qwen original") + " |",
            "|---|---:|" + "---:|" * (len(names) - 1)]
    for bucket, row in payload["by_length"].items():
        cells = " | ".join(f"{row['reduction'][n]:.2f}%" for n in names if n != "Qwen original")
        out.append(f"| {bucket} | {row['articles']:,} | {cells} |")

    out += ["", "## By ASCII share of the article", "", "| Bucket | Articles | " +
            " | ".join(n for n in names if n != "Qwen original") + " |",
            "|---|---:|" + "---:|" * (len(names) - 1)]
    for bucket, row in payload["by_ascii"].items():
        cells = " | ".join(f"{row['reduction'][n]:.2f}%" for n in names if n != "Qwen original")
        out.append(f"| {bucket} | {row['articles']:,} | {cells} |")

    out += ["", "## Byte fragmentation", "",
            "| Tokenizer | byte-like tokens | invalid UTF-8 surfaces | unknown |",
            "|---|---:|---:|---:|"]
    for name in names:
        row = payload["fragmentation"][name]
        out.append(
            f"| {name} | {row['byte_like_percent']:.3f}% | {row['invalid_surface_percent']:.3f}% | "
            f"{row['unknown']:,} |"
        )

    ci = payload.get("bootstrap")
    if ci:
        out += ["", "## Bootstrap confidence intervals", "",
                f"Article-level bootstrap, {ci['samples']:,} resamples, seed {ci['seed']}.", "",
                "| Comparison | Mean difference | 95% CI | Significant |", "|---|---:|---|---|"]
        for label, row in ci["comparisons"].items():
            significant = "yes" if (row["ci_low"] > 0) or (row["ci_high"] < 0) else "**no**"
            out.append(
                f"| {label} | {row['mean_difference_points']:+.4f} pp | "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {significant} |"
            )

    out += ["", "## Representative articles", "",
            f"Selection rule: for each topic, the articles whose `text_hash` sorts first "
            f"(deterministic, content-derived, independent of any tokenizer).", "",
            "| Topic | Article | Chars | " + " | ".join(names) + " |",
            "|---|---|---:|" + "---:|" * len(names)]
    for row in payload["examples"]:
        cells = " | ".join(str(row["tokens"][n]) for n in names)
        out.append(f"| {row['topic']} | `{row['article_id']}` | {row['chars']:,} | {cells} |")
    out.append("")
    return "\n".join(out) + "\n"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path}; run corpus/build_nikl_eval.py first")

    included: dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            record = json.loads(raw)
            included[record["article_id"]] = record
    print(f"manifest: {len(included):,} articles", file=sys.stderr)

    if args.mode == "stratified":
        rng = random.Random(args.seed)
        by_topic: dict[str, list[str]] = defaultdict(list)
        for article_id, record in included.items():
            by_topic[record["topic"]].append(article_id)
        keep: set[str] = set()
        for topic, ids in by_topic.items():
            ids.sort()
            rng.shuffle(ids)
            keep.update(ids[: args.articles_per_topic])
        included = {k: v for k, v in included.items() if k in keep}
        print(f"stratified subset: {len(included):,} articles", file=sys.stderr)

    config = AutoConfig.from_pretrained(args.base_model)
    hidden = get_hidden_size(config)
    tied = bool(getattr(config, "tie_word_embeddings", True))

    order = list(CANDIDATES)
    tokenizers = {"Qwen original": AutoTokenizer.from_pretrained(args.base_model)}
    for name, path in CANDIDATES.items():
        if path:
            tokenizers[name] = load_tokenizer(path)
    if args.include_supplemental:
        for name, path in SUPPLEMENTAL.items():
            if Path(path).exists():
                tokenizers[name] = load_tokenizer(path)
                order.append(name)

    invariants = {}
    for name, tokenizer in tokenizers.items():
        entry = {"tokenizer_size": len(tokenizer),
                 "embedding_vocab_rows": len(tokenizer) if name.startswith("KO-") else 151_936}
        if name.startswith("KO-"):
            entry["control_tokens_atomic"] = assert_control_tokens_atomic(tokenizer)
        entry["embedding_estimate"] = embedding_estimate(entry["embedding_vocab_rows"], hidden, tied)
        invariants[name] = entry
        print(f"  loaded {name}: vocab {len(tokenizer):,}", file=sys.stderr)

    totals = {name: 0 for name in order}
    counts_per_article: dict[str, list[int]] = {name: [] for name in order}
    frag = {name: Fragmentation(tokenizers[name]) for name in order}
    by_topic: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_publisher: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_length: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_ascii: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    topic_examples: dict[str, list] = defaultdict(list)

    characters = 0
    articles = 0
    started = time.time()
    batch: list[tuple[str, str, dict]] = []

    def flush(batch):
        nonlocal characters, articles
        if not batch:
            return
        texts = [text for _, text, _ in batch]
        encoded = {name: tokenizers[name](texts, add_special_tokens=False)["input_ids"] for name in order}
        for name in order:
            rows = encoded[name]
            frag[name].add_batch([tid for row in rows for tid in row])
        for index, (article_id, text, record) in enumerate(batch):
            n = len(text)
            profile = character_profile(text)
            ratio = (profile["english"] + profile["digits"]) / max(1, n)
            lb = length_bucket(n)
            ab = ascii_bucket(ratio)
            characters += n
            articles += 1
            for name in order:
                ids = encoded[name][index]
                token_count = len(ids)
                totals[name] += token_count
                counts_per_article[name].append(token_count)
                by_topic[record["topic"]][name] += token_count
                by_publisher[record["publisher"]][name] += token_count
                by_length[lb][name] += token_count
                by_ascii[ab][name] += token_count
            by_topic[record["topic"]]["articles"] += 1
            by_publisher[record["publisher"]]["articles"] += 1
            by_length[lb]["articles"] += 1
            by_ascii[ab]["articles"] += 1
            bucket = topic_examples[record["topic"]]
            entry = (record["text_hash"], article_id, n, {name: len(encoded[name][index]) for name in order})
            bucket.append(entry)
            bucket.sort(key=lambda item: item[0])
            del bucket[2:]

    for article in iter_articles(corpus_files(args.root)):
        record = included.get(article.article_id)
        if record is None:
            continue
        text = article.title_and_body if args.with_title else article.body
        if not text:
            continue
        batch.append((article.article_id, text, record))
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
            if articles % args.progress_every < args.batch_size:
                elapsed = time.time() - started
                print(f"[NIKL] articles {articles:,} / {len(included):,}  chars {characters:,}  "
                      f"Qwen tokens {totals['Qwen original']:,}  elapsed {elapsed:,.0f}s", flush=True)
    flush(batch)

    qwen_total = max(1, totals["Qwen original"])
    overall = {}
    for name in order:
        counts = sorted(counts_per_article[name])
        overall[name] = {
            "total_tokens": totals[name],
            "total_characters": characters,
            "characters_per_token": characters / max(1, totals[name]),
            "tokens_per_character": totals[name] / max(1, characters),
            "mean_tokens_per_article": totals[name] / max(1, articles),
            "median_tokens_per_article": percentile(counts, 0.5),
            "p90": percentile(counts, 0.90),
            "p95": percentile(counts, 0.95),
            "p99": percentile(counts, 0.99),
            "token_reduction_percent": 100.0 * (1.0 - totals[name] / qwen_total),
            "sequence_length_ratio": totals[name] / qwen_total,
            "embedding_vocab_rows": invariants[name]["embedding_vocab_rows"],
        }

    def grouped(table) -> dict:
        out = {}
        for key, row in table.items():
            qwen = max(1, row["Qwen original"])
            out[key] = {
                "articles": row["articles"],
                "qwen_tokens": row["Qwen original"],
                "tokens": {name: row[name] for name in order},
                "reduction": {name: 100.0 * (1.0 - row[name] / qwen) for name in order},
            }
        return out

    topics = grouped(by_topic)
    publishers = grouped(by_publisher)
    lengths = {k: grouped(by_length)[k] for k in sorted(by_length, key=lambda s: int(s.split("-")[0].rstrip("+")))}
    asciis = {k: grouped(by_ascii)[k] for k in [b[0] for b in ASCII_BUCKETS] if k in by_ascii}

    robustness = {}
    for name in order:
        values = [row["reduction"][name] for row in publishers.values()]
        mean = sum(values) / max(1, len(values))
        variance = sum((v - mean) ** 2 for v in values) / max(1, len(values))
        robustness[name] = {"min": min(values), "max": max(values), "mean": mean,
                            "std": math.sqrt(variance), "publishers": len(values)}

    fragmentation = {}
    for name in order:
        item = frag[name]
        fragmentation[name] = {
            "tokens": item.tokens,
            "byte_like_percent": 100.0 * item.byte_like / max(1, item.tokens),
            "invalid_surface_percent": 100.0 * item.invalid / max(1, item.tokens),
            "unknown": item.unknown,
        }

    bootstrap = None
    if args.bootstrap_samples > 0 and articles > 1:
        rng = random.Random(args.seed)
        qwen_counts = counts_per_article["Qwen original"]
        pairs = [
            ("structured-v2 minus structured-v1",
             "KO-32K-mixed10-structured-v2", "KO-32K-mixed10-structured"),
            ("structured-v1 minus mixed10",
             "KO-32K-mixed10-structured", "KO-32K-mixed10"),
            ("mixed10 minus current",
             "KO-32K-mixed10", "KO-32K-current"),
        ]
        comparisons = {}
        n = articles
        rounds = min(args.bootstrap_samples, 1_000)
        generator = np.random.default_rng(args.seed)
        qwen_array = np.asarray(qwen_counts, dtype=np.int64)
        arrays = {name: np.asarray(counts_per_article[name], dtype=np.int64) for name in order}
        # Resampling is done in numpy: a Python loop over ~1M indices for every
        # one of 1,000 rounds and 3 comparisons would be billions of operations.
        for label, a_name, b_name in pairs:
            if a_name not in arrays or b_name not in arrays:
                continue
            a_array, b_array = arrays[a_name], arrays[b_name]
            diffs = np.empty(rounds, dtype=np.float64)
            for round_index in range(rounds):
                indices = generator.integers(0, n, size=n)
                qa = int(qwen_array[indices].sum())
                diffs[round_index] = 100.0 * (int(b_array[indices].sum()) - int(a_array[indices].sum())) / max(1, qa)
            diffs.sort()
            point = 100.0 * (int(b_array.sum()) - int(a_array.sum())) / max(1, int(qwen_array.sum()))
            comparisons[label] = {
                "mean_difference_points": point,
                "ci_low": float(diffs[int(0.025 * rounds)]),
                "ci_high": float(diffs[min(rounds - 1, int(0.975 * rounds))]),
            }
        bootstrap = {"samples": rounds, "seed": args.seed,
                     "method": "article-level bootstrap resampling of the evaluation set",
                     "comparisons": comparisons}

    examples = []
    for topic in sorted(topic_examples):
        for text_hash, article_id, chars, tokens in topic_examples[topic]:
            examples.append({"topic": topic, "article_id": article_id, "chars": chars,
                             "text_hash": text_hash, "tokens": tokens})

    payload = {
        "mode": args.mode,
        "with_title": bool(args.with_title),
        "articles": articles,
        "characters": characters,
        "order": order,
        "invariants": invariants,
        "overall": overall,
        "by_topic": topics,
        "by_publisher": publishers,
        "publisher_robustness": robustness,
        "by_length": lengths,
        "by_ascii": asciis,
        "fragmentation": fragmentation,
        "bootstrap": bootstrap,
        "examples": examples,
        "prior_in_domain": PRIOR,
        "elapsed_seconds": time.time() - started,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_markdown).write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({name: round(overall[name]["token_reduction_percent"], 3) for name in order}, indent=2))
    print(f"wrote {args.output_json} and {args.output_markdown}")


if __name__ == "__main__":
    main()
