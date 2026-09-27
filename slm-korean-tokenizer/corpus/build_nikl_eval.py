#!/usr/bin/env python3
"""Scan the NIKL Newspaper Corpus 2025, check leakage, and emit the eval manifest.

NIKL is used for evaluation only. Nothing here writes to any training corpus.

The manifest fixes the article set once so every tokenizer is scored on exactly
the same articles; the benchmark reads it back as an inclusion filter.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nikl import (  # noqa: E402
    DEFAULT_ROOT,
    KNOWN_TOPICS,
    TOPIC_ALIASES,
    Article,
    character_profile,
    corpus_files,
    digest,
    iter_articles,
    short_digest,
)

TRAINING_CORPORA = {
    "tokenizer_train_real": "data/processed/tokenizer_train_real.txt",
    "mixed10": "data/processed/tokenizer_train_mixed10.txt",
    "structured_v1": "data/processed/tokenizer_train_mixed10_structured.txt",
    "structured_v2": "data/processed/tokenizer_train_mixed10_structured_v2.txt",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the NIKL external evaluation manifest.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--manifest", default="data/eval/nikl_eval_manifest.jsonl")
    parser.add_argument("--report-json", default="outputs/nikl_corpus_report.json")
    parser.add_argument("--report-markdown", default="outputs/NIKL_CORPUS_REPORT.md")
    parser.add_argument("--mode", choices=["full", "stratified"], default="full")
    parser.add_argument("--articles-per-topic", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-body-chars", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100_000)
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


def load_training_hashes() -> tuple[dict[int, int], dict[str, int], list[str]]:
    """Map a truncated line hash to a bitmask of the corpora containing it."""
    names = list(TRAINING_CORPORA)
    table: dict[int, int] = {}
    per_corpus: dict[str, int] = {}
    for index, name in enumerate(names):
        path = Path(TRAINING_CORPORA[name])
        if not path.exists():
            print(f"  (missing) {path}", file=sys.stderr)
            per_corpus[name] = 0
            continue
        bit = 1 << index
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                if stripped:
                    key = short_digest(stripped)
                    table[key] = table.get(key, 0) | bit
                    count += 1
        per_corpus[name] = count
        print(f"  {name}: {count:,} lines hashed", file=sys.stderr)
    return table, per_corpus, names


class Accumulator:
    """Incremental statistics; no article list is ever retained."""

    def __init__(self) -> None:
        self.articles = 0
        self.paragraphs = 0
        self.characters = 0
        self.profile = Counter()
        self.lengths: list[int] = []
        self.hangul_syllables: set[str] = set()
        self.unicode_chars: set[str] = set()
        self.by_topic: dict[str, Counter] = defaultdict(Counter)
        self.by_publisher: dict[str, Counter] = defaultdict(Counter)
        self.by_length: dict[str, Counter] = defaultdict(Counter)
        self.by_ascii: dict[str, Counter] = defaultdict(Counter)
        self.original_topics: dict[str, Counter] = defaultdict(Counter)

    def add(self, article: Article, body: str, paragraphs: int) -> None:
        n = len(body)
        profile = character_profile(body)
        self.articles += 1
        self.paragraphs += paragraphs
        self.characters += n
        self.lengths.append(n)
        for key, value in profile.items():
            self.profile[key] += value
        for char in body:
            if "가" <= char <= "힣":
                self.hangul_syllables.add(char)
            self.unicode_chars.add(char)
        ratio = (profile["english"] + profile["digits"]) / max(1, n)
        for bucket, key in (
            (self.by_topic, article.topic),
            (self.by_publisher, article.publisher),
            (self.by_length, length_bucket(n)),
            (self.by_ascii, ascii_bucket(ratio)),
        ):
            bucket[key]["articles"] += 1
            bucket[key]["characters"] += n
            bucket[key]["hangul"] += profile["hangul"]
            bucket[key]["ascii"] += profile["ascii"]
            bucket[key]["english"] += profile["english"]
            bucket[key]["digits"] += profile["digits"]
        self.original_topics[article.topic][article.original_topic] += 1


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, int(len(values) * p))
    return values[index]


def summarize(acc: Accumulator) -> dict:
    lengths = sorted(acc.lengths)
    chars = max(1, acc.characters)
    return {
        "articles": acc.articles,
        "paragraphs": acc.paragraphs,
        "characters": acc.characters,
        "median_article_chars": percentile(lengths, 0.5),
        "mean_article_chars": acc.characters / max(1, acc.articles),
        "p95_article_chars": percentile(lengths, 0.95),
        "p99_article_chars": percentile(lengths, 0.99),
        "max_article_chars": lengths[-1] if lengths else 0,
        "hangul_ratio_percent": 100.0 * acc.profile["hangul"] / chars,
        "ascii_ratio_percent": 100.0 * acc.profile["ascii"] / chars,
        "english_alphabet_ratio_percent": 100.0 * acc.profile["english"] / chars,
        "digit_ratio_percent": 100.0 * acc.profile["digits"] / chars,
        "unique_hangul_syllables": len(acc.hangul_syllables),
        "unique_unicode_characters": len(acc.unicode_chars),
    }


def group_table(groups: dict[str, Counter]) -> dict:
    out = {}
    for name, counts in groups.items():
        chars = max(1, counts["characters"])
        out[name] = {
            "articles": counts["articles"],
            "characters": counts["characters"],
            "mean_article_chars": counts["characters"] / max(1, counts["articles"]),
            "hangul_ratio_percent": 100.0 * counts["hangul"] / chars,
            "ascii_ratio_percent": 100.0 * counts["ascii"] / chars,
            "english_alphabet_ratio_percent": 100.0 * counts["english"] / chars,
            "digit_ratio_percent": 100.0 * counts["digits"] / chars,
        }
    return out


def markdown(report: dict) -> str:
    overall = report["overall"]
    out = [
        "# NIKL Newspaper Corpus 2025 — external evaluation corpus",
        "",
        "> Evaluation only. No NIKL text is added to any tokenizer training corpus.",
        "",
        f"Mode: **{report['mode'].upper()}**"
        + ("" if report["mode"] == "full" else f" ({report['articles_per_topic']:,} articles/topic, seed {report['seed']})"),
        "",
        "## Schema as found",
        "",
        "| Path | Role |",
        "|---|---|",
        "| `document[].id` | article id |",
        "| `document[].metadata.publisher` | newspaper |",
        "| `document[].metadata.topic` | one of nine topics |",
        "| `document[].metadata.original_topic` | publisher's own section label |",
        "| `document[].metadata.date` | publication date |",
        "| `document[].metadata.title` | collection label, one distinct value per file — **not** a headline |",
        "| `document[].paragraph[].form` | text, wrapped in `<p>…</p>` |",
        "",
        "`paragraph[0]` is the headline and is held out of the body; `paragraph[1:]` joined in order is the article body.",
        "",
        "## Scale",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| JSON files | {report['files']} |",
        f"| Articles | {overall['articles']:,} |",
        f"| Paragraphs | {overall['paragraphs']:,} |",
        f"| Body characters | {overall['characters']:,} |",
        f"| Median article | {overall['median_article_chars']:,} |",
        f"| Mean article | {overall['mean_article_chars']:,.0f} |",
        f"| p95 article | {overall['p95_article_chars']:,} |",
        f"| p99 article | {overall['p99_article_chars']:,} |",
        f"| Max article | {overall['max_article_chars']:,} |",
        f"| Hangul ratio | {overall['hangul_ratio_percent']:.2f}% |",
        f"| ASCII ratio | {overall['ascii_ratio_percent']:.2f}% |",
        f"| English letters | {overall['english_alphabet_ratio_percent']:.2f}% |",
        f"| Digits | {overall['digit_ratio_percent']:.2f}% |",
        f"| Unique Hangul syllables | {overall['unique_hangul_syllables']:,} |",
        f"| Unique Unicode characters | {overall['unique_unicode_characters']:,} |",
        "",
        "## Topics",
        "",
        "| Topic | Articles | Characters | Mean chars | Hangul | ASCII |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(report["by_topic"].items(), key=lambda kv: -kv[1]["articles"]):
        out.append(
            f"| {name} | {row['articles']:,} | {row['characters']:,} | {row['mean_article_chars']:,.0f} | "
            f"{row['hangul_ratio_percent']:.2f}% | {row['ascii_ratio_percent']:.2f}% |"
        )
    out += [
        "",
        f"The specification calls the science topic `정보통신/과학`; the data spells it "
        f"`IT/과학`. Alias map: `{TOPIC_ALIASES}`. `original_topic` is preserved per article.",
        "",
        "## Publishers",
        "",
        "| Publisher | Articles | Characters | Mean chars |",
        "|---|---:|---:|---:|",
    ]
    for name, row in sorted(report["by_publisher"].items(), key=lambda kv: -kv[1]["articles"]):
        out.append(f"| {name} | {row['articles']:,} | {row['characters']:,} | {row['mean_article_chars']:,.0f} |")

    leak = report["leakage"]
    out += [
        "",
        "## Leakage against the training corpora",
        "",
        "Source-disjoint is not automatically content-disjoint, so exact overlap is measured "
        "against every training corpus by SHA-256-class hashing.",
        "",
        "| Level | Count | Share |",
        "|---|---:|---:|",
        f"| articles whose whole body matches a training line | {leak['article_exact_overlap']:,} | "
        f"{100.0 * leak['article_exact_overlap'] / max(1, overall['articles']):.4f}% |",
        f"| paragraphs matching a training line | {leak['paragraph_exact_overlap']:,} | "
        f"{100.0 * leak['paragraph_exact_overlap'] / max(1, overall['paragraphs']):.4f}% |",
        "",
        "Article-level matches are excluded from the evaluation set. Paragraph-level matches are "
        "reported but not excluded: newspapers share short boilerplate sentences, and treating "
        "those as article leakage would remove genuine articles.",
        "",
        "| Training corpus | Articles matched |",
        "|---|---:|",
    ]
    for name, value in leak["article_overlap_by_corpus"].items():
        out.append(f"| {name} | {value:,} |")
    out += [
        "",
        f"**Evaluation set: {report['manifest_articles']:,} articles / "
        f"{report['manifest_characters']:,} characters.**",
        "",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    args = parse_args()
    paths = corpus_files(args.root)
    if not paths:
        raise SystemExit(f"no NIKL json files under {args.root}")
    print(f"found {len(paths)} NIKL files", file=sys.stderr)
    print("hashing training corpora for leakage detection...", file=sys.stderr)
    train_hashes, per_corpus, corpus_names = load_training_hashes()
    print(f"training hash table: {len(train_hashes):,} entries", file=sys.stderr)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    acc = Accumulator()
    excluded = Accumulator()
    article_overlap_by_corpus = Counter()
    article_exact = 0
    paragraph_exact = 0
    per_topic_kept = Counter()
    seen_ids: set[str] = set()
    started = time.time()

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for article in iter_articles(paths):
            body = article.body
            if len(body) < args.min_body_chars:
                continue
            paragraphs = body.count("\n") + 1 if body else 0

            # Exact-overlap detection against every training corpus.
            mask = train_hashes.get(short_digest(body), 0)
            leaked = mask != 0
            for index, name in enumerate(corpus_names):
                if mask & (1 << index):
                    article_overlap_by_corpus[name] += 1
            for paragraph in body.split("\n"):
                if paragraph and train_hashes.get(short_digest(paragraph), 0):
                    paragraph_exact += 1

            if leaked:
                article_exact += 1
                excluded.add(article, body, paragraphs)
                continue

            if args.mode == "stratified" and per_topic_kept[article.topic] >= args.articles_per_topic:
                continue

            acc.add(article, body, paragraphs)
            per_topic_kept[article.topic] += 1
            seen_ids.add(article.article_id)
            manifest.write(json.dumps({
                "article_id": article.article_id,
                "topic": article.topic,
                "original_topic": article.original_topic,
                "publisher": article.publisher,
                "date": article.date,
                "chars": len(body),
                "paragraphs": paragraphs,
                "text_hash": digest(body),
            }, ensure_ascii=False) + "\n")

            if acc.articles % args.progress_every == 0:
                elapsed = time.time() - started
                print(f"[NIKL] articles {acc.articles:,}  chars {acc.characters:,}  "
                      f"elapsed {elapsed:,.0f}s", flush=True)

    report = {
        "root": args.root,
        "files": len(paths),
        "mode": args.mode,
        "seed": args.seed,
        "articles_per_topic": args.articles_per_topic if args.mode == "stratified" else None,
        "known_topics": list(KNOWN_TOPICS),
        "topic_aliases": TOPIC_ALIASES,
        "overall": summarize(acc),
        "by_topic": group_table(acc.by_topic),
        "by_publisher": group_table(acc.by_publisher),
        "by_length_bucket": group_table(acc.by_length),
        "by_ascii_bucket": group_table(acc.by_ascii),
        "original_topic_by_topic": {k: dict(v.most_common(15)) for k, v in acc.original_topics.items()},
        "training_corpus_lines": per_corpus,
        "leakage": {
            "article_exact_overlap": article_exact,
            "paragraph_exact_overlap": paragraph_exact,
            "article_overlap_by_corpus": dict(article_overlap_by_corpus),
            "excluded_characters": excluded.characters,
            "method": "blake2b-64 over the NFC-normalized article body and each paragraph, "
                      "matched against every line of every training corpus",
        },
        "manifest": str(manifest_path),
        "manifest_articles": acc.articles,
        "manifest_characters": acc.characters,
        "elapsed_seconds": time.time() - started,
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.report_markdown).write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "articles": report["manifest_articles"],
        "characters": report["manifest_characters"],
        "article_exact_overlap": article_exact,
        "paragraph_exact_overlap": paragraph_exact,
        "elapsed_seconds": round(report["elapsed_seconds"]),
    }, indent=2))
    print(f"wrote {manifest_path}, {args.report_json}, {args.report_markdown}")


if __name__ == "__main__":
    main()
