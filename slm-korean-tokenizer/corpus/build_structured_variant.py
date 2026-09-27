#!/usr/bin/env python3
"""Build ``tokenizer_train_mixed10_structured`` from local + external mixed sources.

This holds every KO-32K-mixed10 control fixed (total characters, 10% mixed
share, seed, NFC normalization, deterministic hash sampling, no repetition) and
changes exactly one thing: the composition *inside* the 10% mixed budget.
General source code is reduced so external JSON and SQL can take its place.

Unlike ``build_mixed_sources.build_mixed_pool``, scarce sources claim their lines
before ``source_code`` does. In the original ordering source_code ran first, so
6,925 of 7,477 cli_paths lines and all 24 sql lines were deduplicated away into
the source_code label; that starved the very categories this experiment targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

LOCAL_MIXED_DIR = "data/processed/mixed"
STRUCTURED_DIR = "data/processed/structured"
BASE_JSONL = "data/processed/tokenizer_train_real.jsonl"

# Scarce and experiment-critical sources are visited first so that global
# deduplication cannot reassign their lines to the source_code bucket.
SOURCE_ORDER = (
    "json",
    "sql",
    "urls_api",
    "cli_paths",
    "identifiers_technical",
    "korean_english_mixed",
    "source_code",
)

# Fractions of the *total* corpus, per the experiment plan. ``None`` means
# "take everything available" for sources known to be data-limited; the
# remainder of the mixed budget then falls to source_code.
TARGET_SHARE = {
    "json": None,
    "sql": None,
    "urls_api": None,
    "cli_paths": None,
    "identifiers_technical": 0.01,
    "korean_english_mixed": 0.02,
    "source_code": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the mixed10-structured training corpus.")
    parser.add_argument("--local-mixed-dir", default=LOCAL_MIXED_DIR)
    parser.add_argument("--structured-dir", default=STRUCTURED_DIR)
    parser.add_argument("--base-jsonl", default=BASE_JSONL)
    parser.add_argument("--label", default="mixed10_structured")
    parser.add_argument("--mixed-ratio", type=float, default=0.10)
    parser.add_argument(
        "--target-characters",
        type=int,
        default=92_514_012,
        help="Total corpus size; defaults to the measured KO-32K-mixed10 size",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="outputs/structured_variant_report.json")
    parser.add_argument(
        "--extra-json-jsonl",
        default=None,
        help="Additional JSON training rows (e.g. the harvested StarCoder extra split). "
             "Omitted by default so the original structured variant stays reproducible.",
    )
    parser.add_argument(
        "--json-share",
        type=float,
        default=None,
        help="Fraction of the total corpus to spend on the json family. Without it the json "
             "family simply takes everything available, which is the original behaviour.",
    )
    return parser.parse_args()


def normalize_line(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\t", "    ")).strip()


def text_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def score(seed: int, namespace: str, value: str) -> float:
    raw = hashlib.blake2b(f"{seed}\0{namespace}\0{value}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(raw, "big") / float(1 << 64)


def iter_local(path: Path, source: str):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.rstrip("\r\n")
            if text:
                yield text, source


def iter_structured(path: Path, wanted: set[str]):
    """Yield external structured-train rows, relabelled to their family."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            record = json.loads(raw)
            source = str(record["source"])
            family = "json" if "json" in source else "sql"
            if family in wanted:
                yield str(record["text"]), source


def collect(args: argparse.Namespace) -> dict[str, list[tuple[str, str]]]:
    """Gather candidate lines per family, globally deduplicated in SOURCE_ORDER."""
    local = Path(args.local_mixed_dir)
    structured = Path(args.structured_dir) / "structured_train.jsonl"
    buckets: dict[str, list[tuple[str, str]]] = {name: [] for name in SOURCE_ORDER}
    seen: set[bytes] = set()
    duplicates: Counter = Counter()

    def absorb(iterator, family: str) -> None:
        for text, label in iterator:
            key = text_digest(text)
            if key in seen:
                duplicates[family] += 1
                continue
            seen.add(key)
            buckets[family].append((text, label))

    for family in SOURCE_ORDER:
        if family == "json":
            absorb(iter_structured(structured, {"json"}), "json")
            if args.extra_json_jsonl:
                extra = Path(args.extra_json_jsonl)
                absorb(
                    ((str(json.loads(raw)["text"]), "starcoder_json_extra")
                     for raw in extra.open("r", encoding="utf-8")),
                    "json",
                )
            absorb(iter_local(local / "structured_data.txt", "local_structured_data"), "json")
        elif family == "sql":
            absorb(iter_structured(structured, {"sql"}), "sql")
            absorb(iter_local(local / "sql.txt", "local_sql"), "sql")
        else:
            absorb(iter_local(local / f"{family}.txt", family), family)
    collect.duplicates = dict(duplicates)  # type: ignore[attr-defined]
    return buckets


def excluded_fineweb_hashes(local: Path) -> set[bytes]:
    """FineWeb lines already spent as korean_english_mixed must not be re-counted."""
    hashes: set[bytes] = set()
    for text, _ in iter_local(local / "korean_english_mixed.txt", "korean_english_mixed"):
        hashes.add(text_digest(text))
    return hashes


def main() -> None:
    args = parse_args()
    local = Path(args.local_mixed_dir)
    out_dir = Path(args.base_jsonl).parent
    output_txt = out_dir / f"tokenizer_train_{args.label}.txt"
    output_jsonl = out_dir / f"tokenizer_train_{args.label}.jsonl"

    buckets = collect(args)
    available = {name: sum(len(t) for t, _ in rows) for name, rows in buckets.items()}
    mixed_target = int(args.target_characters * args.mixed_ratio)

    # Sources with an explicit share get it (capped by availability); the rest
    # take everything they have, and source_code absorbs whatever is left so the
    # total mixed share still lands on the controlled 10%.
    shares = dict(TARGET_SHARE)
    if args.json_share is not None:
        shares["json"] = args.json_share
    budget: dict[str, int] = {}
    for name in SOURCE_ORDER:
        share = shares[name]
        if name == "source_code":
            continue
        budget[name] = min(available[name], int(args.target_characters * share)) if share else available[name]
    budget["source_code"] = max(0, min(available["source_code"], mixed_target - sum(budget.values())))

    seen: set[bytes] = set()
    counts: Counter = Counter()
    per_label: dict[str, Counter] = {}
    per_family: dict[str, Counter] = {}

    with output_txt.open("w", encoding="utf-8") as txt, output_jsonl.open("w", encoding="utf-8") as jsonl:
        for family in SOURCE_ORDER:
            rows = buckets[family]
            allowance = budget[family]
            if allowance <= 0 or not rows:
                continue
            # Deterministic hash order + greedy fill hits the budget exactly.
            # A probability filter under-fills badly on families made of a few
            # very long documents (korean_english_mixed averages ~7.5k chars).
            ordered = sorted(rows, key=lambda row: score(args.seed, f"{args.label}-{family}", f"{row[1]}\0{row[0]}"))
            used = 0
            for text, label in ordered:
                if used >= allowance:
                    break
                key = text_digest(text)
                if key in seen:
                    continue
                seen.add(key)
                txt.write(text + "\n")
                jsonl.write(json.dumps({"text": text, "source": label}, ensure_ascii=False) + "\n")
                used += len(text)
                counts["mixed_lines"] += 1
                counts["mixed_characters"] += len(text)
                per_label.setdefault(label, Counter())
                per_label[label]["lines"] += 1
                per_label[label]["characters"] += len(text)
                per_family.setdefault(family, Counter())
                per_family[family]["lines"] += 1
                per_family[family]["characters"] += len(text)

        # Korean-centric remainder, sampled exactly as build_mixed_sources does.
        excluded = excluded_fineweb_hashes(local)
        base_available = 0
        base_path = Path(args.base_jsonl)
        with base_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                record = json.loads(raw)
                text = normalize_line(str(record["text"]))
                if record.get("source") == "fineweb" and text_digest(text) in excluded:
                    continue
                base_available += len(text)
        korean_target = max(0, args.target_characters - counts["mixed_characters"])
        korean_probability = min(1.0, korean_target / (base_available or 1))
        with base_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                record = json.loads(raw)
                text = normalize_line(str(record["text"]))
                source = str(record.get("source", "unknown"))
                key = text_digest(text)
                if source == "fineweb" and key in excluded:
                    continue
                if counts["korean_characters"] >= korean_target:
                    continue
                if score(args.seed, f"{args.label}-korean", f"{source}\0{text}") >= korean_probability:
                    continue
                if key in seen:
                    counts["duplicates_removed"] += 1
                    continue
                seen.add(key)
                txt.write(text + "\n")
                jsonl.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
                counts["korean_lines"] += 1
                counts["korean_characters"] += len(text)
                per_label.setdefault(source, Counter())
                per_label[source]["lines"] += 1
                per_label[source]["characters"] += len(text)

    total_chars = counts["mixed_characters"] + counts["korean_characters"]
    report = {
        "label": args.label,
        "seed": args.seed,
        "target_characters": args.target_characters,
        "mixed_ratio_target_percent": 100.0 * args.mixed_ratio,
        "mixed_target_characters": mixed_target,
        "available_characters_by_family": available,
        "budget_characters_by_family": budget,
        "cross_source_duplicates": getattr(collect, "duplicates", {}),
        "mixed_lines": counts["mixed_lines"],
        "mixed_characters": counts["mixed_characters"],
        "korean_lines": counts["korean_lines"],
        "korean_characters": counts["korean_characters"],
        "duplicates_removed": counts["duplicates_removed"],
        "total_lines": counts["mixed_lines"] + counts["korean_lines"],
        "total_characters": total_chars,
        "actual_mixed_percent": 100.0 * counts["mixed_characters"] / (total_chars or 1),
        "family_percent_of_total": {
            k: 100.0 * v["characters"] / (total_chars or 1) for k, v in per_family.items()
        },
        "sources": {k: dict(v) for k, v in sorted(per_label.items())},
        "txt": str(output_txt),
        "jsonl": str(output_jsonl),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "total_characters", "total_lines", "mixed_characters", "actual_mixed_percent",
        "budget_characters_by_family", "family_percent_of_total")}, indent=2, ensure_ascii=False))
    print(f"wrote {output_txt} and {output_jsonl}")


if __name__ == "__main__":
    main()
