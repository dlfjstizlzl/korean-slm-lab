#!/usr/bin/env python3
"""Stream extra clean JSON documents from bigcode/starcoderdata (data_dir="json").

Stopping is governed by *usable clean characters*, never by downloaded bytes.
Each dataset row is treated as one independent document, so the brace-balance
reconstruction the previous corpus needed is gone. StarCoder embeds its
``<reponame>``/``<filename>``/``<gh_stars>`` markers inside ``content`` itself,
so those headers are stripped per row before parsing.

Nothing is ever duplicated to reach the target: if the stream is exhausted (or
the row cap is hit) first, the script reports the real figure and stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HEADER_RE = re.compile(r"^(?:<reponame>[^\n]*\n?|<filename>[^\n]*\n?|<gh_stars>[^\n]*\n?)+")
INLINE_RE = re.compile(r"<(?:reponame|filename|gh_stars)>[^\n]*\n?")
REDACTION_RE = re.compile(r"<(?:KEY|NAME|EMAIL|PASSWORD|GH_[A-Z_]+)>")

LOCK_BASENAMES = {
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml",
    "composer.lock", "cargo.lock", "gemfile.lock", "poetry.lock",
    "packages.lock.json", "project.assets.json", "flake.lock",
}
LOCK_KEYS = {"lockfileVersion", "integrity", "resolved", "_meta"}

EXISTING_SOURCES = (
    "data/processed/tokenizer_train_mixed10_structured.txt",
    "data/processed/structured/structured_train.txt",
    "data/processed/structured/structured_valid.txt",
    "data/processed/mixed/structured_data.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest extra clean JSON from StarCoderData.")
    parser.add_argument("--dataset", default="bigcode/starcoderdata")
    parser.add_argument("--data-dir", default="json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-chars", type=int, default=3_000_000)
    parser.add_argument("--max-rows", type=int, default=3_000_000, help="Safety bound on rows scanned")
    parser.add_argument("--max-doc-chars", type=int, default=200_000)
    parser.add_argument("--min-doc-chars", type=int, default=20)
    parser.add_argument("--output", default="data/external/json/starcoder_json_extra.jsonl")
    parser.add_argument("--report-json", default="outputs/json_extra_corpus_report.json")
    parser.add_argument("--report-markdown", default="outputs/JSON_EXTRA_CORPUS_REPORT.md")
    parser.add_argument("--progress-every", type=int, default=5_000)
    return parser.parse_args()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_content(text: str) -> str:
    text = text.replace("\x00", "").lstrip("﻿")
    text = HEADER_RE.sub("", text)
    if "<filename>" in text or "<reponame>" in text or "<gh_stars>" in text:
        text = INLINE_RE.sub("", text)
    return unicodedata.normalize("NFC", text).strip()


def looks_binary(text: str) -> bool:
    if not text:
        return True
    control = sum(1 for c in text[:4000] if ord(c) < 32 and c not in "\t\n\r")
    return control / max(1, min(len(text), 4000)) > 0.01


def mechanical_dump(parsed, text: str) -> bool:
    """Detect machine-generated dumps that repeat one shape many times."""
    if isinstance(parsed, list) and len(parsed) >= 200:
        shapes = set()
        for item in parsed[:500]:
            if isinstance(item, dict):
                shapes.add(tuple(sorted(item.keys())))
            else:
                shapes.add(type(item).__name__)
            if len(shapes) > 1:
                return False
        return len(text) > 20_000
    if isinstance(parsed, dict) and len(parsed) >= 2_000:
        return True
    return False


def is_lock(parsed, basename: str) -> bool:
    if basename in LOCK_BASENAMES:
        return True
    return isinstance(parsed, dict) and bool(set(parsed.keys()) & LOCK_KEYS)


def load_existing_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for relative in EXISTING_SOURCES:
        path = root / relative
        if not path.exists():
            print(f"  (missing, skipped) {relative}", file=sys.stderr)
            continue
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                if stripped:
                    hashes.add(sha256(stripped))
                    count += 1
        print(f"  {relative}: {count:,} lines hashed", file=sys.stderr)
    # Document-level forms already kept from the first StarCoder pass.
    jsonl = root / "data/processed/structured/structured_train.jsonl"
    if jsonl.exists():
        with jsonl.open("r", encoding="utf-8") as handle:
            for raw in handle:
                record = json.loads(raw)
                hashes.add(sha256(str(record["text"])))
    return hashes


def markdown(report: dict) -> str:
    counts = report["counts"]
    dist = report["size_distribution"]
    lines = [
        "# Extra StarCoder JSON corpus",
        "",
        f"Source: `{report['dataset']}` (`data_dir=\"{report['data_dir']}\"`, streaming), "
        f"stopping rule: usable clean characters, target {report['target_chars']:,}.",
        "",
        "## Detected schema",
        "",
        "| Field | Role |",
        "|---|---|",
    ]
    for field, role in report["schema"].items():
        lines.append(f"| `{field}` | {role} |")
    lines += [
        "",
        f"No license column exists in this subset, so no license metadata could be preserved.",
        "",
        "## Funnel",
        "",
        "| Stage | Documents | Characters |",
        "|---|---:|---:|",
        f"| rows scanned | {counts['scanned']:,} | {report['raw_characters_inspected']:,} |",
        f"| parsed as valid JSON after header strip | {counts['valid']:,} | — |",
        f"| rejected: malformed | {counts['malformed']:,} | — |",
        f"| rejected: redaction placeholder | {counts['redacted']:,} | — |",
        f"| rejected: lock / dependency metadata | {counts['lock']:,} | — |",
        f"| rejected: mechanical generated dump | {counts['mechanical']:,} | — |",
        f"| rejected: too large (> {report['max_doc_chars']:,}) | {counts['too_large']:,} | — |",
        f"| rejected: too small / empty / binary | {counts['too_small'] + counts['binary']:,} | — |",
        f"| rejected: duplicate within new data | {counts['duplicate_new']:,} | — |",
        f"| rejected: overlaps existing corpus | {counts['overlap_existing']:,} | — |",
        f"| **kept** | **{counts['kept']:,}** | **{report['clean_characters']:,}** |",
        "",
        "## Kept-document shape",
        "",
        f"- JSON objects: {counts['object']:,} ({100.0 * counts['object'] / max(1, counts['kept']):.1f}%)",
        f"- JSON arrays: {counts['array']:,} ({100.0 * counts['array'] / max(1, counts['kept']):.1f}%)",
        f"- distinct repositories: {report['repository_count']:,}",
        "",
        "## Kept document size distribution (characters)",
        "",
        "| min | median | p90 | p95 | p99 | max | mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {dist['min']:,} | {dist['median']:,} | {dist['p90']:,} | {dist['p95']:,} | "
        f"{dist['p99']:,} | {dist['max']:,} | {dist['mean']:,.0f} |",
        "",
        f"Hard cap was {report['max_doc_chars']:,} characters; it rejected {counts['too_large']:,} documents.",
        "",
        "## Train / validation split",
        "",
        f"Repository-level, seed {report['split']['seed']}, target validation ratio "
        f"{report['split']['valid_ratio']:.0%}.",
        "",
        "| Split | Repositories | Documents | Characters |",
        "|---|---:|---:|---:|",
        f"| train | {report['split']['train_repositories']:,} | {report['split']['train_documents']:,} | "
        f"{report['split']['train_characters']:,} |",
        f"| validation | {report['split']['valid_repositories']:,} | {report['split']['valid_documents']:,} | "
        f"{report['split']['valid_characters']:,} |",
        "",
        f"Exact overlap between the two splits: {report['split']['exact_overlap']}.",
        "",
    ]
    if report["top_repositories"]:
        lines += ["## Most frequent repositories", "", "| Repository | Documents |", "|---|---:|"]
        for name, value in report["top_repositories"]:
            lines.append(f"| `{name}` | {value:,} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    from datasets import load_dataset

    print("hashing existing corpora for cross-dedup...", file=sys.stderr)
    existing = load_existing_hashes(root)
    print(f"existing hash set: {len(existing):,}", file=sys.stderr)

    try:
        stream = load_dataset(args.dataset, data_dir=args.data_dir, split=args.split, streaming=True)
    except Exception as exc:  # gated/auth failures must surface, not be worked around
        print(f"DATASET ACCESS ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)

    counts: Counter = Counter()
    seen_new: set[str] = set()
    sizes: list[int] = []
    repositories: Counter = Counter()
    schema: dict[str, str] = {}
    raw_chars = 0
    clean_chars = 0

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for row in stream:
            if counts["scanned"] >= args.max_rows or clean_chars >= args.target_chars:
                break
            counts["scanned"] += 1
            if not schema:
                schema = {k: type(v).__name__ for k, v in row.items()}
            content = row.get("content") or ""
            raw_chars += len(content)

            if counts["scanned"] % args.progress_every == 0:
                print(
                    f"[JSON STREAM] scanned={counts['scanned']:,} valid={counts['valid']:,} "
                    f"kept={counts['kept']:,} clean_chars={clean_chars:,} / {args.target_chars:,}",
                    flush=True,
                )

            text = clean_content(content)
            if len(text) < args.min_doc_chars:
                counts["too_small"] += 1
                continue
            if len(text) > args.max_doc_chars:
                counts["too_large"] += 1
                continue
            if looks_binary(text):
                counts["binary"] += 1
                continue
            if REDACTION_RE.search(text):
                counts["redacted"] += 1
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                counts["malformed"] += 1
                continue
            counts["valid"] += 1
            if parsed is None or (isinstance(parsed, (dict, list)) and not parsed):
                counts["too_small"] += 1
                continue
            if not isinstance(parsed, (dict, list)):
                counts["too_small"] += 1
                continue

            path = str(row.get("max_stars_repo_path") or "")
            basename = path.rsplit("/", 1)[-1].lower()
            if is_lock(parsed, basename):
                counts["lock"] += 1
                continue
            if mechanical_dump(parsed, text):
                counts["mechanical"] += 1
                continue

            key = sha256(text)
            if key in seen_new:
                counts["duplicate_new"] += 1
                continue
            if key in existing:
                counts["overlap_existing"] += 1
                continue
            seen_new.add(key)

            repository = str(row.get("max_stars_repo_name") or "")
            record = {
                "text": text,
                "source": "starcoder_json_extra",
                "repository": repository,
                "filename": path,
                "stars": row.get("max_stars_count"),
                "dataset_id": row.get("id"),
                "license": None,  # absent from this subset
                "sha256": key,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["kept"] += 1
            counts["array" if isinstance(parsed, list) else "object"] += 1
            clean_chars += len(text)
            sizes.append(len(text))
            repositories[repository] += 1

    stopped = "target reached" if clean_chars >= args.target_chars else (
        "row cap reached" if counts["scanned"] >= args.max_rows else "stream exhausted"
    )
    sizes.sort()
    n = max(1, len(sizes))
    q = lambda p: sizes[min(n - 1, int(n * p))] if sizes else 0
    report = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "schema": {
            "content": "document text (StarCoder markers embedded; stripped here)",
            "max_stars_repo_path": "file path within the repository",
            "max_stars_repo_name": "repository name",
            "max_stars_count": "repository star count",
            "id": "row identifier",
        } if "content" in schema else schema,
        "raw_schema_types": schema,
        "target_chars": args.target_chars,
        "max_doc_chars": args.max_doc_chars,
        "stop_reason": stopped,
        "raw_characters_inspected": raw_chars,
        "clean_characters": clean_chars,
        "counts": {k: counts[k] for k in (
            "scanned", "valid", "malformed", "redacted", "lock", "mechanical",
            "too_large", "too_small", "binary", "duplicate_new", "overlap_existing",
            "kept", "object", "array")},
        "size_distribution": {
            "min": sizes[0] if sizes else 0,
            "median": q(0.5),
            "p90": q(0.9),
            "p95": q(0.95),
            "p99": q(0.99),
            "max": sizes[-1] if sizes else 0,
            "mean": statistics.fmean(sizes) if sizes else 0.0,
        },
        "repository_count": len(repositories),
        "top_repositories": repositories.most_common(10),
        "output": str(output_path),
    }
    report_path = root / args.report_json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("stop_reason", "clean_characters", "counts")}, indent=2))
    print(f"wrote {output_path} and {report_path}")


if __name__ == "__main__":
    main()
